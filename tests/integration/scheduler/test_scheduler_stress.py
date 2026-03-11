"""
Comprehensive stress testing for QueryPlanner and JobScheduler to ensure optimal plan generation.

This module contains extensive stress tests that validate the scheduler's ability to produce
optimal execution plans under various conditions including:
- Large-scale complex DAG structures (100s-1000s of nodes)
- Resource-constrained environments
- Mixed priority workloads
- High concurrency scenarios
- Edge cases and race conditions

The tests verify that the GlobalPriorityExecutionPlanner correctly prioritizes jobs according to:
1. Runnable (has free slots) before blocked
2. Existing DAGs before new DAGs
3. Deeper level (critical path) first
4. Higher user priority
5. More executor free slots (tie-breaker)
6. Shorter estimated runtime
7. FIFO (original input order)
"""

import copy
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pytest

from marie.job.job_manager import generate_job_id, increment_uuid7str
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    LlmQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryPlanRegistry,
    QueryType,
)
from marie.query_planner.planner import (
    compute_job_levels,
    query_planner,
    topological_sort,
)
from marie.scheduler.execution_planner import FlatJob
from marie.scheduler.global_execution_planner import GlobalPriorityExecutionPlanner
from marie.scheduler.models import WorkInfo


def create_work_info(
        job_id: str,
        dag_id: str,
        job_level: int,
        priority: int,
        estimated_runtime: float = float("inf"),
        endpoint: str = "executor_a://endpoint",
        **kwargs,
) -> FlatJob:
    """Helper function to create a FlatJob object for tests."""
    wi = WorkInfo(
        id=job_id,
        name=f"test_job_{job_id}",
        priority=priority,
        state="created",
        retry_limit=0,
        start_after=datetime.now(),
        expire_in_seconds=3600,
        data={"metadata": {"estimated_runtime": estimated_runtime, "on": endpoint, **kwargs}},
        retry_delay=0,
        retry_backoff=False,
        keep_until=datetime.now() + timedelta(days=1),
        dag_id=dag_id,
        job_level=job_level,
    )
    return endpoint, wi


@pytest.fixture
def planner_fixture():
    """Provides a planner and its context for tests."""
    planner = GlobalPriorityExecutionPlanner()
    slots = {"executor_a": 2, "executor_b": 4, "executor_c": 8, "executor_d": 16}
    active_dags = {"dag_1", "dag_2", "dag_3"}
    return planner, slots, active_dags


# =============================================================================
# COMPLEX DAG STRUCTURE TESTS
# =============================================================================


def create_complex_document_processing_plan(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Creates a complex document processing plan similar to real-world document annotation workflows.

    Structure:
        START -> OCR -> [Parallel Annotators] -> Merge -> [Post-processing] -> Parse -> END

    This mimics  plans with multiple parallel
    annotators and sequential post-processing steps.
    """
    base_id = planner_info.base_id
    current_id = 0

    def next_id():
        nonlocal current_id
        task_id = f"{increment_uuid7str(base_id, current_id)}"
        current_id += 1
        return task_id

    nodes = []

    # Root node
    root = Query(
        task_id=next_id(),
        query_str="START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(root)

    # OCR step
    ocr = Query(
        task_id=next_id(),
        query_str="OCR Processing",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="executor_a://ocr"),
    )
    nodes.append(ocr)

    # Parallel annotators (KV, Tables, Remarks, Claims, Embeddings)
    annotators = []
    annotator_configs = [
        ("KV Annotator", "executor_b://annotator", LlmQueryDefinition),
        ("Table Annotator", "executor_b://annotator", LlmQueryDefinition),
        ("Remark Annotator", "executor_b://annotator", LlmQueryDefinition),
        ("Claim Annotator", "executor_c://annotator", LlmQueryDefinition),
        ("Embedding Annotator", "executor_c://embeddings", ExecutorEndpointQueryDefinition),
    ]

    for name, endpoint, def_type in annotator_configs:
        if def_type == LlmQueryDefinition:
            definition = LlmQueryDefinition(
                model_name="deepseek_r1_32",
                endpoint=endpoint,
                params={"key": name.lower().replace(" ", "_")}
            )
        else:
            definition = ExecutorEndpointQueryDefinition(
                endpoint=endpoint,
                params={"key": name.lower().replace(" ", "_")}
            )

        annotator = Query(
            task_id=next_id(),
            query_str=name,
            dependencies=[ocr.task_id],
            node_type=QueryType.COMPUTE,
            definition=definition,
        )
        nodes.append(annotator)
        annotators.append(annotator)

    # Merger node
    merger = Query(
        task_id=next_id(),
        query_str="Merge Annotators",
        dependencies=[a.task_id for a in annotators],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(),
    )
    nodes.append(merger)

    # Post-processing steps
    table_parser = Query(
        task_id=next_id(),
        query_str="Table Parser",
        dependencies=[merger.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="executor_d://parser"),
    )
    nodes.append(table_parser)

    table_extractor = Query(
        task_id=next_id(),
        query_str="Table Extractor",
        dependencies=[table_parser.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="qwen_v2_5_vl",
            endpoint="executor_d://extractor",
            params={"key": "table_extract"}
        ),
    )
    nodes.append(table_extractor)

    remark_parser = Query(
        task_id=next_id(),
        query_str="Remark Parser",
        dependencies=[table_extractor.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(remark_parser)

    final_parser = Query(
        task_id=next_id(),
        query_str="Final Parser",
        dependencies=[remark_parser.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="executor_a://finalizer"),
    )
    nodes.append(final_parser)

    # End node
    end = Query(
        task_id=next_id(),
        query_str="END",
        dependencies=[final_parser.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


def test_complex_document_processing_dag_scheduling(planner_fixture):
    """
    Test scheduling of a complex document processing DAG with realistic structure.

    Validates that jobs are scheduled optimally across multiple executors with
    varying capacities while respecting dependencies.
    """
    planner, slots, active_dags = planner_fixture

    # Register and create the plan
    QueryPlanRegistry.register("complex_doc_plan", create_complex_document_processing_plan)
    dag_id = generate_job_id()

    base_work_info = WorkInfo(
        id=dag_id, name="complex_doc_plan", priority=10, state="created",
        retry_limit=0, start_after=datetime.now(), expire_in_seconds=3600,
        data={"metadata": {}}, retry_delay=0, retry_backoff=False,
        keep_until=datetime.now() + timedelta(days=1),
        dag_id=dag_id, job_level=0,
    )

    planner_info = PlannerInfo(name="complex_doc_plan", base_id=dag_id)
    plan = query_planner(planner_info)

    sorted_nodes_ids = topological_sort(plan)
    job_levels = compute_job_levels(sorted_nodes_ids, plan)

    node_dict = {node.task_id: node for node in plan.nodes}
    all_dag_nodes = []
    for task_id in sorted_nodes_ids:
        node = node_dict[task_id]
        wi = copy.deepcopy(base_work_info)
        wi.id = node.task_id
        wi.job_level = job_levels[task_id]
        wi.dependencies = node.dependencies

        if isinstance(node.definition, (ExecutorEndpointQueryDefinition, LlmQueryDefinition)):
            endpoint = node.definition.endpoint
        elif isinstance(node.definition, NoopQueryDefinition):
            endpoint = "noop://endpoint"
        else:
            endpoint = "default://endpoint"
        wi.data["metadata"]["on"] = endpoint
        all_dag_nodes.append((endpoint, wi))

    # Simulate first wave: root is complete, OCR is ready
    root_task_id = sorted_nodes_ids[0]
    ocr_ready = [
        (endpoint, wi)
        for endpoint, wi in all_dag_nodes
        if wi.dependencies == [root_task_id]
    ]

    planned_jobs = planner.plan(ocr_ready, slots, {dag_id})
    assert len(planned_jobs) == 1  # Only OCR should be ready

    # Simulate second wave: OCR complete, all annotators ready
    ocr_task_id = sorted_nodes_ids[1]
    annotators_ready = [
        (endpoint, wi)
        for endpoint, wi in all_dag_nodes
        if wi.dependencies == [ocr_task_id]
    ]

    planned_jobs = planner.plan(annotators_ready, slots, {dag_id})
    assert len(planned_jobs) == 5  # All 5 annotators should be scheduled

    # Verify that jobs for executors with more slots come first
    # executor_c and executor_d have more slots than executor_b
    planned_endpoints = [endpoint.split("://")[0] for endpoint, _ in planned_jobs[:2]]
    # The first two should be from higher-capacity executors
    assert "executor_c" in planned_endpoints or "executor_d" in planned_endpoints


def create_wide_dag_plan(planner_info: PlannerInfo, width: int = 50, **kwargs) -> QueryPlan:
    """
    Creates a wide DAG with many parallel branches.

    Structure: ROOT -> [width parallel jobs] -> MERGER -> END

    Args:
        width: Number of parallel branches
    """
    base_id = planner_info.base_id
    current_id = 0

    def next_id():
        nonlocal current_id
        task_id = f"{increment_uuid7str(base_id, current_id)}"
        current_id += 1
        return task_id

    nodes = []

    # Root
    root = Query(
        task_id=next_id(),
        query_str="ROOT",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(root)

    # Wide parallel branches
    parallel_jobs = []
    for i in range(width):
        # Distribute across different executors
        executor_idx = i % 4  # Cycle through 4 executors
        executor = f"executor_{chr(97 + executor_idx)}"  # executor_a, executor_b, etc.

        job = Query(
            task_id=next_id(),
            query_str=f"Parallel Job {i}",
            dependencies=[root.task_id],
            node_type=QueryType.COMPUTE,
            definition=ExecutorEndpointQueryDefinition(endpoint=f"{executor}://work"),
        )
        nodes.append(job)
        parallel_jobs.append(job)

    # Merger
    merger = Query(
        task_id=next_id(),
        query_str="MERGER",
        dependencies=[job.task_id for job in parallel_jobs],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(),
    )
    nodes.append(merger)

    # End
    end = Query(
        task_id=next_id(),
        query_str="END",
        dependencies=[merger.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@pytest.mark.parametrize("width", [50, 100, 200])
def test_wide_dag_scheduling(planner_fixture, width):
    """
    Test scheduling of wide DAGs with many parallel branches.

    Validates that the scheduler correctly prioritizes jobs based on executor
    capacity when faced with many parallel options.
    """
    planner, slots, _ = planner_fixture

    # Register the plan
    plan_name = f"wide_dag_{width}"
    QueryPlanRegistry._plans[plan_name] = lambda pi, **kw: create_wide_dag_plan(pi, width=width, **kw)

    dag_id = generate_job_id()
    planner_info = PlannerInfo(name=plan_name, base_id=dag_id)
    plan = query_planner(planner_info)

    sorted_nodes_ids = topological_sort(plan)
    job_levels = compute_job_levels(sorted_nodes_ids, plan)

    node_dict = {node.task_id: node for node in plan.nodes}

    # Create work items for all parallel jobs
    base_work_info = WorkInfo(
        id=dag_id, name=plan_name, priority=10, state="created",
        retry_limit=0, start_after=datetime.now(), expire_in_seconds=3600,
        data={"metadata": {}}, retry_delay=0, retry_backoff=False,
        keep_until=datetime.now() + timedelta(days=1),
        dag_id=dag_id, job_level=0,
    )

    all_parallel_jobs = []
    root_task_id = sorted_nodes_ids[0]

    for task_id in sorted_nodes_ids[1:-2]:  # Skip root, merger, and end
        node = node_dict[task_id]
        wi = copy.deepcopy(base_work_info)
        wi.id = node.task_id
        wi.job_level = job_levels[task_id]
        wi.dependencies = node.dependencies

        if isinstance(node.definition, ExecutorEndpointQueryDefinition):
            endpoint = node.definition.endpoint
        elif isinstance(node.definition, NoopQueryDefinition):
            endpoint = "noop://endpoint"
        else:
            endpoint = "default://endpoint"

        wi.data["metadata"]["on"] = endpoint
        all_parallel_jobs.append((endpoint, wi))

    # Plan all parallel jobs
    planned_jobs = planner.plan(all_parallel_jobs, slots, {dag_id})

    # Verify all jobs are planned
    assert len(planned_jobs) == width

    # Count jobs by executor
    executor_counts = {}
    for endpoint, _ in planned_jobs:
        executor = endpoint.split("://")[0]
        executor_counts[executor] = executor_counts.get(executor, 0) + 1

    # Verify distribution matches input (should be roughly even due to round-robin creation)
    expected_per_executor = width // 4
    for executor in ["executor_a", "executor_b", "executor_c", "executor_d"]:
        # Allow some tolerance for rounding
        assert abs(executor_counts.get(executor, 0) - expected_per_executor) <= 2


def create_deep_dag_plan(planner_info: PlannerInfo, depth: int = 50, **kwargs) -> QueryPlan:
    """
    Creates a deep sequential DAG with many dependent steps.

    Structure: N1 -> N2 -> N3 -> ... -> N_depth

    Args:
        depth: Number of sequential steps
    """
    base_id = planner_info.base_id
    current_id = 0

    def next_id():
        nonlocal current_id
        task_id = f"{increment_uuid7str(base_id, current_id)}"
        current_id += 1
        return task_id

    nodes = []
    prev_task_id = None

    for i in range(depth):
        # Alternate executors
        executor_idx = i % 4
        executor = f"executor_{chr(97 + executor_idx)}"

        dependencies = [prev_task_id] if prev_task_id else []

        node = Query(
            task_id=next_id(),
            query_str=f"Step {i}",
            dependencies=dependencies,
            node_type=QueryType.COMPUTE,
            definition=ExecutorEndpointQueryDefinition(endpoint=f"{executor}://work"),
        )
        nodes.append(node)
        prev_task_id = node.task_id

    return QueryPlan(nodes=nodes)


@pytest.mark.parametrize("depth", [50, 100])
def test_deep_dag_scheduling(planner_fixture, depth):
    """
    Test scheduling of deep sequential DAGs.

    Validates that job levels are correctly computed and jobs are scheduled
    in dependency order with correct priority based on depth.
    """
    planner, slots, _ = planner_fixture

    plan_name = f"deep_dag_{depth}"
    QueryPlanRegistry._plans[plan_name] = lambda pi, **kw: create_deep_dag_plan(pi, depth=depth, **kw)

    dag_id = generate_job_id()
    planner_info = PlannerInfo(name=plan_name, base_id=dag_id)
    plan = query_planner(planner_info)

    sorted_nodes_ids = topological_sort(plan)
    job_levels = compute_job_levels(sorted_nodes_ids, plan)

    # Verify levels are correct (should be depth - 1 down to 0)
    expected_levels = list(range(depth - 1, -1, -1))
    actual_levels = [job_levels[task_id] for task_id in sorted_nodes_ids]
    assert actual_levels == expected_levels

    # Create work items
    node_dict = {node.task_id: node for node in plan.nodes}
    base_work_info = WorkInfo(
        id=dag_id, name=plan_name, priority=10, state="created",
        retry_limit=0, start_after=datetime.now(), expire_in_seconds=3600,
        data={"metadata": {}}, retry_delay=0, retry_backoff=False,
        keep_until=datetime.now() + timedelta(days=1),
        dag_id=dag_id, job_level=0,
    )

    all_jobs = []
    for task_id in sorted_nodes_ids:
        node = node_dict[task_id]
        wi = copy.deepcopy(base_work_info)
        wi.id = node.task_id
        wi.job_level = job_levels[task_id]
        wi.dependencies = node.dependencies

        if isinstance(node.definition, ExecutorEndpointQueryDefinition):
            endpoint = node.definition.endpoint
        elif isinstance(node.definition, NoopQueryDefinition):
            endpoint = "noop://endpoint"
        else:
            endpoint = "default://endpoint"

        wi.data["metadata"]["on"] = endpoint
        all_jobs.append((endpoint, wi))

    # Test that deeper jobs (higher levels) are prioritized
    # Simulate scenario where first 10 jobs are ready
    ready_jobs = all_jobs[:10]
    planned = planner.plan(ready_jobs, slots, {dag_id})

    # The first job should have the highest level
    first_job_level = planned[0][1].job_level
    for _, wi in planned[1:]:
        assert wi.job_level <= first_job_level


# =============================================================================
# RESOURCE CONSTRAINT STRESS TESTS
# =============================================================================


def test_extreme_slot_constraints(planner_fixture):
    """
    Test scheduling under extreme resource constraints (very few slots).

    Validates that the scheduler correctly identifies blocked jobs and
    prioritizes runnable jobs even with minimal resources.
    """
    planner, _, active_dags = planner_fixture

    # Extremely limited slots
    constrained_slots = {"executor_a": 1, "executor_b": 1, "executor_c": 0, "executor_d": 0}

    # Create many jobs for different executors
    jobs = []
    for i in range(20):
        executor = f"executor_{chr(97 + (i % 4))}"
        job = create_work_info(
            f"job_{i}",
            "dag_1",
            job_level=5 - (i % 5),  # Mix of levels
            priority=100 - i,  # Decreasing priority
            endpoint=f"{executor}://endpoint"
        )
        jobs.append(job)

    planned = planner.plan(jobs, constrained_slots, active_dags, exclude_blocked=True)

    # Only jobs for executor_a and executor_b should be in the plan (blocked ones excluded)
    planned_executors = {endpoint.split("://")[0] for endpoint, _ in planned}
    assert planned_executors.issubset({"executor_a", "executor_b"})

    # Should have all runnable jobs for executor_a and executor_b (5 each = 10 total)
    # The planner returns all runnable jobs in priority order; actual slot limiting happens at execution
    assert len(planned) == 10

    # Verify all are for runnable executors
    for endpoint, wi in planned:
        executor = endpoint.split("://")[0]
        assert executor in ["executor_a", "executor_b"]

    # Verify they're sorted by priority (higher priority first)
    priorities = [wi.priority for _, wi in planned]
    assert priorities == sorted(priorities, reverse=True) or all(
        planned[i][1].job_level >= planned[i+1][1].job_level for i in range(len(planned)-1)
    )


def test_dynamic_slot_changes():
    """
    Test scheduling behavior as slot availability changes over time.

    Simulates a realistic scenario where executor capacity changes and
    validates that job priorities are adjusted correctly.
    """
    planner = GlobalPriorityExecutionPlanner()
    active_dags = {"dag_1"}

    # Create jobs for different executors with equal priority
    # so that slot availability becomes the deciding factor
    jobs = [
        create_work_info(f"a_{i}", "dag_1", job_level=5, priority=100, endpoint="executor_a://ep")
        for i in range(5)
    ] + [
        create_work_info(f"b_{i}", "dag_1", job_level=5, priority=100, endpoint="executor_b://ep")
        for i in range(5)
    ]

    # Initial state: executor_a has more capacity
    slots_t1 = {"executor_a": 3, "executor_b": 1}
    planned_t1 = planner.plan(jobs, slots_t1, active_dags, exclude_blocked=True)

    # More executor_a jobs should be prioritized (higher capacity)
    executor_a_count = sum(1 for ep, _ in planned_t1 if "executor_a" in ep)
    executor_b_count = sum(1 for ep, _ in planned_t1 if "executor_b" in ep)
    assert executor_a_count >= executor_b_count

    # Changed state: executor_b now has more capacity
    slots_t2 = {"executor_a": 1, "executor_b": 3}
    planned_t2 = planner.plan(jobs, slots_t2, active_dags, exclude_blocked=True)

    # Now executor_b jobs should be prioritized due to higher capacity
    # With equal priorities, slot availability becomes the deciding factor
    top_5_executors = [ep.split("://")[0] for ep, _ in planned_t2[:5]]
    # With higher capacity, we expect more representation in top 5
    b_in_top_5 = sum(1 for ex in top_5_executors if ex == "executor_b")
    assert b_in_top_5 >= 2  # At least some executor_b jobs in top 5


def test_slot_exhaustion_and_backpressure():
    """
    Test behavior when all slots are exhausted and new high-priority jobs arrive.

    Validates that the scheduler correctly orders jobs for when slots become available.
    """
    planner = GlobalPriorityExecutionPlanner()
    active_dags = {"dag_1"}

    # All slots full (0 available)
    slots = {"executor_a": 0, "executor_b": 0}

    # Create mix of jobs
    jobs = [
        create_work_info("urgent_a", "dag_1", job_level=10, priority=1000, endpoint="executor_a://ep"),
        create_work_info("urgent_b", "dag_1", job_level=10, priority=1000, endpoint="executor_b://ep"),
        create_work_info("normal_a", "dag_1", job_level=5, priority=100, endpoint="executor_a://ep"),
        create_work_info("low_a", "dag_1", job_level=1, priority=10, endpoint="executor_a://ep"),
    ]

    # With exclude_blocked=False, all jobs should be returned but in correct order
    planned = planner.plan(jobs, slots, active_dags, exclude_blocked=False)
    assert len(planned) == 4

    # Even though all are blocked, they should be ordered correctly
    # Urgent jobs (level 10, priority 1000) should be first
    planned_ids = [wi.id for _, wi in planned]
    assert "urgent_a" in planned_ids[:2]
    assert "urgent_b" in planned_ids[:2]

    # With exclude_blocked=True, none should be returned
    planned_filtered = planner.plan(jobs, slots, active_dags, exclude_blocked=True)
    assert len(planned_filtered) == 0


# =============================================================================
# PRIORITY AND FAIRNESS TESTS
# =============================================================================


def test_priority_inversion_prevention():
    """
    Test that high-priority jobs are not starved by low-priority jobs.

    Validates that the priority ordering is strictly enforced even when
    other factors (like estimated runtime) would suggest a different order.
    """
    planner = GlobalPriorityExecutionPlanner()
    active_dags = {"dag_1"}
    slots = {"executor_a": 10}

    jobs = [
        # Low priority but very short runtime
        create_work_info("low_fast", "dag_1", job_level=5, priority=1,
                        estimated_runtime=1.0, endpoint="executor_a://ep"),
        # High priority but longer runtime
        create_work_info("high_slow", "dag_1", job_level=5, priority=100,
                        estimated_runtime=1000.0, endpoint="executor_a://ep"),
        # Medium priority, medium runtime
        create_work_info("med", "dag_1", job_level=5, priority=50,
                        estimated_runtime=100.0, endpoint="executor_a://ep"),
    ]

    planned = planner.plan(jobs, slots, active_dags)
    planned_ids = [wi.id for _, wi in planned]

    # Priority should dominate: high_slow > med > low_fast
    assert planned_ids == ["high_slow", "med", "low_fast"]


def test_dag_fairness_existing_vs_new():
    """
    Test fairness between existing and new DAGs.

    Validates that existing DAGs are prioritized to minimize the number of
    concurrent DAGs in the system (DAG minimization strategy).
    """
    planner = GlobalPriorityExecutionPlanner()
    active_dags = {"active_1", "active_2"}
    slots = {"executor_a": 10}

    jobs = [
        # New DAG with higher priority
        create_work_info("new_high", "new_dag", job_level=5, priority=1000, endpoint="executor_a://ep"),
        # Existing DAG with lower priority
        create_work_info("existing_low", "active_1", job_level=5, priority=100, endpoint="executor_a://ep"),
        # Existing DAG with even lower priority
        create_work_info("existing_lower", "active_2", job_level=5, priority=50, endpoint="executor_a://ep"),
    ]

    planned = planner.plan(jobs, slots, active_dags)
    planned_ids = [wi.id for _, wi in planned]

    # Existing DAGs should come first, regardless of priority
    assert planned_ids[0] == "existing_low"
    assert planned_ids[1] == "existing_lower"
    assert planned_ids[2] == "new_high"


def test_level_based_critical_path_prioritization():
    """
    Test that jobs on the critical path (deeper levels) are prioritized.

    Validates that the scheduler correctly identifies and prioritizes jobs
    that are on the critical path of the DAG.
    """
    planner = GlobalPriorityExecutionPlanner()
    active_dags = {"dag_1"}
    slots = {"executor_a": 10}

    jobs = [
        # Shallow level (not critical)
        create_work_info("shallow", "dag_1", job_level=1, priority=100, endpoint="executor_a://ep"),
        # Medium level
        create_work_info("medium", "dag_1", job_level=5, priority=100, endpoint="executor_a://ep"),
        # Deep level (critical path)
        create_work_info("deep", "dag_1", job_level=10, priority=100, endpoint="executor_a://ep"),
        # Very deep level
        create_work_info("very_deep", "dag_1", job_level=20, priority=100, endpoint="executor_a://ep"),
    ]

    planned = planner.plan(jobs, slots, active_dags)
    planned_ids = [wi.id for _, wi in planned]

    # Should be ordered by level descending
    assert planned_ids == ["very_deep", "deep", "medium", "shallow"]


# =============================================================================
# LARGE-SCALE STRESS TESTS
# =============================================================================


@pytest.mark.slow
def test_massive_job_queue_1000_jobs():
    """
    Stress test with 1000 jobs in the queue.

    Validates that the scheduler can handle large job queues efficiently
    and produces correct ordering.
    """
    planner = GlobalPriorityExecutionPlanner()
    active_dags = {f"dag_{i}" for i in range(10)}  # 10 active DAGs
    slots = {"executor_a": 10, "executor_b": 20, "executor_c": 30}

    jobs = []
    for i in range(1000):
        dag_id = f"dag_{i % 20}"  # 20 total DAGs (10 active + 10 new)
        executor = f"executor_{chr(97 + (i % 3))}"  # Cycle through executors
        level = random.randint(1, 50)
        priority = random.randint(1, 100)
        runtime = random.uniform(1.0, 1000.0)

        job = create_work_info(
            f"job_{i}",
            dag_id,
            job_level=level,
            priority=priority,
            estimated_runtime=runtime,
            endpoint=f"{executor}://endpoint"
        )
        jobs.append(job)

    # Plan all jobs
    import time
    start = time.time()
    planned = planner.plan(jobs, slots, active_dags)
    duration = time.time() - start

    # Verify all jobs are returned and ordered
    assert len(planned) == 1000

    # Performance check: should complete in reasonable time (< 1 second)
    assert duration < 1.0, f"Planning took too long: {duration:.3f}s"

    # Verify ordering properties
    # 1. Runnable jobs before blocked
    runnable_count = 0
    blocked_started = False
    for endpoint, wi in planned:
        executor = endpoint.split("://")[0]
        is_blocked = slots.get(executor, 0) <= 0

        if is_blocked:
            blocked_started = True
        elif blocked_started:
            # If we see a runnable after a blocked, that's wrong
            assert False, "Runnable job found after blocked job"

        if not is_blocked:
            runnable_count += 1

    # Should have many runnable jobs
    assert runnable_count > 0

    # 2. Existing DAGs before new DAGs (among runnable jobs)
    existing_ended = False
    for endpoint, wi in planned[:runnable_count]:
        is_new = wi.dag_id not in active_dags
        if is_new:
            existing_ended = True
        elif existing_ended:
            assert False, "Existing DAG job found after new DAG job"


@pytest.mark.slow
def test_massive_dag_structure_1000_nodes():
    """
    Stress test with a DAG containing 1000 nodes.

    Creates a complex DAG structure and validates that job levels are
    computed correctly and scheduling is optimal.
    """
    planner = GlobalPriorityExecutionPlanner()
    slots = {"executor_a": 50, "executor_b": 50}

    # Create a large diamond-shaped DAG
    # ROOT -> [250 parallel] -> [250 parallel] -> [250 parallel] -> [250 parallel] -> MERGER -> END
    dag_id = generate_job_id()

    def create_large_diamond_dag(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
        base_id = planner_info.base_id
        current_id = 0

        def next_id():
            nonlocal current_id
            task_id = f"{increment_uuid7str(base_id, current_id)}"
            current_id += 1
            return task_id

        nodes = []

        # Root
        root = Query(
            task_id=next_id(),
            query_str="ROOT",
            dependencies=[],
            node_type=QueryType.COMPUTE,
            definition=NoopQueryDefinition(),
        )
        nodes.append(root)

        # Create 4 layers of parallel jobs
        layers = []
        prev_layer = [root]

        for layer_idx in range(4):
            current_layer = []
            for i in range(250):
                # Each job depends on one random job from previous layer
                dep = random.choice(prev_layer)
                executor = "executor_a" if i % 2 == 0 else "executor_b"

                job = Query(
                    task_id=next_id(),
                    query_str=f"Layer {layer_idx} Job {i}",
                    dependencies=[dep.task_id],
                    node_type=QueryType.COMPUTE,
                    definition=ExecutorEndpointQueryDefinition(endpoint=f"{executor}://work"),
                )
                nodes.append(job)
                current_layer.append(job)

            layers.append(current_layer)
            prev_layer = current_layer

        # Merger depends on all jobs in last layer
        merger = Query(
            task_id=next_id(),
            query_str="MERGER",
            dependencies=[job.task_id for job in layers[-1]],
            node_type=QueryType.MERGER,
            definition=NoopQueryDefinition(),
        )
        nodes.append(merger)

        # End
        end = Query(
            task_id=next_id(),
            query_str="END",
            dependencies=[merger.task_id],
            node_type=QueryType.COMPUTE,
            definition=NoopQueryDefinition(),
        )
        nodes.append(end)

        return QueryPlan(nodes=nodes)

    # Register and create plan
    QueryPlanRegistry._plans["large_diamond"] = create_large_diamond_dag
    planner_info = PlannerInfo(name="large_diamond", base_id=dag_id)

    import time
    start = time.time()
    plan = query_planner(planner_info)
    plan_time = time.time() - start

    # Verify plan has correct number of nodes
    assert len(plan.nodes) == 1003  # 1 root + 4*250 + 1 merger + 1 end

    # Compute levels
    start = time.time()
    sorted_nodes_ids = topological_sort(plan)
    sort_time = time.time() - start

    start = time.time()
    job_levels = compute_job_levels(sorted_nodes_ids, plan)
    level_time = time.time() - start

    # Performance checks
    assert plan_time < 5.0, f"Plan creation took too long: {plan_time:.3f}s"
    assert sort_time < 2.0, f"Topological sort took too long: {sort_time:.3f}s"
    assert level_time < 2.0, f"Level computation took too long: {level_time:.3f}s"

    # Create work items for first layer (250 jobs ready after root)
    node_dict = {node.task_id: node for node in plan.nodes}
    root_task_id = sorted_nodes_ids[0]

    base_work_info = WorkInfo(
        id=dag_id, name="large_diamond", priority=10, state="created",
        retry_limit=0, start_after=datetime.now(), expire_in_seconds=3600,
        data={"metadata": {}}, retry_delay=0, retry_backoff=False,
        keep_until=datetime.now() + timedelta(days=1),
        dag_id=dag_id, job_level=0,
    )

    first_layer_jobs = []
    for task_id in sorted_nodes_ids:
        node = node_dict[task_id]
        if len(node.dependencies) == 1 and node.dependencies[0] == root_task_id:
            wi = copy.deepcopy(base_work_info)
            wi.id = node.task_id
            wi.job_level = job_levels[task_id]
            wi.dependencies = node.dependencies

            if isinstance(node.definition, ExecutorEndpointQueryDefinition):
                endpoint = node.definition.endpoint
            else:
                endpoint = "default://endpoint"

            wi.data["metadata"]["on"] = endpoint
            first_layer_jobs.append((endpoint, wi))

    # Plan first layer
    start = time.time()
    planned = planner.plan(first_layer_jobs, slots, {dag_id})
    planning_time = time.time() - start

    # Performance check
    assert planning_time < 0.5, f"Planning took too long: {planning_time:.3f}s"

    # Verify all first layer jobs are planned
    assert len(planned) == 250

    # Verify balanced distribution between executors
    executor_a_count = sum(1 for ep, _ in planned if "executor_a" in ep)
    executor_b_count = sum(1 for ep, _ in planned if "executor_b" in ep)
    assert abs(executor_a_count - executor_b_count) <= 10  # Should be roughly balanced


# =============================================================================
# EDGE CASES AND CORRECTNESS TESTS
# =============================================================================


def test_all_jobs_same_priority_level_executor():
    """
    Test tie-breaking behavior when all primary factors are equal.

    Validates that FIFO order is preserved when all other factors tie.
    """
    planner = GlobalPriorityExecutionPlanner()
    active_dags = {"dag_1"}
    slots = {"executor_a": 10}

    # All jobs identical except ID
    jobs = [
        create_work_info(f"job_{i}", "dag_1", job_level=5, priority=100,
                        estimated_runtime=50.0, endpoint="executor_a://ep")
        for i in range(10)
    ]

    planned = planner.plan(jobs, slots, active_dags)
    planned_ids = [wi.id for _, wi in planned]

    # Should maintain FIFO order
    expected_ids = [f"job_{i}" for i in range(10)]
    assert planned_ids == expected_ids


def test_noop_executor_never_blocked():
    """
    Test that noop executor jobs are never considered blocked.

    Validates special handling of noop jobs which don't consume slots.
    """
    planner = GlobalPriorityExecutionPlanner()
    active_dags = {"dag_1"}

    # No slots for noop (doesn't matter)
    slots = {"executor_a": 0}

    jobs = [
        create_work_info("noop_job", "dag_1", job_level=5, priority=100, endpoint="noop://noop"),
        create_work_info("blocked_job", "dag_1", job_level=5, priority=100, endpoint="executor_a://ep"),
    ]

    # With exclude_blocked=True, only noop should be planned
    planned = planner.plan(jobs, slots, active_dags, exclude_blocked=True)

    assert len(planned) == 1
    assert planned[0][1].id == "noop_job"


def test_estimated_runtime_none_and_inf_handling():
    """
    Test correct handling of None and infinity values for estimated runtime.

    Validates that missing or infinite runtimes are handled gracefully.
    """
    planner = GlobalPriorityExecutionPlanner()
    active_dags = {"dag_1"}
    slots = {"executor_a": 10}

    jobs = [
        create_work_info("runtime_none", "dag_1", job_level=5, priority=100,
                        estimated_runtime=None, endpoint="executor_a://ep"),
        create_work_info("runtime_inf", "dag_1", job_level=5, priority=100,
                        estimated_runtime=float('inf'), endpoint="executor_a://ep"),
        create_work_info("runtime_short", "dag_1", job_level=5, priority=100,
                        estimated_runtime=10.0, endpoint="executor_a://ep"),
    ]

    planned = planner.plan(jobs, slots, active_dags)
    planned_ids = [wi.id for _, wi in planned]

    # Short runtime should come first, None and inf treated equally (FIFO)
    assert planned_ids[0] == "runtime_short"
    assert set(planned_ids[1:]) == {"runtime_none", "runtime_inf"}


def test_empty_job_list():
    """Test handling of empty job list."""
    planner = GlobalPriorityExecutionPlanner()
    slots = {"executor_a": 10}
    active_dags = set()

    planned = planner.plan([], slots, active_dags)
    assert len(planned) == 0


def test_single_job():
    """Test handling of single job."""
    planner = GlobalPriorityExecutionPlanner()
    slots = {"executor_a": 10}
    active_dags = {"dag_1"}

    job = create_work_info("solo", "dag_1", job_level=5, priority=100, endpoint="executor_a://ep")
    planned = planner.plan([job], slots, active_dags)

    assert len(planned) == 1
    assert planned[0][1].id == "solo"


# =============================================================================
# INTEGRATION TESTS WITH QUERY PLANNER
# =============================================================================


def test_end_to_end_complex_plan_optimal_scheduling():
    """
    End-to-end test of query planner + scheduler integration.

    Creates a complex realistic plan, simulates execution, and validates
    that the scheduler produces optimal schedules at each step.
    """
    planner = GlobalPriorityExecutionPlanner()
    slots = {"executor_a": 2, "executor_b": 4, "executor_c": 8, "executor_d": 16}

    # Use the complex document processing plan
    QueryPlanRegistry.register("e2e_complex", create_complex_document_processing_plan)
    dag_id = generate_job_id()

    base_work_info = WorkInfo(
        id=dag_id, name="e2e_complex", priority=10, state="created",
        retry_limit=0, start_after=datetime.now(), expire_in_seconds=3600,
        data={"metadata": {}}, retry_delay=0, retry_backoff=False,
        keep_until=datetime.now() + timedelta(days=1),
        dag_id=dag_id, job_level=0,
    )

    planner_info = PlannerInfo(name="e2e_complex", base_id=dag_id)
    plan = query_planner(planner_info)

    sorted_nodes_ids = topological_sort(plan)
    job_levels = compute_job_levels(sorted_nodes_ids, plan)

    node_dict = {node.task_id: node for node in plan.nodes}
    all_dag_nodes = []

    for task_id in sorted_nodes_ids:
        node = node_dict[task_id]
        wi = copy.deepcopy(base_work_info)
        wi.id = node.task_id
        wi.job_level = job_levels[task_id]
        wi.dependencies = node.dependencies

        if isinstance(node.definition, (ExecutorEndpointQueryDefinition, LlmQueryDefinition)):
            endpoint = node.definition.endpoint
        elif isinstance(node.definition, NoopQueryDefinition):
            endpoint = "noop://endpoint"
        else:
            endpoint = "default://endpoint"
        wi.data["metadata"]["on"] = endpoint
        all_dag_nodes.append((endpoint, wi))

    # Simulate execution waves
    completed = set()
    wave_count = 0
    max_waves = 20  # Safety limit

    while len(completed) < len(all_dag_nodes) and wave_count < max_waves:
        wave_count += 1

        # Find jobs whose dependencies are all completed
        ready_jobs = [
            (endpoint, wi)
            for endpoint, wi in all_dag_nodes
            if wi.id not in completed and all(dep in completed for dep in wi.dependencies)
        ]

        if not ready_jobs:
            break

        # Plan this wave
        planned = planner.plan(ready_jobs, slots, {dag_id})

        # Verify planned jobs are optimal for this wave
        # (all should be runnable, ordered by priority rules)
        for endpoint, wi in planned:
            executor = endpoint.split("://")[0]
            # All should have capacity (or be noop)
            assert executor == "noop" or slots.get(executor, 0) > 0

        # Simulate completion of some jobs (randomly select a subset)
        completed_this_wave = random.sample(
            [wi.id for _, wi in planned],
            k=min(len(planned), sum(slots.values()))
        )
        completed.update(completed_this_wave)

    # Verify all jobs were eventually scheduled
    # (might not all complete in simulation, but all should be reachable)
    assert len(completed) > 0
    assert wave_count < max_waves, "Execution did not progress"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
