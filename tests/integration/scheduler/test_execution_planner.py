import copy
import random
from datetime import datetime, timedelta

import pytest

from marie.job.job_manager import generate_job_id, increment_uuid7str
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
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


def query_planner_for_test(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """A query planner for testing with 4 jobs and 2 mergers."""
    base_id = planner_info.base_id
    current_id = 0

    def next_id():
        nonlocal current_id
        task_id = f"{increment_uuid7str(base_id, current_id)}"
        current_id += 1
        return task_id

    root = Query(
        task_id=next_id(),
        query_str="start",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    job_a = Query(
        task_id=next_id(),
        query_str="job_a",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="executor_a://endpoint"),
    )
    job_b = Query(
        task_id=next_id(),
        query_str="job_b",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="executor_a://endpoint"),
    )
    job_c = Query(
        task_id=next_id(),
        query_str="job_c",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="executor_b://endpoint"),
    )
    job_d = Query(
        task_id=next_id(),
        query_str="job_d",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="executor_b://endpoint"),
    )

    merger1 = Query(
        task_id=next_id(),
        query_str="merger1",
        dependencies=[job_a.task_id, job_b.task_id],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(),
    )
    merger2 = Query(
        task_id=next_id(),
        query_str="merger2",
        dependencies=[job_c.task_id, job_d.task_id],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(),
    )

    final_job = Query(
        task_id=next_id(),
        query_str="final_job",
        dependencies=[merger1.task_id, merger2.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(endpoint="executor_a://endpoint"),
    )

    end = Query(
        task_id=next_id(),
        query_str="end",
        dependencies=[final_job.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    return QueryPlan(nodes=[root, job_a, job_b, job_c, job_d, merger1, merger2, final_job, end])


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
    slots = {"executor_a": 2, "executor_b": 4}
    active_dags = {"dag_1", "dag_2"}
    recently_activated_dags = {"dag_3"}
    return planner, slots, active_dags, recently_activated_dags


def test_sort_by_level(planner_fixture):
    """Jobs should be sorted by level descending."""
    planner, slots, active_dags, recently_activated_dags = planner_fixture
    job1 = create_work_info("job1", "dag_1", 1, 10)
    job2 = create_work_info("job2", "dag_1", 2, 10)

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)

    assert planned_jobs[0][1].id == "job2"
    assert planned_jobs[1][1].id == "job1"


def test_sort_by_priority(planner_fixture):
    """Jobs should be sorted by priority descending."""
    planner, slots, active_dags, recently_activated_dags = planner_fixture
    job1 = create_work_info("job1", "dag_1", 1, 10)
    job2 = create_work_info("job2", "dag_1", 1, 20)

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)

    assert planned_jobs[0][1].id == "job2"
    assert planned_jobs[1][1].id == "job1"


def test_sort_by_free_slots(planner_fixture):
    """Jobs for executors with more free slots should be prioritized."""
    planner, slots, active_dags, recently_activated_dags = planner_fixture
    job1 = create_work_info("job1", "dag_1", 1, 10, endpoint="executor_a://endpoint")  # 2 slots
    job2 = create_work_info("job2", "dag_1", 1, 10, endpoint="executor_b://endpoint")  # 4 slots

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)

    assert planned_jobs[0][1].id == "job2"
    assert planned_jobs[1][1].id == "job1"


def test_sort_by_existing_dag(planner_fixture):
    """Jobs from existing DAGs should be prioritized over new ones."""
    planner, slots, active_dags, recently_activated_dags = planner_fixture
    job1 = create_work_info("job1", "dag_1", 1, 10)  # existing
    job2 = create_work_info("job2", "dag_new", 1, 10)  # new

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)

    assert planned_jobs[0][1].id == "job1"
    assert planned_jobs[1][1].id == "job2"


def test_sort_by_estimated_runtime(planner_fixture):
    """Jobs with shorter estimated runtime should be prioritized."""
    planner, slots, active_dags, recently_activated_dags = planner_fixture
    job1 = create_work_info("job1", "dag_1", 1, 10, estimated_runtime=100.0)
    job2 = create_work_info("job2", "dag_1", 1, 10, estimated_runtime=50.0)

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)

    assert planned_jobs[0][1].id == "job2"
    assert planned_jobs[1][1].id == "job1"


def test_sort_by_burst_boost(planner_fixture):
    """Jobs from recently activated DAGs should get a burst boost."""
    planner, slots, active_dags, recently_activated_dags = planner_fixture
    job1 = create_work_info("job1", "dag_new_1", 1, 10)  # new, not recent
    job2 = create_work_info("job2", "dag_3", 1, 10)  # new, recent and boosted

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)

    # Both are new DAGs, but job2 has a burst boost, so it should be prioritized.
    assert planned_jobs[0][1].id == "job2"
    assert planned_jobs[1][1].id == "job1"


def test_complex_plan(planner_fixture):
    """Test a more complex scenario with multiple jobs and mixed criteria."""
    planner, slots, active_dags, recently_activated_dags = planner_fixture
    # job1: high level, low prio
    # job2: low level, high prio
    # job3: different executor with more slots
    # job4: new DAG, no boost
    # job5: short runtime
    # job6: new DAG with burst boost
    job1 = create_work_info("j1", "dag_1", 3, 5, endpoint="executor_a://ep")
    job2 = create_work_info("j2", "dag_1", 1, 15, endpoint="executor_a://ep")
    job3 = create_work_info("j3", "dag_2", 1, 10, endpoint="executor_b://ep")
    job4 = create_work_info("j4", "dag_new", 1, 10, endpoint="executor_a://ep")
    job5 = create_work_info("j5", "dag_2", 1, 10, endpoint="executor_a://ep", estimated_runtime=10.0)
    job6 = create_work_info("j6", "dag_3", 1, 10, endpoint="executor_a://ep")  # new, boosted

    jobs = [job1, job2, job3, job4, job5, job6]
    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)
    planned_ids = [wi.id for _, wi in planned_jobs]

    # Expected order with new logic (slots > priority):
    # 1. j1 (highest level)
    # 2. j3 (most free slots)
    # 3. j2 (highest priority)
    # 4. j5 (existing DAG, shortest runtime)
    # 5. j6 (new DAG with boost)
    # 6. j4 (new DAG, no boost)
    assert planned_ids == ["j1", "j3", "j2", "j5", "j6", "j4"]


def test_another_complex_plan(planner_fixture):
    """Test another complex scenario to confirm sorting logic."""
    planner, slots, active_dags, recently_activated_dags = planner_fixture

    # jA: existing DAG, high priority, but for an executor with few slots.
    # jB: new DAG, low priority, but for an executor with many slots.
    # jC: existing DAG, medium priority, but very high job level (critical path).
    # jD: new DAG, burst-boosted, medium priority.
    # jE: existing DAG, medium priority, but very short runtime.
    # jF: new DAG, no boost, lowest priority.
    jA = create_work_info("jA", "dag_1", 2, 20, endpoint="executor_a://ep")
    jB = create_work_info("jB", "dag_new_1", 2, 5, endpoint="executor_b://ep")
    jC = create_work_info("jC", "dag_1", 5, 10, endpoint="executor_a://ep")
    jD = create_work_info("jD", "dag_3", 2, 10, endpoint="executor_a://ep")  # boosted
    jE = create_work_info("jE", "dag_2", 2, 10, endpoint="executor_a://ep", estimated_runtime=10.0)
    jF = create_work_info("jF", "dag_new_2", 2, 5, endpoint="executor_a://ep")

    jobs = [jA, jB, jC, jD, jE, jF]
    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)
    planned_ids = [wi.id for _, wi in planned_jobs]

    # Expected order based on GlobalPriorityExecutionPlanner logic (slots > priority):
    # 1. jC (highest level: 5)
    # 2. jB (level 2, most free slots: 4)
    # 3. jA (level 2, prio 20)
    # 4. jE (level 2, prio 10, existing DAG, shortest runtime)
    # 5. jD (level 2, prio 10, new DAG with boost)
    # 6. jF (level 2, prio 5)
    assert planned_ids == ["jC", "jB", "jA", "jE", "jD", "jF"]


def test_large_complex_plan_top_6(planner_fixture):
    """Test with 24 jobs and assert the top 6 are prioritized correctly."""
    planner, slots, active_dags, recently_activated_dags = planner_fixture
    jobs = []

    # Create 6 jobs that should be at the top
    jobs.append(create_work_info("top1_level", "dag_1", 10, 1))
    jobs.append(create_work_info("top2_prio", "dag_1", 9, 100))
    jobs.append(create_work_info("top3_slots", "dag_2", 9, 50, endpoint="executor_b://ep"))
    jobs.append(create_work_info("top4_existing", "dag_1", 9, 50, endpoint="executor_a://ep"))
    jobs.append(create_work_info("top5_runtime", "dag_2", 9, 50, endpoint="executor_a://ep", estimated_runtime=1.0))
    jobs.append(create_work_info("top6_boosted", "dag_3", 9, 50, endpoint="executor_a://ep"))

    # Create 18 other jobs with lower priority/level
    for i in range(18):
        jobs.append(
            create_work_info(f"other_{i}", f"dag_other_{i}", 1, 1, endpoint="executor_a://ep")
        )

    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)
    top_6_planned_ids = [wi.id for _, wi in planned_jobs[:6]]

    # Expected order of the top 6 (slots > priority):
    # 1. top1_level (highest level)
    # 2. top3_slots (level 9, most free slots)
    # 3. top2_prio (level 9, highest priority)
    # 4. top5_runtime (level 9, existing DAG, shortest runtime)
    # 5. top4_existing (level 9, existing DAG, infinite runtime)
    # 6. top6_boosted (level 9, new DAG with boost)
    expected_top_6_ids = [
        "top1_level",
        "top3_slots",
        "top2_prio",
        "top5_runtime",
        "top4_existing",
        "top6_boosted",
    ]

    assert top_6_planned_ids == expected_top_6_ids


def test_edge_case_and_tie_breaking(planner_fixture):
    """
    Tests edge cases that could break the planner:
    - Jobs with identical top-level priorities.
    - High-priority job for an executor with zero slots.
    - Mix of None and infinite runtimes.
    - Sort stability with identical jobs.
    """
    planner, _, active_dags, recently_activated_dags = planner_fixture
    slots = {"executor_a": 1, "executor_b": 2, "executor_c": 0}

    jobs = [
        # High-priority job for an executor with ZERO slots. Should be de-prioritized.
        create_work_info("zero_slots_job", "dag_1", 5, 100, endpoint="executor_c://ep"),

        # A lower-priority job for an executor that HAS slots. Should come first.
        create_work_info("available_slots_job", "dag_1", 5, 90, endpoint="executor_b://ep"),

        # A set of jobs to test runtime and boost tie-breaking.
        # All have the same level (4) and priority (50).
        create_work_info("tie_breaker_runtime_inf", "dag_2", 4, 50, endpoint="executor_a://ep",
                         estimated_runtime=float('inf')),
        create_work_info("tie_breaker_runtime_none", "dag_2", 4, 50, endpoint="executor_a://ep",
                         estimated_runtime=None),
        create_work_info("tie_breaker_shortest_rt", "dag_2", 4, 50, endpoint="executor_a://ep", estimated_runtime=10.0),

        # Test if an existing DAG is always prioritized over a new, boosted DAG.
        create_work_info("existing_dag_job", "dag_1", 3, 50, endpoint="executor_a://ep"),
        create_work_info("new_boosted_dag_job", "dag_3", 3, 50, endpoint="executor_a://ep"),
        # dag_3 is in recently_activated_dags

        # Two identical jobs to test sort stability. Their relative order should be preserved.
        create_work_info("identical_1", "dag_1", 2, 50, endpoint="executor_a://ep"),
        create_work_info("identical_2", "dag_1", 2, 50, endpoint="executor_a://ep"),
    ]

    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)
    planned_ids = [wi.id for _, wi in planned_jobs]

    # relative to other jobs at the same level.
    expected_order = [
        # 1. Level 5, 2 slots
        "available_slots_job",
        # 2. Level 5, 0 slots
        "zero_slots_job",
        # 3. Level 4, ordered by runtime
        "tie_breaker_shortest_rt",
        # 4 & 5. Runtimes are inf, order between them is not guaranteed.
        "tie_breaker_runtime_inf",
        "tie_breaker_runtime_none",
        # 6. Level 3, existing DAG
        "existing_dag_job",
        # 7. Level 3, new boosted DAG
        "new_boosted_dag_job",
        # 8 & 9. Identical jobs, stable sort preserves order
        "identical_1",
        "identical_2",
    ]

    # Check the top N jobs for the key behaviors
    assert planned_ids[0] == "available_slots_job"
    assert planned_ids[1] == "zero_slots_job"
    assert planned_ids[2] == "tie_breaker_shortest_rt"
    assert set(planned_ids[3:5]) == {"tie_breaker_runtime_inf", "tie_breaker_runtime_none"}
    assert planned_ids[5] == "existing_dag_job"
    assert planned_ids[6] == "new_boosted_dag_job"
    assert planned_ids[7:9] == ["identical_1", "identical_2"]


def test_with_planner_jobs(planner_fixture):
    """Test planning jobs generated from a query planner."""
    planner, slots, _, _ = planner_fixture
    # 1. Setup for planner
    QueryPlanRegistry.register("test_planner", query_planner_for_test)
    planner_name = "test_planner"
    dag_id = generate_job_id()

    base_work_info = WorkInfo(
        id=dag_id, name=planner_name, priority=10, state="created",
        retry_limit=0, start_after=datetime.now(), expire_in_seconds=3600,
        data={"metadata": {}}, retry_delay=0, retry_backoff=False,
        keep_until=datetime.now() + timedelta(days=1),
        dag_id=dag_id, job_level=0,
    )

    planner_info = PlannerInfo(name=planner_name, base_id=dag_id)
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

        endpoint = "default://endpoint"
        if isinstance(node.definition, ExecutorEndpointQueryDefinition):
            endpoint = node.definition.endpoint
        wi.data["metadata"]["on"] = endpoint
        all_dag_nodes.append((endpoint, wi))

    # Assume root node is complete, so the next level of jobs is ready.
    root_task_id = sorted_nodes_ids[0]
    ready_jobs = [
        (endpoint, wi)
        for endpoint, wi in all_dag_nodes
        if wi.dependencies == [root_task_id]
    ]

    assert len(ready_jobs) == 4

    # Plan these ready jobs
    planned_jobs = planner.plan(ready_jobs, slots, set(), set())
    planned_job_queries = [node_dict[job[1].id].query_str for job in planned_jobs]

    # executor_b has 4 slots, executor_a has 2. Jobs for executor_b should come first.
    assert "job_c" in planned_job_queries[:2]
    assert "job_d" in planned_job_queries[:2]
    assert "job_a" in planned_job_queries[2:]
    assert "job_b" in planned_job_queries[2:]


def test_minimize_dags_with_6_dags_and_tie_breaking(planner_fixture):
    """
    Tests that with 6 DAGs of equal level and priority, the planner clears
    jobs from active DAGs before starting jobs from new DAGs.
    """
    planner, slots, _, recently_activated_dags = planner_fixture
    active_dags = {'active_dag_1', 'active_dag_2', 'active_dag_3'}

    # All jobs have the same level, priority, and endpoint to force tie-breaking on the 'is_new' status.
    jobs = [
        create_work_info("job_active_1", "active_dag_1", 5, 50, endpoint="executor_a://ep"),
        create_work_info("job_active_2", "active_dag_2", 5, 50, endpoint="executor_a://ep"),
        create_work_info("job_active_3", "active_dag_3", 5, 50, endpoint="executor_a://ep"),
        create_work_info("job_new_1", "new_dag_1", 5, 50, endpoint="executor_a://ep"),
        create_work_info("job_new_2", "new_dag_2", 5, 50, endpoint="executor_a://ep"),
        create_work_info("job_new_3", "new_dag_3", 5, 50, endpoint="executor_a://ep"),
    ]

    # Shuffle the jobs to ensure the sorting is not dependent on the initial order.
    random.shuffle(jobs)

    planned_jobs = planner.plan(jobs, slots, active_dags, recently_activated_dags)
    planned_ids = [wi.id for _, wi in planned_jobs]

    # The first 3 jobs should be from the active DAGs.
    # The next 3 jobs should be from the new DAGs.
    active_job_ids = {"job_active_1", "job_active_2", "job_active_3"}
    new_job_ids = {"job_new_1", "job_new_2", "job_new_3"}

    assert set(planned_ids[:3]) == active_job_ids
    assert set(planned_ids[3:]) == new_job_ids
