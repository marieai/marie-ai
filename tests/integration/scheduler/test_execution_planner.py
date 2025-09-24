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
    return planner, slots, active_dags


def test_sort_by_level(planner_fixture):
    """Jobs should be sorted by level descending."""
    planner, slots, active_dags = planner_fixture
    job1 = create_work_info("job1", "dag_1", 1, 10)
    job2 = create_work_info("job2", "dag_1", 2, 10)

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags)

    assert planned_jobs[0][1].id == "job2"
    assert planned_jobs[1][1].id == "job1"


def test_sort_by_priority(planner_fixture):
    """Jobs should be sorted by priority descending."""
    planner, slots, active_dags = planner_fixture
    job1 = create_work_info("job1", "dag_1", 1, 10)
    job2 = create_work_info("job2", "dag_1", 1, 20)

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags)

    assert planned_jobs[0][1].id == "job2"
    assert planned_jobs[1][1].id == "job1"


def test_sort_by_free_slots(planner_fixture):
    """Jobs for executors with more free slots should be prioritized."""
    planner, slots, active_dags = planner_fixture
    job1 = create_work_info("job1", "dag_1", 1, 10, endpoint="executor_a://endpoint")  # 2 slots
    job2 = create_work_info("job2", "dag_1", 1, 10, endpoint="executor_b://endpoint")  # 4 slots

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags)

    assert planned_jobs[0][1].id == "job2"
    assert planned_jobs[1][1].id == "job1"


def test_sort_by_existing_dag(planner_fixture):
    """Jobs from existing DAGs should be prioritized over new ones."""
    planner, slots, active_dags = planner_fixture
    job1 = create_work_info("job1", "dag_1", 1, 10)  # existing
    job2 = create_work_info("job2", "dag_new", 1, 10)  # new

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags)

    assert planned_jobs[0][1].id == "job1"
    assert planned_jobs[1][1].id == "job2"


def test_sort_by_estimated_runtime(planner_fixture):
    """Jobs with shorter estimated runtime should be prioritized."""
    planner, slots, active_dags = planner_fixture
    job1 = create_work_info("job1", "dag_1", 1, 10, estimated_runtime=100.0)
    job2 = create_work_info("job2", "dag_1", 1, 10, estimated_runtime=50.0)

    jobs = [job1, job2]
    planned_jobs = planner.plan(jobs, slots, active_dags)

    assert planned_jobs[0][1].id == "job2"
    assert planned_jobs[1][1].id == "job1"


def test_complex_plan(planner_fixture):
    """Pure ordering: runnable(exist) → runnable(new) → blocked, with tie-breaks:
       level ↓, priority ↓, free_slots ↓, est_runtime ↑, FIFO."""
    planner, slots, active_dags = planner_fixture

    # job1: higher level
    job1 = create_work_info("j1", "dag_1", 3, 5, endpoint="executor_a://ep")
    # job2: same DAG as j1, lower level but higher priority
    job2 = create_work_info("j2", "dag_1", 1, 15, endpoint="executor_a://ep")
    # job3: different executor with more slots (tie-breaker after priority)
    job3 = create_work_info("j3", "dag_2", 1, 10, endpoint="executor_b://ep")
    # job4: NEW DAG → should be after all existing runnable jobs
    job4 = create_work_info("j4", "dag_new", 1, 10, endpoint="executor_a://ep")
    # job5: existing DAG, same prio as j3, shorter runtime → tie-breaker after free_slots
    job5 = create_work_info("j5", "dag_2", 1, 10, endpoint="executor_a://ep", estimated_runtime=10.0)

    jobs = [job1, job2, job3, job4, job5]
    planned_jobs = planner.plan(jobs, slots, active_dags)
    planned_ids = [wi.id for _, wi in planned_jobs]

    # Expected order with new logic:
    # 1. j1 (existing, highest level)
    # 2. j2 (existing, higher priority than j3/j5)
    # 3. j3 (existing, ties j5 on prio; wins on more free slots)
    # 4. j5 (existing, loses to j3 on free slots; runtime tie-breaker would apply if needed)
    # 5. j4 (new DAG)
    assert planned_ids == ["j1", "j2", "j3", "j5", "j4"]


def test_blocked_jobs_last(planner_fixture):
    planner, slots, active_dags = planner_fixture
    # Force executor_a to have 0 slots; executor_b has >0 in the fixture
    slots = {**slots, "executor_a": 0}

    j1 = create_work_info("a1", "dag_1", 2, 10, endpoint="executor_a://ep")  # blocked
    j2 = create_work_info("b1", "dag_2", 1, 5, endpoint="executor_b://ep")  # runnable

    planned = planner.plan([j1, j2], slots, active_dags)
    ids = [wi.id for _, wi in planned]
    # runnable first, blocked last
    assert ids == ["b1", "a1"]


def test_another_complex_plan(planner_fixture):
    """Pure ordering: runnable(existing) → runnable(new) → blocked.
       Within each: level ↓, priority ↓, free_slots ↓, est_runtime ↑, FIFO."""
    planner, slots, active_dags = planner_fixture

    jA = create_work_info("jA", "dag_1", 2, 20, endpoint="executor_a://ep")
    jB = create_work_info("jB", "dag_new_1", 2, 5, endpoint="executor_b://ep")
    jC = create_work_info("jC", "dag_1", 5, 10, endpoint="executor_a://ep")
    jD = create_work_info("jD", "dag_3", 2, 10, endpoint="executor_a://ep")  # no boost in new logic
    jE = create_work_info("jE", "dag_2", 2, 10, endpoint="executor_a://ep", estimated_runtime=10.0)
    jF = create_work_info("jF", "dag_new_2", 2, 5, endpoint="executor_a://ep")

    jobs = [jA, jB, jC, jD, jE, jF]
    planned_jobs = planner.plan(jobs, slots, active_dags)
    planned_ids = [wi.id for _, wi in planned_jobs]

    # Expected: existing first (C, A, E), then new (D, B, F).
    assert planned_ids == ["jC", "jA", "jE", "jD", "jB", "jF"]


def test_large_complex_plan_top_6(planner_fixture):
    """Pure ordering: runnable(existing) → runnable(new) → blocked.
       Within each: level ↓, priority ↓, free_slots ↓, est_runtime ↑, FIFO."""
    planner, slots, active_dags = planner_fixture
    jobs = []

    # Top candidates
    jobs.append(create_work_info("top1_level", "dag_1", 10, 1))  # existing, highest level
    jobs.append(create_work_info("top2_prio", "dag_1", 9, 100))  # existing, highest priority
    jobs.append(create_work_info("top3_slots", "dag_2", 9, 50, endpoint="executor_b://ep"))  # existing, more slots
    jobs.append(create_work_info("top4_existing", "dag_1", 9, 50, endpoint="executor_a://ep"))  # existing
    jobs.append(create_work_info("top5_runtime", "dag_2", 9, 50, endpoint="executor_a://ep",
                                 estimated_runtime=1.0))  # existing, shortest runtime
    jobs.append(create_work_info("top6_boosted", "dag_3", 9, 50, endpoint="executor_a://ep"))  # NEW DAG (no boost)

    # Fillers
    for i in range(18):
        jobs.append(create_work_info(f"other_{i}", f"dag_other_{i}", 1, 1, endpoint="executor_a://ep"))

    planned_jobs = planner.plan(jobs, slots, active_dags)
    top_6_planned_ids = [wi.id for _, wi in planned_jobs[:6]]

    expected_top_6_ids = [
        "top1_level",
        "top2_prio",  # priority outranks free-slots within same level
        "top3_slots",
        "top5_runtime",  # same prio/level/executor as top4 → shorter runtime first
        "top4_existing",
        "top6_boosted",  # new DAGs come after all existing
    ]

    assert top_6_planned_ids == expected_top_6_ids


def test_edge_case_and_tie_breaking(planner_fixture):
    """
    Edge cases for the pure-ordering planner:
    - Jobs with identical top-level priorities.
    - High-priority job for an executor with zero slots (blocked → last).
    - Mix of None and infinite runtimes (treated as ∞; FIFO ties).
    - Sort stability with identical jobs.
    """
    planner, _, active_dags = planner_fixture
    slots = {"executor_a": 1, "executor_b": 2, "executor_c": 0}

    jobs = [
        # Blocked: executor_c has 0 slots → should be LAST overall.
        create_work_info("zero_slots_job", "dag_1", 5, 100, endpoint="executor_c://ep"),

        # Runnable: same level (5), but lower priority than blocked one → still ranks before it because runnable > blocked.
        create_work_info("available_slots_job", "dag_1", 5, 90, endpoint="executor_b://ep"),

        # Level 4 ties (same level/priority/executor): runtime asc, then FIFO
        create_work_info("tie_breaker_runtime_inf", "dag_2", 4, 50, endpoint="executor_a://ep",
                         estimated_runtime=float('inf')),
        create_work_info("tie_breaker_runtime_none", "dag_2", 4, 50, endpoint="executor_a://ep",
                         estimated_runtime=None),
        create_work_info("tie_breaker_shortest_rt", "dag_2", 4, 50, endpoint="executor_a://ep", estimated_runtime=10.0),

        # Existing vs new at level 3: existing first
        create_work_info("existing_dag_job", "dag_1", 3, 50, endpoint="executor_a://ep"),
        create_work_info("new_boosted_dag_job", "dag_3", 3, 50, endpoint="executor_a://ep"),

        # Identical jobs (level 2, same DAG/priority/runtime/executor): FIFO preserved
        create_work_info("identical_1", "dag_1", 2, 50, endpoint="executor_a://ep"),
        create_work_info("identical_2", "dag_1", 2, 50, endpoint="executor_a://ep"),
    ]

    planned = planner.plan(jobs, slots, active_dags)
    ids = [wi.id for _, wi in planned]

    # Full expected order:
    expected = [
        # Runnable + existing
        "available_slots_job",  # level 5, runnable
        "tie_breaker_shortest_rt",  # level 4, shortest runtime among ties
        "tie_breaker_runtime_inf",  # level 4, ∞ runtime (ties with None) → FIFO before _none
        "tie_breaker_runtime_none",  # level 4, treated as ∞
        "existing_dag_job",  # level 3, existing
        "identical_1",  # level 2, existing; FIFO
        "identical_2",  # level 2, existing; FIFO
        # Runnable + new
        "new_boosted_dag_job",  # level 3, new → after all existing runnable
        # Blocked (no capacity)
        "zero_slots_job",  # last
    ]

    assert ids == expected


def test_with_planner_jobs(planner_fixture):
    """Test planning jobs generated from a query planner."""
    planner, slots, _ = planner_fixture
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
    planned_jobs = planner.plan(ready_jobs, slots, set())
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
    planner, slots, _ = planner_fixture
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

    planned_jobs = planner.plan(jobs, slots, active_dags)
    planned_ids = [wi.id for _, wi in planned_jobs]

    # The first 3 jobs should be from the active DAGs.
    # The next 3 jobs should be from the new DAGs.
    active_job_ids = {"job_active_1", "job_active_2", "job_active_3"}
    new_job_ids = {"job_new_1", "job_new_2", "job_new_3"}

    assert set(planned_ids[:3]) == active_job_ids
    assert set(planned_ids[3:]) == new_job_ids


def test_noop_jobs_are_not_blocked(planner_fixture):
    """NOOP jobs should never be considered blocked, even with no slots."""
    planner, slots, active_dags = planner_fixture

    slots.pop("noop", None)

    noop_job = create_work_info("noop_job", "dag_1", 1, 10, endpoint="noop://noop")
    regular_job_blocked = create_work_info(
        "blocked_job", "dag_1", 1, 10, endpoint="executor_c://endpoint"
    )

    slots["executor_c"] = 0

    jobs = [noop_job, regular_job_blocked]

    # With exclude_blocked=True, the noop job should still be planned.
    planned_jobs = planner.plan(jobs, slots, active_dags, exclude_blocked=True)

    assert len(planned_jobs) == 1
