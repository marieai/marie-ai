import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import marie.scheduler.psql as scheduler_psql
from marie.scheduler.psql import (
    SLOT_POLL_INTERVAL,
    ControlFlowBatchResult,
    DispatchCycleResult,
    PostgreSQLJobScheduler,
)
from marie.scheduler.services import ControlFlowExecutionOutcome
from marie.scheduler.state import WorkState


def build_scheduler() -> PostgreSQLJobScheduler:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler.running = True
    scheduler.frontier_batch_size = 100
    scheduler.max_concurrent_dags = 16
    scheduler.lease_ttl_seconds = 5
    scheduler.active_dags = {}
    scheduler._next_priority_refresh_at = float("inf")
    scheduler.submission_service = SimpleNamespace(
        submission_count=0,
        queue_size=0,
        pending_count=0,
    )
    scheduler.priority_refresh_interval_seconds = 5.0
    scheduler._event_queue = asyncio.Queue()
    scheduler._gateway_ready_event = None
    scheduler._paused = False
    scheduler._semaphore_store = MagicMock()
    scheduler._sem_default_ttl = 30
    scheduler.lease_owner = "scheduler-1"
    scheduler.gateway_instance_id = "gateway-1"
    scheduler.cycle_log_every = 100
    scheduler._request_priority_refresh = MagicMock()
    scheduler._wait_for_dispatch_wake = AsyncMock(return_value=False)
    scheduler.frontier = SimpleNamespace(
        reap_expired_soft_leases=AsyncMock(return_value=0),
        peek_ready=AsyncMock(return_value=[]),
        summary=MagicMock(return_value={}),
        dag_remaining_counts=MagicMock(return_value={}),
        take=AsyncMock(return_value=[]),
        release_lease_local=AsyncMock(),
        compact_ready_heap=AsyncMock(return_value=0),
    )
    scheduler.execution_planner = SimpleNamespace(plan=MagicMock(return_value=[]))
    scheduler.dag_service = SimpleNamespace(admit_dag=AsyncMock(return_value=True))
    scheduler._lease_jobs_db = AsyncMock(return_value=set())
    scheduler._release_lease_db = AsyncMock()
    scheduler._reconcile_db_lease_shortfall = AsyncMock(return_value=0)
    scheduler._reserve_semaphore_slots = AsyncMock(return_value=set())
    scheduler._activate_from_lease_db = AsyncMock(return_value={})
    scheduler._activate_and_enqueue_job = AsyncMock(return_value=True)
    scheduler._handle_dispatch_failure = AsyncMock()
    scheduler.get_dag_by_id = AsyncMock(return_value=None)
    scheduler.notify_event = AsyncMock(return_value=True)
    return scheduler


@pytest.mark.asyncio
async def test_dispatch_cycle_returns_short_poll_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_scheduler()
    monkeypatch.setattr(scheduler_psql, "available_slots_by_executor", lambda _sem: {})

    result = await scheduler.run_dispatch_cycle(cycle_index=1)

    assert result == DispatchCycleResult(
        scheduled=False,
        wait_interval=SLOT_POLL_INTERVAL,
    )
    scheduler._wait_for_dispatch_wake.assert_not_awaited()
    scheduler.frontier.peek_ready.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_cycle_plans_leases_activates_and_dispatches_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_scheduler()
    work_item = SimpleNamespace(
        id="job-1",
        dag_id="dag-1",
        name="extract",
        data={"metadata": {"on": "extract://default"}},
        state=WorkState.CREATED,
    )
    scheduler.active_dags[work_item.dag_id] = object()
    scheduler.frontier.peek_ready.return_value = [work_item]
    scheduler.frontier.take.return_value = [work_item]
    scheduler.execution_planner.plan.return_value = [("extract://default", work_item)]
    scheduler._lease_jobs_db.return_value = {work_item.id}
    scheduler._reserve_semaphore_slots.return_value = {work_item.id}
    scheduler._activate_from_lease_db.return_value = {work_item.id: "attempt-1"}
    monkeypatch.setattr(
        scheduler_psql,
        "available_slots_by_executor",
        lambda _sem: {"extract": 1},
    )
    monkeypatch.setattr(
        scheduler_psql,
        "debug_candidates_and_plan",
        AsyncMock(),
    )

    result = await scheduler.run_dispatch_cycle(cycle_index=1)

    assert result == DispatchCycleResult(scheduled=True)
    scheduler.frontier.take.assert_awaited_once_with([work_item.id], lease_ttl=5)
    scheduler._lease_jobs_db.assert_awaited_once_with("extract", [work_item.id])
    scheduler._reserve_semaphore_slots.assert_awaited_once_with("extract", [work_item])
    scheduler._activate_from_lease_db.assert_awaited_once_with([work_item.id])
    scheduler._activate_and_enqueue_job.assert_awaited_once_with(
        work_item,
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )
    scheduler.notify_event.assert_awaited_once_with()
    scheduler._wait_for_dispatch_wake.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_cycle_preserves_control_flow_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_scheduler()
    regular = SimpleNamespace(
        id="job-1",
        dag_id="dag-1",
        name="extract",
        data={"metadata": {"on": "extract://default"}},
        state=WorkState.CREATED,
    )
    control_flow = SimpleNamespace(
        id="noop-1",
        dag_id="dag-1",
        name="extract",
        data={"metadata": {"on": "noop://default"}},
        state=WorkState.CREATED,
    )
    scheduler.active_dags[regular.dag_id] = object()
    scheduler.frontier.peek_ready.side_effect = [
        [regular],
        [control_flow, regular],
        [regular],
    ]
    scheduler.frontier.take.return_value = [regular]
    scheduler.execution_planner.plan.return_value = [("extract://default", regular)]
    scheduler._process_control_flow_candidates = AsyncMock(
        return_value=ControlFlowBatchResult(
            outcomes=(ControlFlowExecutionOutcome.COMPLETED,)
        )
    )
    scheduler._lease_jobs_db.return_value = {regular.id}
    scheduler._reserve_semaphore_slots.return_value = set()
    monkeypatch.setattr(
        scheduler_psql,
        "available_slots_by_executor",
        lambda _sem: {"extract": 2},
    )
    monkeypatch.setattr(scheduler_psql, "debug_candidates_and_plan", AsyncMock())

    result = await scheduler.run_dispatch_cycle(cycle_index=1)

    assert result == DispatchCycleResult(scheduled=True)
    scheduler._process_control_flow_candidates.assert_awaited_once_with(
        [control_flow], 5
    )
    scheduler._activate_and_enqueue_job.assert_not_awaited()
    scheduler.notify_event.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_dispatch_cycle_does_not_count_control_flow_failure_as_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_scheduler()
    control_flow = SimpleNamespace(
        id="noop-1",
        dag_id="dag-1",
        name="extract",
        data={"metadata": {"on": "noop://default"}},
        state=WorkState.CREATED,
    )
    scheduler.active_dags[control_flow.dag_id] = object()
    scheduler.frontier.peek_ready.side_effect = [
        [],
        [control_flow],
        [],
        [],
    ]
    scheduler._process_control_flow_candidates = AsyncMock(
        return_value=ControlFlowBatchResult(
            outcomes=(ControlFlowExecutionOutcome.FAILED,)
        )
    )
    monkeypatch.setattr(
        scheduler_psql,
        "available_slots_by_executor",
        lambda _sem: {"extract": 1},
    )

    result = await scheduler.run_dispatch_cycle(cycle_index=1)

    assert result == DispatchCycleResult(
        scheduled=False,
        wait_interval=scheduler_psql.SHORT_POLL_INTERVAL,
    )
    scheduler.notify_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_control_flow_batch_separates_outcomes_from_reconciliation() -> None:
    scheduler = build_scheduler()
    jobs = [
        SimpleNamespace(
            id=job_id,
            dag_id="dag-1",
            name="extract",
            state=WorkState.CREATED,
        )
        for job_id in ("completed", "refused", "failed", "reconciled")
    ]
    scheduler.frontier.take.return_value = [object()]
    scheduler._lease_jobs_db.side_effect = [
        {"completed"},
        {"refused"},
        {"failed"},
        set(),
    ]
    scheduler.control_flow_service = SimpleNamespace(
        process_node=AsyncMock(
            side_effect=[
                ControlFlowExecutionOutcome.COMPLETED,
                ControlFlowExecutionOutcome.ACTIVATION_REFUSED,
                ControlFlowExecutionOutcome.FAILED,
            ]
        )
    )
    scheduler.repository = SimpleNamespace(
        get_job_by_id=AsyncMock(return_value=SimpleNamespace(state=WorkState.COMPLETED))
    )
    scheduler._reconcile_control_flow_lease_miss = AsyncMock(return_value=True)

    result = await scheduler._process_control_flow_candidates(jobs, lease_ttl=5)

    assert result == ControlFlowBatchResult(
        outcomes=(
            ControlFlowExecutionOutcome.COMPLETED,
            ControlFlowExecutionOutcome.ACTIVATION_REFUSED,
            ControlFlowExecutionOutcome.FAILED,
        ),
        reconciled=1,
    )
    assert result.completed == 1
    assert result.made_progress is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_path",
    [
        "gateway_not_ready",
        "paused",
        "no_candidates",
        "no_planner_picks",
        "no_database_leases",
    ],
)
async def test_dispatch_cycle_compacts_on_cadence_before_early_return(
    exit_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_scheduler()
    work_item = SimpleNamespace(
        id="job-1",
        dag_id="dag-1",
        name="extract",
        data={"metadata": {"on": "extract://default"}},
        state=WorkState.CREATED,
    )
    monkeypatch.setattr(
        scheduler_psql,
        "available_slots_by_executor",
        lambda _sem: {"extract": 1},
    )
    monkeypatch.setattr(scheduler_psql, "debug_candidates_and_plan", AsyncMock())

    if exit_path == "gateway_not_ready":
        scheduler._gateway_ready_event = asyncio.Event()
    elif exit_path == "paused":
        scheduler._paused = True
    elif exit_path in {"no_planner_picks", "no_database_leases"}:
        scheduler.frontier.peek_ready.return_value = [work_item]
        if exit_path == "no_database_leases":
            scheduler.execution_planner.plan.return_value = [
                ("extract://default", work_item)
            ]
            scheduler.frontier.take.return_value = [work_item]

    await scheduler.run_dispatch_cycle(cycle_index=20)

    scheduler.frontier.compact_ready_heap.assert_awaited_once_with(max_scan=10000)


@pytest.mark.asyncio
async def test_dispatch_cycle_skips_compaction_off_cadence() -> None:
    scheduler = build_scheduler()
    scheduler._gateway_ready_event = asyncio.Event()

    await scheduler.run_dispatch_cycle(cycle_index=19)

    scheduler.frontier.compact_ready_heap.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_uses_cycle_wait_interval_for_the_next_wake() -> None:
    scheduler = build_scheduler()
    results = [
        DispatchCycleResult(scheduled=False, wait_interval=0.1),
        DispatchCycleResult(scheduled=True),
    ]

    async def run_dispatch_cycle(_cycle_index: int) -> DispatchCycleResult:
        result = results.pop(0)
        if not results:
            scheduler.running = False
        return result

    scheduler.run_dispatch_cycle = AsyncMock(side_effect=run_dispatch_cycle)

    await scheduler._poll()

    assert scheduler._wait_for_dispatch_wake.await_args_list[1].args == (0.1,)
    assert scheduler.run_dispatch_cycle.await_count == 2
    assert [call.args[0] for call in scheduler.run_dispatch_cycle.await_args_list] == [
        0,
        1,
    ]
