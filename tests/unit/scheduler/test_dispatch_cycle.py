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
    SemaphoreReservationStatus,
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
    scheduler.priority_refresh_enabled = False
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
    scheduler.repository = SimpleNamespace(defer_leased_job=AsyncMock())
    scheduler._sem_default_ttl = 30
    scheduler._ticket_collision_counts = {}
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
    scheduler._reserve_semaphore_slots = AsyncMock(return_value={})
    scheduler._activate_from_lease_db = AsyncMock(return_value={})
    scheduler._activate_and_enqueue_job = AsyncMock(return_value=True)
    scheduler._handle_dispatch_failure = AsyncMock()
    scheduler.get_dag_by_id = AsyncMock(return_value=None)
    scheduler.notify_event = AsyncMock(return_value=True)
    return scheduler


@pytest.mark.asyncio
async def test_semaphore_reservation_reports_reason_per_job() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler._sem_default_ttl = 30
    scheduler._semaphore_store = MagicMock()
    scheduler._semaphore_store.reserve_many.return_value = {"job-reserved"}
    scheduler._semaphore_store.get_holder.side_effect = lambda _executor, job_id: (
        object() if job_id == "job-existing" else None
    )
    scheduler._semaphore_store.available_slot_count.return_value = 0
    jobs = [
        SimpleNamespace(id="job-reserved", dag_id="dag-1"),
        SimpleNamespace(id="job-existing", dag_id="dag-1"),
        SimpleNamespace(id="job-full", dag_id="dag-1"),
    ]
    run_attempt_ids = {job.id: f"attempt-{job.id}" for job in jobs}

    results = await scheduler._reserve_semaphore_slots("extract", jobs, run_attempt_ids)

    assert results == {
        "job-reserved": SemaphoreReservationStatus.RESERVED,
        "job-existing": SemaphoreReservationStatus.TICKET_EXISTS,
        "job-full": SemaphoreReservationStatus.CAPACITY_FULL,
    }
    scheduler._semaphore_store.reserve_many.assert_called_once_with(
        "extract",
        [job.id for job in jobs],
        node='',
        ttl=30,
        owner_by_ticket={job.id: job.id for job in jobs},
        run_attempt_id_by_ticket=run_attempt_ids,
    )


@pytest.mark.asyncio
async def test_semaphore_reservation_fallback_preserves_attempt_id() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler._sem_default_ttl = 30
    scheduler._semaphore_store = MagicMock()
    scheduler._semaphore_store.reserve_many.side_effect = RuntimeError("etcd retry")
    scheduler._semaphore_store.reserve.return_value = True
    job = SimpleNamespace(id="job-1", dag_id="dag-1")

    results = await scheduler._reserve_semaphore_slots(
        "extract", [job], {job.id: "attempt-1"}
    )

    assert results == {job.id: SemaphoreReservationStatus.RESERVED}
    scheduler._semaphore_store.reserve.assert_called_once_with(
        "extract",
        job.id,
        node='',
        ttl=30,
        owner=job.id,
        run_attempt_id="attempt-1",
    )


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
    scheduler._reserve_semaphore_slots.return_value = {
        work_item.id: SemaphoreReservationStatus.RESERVED
    }
    scheduler._activate_from_lease_db.return_value = {work_item.id: "attempt-1"}
    monkeypatch.setattr(
        scheduler_psql,
        "_uuid",
        SimpleNamespace(uuid4=lambda: "attempt-1"),
    )
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
    scheduler.frontier.dag_remaining_counts.return_value = {work_item.dag_id: 1}

    result = await scheduler.run_dispatch_cycle(cycle_index=1)

    assert result == DispatchCycleResult(scheduled=True)
    scheduler.execution_planner.plan.assert_called_once_with(
        [("extract://default", work_item)],
        {"extract": 1},
        scheduler.active_dags,
        exclude_blocked=True,
        dag_remaining={work_item.dag_id: 1},
    )
    scheduler.frontier.dag_remaining_counts.assert_called_once_with()
    scheduler.frontier.take.assert_awaited_once_with([work_item.id], lease_ttl=5)
    scheduler._lease_jobs_db.assert_awaited_once_with("extract", [work_item.id])
    scheduler._reserve_semaphore_slots.assert_awaited_once_with(
        "extract", [work_item], {work_item.id: "attempt-1"}
    )
    scheduler._activate_from_lease_db.assert_awaited_once_with(
        [work_item.id], {work_item.id: "attempt-1"}
    )
    scheduler._activate_and_enqueue_job.assert_awaited_once_with(
        work_item,
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )
    scheduler._semaphore_store.bind_run_attempt.assert_not_called()
    scheduler.notify_event.assert_not_awaited()
    scheduler._wait_for_dispatch_wake.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_cycle_reconciles_partial_database_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_scheduler()
    leased = SimpleNamespace(
        id="job-leased",
        dag_id="dag-1",
        name="extract",
        data={"metadata": {"on": "extract://default"}},
        state=WorkState.CREATED,
    )
    missing = SimpleNamespace(
        id="job-missing",
        dag_id="dag-1",
        name="extract",
        data={"metadata": {"on": "extract://default"}},
        state=WorkState.CREATED,
    )
    selected = [leased, missing]
    scheduler.active_dags[leased.dag_id] = object()
    scheduler.frontier.peek_ready.return_value = selected
    scheduler.frontier.take.return_value = selected
    scheduler.execution_planner.plan.return_value = [
        ("extract://default", leased),
        ("extract://default", missing),
    ]
    scheduler._lease_jobs_db.return_value = {leased.id}
    scheduler._reconcile_db_lease_shortfall.return_value = 1
    scheduler._reserve_semaphore_slots.return_value = {
        leased.id: SemaphoreReservationStatus.RESERVED
    }
    scheduler._activate_from_lease_db.return_value = {leased.id: "attempt-1"}
    monkeypatch.setattr(
        scheduler_psql,
        "_uuid",
        SimpleNamespace(uuid4=lambda: "attempt-1"),
    )
    monkeypatch.setattr(
        scheduler_psql,
        "available_slots_by_executor",
        lambda _sem: {"extract": 2},
    )
    monkeypatch.setattr(scheduler_psql, "debug_candidates_and_plan", AsyncMock())

    result = await scheduler.run_dispatch_cycle(cycle_index=1)

    assert result == DispatchCycleResult(scheduled=True)
    scheduler._lease_jobs_db.assert_awaited_once_with(
        "extract", [leased.id, missing.id]
    )
    scheduler._reconcile_db_lease_shortfall.assert_awaited_once_with(
        selected, {leased.id}
    )
    scheduler._reserve_semaphore_slots.assert_awaited_once_with(
        "extract", [leased], {leased.id: "attempt-1"}
    )
    scheduler._activate_from_lease_db.assert_awaited_once_with(
        [leased.id], {leased.id: "attempt-1"}
    )
    scheduler._activate_and_enqueue_job.assert_awaited_once_with(
        leased,
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )


@pytest.mark.asyncio
async def test_dispatch_cycle_releases_resources_when_database_activation_fails(
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
    scheduler._reserve_semaphore_slots.return_value = {
        work_item.id: SemaphoreReservationStatus.RESERVED
    }
    scheduler._activate_from_lease_db.side_effect = RuntimeError("activation failed")
    monkeypatch.setattr(
        scheduler_psql,
        "_uuid",
        SimpleNamespace(uuid4=lambda: "attempt-1"),
    )
    monkeypatch.setattr(
        scheduler_psql,
        "available_slots_by_executor",
        lambda _sem: {"extract": 1},
    )
    monkeypatch.setattr(scheduler_psql, "debug_candidates_and_plan", AsyncMock())

    result = await scheduler.run_dispatch_cycle(cycle_index=1)

    assert result == DispatchCycleResult(scheduled=False)
    scheduler._release_lease_db.assert_awaited_once_with([work_item.id])
    scheduler.frontier.release_lease_local.assert_awaited_once_with(work_item.id)
    scheduler._semaphore_store.release_owned.assert_called_once_with(
        "extract",
        work_item.id,
        owner=work_item.id,
        run_attempt_id="attempt-1",
    )
    scheduler._activate_and_enqueue_job.assert_not_awaited()
    scheduler._handle_dispatch_failure.assert_not_awaited()
    scheduler.notify_event.assert_not_awaited()


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
    scheduler._reserve_semaphore_slots.return_value = {
        regular.id: SemaphoreReservationStatus.CAPACITY_FULL
    }
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
    scheduler.notify_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_cycle_defers_existing_ticket_without_hot_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_scheduler()
    work_item = SimpleNamespace(
        id="job-1",
        dag_id="dag-1",
        name="extract",
        data={"metadata": {"on": "extract://default"}},
        state=WorkState.RETRY,
        start_after=None,
    )
    scheduler.active_dags[work_item.dag_id] = object()
    scheduler.frontier.peek_ready.return_value = [work_item]
    scheduler.frontier.take.return_value = [work_item]
    scheduler.execution_planner.plan.return_value = [("extract://default", work_item)]
    scheduler._lease_jobs_db.return_value = {work_item.id}
    scheduler._reserve_semaphore_slots.return_value = {
        work_item.id: SemaphoreReservationStatus.TICKET_EXISTS
    }
    scheduler.repository.defer_leased_job.return_value = True
    monkeypatch.setattr(
        scheduler_psql,
        "available_slots_by_executor",
        lambda _sem: {"extract": 1},
    )
    monkeypatch.setattr(scheduler_psql, "debug_candidates_and_plan", AsyncMock())

    result = await scheduler.run_dispatch_cycle(cycle_index=1)

    assert result == DispatchCycleResult(scheduled=False)
    scheduler.repository.defer_leased_job.assert_awaited_once()
    assert work_item.start_after is not None
    assert scheduler._ticket_collision_counts == {work_item.id: 1}
    scheduler._release_lease_db.assert_not_awaited()
    scheduler.frontier.release_lease_local.assert_awaited_once_with(work_item.id)
    scheduler._activate_from_lease_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_ticket_collision_delay_backs_off_and_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_scheduler()
    work_item = SimpleNamespace(
        id="job-1",
        dag_id="dag-1",
        name="extract",
        data={"metadata": {"on": "extract://default"}},
        state=WorkState.RETRY,
        start_after=None,
    )
    scheduler.active_dags[work_item.dag_id] = object()
    scheduler.frontier.peek_ready.return_value = [work_item]
    scheduler.frontier.take.return_value = [work_item]
    scheduler.execution_planner.plan.return_value = [("extract://default", work_item)]
    scheduler._lease_jobs_db.return_value = {work_item.id}
    scheduler._reserve_semaphore_slots.return_value = {
        work_item.id: SemaphoreReservationStatus.TICKET_EXISTS
    }
    scheduler.repository.defer_leased_job.return_value = True
    scheduler._ticket_collision_counts[work_item.id] = 20
    monkeypatch.setattr(
        scheduler_psql,
        "available_slots_by_executor",
        lambda _sem: {"extract": 1},
    )
    monkeypatch.setattr(scheduler_psql, "debug_candidates_and_plan", AsyncMock())
    monkeypatch.setattr(scheduler_psql.random, "uniform", lambda _low, _high: 1.2)

    await scheduler.run_dispatch_cycle(cycle_index=1)

    assert scheduler._ticket_collision_counts[work_item.id] == 21
    deferred_call = scheduler.repository.defer_leased_job.await_args
    assert deferred_call.kwargs["delay_seconds"] == 30.0


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
    scheduler.frontier.take.return_value = jobs
    scheduler._lease_jobs_db.return_value = {"completed", "refused", "failed"}
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
    scheduler.frontier.take.assert_awaited_once_with(
        ["completed", "refused", "failed", "reconciled"], lease_ttl=5
    )
    scheduler._lease_jobs_db.assert_awaited_once_with(
        "extract", ["completed", "refused", "failed", "reconciled"]
    )


@pytest.mark.asyncio
async def test_control_flow_batch_releases_group_after_lease_error() -> None:
    scheduler = build_scheduler()
    extract_jobs = [
        SimpleNamespace(
            id=job_id,
            dag_id="dag-1",
            name="extract",
            state=WorkState.CREATED,
        )
        for job_id in ("extract-1", "extract-2")
    ]
    classify_job = SimpleNamespace(
        id="classify-1",
        dag_id="dag-1",
        name="classify",
        state=WorkState.CREATED,
    )
    scheduler.frontier.take.return_value = [*extract_jobs, classify_job]
    scheduler._lease_jobs_db.side_effect = [
        RuntimeError("database unavailable"),
        {classify_job.id},
    ]
    scheduler.control_flow_service = SimpleNamespace(
        process_node=AsyncMock(return_value=ControlFlowExecutionOutcome.COMPLETED)
    )

    result = await scheduler._process_control_flow_candidates(
        [*extract_jobs, classify_job], lease_ttl=5
    )

    assert result.completed == 1
    assert scheduler._lease_jobs_db.await_args_list[0].args == (
        "extract",
        ["extract-1", "extract-2"],
    )
    assert scheduler._lease_jobs_db.await_args_list[1].args == (
        "classify",
        ["classify-1"],
    )
    assert [
        call.args for call in scheduler.frontier.release_lease_local.await_args_list
    ] == [
        ("extract-1",),
        ("extract-2",),
    ]


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


@pytest.mark.asyncio
async def test_poll_drains_scheduled_cycles_before_waiting_again() -> None:
    scheduler = build_scheduler()
    results = [
        DispatchCycleResult(scheduled=True),
        DispatchCycleResult(scheduled=False, wait_interval=0.1),
        DispatchCycleResult(scheduled=False),
    ]

    async def run_dispatch_cycle(_cycle_index: int) -> DispatchCycleResult:
        result = results.pop(0)
        if not results:
            scheduler.running = False
        return result

    scheduler.run_dispatch_cycle = AsyncMock(side_effect=run_dispatch_cycle)

    await scheduler._poll()

    assert scheduler._wait_for_dispatch_wake.await_count == 2
    assert [
        call.args[0] for call in scheduler._wait_for_dispatch_wake.await_args_list
    ] == [
        scheduler_psql.INIT_POLL_PERIOD,
        0.1,
    ]
    assert [call.args[0] for call in scheduler.run_dispatch_cycle.await_args_list] == [
        0,
        1,
        2,
    ]
