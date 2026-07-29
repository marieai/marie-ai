import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.job.common import JobStatus
from marie.job.event_publisher import EventPublisher
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.psql import PostgreSQLJobScheduler, _PendingDispatch
from marie.scheduler.repository import JobRepository
from marie.scheduler.services import SchedulerRuntime
from marie.scheduler.state import WorkState
from marie.serve.runtimes.servers.cluster_state import ClusterState


async def _wait_forever() -> None:
    await asyncio.Event().wait()


def _scheduler_for_stop() -> PostgreSQLJobScheduler:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler.running = True
    scheduler._resources_closed = False
    scheduler._lifecycle_lock = asyncio.Lock()
    scheduler._event_subscriptions_active = True
    scheduler.job_manager = SimpleNamespace(
        event_publisher=SimpleNamespace(
            join=AsyncMock(),
            unsubscribe=MagicMock(),
        )
    )
    scheduler.notification_service = SimpleNamespace(stop=AsyncMock())
    scheduler.maintenance_service = SimpleNamespace(stop=AsyncMock())
    scheduler.dag_service = SimpleNamespace(
        stop_admission=AsyncMock(),
        stop_sync=AsyncMock(),
    )
    scheduler.runtime = SchedulerRuntime(scheduler.logger)
    scheduler.dispatch_confirmation_max_in_flight = 256
    scheduler._pending_dispatches = {}
    scheduler._semaphore_store = MagicMock()
    scheduler._activate_and_enqueue_job = AsyncMock(return_value=True)
    scheduler._handle_dispatch_failure = AsyncMock()
    scheduler.notify_event = AsyncMock(return_value=True)
    return scheduler


def _scheduler_for_start() -> PostgreSQLJobScheduler:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler.running = False
    scheduler._resources_closed = False
    scheduler._lifecycle_lock = asyncio.Lock()
    scheduler._priority_refresh_event = asyncio.Event()
    scheduler.priority_refresh_enabled = False
    scheduler.known_queues = set()
    scheduler.repository = SimpleNamespace(
        initialize=AsyncMock(),
        is_installed=AsyncMock(return_value=True),
        validate_durable_scheduler_schema=AsyncMock(),
        get_defined_queues=AsyncMock(return_value=set()),
    )
    scheduler.notification_service = SimpleNamespace(
        start=AsyncMock(),
        stop=AsyncMock(),
    )
    scheduler.maintenance_service = SimpleNamespace(
        maintenance_interval=30,
        start=AsyncMock(),
        stop=AsyncMock(),
    )
    scheduler.dag_service = SimpleNamespace(
        start_admission=AsyncMock(),
        stop_admission=AsyncMock(),
        start_sync=AsyncMock(),
        stop_sync=AsyncMock(),
    )
    scheduler.submission_service = SimpleNamespace()
    scheduler.job_manager = SimpleNamespace(
        event_publisher=SimpleNamespace(join=AsyncMock())
    )
    scheduler.runtime = SchedulerRuntime(scheduler.logger)
    scheduler._pending_dispatches = {}
    scheduler._setup_event_subscriptions = MagicMock()
    scheduler._remove_event_subscriptions = MagicMock()
    scheduler.notify_event = AsyncMock(return_value=True)
    scheduler._renew_active_run_leases = AsyncMock()
    scheduler._semaphore_store = MagicMock()
    scheduler._semaphore_store.reconcile_all.return_value = {}

    async def close_resources() -> None:
        scheduler._resources_closed = True

    scheduler._close_runtime_resources = AsyncMock(side_effect=close_resources)
    return scheduler


@pytest.mark.asyncio
async def test_start_tasks_observe_running_after_suspending_dependency_start() -> None:
    scheduler = _scheduler_for_start()
    observed_running: list[bool] = []

    async def suspending_maintenance_start() -> None:
        await asyncio.sleep(0)

    async def observe_running() -> None:
        observed_running.append(scheduler.running)

    scheduler.maintenance_service.start = AsyncMock(
        side_effect=suspending_maintenance_start
    )
    scheduler._priority_refresh_loop = observe_running
    scheduler._sync = observe_running
    scheduler._renew_run_leases = observe_running
    scheduler._poll = observe_running
    scheduler._PostgreSQLJobScheduler__monitor_deployment_updates = observe_running
    await scheduler._start_locked()
    await asyncio.sleep(0)

    assert observed_running == [True] * 4
    scheduler._semaphore_store.reconcile_all.assert_called_once_with(
        delete_orphan_holders=True,
        fix_counters=True,
    )


@pytest.mark.asyncio
async def test_deployment_update_requests_durable_admission(monkeypatch) -> None:
    scheduler = _scheduler_for_start()
    scheduler.running = True
    scheduler.dag_service.request_admission = AsyncMock()
    deployment_update = asyncio.Event()
    monkeypatch.setattr(ClusterState, "deployment_update_event", deployment_update)

    monitor = asyncio.create_task(
        scheduler._PostgreSQLJobScheduler__monitor_deployment_updates()
    )
    await asyncio.sleep(0)
    ClusterState.notify_deployment_update()
    async with asyncio.timeout(1):
        while scheduler.dag_service.request_admission.await_count == 0:
            await asyncio.sleep(0)
    monitor.cancel()
    await monitor

    scheduler.dag_service.request_admission.assert_awaited_once_with(
        "deployment_update"
    )
    scheduler.notify_event.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_event_loop_lag_watchdog_traces_runtime_delay(monkeypatch) -> None:
    scheduler = _scheduler_for_stop()
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        'marie.scheduler.psql.scheduler_trace',
        lambda event, **fields: events.append((event, fields)),
    )

    watchdog = asyncio.create_task(
        scheduler._event_loop_lag_watchdog(interval_seconds=0.001)
    )
    async with asyncio.timeout(1):
        while len(events) < 10:
            await asyncio.sleep(0)
    scheduler.running = False
    await watchdog

    event, fields = events[9]
    assert event == 'gateway_event_loop_lag'
    assert fields['lag_ms'] >= 0.0
    assert fields['interval_ms'] == 1.0
    assert fields['task_count'] >= 1
    assert len(fields['task_names']) <= 10
    assert (
        sum(fields['task_names'].values()) + fields['task_names_other']
        == fields['task_count']
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["admission", "initial_wake"])
async def test_start_rolls_back_partial_startup(failure_stage: str) -> None:
    scheduler = _scheduler_for_start()

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    scheduler._priority_refresh_loop = wait_forever
    scheduler._sync = wait_forever
    scheduler._renew_run_leases = wait_forever
    scheduler._poll = wait_forever
    scheduler._PostgreSQLJobScheduler__monitor_deployment_updates = wait_forever
    if failure_stage == "admission":
        scheduler.dag_service.start_admission.side_effect = RuntimeError(
            "admission failed"
        )
    else:
        scheduler.notify_event.side_effect = RuntimeError("initial wake failed")

    try:
        with pytest.raises(RuntimeError, match="failed"):
            await scheduler.start()

        assert scheduler.running is False
        assert scheduler._resources_closed is True
        scheduler.notification_service.stop.assert_awaited_once_with()
        scheduler.maintenance_service.stop.assert_awaited_once_with()
        scheduler.dag_service.stop_admission.assert_awaited_once_with()
        scheduler.dag_service.stop_sync.assert_awaited_once_with()
        scheduler._remove_event_subscriptions.assert_called_once_with()
        scheduler._close_runtime_resources.assert_awaited_once_with()
        assert scheduler.runtime.tasks() == []
    finally:
        scheduler.running = False
        await scheduler.runtime.stop({}, timeout=0.05)


@pytest.mark.asyncio
async def test_stop_cancels_scheduler_tasks_before_returning() -> None:
    scheduler = _scheduler_for_stop()
    poll_task = scheduler.runtime.create_task(_wait_forever(), name="scheduler-poll")
    sync_task = scheduler.runtime.create_task(_wait_forever(), name="scheduler-sync")
    event_task = asyncio.create_task(_wait_forever(), name="scheduler-dispatch")
    scheduler.runtime.track_event_task(event_task)

    async def close_resources() -> None:
        scheduler._resources_closed = True

    scheduler._close_runtime_resources = AsyncMock(side_effect=close_resources)

    await asyncio.sleep(0)
    await scheduler.stop(timeout=0.05)

    assert poll_task.cancelled()
    assert sync_task.cancelled()
    assert event_task.cancelled()
    scheduler.notification_service.stop.assert_awaited_once()
    scheduler.maintenance_service.stop.assert_awaited_once()
    scheduler.dag_service.stop_admission.assert_awaited_once()
    scheduler.dag_service.stop_sync.assert_awaited_once()
    scheduler._close_runtime_resources.assert_awaited_once()
    scheduler.job_manager.event_publisher.join.assert_awaited_once_with()
    assert scheduler.runtime.tasks() == []


@pytest.mark.asyncio
async def test_stop_drains_pending_dispatch_before_runtime_shutdown() -> None:
    scheduler = _scheduler_for_stop()
    confirmation_started = asyncio.Event()
    release_confirmation = asyncio.Event()

    async def wait_for_confirmation(*_args: object, **_kwargs: object) -> bool:
        confirmation_started.set()
        await release_confirmation.wait()
        return True

    async def close_resources() -> None:
        scheduler._resources_closed = True

    scheduler._activate_and_enqueue_job = AsyncMock(side_effect=wait_for_confirmation)
    scheduler._close_runtime_resources = AsyncMock(side_effect=close_resources)
    work_item = SimpleNamespace(id="job-1", dag_id="dag-1")
    scheduler._start_pending_dispatch(
        _PendingDispatch(
            work_info=work_item,
            executor="extract",
            semaphore_owner=work_item.id,
            run_owner="scheduler-1",
            run_attempt_id="attempt-1",
        )
    )
    await confirmation_started.wait()

    stop_task = asyncio.create_task(scheduler.stop(timeout=0.2))
    await asyncio.sleep(0)

    assert not stop_task.done()
    release_confirmation.set()
    await stop_task

    assert scheduler._pending_dispatches == {}
    scheduler._handle_dispatch_failure.assert_not_awaited()
    scheduler._close_runtime_resources.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_stop_keeps_unresolved_dispatch_attempt_fenced() -> None:
    scheduler = _scheduler_for_stop()
    confirmation_started = asyncio.Event()

    async def wait_for_confirmation(*_args: object, **_kwargs: object) -> bool:
        confirmation_started.set()
        await asyncio.Event().wait()
        return True

    async def close_resources() -> None:
        scheduler._resources_closed = True

    scheduler._activate_and_enqueue_job = AsyncMock(side_effect=wait_for_confirmation)
    scheduler._close_runtime_resources = AsyncMock(side_effect=close_resources)
    work_item = SimpleNamespace(id="job-1", dag_id="dag-1")
    scheduler._start_pending_dispatch(
        _PendingDispatch(
            work_info=work_item,
            executor="extract",
            semaphore_owner=work_item.id,
            run_owner="scheduler-1",
            run_attempt_id="attempt-1",
        )
    )
    await confirmation_started.wait()

    await scheduler.stop(timeout=0.01)

    assert scheduler._pending_dispatches == {}
    scheduler._handle_dispatch_failure.assert_not_awaited()
    scheduler._semaphore_store.release_owned.assert_not_called()
    scheduler._close_runtime_resources.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_start_reopens_resources_once_after_stop() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler.running = False
    scheduler._resources_closed = True
    scheduler._lifecycle_lock = asyncio.Lock()

    async def reopen() -> None:
        scheduler._resources_closed = False

    scheduler._reopen_runtime_resources = AsyncMock(side_effect=reopen)
    scheduler._setup_event_subscriptions = MagicMock()

    async def start_locked() -> None:
        scheduler.running = True

    scheduler._start_locked = AsyncMock(side_effect=start_locked)

    await scheduler.start()
    await scheduler.start()

    scheduler._reopen_runtime_resources.assert_awaited_once_with()
    scheduler._setup_event_subscriptions.assert_called_once_with()
    scheduler._start_locked.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_stopped_scheduler_rejects_submissions() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.running = False
    scheduler.submission_service = SimpleNamespace(
        submit=AsyncMock(side_effect=RuntimeError("Job scheduler is not running"))
    )

    with pytest.raises(RuntimeError, match="not running"):
        await scheduler.submit_job(MagicMock())


@pytest.mark.asyncio
async def test_running_scheduler_handles_job_events_from_publisher() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.running = True
    scheduler.logger = MagicMock()
    work_item = SimpleNamespace(id='job-1', dag_id='dag-1')
    scheduler.get_job = AsyncMock(return_value=work_item)
    scheduler.attempt_lifecycle_service = SimpleNamespace(
        transition_terminal=AsyncMock()
    )
    message = {
        'job_id': 'job-1',
        'run_owner': 'worker-1',
        'run_attempt_id': 'attempt-1',
    }

    await scheduler.handle_job_event(JobStatus.SUCCEEDED.value, message)

    scheduler.attempt_lifecycle_service.transition_terminal.assert_awaited_once_with(
        'job-1',
        work_item,
        JobStatus.SUCCEEDED,
        run_owner='worker-1',
        run_attempt_id='attempt-1',
        source='job_event',
        message=None,
        runtime_env=None,
    )


@pytest.mark.asyncio
async def test_keyed_publisher_delivers_pending_running_terminal_in_order(
    monkeypatch,
) -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.running = True
    scheduler.logger = MagicMock()
    work_item = SimpleNamespace(
        id='job-1',
        dag_id='dag-1',
        state=WorkState.CREATED,
        run_owner=None,
        run_attempt_id=None,
    )
    scheduler.get_job = AsyncMock(return_value=work_item)
    scheduler._job_cache = {}
    scheduler.frontier = SimpleNamespace(update_job_state=AsyncMock())
    scheduler._ha_trace_fields = MagicMock(return_value={})
    observed: list[str] = []

    async def extend_run_lease(
        job_ids: list[str], *, run_owner: str, run_attempt_id: str
    ) -> set[str]:
        observed.append(JobStatus.RUNNING.value)
        return set(job_ids)

    async def transition_terminal(*_args: object, **_kwargs: object) -> None:
        observed.append(JobStatus.SUCCEEDED.value)

    scheduler._extend_run_lease_db = extend_run_lease
    scheduler.attempt_lifecycle_service = SimpleNamespace(
        transition_terminal=transition_terminal
    )
    received: list[str] = []

    def record_trace(event: str, **fields: object) -> None:
        if event == 'scheduler_job_event_received':
            received.append(str(fields['status']))

    monkeypatch.setattr(
        'marie.scheduler.psql.scheduler_trace',
        record_trace,
    )
    publisher = EventPublisher(
        max_queue_size=3,
        worker_count=1,
        publish_blocking=True,
        subscriber_timeout_s=0,
    )
    publisher.subscribe(
        [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED],
        scheduler.handle_job_event,
    )
    event = {
        'job_id': 'job-1',
        'run_owner': 'worker-1',
        'run_attempt_id': 'attempt-1',
    }
    try:
        await publisher.publish(JobStatus.PENDING, event)
        await publisher.publish(JobStatus.RUNNING, event)
        await publisher.publish(JobStatus.SUCCEEDED, event)
        await asyncio.wait_for(publisher.join(), timeout=1)
    finally:
        await publisher.stop()

    assert received == ['PENDING', 'RUNNING', 'SUCCEEDED']
    assert observed == ['RUNNING', 'SUCCEEDED']
    scheduler.frontier.update_job_state.assert_awaited_once_with(
        'job-1', WorkState.ACTIVE
    )


def test_event_subscriptions_are_idempotent() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    publisher = SimpleNamespace(subscribe=MagicMock(), unsubscribe=MagicMock())
    scheduler.job_manager = SimpleNamespace(event_publisher=publisher)
    scheduler._event_subscriptions_active = False

    scheduler._setup_event_subscriptions()
    scheduler._setup_event_subscriptions()
    scheduler._remove_event_subscriptions()
    scheduler._remove_event_subscriptions()

    publisher.subscribe.assert_called_once()
    assert publisher.unsubscribe.call_count == 5
    assert {call.args[0] for call in publisher.unsubscribe.call_args_list} == {
        JobStatus.RUNNING,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.PENDING,
        JobStatus.STOPPED,
    }


class _Pool:
    def __init__(self) -> None:
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_job_repository_close_is_idempotent() -> None:
    repository = object.__new__(JobRepository)
    repository._closed = False
    repository._owns_pool = True
    repository._pool = _Pool()

    await repository.close()
    await repository.close()

    assert repository._pool.close_count == 1


@pytest.mark.asyncio
async def test_close_runtime_resources_closes_async_pool() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler._resources_closed = False
    scheduler.repository = SimpleNamespace(close=AsyncMock())
    scheduler._db_pool = _Pool()

    await scheduler._close_runtime_resources()

    scheduler.repository.close.assert_awaited_once_with()
    assert scheduler._db_pool.close_count == 1
    assert scheduler._resources_closed


def test_scheduler_services_rebuild_with_current_runtime_resources() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.runtime = SchedulerRuntime(MagicMock())
    scheduler.repository = MagicMock()
    scheduler.frontier = MagicMock()
    scheduler.dag_service = MagicMock()
    scheduler._status_update_lock = AsyncJobLock()
    scheduler._topology_cache = DagTopologyCache()
    scheduler._job_cache = {}
    scheduler.lease_owner = 'scheduler-1'
    scheduler.run_ttl_seconds = 60
    scheduler.gateway_instance_id = 'gateway-1'
    scheduler.notify_event = AsyncMock(return_value=True)
    scheduler._scheduler_counter = MagicMock()

    original_control_flow_service = scheduler._build_control_flow_service()
    scheduler.control_flow_service = original_control_flow_service
    original_attempt_service = scheduler._build_attempt_lifecycle_service()
    scheduler.repository = MagicMock()
    scheduler.dag_service = MagicMock()

    control_flow_service = scheduler._build_control_flow_service()
    scheduler.control_flow_service = control_flow_service
    attempt_service = scheduler._build_attempt_lifecycle_service()

    assert control_flow_service.repository is scheduler.repository
    assert control_flow_service.dag_service is scheduler.dag_service
    assert control_flow_service.frontier is scheduler.frontier
    assert (
        control_flow_service.repository is not original_control_flow_service.repository
    )
    assert (
        control_flow_service.dag_service
        is not original_control_flow_service.dag_service
    )
    assert attempt_service.repository is scheduler.repository
    assert attempt_service.dag_service is scheduler.dag_service
    assert attempt_service.control_flow_service is control_flow_service
    assert attempt_service.repository is not original_attempt_service.repository
    assert attempt_service.dag_service is not original_attempt_service.dag_service
