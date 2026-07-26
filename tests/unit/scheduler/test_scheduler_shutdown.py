import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.job.common import JobStatus
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.psql import PostgreSQLJobScheduler, _PendingDispatch
from marie.scheduler.repository import JobRepository
from marie.scheduler.services import SchedulerRuntime


async def _wait_forever() -> None:
    await asyncio.Event().wait()


def _scheduler_for_stop() -> PostgreSQLJobScheduler:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler.running = True
    scheduler._resources_closed = False
    scheduler._lifecycle_lock = asyncio.Lock()
    scheduler._fetch_event = asyncio.Event()
    scheduler._event_subscriptions_active = True
    scheduler.job_manager = SimpleNamespace(
        event_publisher=SimpleNamespace(unsubscribe=MagicMock())
    )
    scheduler.notification_service = SimpleNamespace(stop=AsyncMock())
    scheduler.maintenance_service = SimpleNamespace(stop=AsyncMock())
    scheduler.heartbeat = SimpleNamespace(stop=AsyncMock())
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
    scheduler.submission_service = SimpleNamespace(abort_pending=MagicMock())
    scheduler.job_event_processor = SimpleNamespace(
        abort_pending=MagicMock(return_value=0)
    )
    return scheduler


def _scheduler_for_start() -> PostgreSQLJobScheduler:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler.running = False
    scheduler._resources_closed = False
    scheduler._lifecycle_lock = asyncio.Lock()
    scheduler._fetch_event = asyncio.Event()
    scheduler._priority_refresh_event = asyncio.Event()
    scheduler.priority_refresh_enabled = False
    scheduler.known_queues = set()
    scheduler.max_workers = 1
    scheduler.job_event_worker_count = 2
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
    scheduler.heartbeat = SimpleNamespace(stop=AsyncMock())
    scheduler.dag_service = SimpleNamespace(
        start_admission=AsyncMock(),
        stop_admission=AsyncMock(),
        start_sync=AsyncMock(),
        stop_sync=AsyncMock(),
    )
    scheduler.submission_service = SimpleNamespace(
        run_worker=AsyncMock(),
        abort_pending=MagicMock(),
    )
    scheduler.job_event_processor = SimpleNamespace(
        run_worker=AsyncMock(),
        abort_pending=MagicMock(return_value=0),
    )
    scheduler.runtime = SchedulerRuntime(scheduler.logger)
    scheduler._pending_dispatches = {}
    scheduler._setup_event_subscriptions = MagicMock()
    scheduler._remove_event_subscriptions = MagicMock()
    scheduler.notify_event = AsyncMock(return_value=True)
    scheduler._renew_active_run_leases = AsyncMock()

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

    async def observe_worker(_worker_id: int) -> None:
        observed_running.append(scheduler.running)

    scheduler.maintenance_service.start = AsyncMock(
        side_effect=suspending_maintenance_start
    )
    scheduler._priority_refresh_loop = observe_running
    scheduler._sync = observe_running
    scheduler._renew_run_leases = observe_running
    scheduler._poll = observe_running
    scheduler._PostgreSQLJobScheduler__monitor_deployment_updates = observe_running
    scheduler.submission_service.run_worker = observe_worker
    scheduler.job_event_processor.run_worker = observe_worker

    await scheduler._start_locked()
    await asyncio.sleep(0)

    assert observed_running == [True] * 7


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["admission", "initial_wake"])
async def test_start_rolls_back_partial_startup(failure_stage: str) -> None:
    scheduler = _scheduler_for_start()

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    async def wait_forever_worker(_worker_id: int) -> None:
        await asyncio.Event().wait()

    scheduler._priority_refresh_loop = wait_forever
    scheduler._sync = wait_forever
    scheduler._renew_run_leases = wait_forever
    scheduler._poll = wait_forever
    scheduler._PostgreSQLJobScheduler__monitor_deployment_updates = wait_forever
    scheduler.submission_service.run_worker = wait_forever_worker
    scheduler.job_event_processor.run_worker = wait_forever_worker

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
        scheduler.heartbeat.stop.assert_awaited_once_with()
        scheduler.dag_service.stop_admission.assert_awaited_once_with()
        scheduler.dag_service.stop_sync.assert_awaited_once_with()
        scheduler.submission_service.abort_pending.assert_called_once_with()
        scheduler.job_event_processor.abort_pending.assert_called_once_with()
        scheduler._remove_event_subscriptions.assert_called_once_with()
        scheduler._close_runtime_resources.assert_awaited_once_with()
        assert scheduler.runtime.tasks() == []
    finally:
        scheduler.running = False
        await scheduler.runtime.stop({}, timeout=0.05)


@pytest.mark.asyncio
async def test_stop_cancels_poll_and_workers_before_returning() -> None:
    scheduler = _scheduler_for_stop()
    poll_task = scheduler.runtime.create_task(_wait_forever(), name="scheduler-poll")
    worker_task = scheduler.runtime.create_task(
        _wait_forever(), name="scheduler-submission-0"
    )
    sync_task = scheduler.runtime.create_task(_wait_forever(), name="scheduler-sync")
    event_task = asyncio.create_task(_wait_forever(), name="scheduler-job-event")
    scheduler.runtime.track_event_task(event_task)

    async def close_resources() -> None:
        scheduler._resources_closed = True

    scheduler._close_runtime_resources = AsyncMock(side_effect=close_resources)

    await asyncio.sleep(0)
    await scheduler.stop(timeout=0.05)

    assert poll_task.cancelled()
    assert worker_task.cancelled()
    assert sync_task.cancelled()
    assert event_task.cancelled()
    scheduler.notification_service.stop.assert_awaited_once()
    scheduler.maintenance_service.stop.assert_awaited_once()
    scheduler.heartbeat.stop.assert_awaited_once()
    scheduler.dag_service.stop_admission.assert_awaited_once()
    scheduler.dag_service.stop_sync.assert_awaited_once()
    scheduler._close_runtime_resources.assert_awaited_once()
    assert scheduler.runtime.tasks() == []
    scheduler.submission_service.abort_pending.assert_called_once_with()
    scheduler.job_event_processor.abort_pending.assert_called_once_with()


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
async def test_stopped_scheduler_rejects_submissions_and_job_events() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.running = False
    scheduler.job_event_processor = SimpleNamespace(enqueue=AsyncMock())
    scheduler.submission_service = SimpleNamespace(
        submit=AsyncMock(side_effect=RuntimeError("Job scheduler is not running"))
    )

    with pytest.raises(RuntimeError, match="not running"):
        await scheduler.submit_job(MagicMock())
    await scheduler.handle_job_event(JobStatus.RUNNING.value, {"job_id": "job-1"})

    scheduler.job_event_processor.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_running_scheduler_queues_job_events_for_background_processing() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.running = True
    scheduler.logger = MagicMock()
    scheduler.runtime = SchedulerRuntime(scheduler.logger)
    scheduler.job_event_processor = SimpleNamespace(enqueue=AsyncMock())
    message = {'job_id': 'job-1'}

    await scheduler.handle_job_event(JobStatus.SUCCEEDED.value, message)

    scheduler.job_event_processor.enqueue.assert_awaited_once_with(
        JobStatus.SUCCEEDED.value,
        message,
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
