import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.job.common import JobStatus
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.psql import PostgreSQLJobScheduler
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
    scheduler.dag_service = SimpleNamespace(stop_sync=AsyncMock())
    scheduler.runtime = SchedulerRuntime(scheduler.logger)
    scheduler.submission_service = SimpleNamespace(abort_pending=MagicMock())
    return scheduler


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
    scheduler.dag_service.stop_sync.assert_awaited_once()
    scheduler._close_runtime_resources.assert_awaited_once()
    assert scheduler.runtime.tasks() == []
    scheduler.submission_service.abort_pending.assert_called_once_with()


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
    scheduler._handle_job_event = AsyncMock()
    scheduler.submission_service = SimpleNamespace(
        submit=AsyncMock(side_effect=RuntimeError("Job scheduler is not running"))
    )

    with pytest.raises(RuntimeError, match="not running"):
        await scheduler.submit_job(MagicMock())
    await scheduler.handle_job_event(JobStatus.RUNNING.value, {"job_id": "job-1"})

    scheduler._handle_job_event.assert_not_awaited()


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
