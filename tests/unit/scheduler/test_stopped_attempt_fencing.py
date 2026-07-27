import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.job.common import JobInfo, JobStatus
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.repository import JobRepository
from marie.scheduler.services.attempt_lifecycle_service import (
    AttemptLifecycleService,
)
from marie.scheduler.state import WorkState

JOB_ID = "00000000-0000-0000-0000-000000000001"
ATTEMPT_A = "00000000-0000-0000-0000-00000000000a"
ATTEMPT_B = "00000000-0000-0000-0000-00000000000b"


def build_work_item() -> SimpleNamespace:
    return SimpleNamespace(
        id=JOB_ID,
        dag_id="00000000-0000-0000-0000-000000000010",
        name="extract",
        data={"metadata": {"on": "extract://default"}},
        state=WorkState.ACTIVE,
        run_owner="owner-b",
        run_attempt_id=ATTEMPT_B,
    )


def build_scheduler(
    work_item: SimpleNamespace, cancelled_ids: set[str]
) -> PostgreSQLJobScheduler:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler.running = True
    scheduler.get_job = AsyncMock(return_value=work_item)
    scheduler.repository = SimpleNamespace(
        get_job_by_id=AsyncMock(return_value=work_item),
        cancel_job=AsyncMock(return_value=1),
        cancel_job_attempt=AsyncMock(return_value=cancelled_ids),
        record_job_attempt_terminal=AsyncMock(),
    )
    scheduler._status_update_lock = AsyncJobLock()
    scheduler._job_cache = {}
    scheduler.frontier = SimpleNamespace(on_job_cancelled=AsyncMock())
    scheduler._scheduler_counter = MagicMock()
    scheduler._semaphore_store = MagicMock()
    scheduler._semaphore_store.renew.return_value = True
    scheduler._ha_trace_fields = MagicMock(return_value={})
    scheduler.dag_service = SimpleNamespace(
        resolve_dag_status_with_retry=AsyncMock(),
        request_admission=AsyncMock(),
    )
    scheduler.notify_event = AsyncMock()
    scheduler.lease_owner = "scheduler"
    scheduler.gateway_instance_id = "gateway"
    scheduler.attempt_lifecycle_service = AttemptLifecycleService(
        repository=scheduler.repository,
        frontier=scheduler.frontier,
        dag_service=scheduler.dag_service,
        control_flow_service=MagicMock(),
        status_update_lock=scheduler._status_update_lock,
        job_cache=scheduler._job_cache,
        scheduler_lease_owner=scheduler.lease_owner,
        gateway_instance_id=scheduler.gateway_instance_id,
        notify_callback=scheduler.notify_event,
        counter_callback=scheduler._scheduler_counter,
    )
    return scheduler


@pytest.mark.asyncio
async def test_stopped_event_requires_attempt_identity() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, {JOB_ID})

    await scheduler.handle_job_event(JobStatus.STOPPED.value, {"job_id": JOB_ID})

    scheduler.repository.cancel_job_attempt.assert_not_awaited()
    assert work_item.state == WorkState.ACTIVE
    assert scheduler._job_cache == {}
    scheduler.frontier.on_job_cancelled.assert_not_awaited()
    scheduler.dag_service.resolve_dag_status_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_job_event_trace_records_scheduler_handler_entry(monkeypatch) -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, {JOB_ID})
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "marie.scheduler.psql.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    await scheduler.handle_job_event(JobStatus.STOPPED.value, {"job_id": JOB_ID})

    assert events[0] == (
        "scheduler_job_event_received",
        {"job_id": JOB_ID, "status": "STOPPED"},
    )


@pytest.mark.asyncio
async def test_stale_stopped_event_does_not_cancel_current_attempt() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())

    await scheduler.handle_job_event(
        JobStatus.STOPPED.value,
        {
            "job_id": JOB_ID,
            "run_owner": "owner-a",
            "run_attempt_id": ATTEMPT_A,
        },
    )

    scheduler.repository.cancel_job_attempt.assert_awaited_once_with(
        job_id=JOB_ID,
        queue_name="extract",
        run_owner="owner-a",
        run_attempt_id=ATTEMPT_A,
        schema="marie_scheduler",
    )
    assert work_item.state == WorkState.ACTIVE
    assert work_item.run_owner == "owner-b"
    assert work_item.run_attempt_id == ATTEMPT_B
    assert scheduler._job_cache == {}
    scheduler.frontier.on_job_cancelled.assert_not_awaited()
    scheduler.dag_service.resolve_dag_status_with_retry.assert_not_awaited()
    assert (
        scheduler.repository.record_job_attempt_terminal.await_args.kwargs["accepted"]
        is False
    )
    scheduler._scheduler_counter.assert_called_once()


@pytest.mark.asyncio
async def test_current_stopped_event_cancels_after_committed_match() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, {JOB_ID})

    await scheduler.handle_job_event(
        JobStatus.STOPPED.value,
        {
            "job_id": JOB_ID,
            "run_owner": "owner-b",
            "run_attempt_id": ATTEMPT_B,
        },
    )

    assert work_item.state == WorkState.CANCELLED
    assert work_item.run_owner is None
    assert work_item.run_attempt_id is None
    assert scheduler._job_cache == {JOB_ID: work_item}
    scheduler.frontier.on_job_cancelled.assert_awaited_once_with(JOB_ID)
    assert (
        scheduler.repository.record_job_attempt_terminal.await_args.kwargs["accepted"]
        is True
    )
    scheduler.dag_service.resolve_dag_status_with_retry.assert_awaited_once()
    scheduler.notify_event.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_stopped_event_database_failure_leaves_memory_unchanged() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())
    scheduler.repository.cancel_job_attempt.side_effect = RuntimeError("database down")

    with pytest.raises(RuntimeError, match="database down"):
        await scheduler.handle_job_event(
            JobStatus.STOPPED.value,
            {
                "job_id": JOB_ID,
                "run_owner": "owner-b",
                "run_attempt_id": ATTEMPT_B,
            },
        )

    assert work_item.state == WorkState.ACTIVE
    assert scheduler._job_cache == {}
    scheduler.frontier.on_job_cancelled.assert_not_awaited()
    scheduler.dag_service.resolve_dag_status_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_storage_sync_rejects_stale_stopped_attempt() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())
    job_info = JobInfo(
        status=JobStatus.STOPPED,
        entrypoint="extract",
        end_time=int(
            (datetime.now(timezone.utc) - timedelta(seconds=1)).timestamp() * 1000
        ),
        run_owner="owner-a",
        run_attempt_id=ATTEMPT_A,
    )

    synchronized = await scheduler._sync_terminal_job_state(
        JOB_ID,
        work_item,
        job_info,
        min_sync_interval_seconds=0,
    )

    assert synchronized is False
    assert work_item.state == WorkState.ACTIVE
    scheduler.frontier.on_job_cancelled.assert_not_awaited()
    scheduler.dag_service.resolve_dag_status_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_storage_sync_defers_recent_terminal_mismatch() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())
    transition_terminal = AsyncMock(return_value=True)
    scheduler.attempt_lifecycle_service.transition_terminal = transition_terminal
    job_info = JobInfo(
        status=JobStatus.STOPPED,
        entrypoint="extract",
        end_time=int(
            (datetime.now(timezone.utc) - timedelta(seconds=5)).timestamp() * 1000
        ),
        run_owner="owner-b",
        run_attempt_id=ATTEMPT_B,
    )

    synchronized = await scheduler._sync_terminal_job_state(
        JOB_ID,
        work_item,
        job_info,
        min_sync_interval_seconds=300,
    )

    assert synchronized is False
    transition_terminal.assert_not_awaited()
    message = scheduler.logger.info.call_args.args[0]
    assert message.startswith("Terminal state mismatch detected")
    assert "repair_grace_seconds=300" in message
    terminal_age = float(message.split("terminal_age_seconds=", 1)[1].split(",", 1)[0])
    assert 5 <= terminal_age < 10


@pytest.mark.asyncio
async def test_storage_sync_ignores_stale_active_snapshot() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())
    current_work_item = build_work_item()
    current_work_item.state = WorkState.COMPLETED
    scheduler.repository.get_job_by_id.return_value = current_work_item
    transition_terminal = AsyncMock(return_value=True)
    scheduler.attempt_lifecycle_service.transition_terminal = transition_terminal
    job_info = JobInfo(
        status=JobStatus.SUCCEEDED,
        entrypoint="extract",
        end_time=int(
            (datetime.now(timezone.utc) - timedelta(seconds=301)).timestamp() * 1000
        ),
        run_owner="owner-b",
        run_attempt_id=ATTEMPT_B,
    )

    synchronized = await scheduler._sync_terminal_job_state(
        JOB_ID,
        work_item,
        job_info,
        min_sync_interval_seconds=300,
    )

    assert synchronized is False
    scheduler.repository.get_job_by_id.assert_awaited_once_with(JOB_ID)
    transition_terminal.assert_not_awaited()
    scheduler.logger.info.assert_not_called()


@pytest.mark.asyncio
async def test_storage_sync_logs_repair_after_terminal_grace() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())
    transition_terminal = AsyncMock(return_value=True)
    scheduler.attempt_lifecycle_service.transition_terminal = transition_terminal
    job_info = JobInfo(
        status=JobStatus.STOPPED,
        entrypoint="extract",
        end_time=int(
            (datetime.now(timezone.utc) - timedelta(seconds=301)).timestamp() * 1000
        ),
        run_owner="owner-b",
        run_attempt_id=ATTEMPT_B,
    )

    synchronized = await scheduler._sync_terminal_job_state(
        JOB_ID,
        work_item,
        job_info,
        min_sync_interval_seconds=300,
    )

    assert synchronized is True
    transition_terminal.assert_awaited_once()
    message = scheduler.logger.info.call_args.args[0]
    assert message.startswith("Repairing terminal state mismatch")
    assert "repair_grace_seconds=300" in message
    terminal_age = float(message.split("terminal_age_seconds=", 1)[1].split(",", 1)[0])
    assert terminal_age >= 300


@pytest.mark.asyncio
async def test_pending_job_renews_matching_run_lease() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())
    scheduler.lease_owner = work_item.run_owner
    get_info = AsyncMock(
        return_value=JobInfo(
            status=JobStatus.PENDING,
            entrypoint="extract",
            run_owner=work_item.run_owner,
            run_attempt_id=work_item.run_attempt_id,
        )
    )
    scheduler.list_jobs = AsyncMock(return_value={JOB_ID: work_item})
    scheduler.job_manager = SimpleNamespace(
        job_info_client=MagicMock(return_value=SimpleNamespace(get_info=get_info))
    )
    scheduler._extend_run_lease_db = AsyncMock(return_value={JOB_ID})

    await scheduler._renew_active_run_leases()

    scheduler._extend_run_lease_db.assert_awaited_once_with(
        [JOB_ID],
        run_owner=work_item.run_owner,
        run_attempt_id=work_item.run_attempt_id,
    )
    scheduler._semaphore_store.renew.assert_called_once_with(
        "extract",
        JOB_ID,
        owner=JOB_ID,
        run_attempt_id=work_item.run_attempt_id,
    )


@pytest.mark.asyncio
async def test_terminal_job_does_not_renew_run_lease() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())
    scheduler.lease_owner = work_item.run_owner
    scheduler.list_jobs = AsyncMock(return_value={JOB_ID: work_item})
    scheduler.job_manager = SimpleNamespace(
        job_info_client=MagicMock(
            return_value=SimpleNamespace(
                get_info=AsyncMock(
                    return_value=JobInfo(
                        status=JobStatus.SUCCEEDED,
                        entrypoint="extract",
                        run_owner=work_item.run_owner,
                        run_attempt_id=work_item.run_attempt_id,
                    )
                )
            )
        )
    )
    scheduler._extend_run_lease_db = AsyncMock(return_value={JOB_ID})

    await scheduler._renew_active_run_leases()

    scheduler._extend_run_lease_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_operator_cancellation_uses_separate_job_scoped_path() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())

    cancelled = await scheduler.cancel_job(JOB_ID, work_item)

    assert cancelled == 1
    scheduler.repository.cancel_job.assert_awaited_once_with(
        job_id=JOB_ID,
        queue_name="extract",
        schema="marie_scheduler",
    )
    scheduler.repository.cancel_job_attempt.assert_not_awaited()


@pytest.mark.asyncio
async def test_repository_cancels_only_matching_active_attempt() -> None:
    class Connection:
        async def fetchrow(self, query, *params):
            self.query = query
            self.params = params
            return (JOB_ID,)

    class Pool:
        def __init__(self):
            self.connection = Connection()

        @asynccontextmanager
        async def acquire(self):
            yield self.connection

    pool = Pool()
    repository = JobRepository({}, pool=pool)

    cancelled_ids = await repository.cancel_job_attempt(
        job_id=JOB_ID,
        queue_name="extract",
        run_owner="owner-b",
        run_attempt_id=ATTEMPT_B,
    )

    query = pool.connection.query
    params = pool.connection.params
    assert "AND state = %s::marie_scheduler.job_state" in query
    assert "AND run_owner = %s" in query
    assert "AND run_attempt_id = %s::uuid" in query
    assert params == (
        WorkState.CANCELLED.value,
        JOB_ID,
        "extract",
        WorkState.ACTIVE.value,
        "owner-b",
        ATTEMPT_B,
    )
    assert cancelled_ids == {JOB_ID}
