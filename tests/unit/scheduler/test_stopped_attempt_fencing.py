import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.job.common import JobInfo, JobStatus
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.repository import JobRepository
from marie.scheduler.state import WorkState

JOB_ID = "00000000-0000-0000-0000-000000000001"
ATTEMPT_A = "00000000-0000-0000-0000-00000000000a"
ATTEMPT_B = "00000000-0000-0000-0000-00000000000b"


def build_work_item() -> SimpleNamespace:
    return SimpleNamespace(
        id=JOB_ID,
        dag_id="00000000-0000-0000-0000-000000000010",
        name="extract",
        state=WorkState.ACTIVE,
        run_owner="owner-b",
        run_attempt_id=ATTEMPT_B,
    )


def build_scheduler(
    work_item: SimpleNamespace, cancelled_ids: set[str]
) -> PostgreSQLJobScheduler:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler.get_job = AsyncMock(return_value=work_item)
    scheduler.repository = SimpleNamespace(
        cancel_job=AsyncMock(return_value=1),
        cancel_job_attempt=AsyncMock(return_value=cancelled_ids),
    )
    scheduler._status_update_lock = AsyncJobLock()
    scheduler._job_cache = {}
    scheduler.frontier = SimpleNamespace(on_job_cancelled=AsyncMock())
    scheduler._record_terminal_attempt_audit = AsyncMock()
    scheduler._scheduler_counter = MagicMock()
    scheduler._ha_trace_fields = MagicMock(return_value={})
    scheduler._resolve_dag_status_with_retry = AsyncMock()
    scheduler.notify_event = AsyncMock()
    return scheduler


@pytest.mark.asyncio
async def test_stopped_event_requires_attempt_identity() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, {JOB_ID})

    await scheduler._handle_job_event(JobStatus.STOPPED.value, {"job_id": JOB_ID})

    scheduler.repository.cancel_job_attempt.assert_not_awaited()
    assert work_item.state == WorkState.ACTIVE
    assert scheduler._job_cache == {}
    scheduler.frontier.on_job_cancelled.assert_not_awaited()
    scheduler._resolve_dag_status_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_stopped_event_does_not_cancel_current_attempt() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())

    await scheduler._handle_job_event(
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
    scheduler._resolve_dag_status_with_retry.assert_not_awaited()
    assert (
        scheduler._record_terminal_attempt_audit.await_args.kwargs["accepted"] is False
    )
    scheduler._scheduler_counter.assert_called_once()


@pytest.mark.asyncio
async def test_current_stopped_event_cancels_after_committed_match() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, {JOB_ID})

    await scheduler._handle_job_event(
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
        scheduler._record_terminal_attempt_audit.await_args.kwargs["accepted"] is True
    )
    scheduler._resolve_dag_status_with_retry.assert_awaited_once()
    scheduler.notify_event.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_stopped_event_database_failure_leaves_memory_unchanged() -> None:
    work_item = build_work_item()
    scheduler = build_scheduler(work_item, set())
    scheduler.repository.cancel_job_attempt.side_effect = RuntimeError("database down")

    await scheduler._handle_job_event(
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
    scheduler._resolve_dag_status_with_retry.assert_not_awaited()
    scheduler.logger.error.assert_called_once()


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
    scheduler._resolve_dag_status_with_retry.assert_not_awaited()


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
    cursor = MagicMock()
    cursor.fetchone.return_value = (JOB_ID,)
    connection = MagicMock()
    connection.cursor.return_value = cursor

    with ThreadPoolExecutor(max_workers=1) as executor:
        repository = object.__new__(JobRepository)
        repository._loop = asyncio.get_running_loop()
        repository._db_executor = executor
        repository._get_connection = MagicMock(return_value=connection)
        repository._close_cursor = MagicMock()
        repository._close_connection = MagicMock()

        cancelled_ids = await repository.cancel_job_attempt(
            job_id=JOB_ID,
            queue_name="extract",
            run_owner="owner-b",
            run_attempt_id=ATTEMPT_B,
        )

    query, params = cursor.execute.call_args.args
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
    connection.commit.assert_called_once_with()
