from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.job.common import JobStatus
from marie.scheduler.psql import PostgreSQLJobScheduler


def build_scheduler(
    work_item: SimpleNamespace, job_info: SimpleNamespace
) -> PostgreSQLJobScheduler:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.list_jobs = AsyncMock(return_value={work_item.id: work_item})
    scheduler.job_manager = SimpleNamespace(
        job_info_client=lambda: SimpleNamespace(
            get_info=AsyncMock(return_value=job_info)
        )
    )
    scheduler._extend_run_lease_db = AsyncMock(return_value={work_item.id})
    scheduler._semaphore_store = MagicMock()
    scheduler._semaphore_store.renew.return_value = True
    scheduler._scheduler_counter = MagicMock()
    scheduler.logger = MagicMock()
    scheduler.gateway_instance_id = "gateway-1"
    scheduler.lease_owner = "scheduler-1"
    return scheduler


@pytest.mark.asyncio
async def test_dedicated_loop_renews_matching_running_attempt() -> None:
    work_item = SimpleNamespace(
        id="job-1",
        dag_id="dag-1",
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )
    job_info = SimpleNamespace(
        status=JobStatus.RUNNING,
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )
    scheduler = build_scheduler(work_item, job_info)

    await scheduler._renew_active_run_leases()

    scheduler._extend_run_lease_db.assert_awaited_once_with(
        ["job-1"],
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )
    scheduler._scheduler_counter.assert_not_called()
    scheduler._semaphore_store.renew.assert_not_called()


@pytest.mark.asyncio
async def test_pending_attempt_renews_ticket_before_run_lease() -> None:
    work_item = SimpleNamespace(
        id="job-1",
        dag_id="dag-1",
        data={"metadata": {"on": "mock_executor_a://document/process"}},
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )
    job_info = SimpleNamespace(
        status=JobStatus.PENDING,
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )
    scheduler = build_scheduler(work_item, job_info)

    await scheduler._renew_active_run_leases()

    scheduler._semaphore_store.renew.assert_called_once_with(
        "mock_executor_a",
        "job-1",
        owner="job-1",
        run_attempt_id="attempt-1",
    )
    scheduler._extend_run_lease_db.assert_awaited_once_with(
        ["job-1"],
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )


@pytest.mark.asyncio
async def test_pending_attempt_does_not_extend_run_lease_when_ticket_is_stale() -> None:
    work_item = SimpleNamespace(
        id="job-1",
        dag_id="dag-1",
        data={"metadata": {"on": "mock_executor_a://document/process"}},
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )
    job_info = SimpleNamespace(
        status=JobStatus.PENDING,
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )
    scheduler = build_scheduler(work_item, job_info)
    scheduler._semaphore_store.renew.return_value = False

    await scheduler._renew_active_run_leases()

    scheduler._extend_run_lease_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_dedicated_loop_rejects_stale_storage_attempt() -> None:
    work_item = SimpleNamespace(
        id="job-1",
        dag_id="dag-1",
        run_owner="scheduler-1",
        run_attempt_id="attempt-2",
    )
    job_info = SimpleNamespace(
        status=JobStatus.RUNNING,
        run_owner="scheduler-1",
        run_attempt_id="attempt-1",
    )
    scheduler = build_scheduler(work_item, job_info)

    await scheduler._renew_active_run_leases()

    scheduler._extend_run_lease_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_storage_error_does_not_block_other_renewals() -> None:
    failed = SimpleNamespace(
        id="job-failed",
        dag_id="dag-1",
        run_owner="scheduler-1",
        run_attempt_id="attempt-failed",
    )
    healthy = SimpleNamespace(
        id="job-healthy",
        dag_id="dag-1",
        run_owner="scheduler-1",
        run_attempt_id="attempt-healthy",
    )
    healthy_info = SimpleNamespace(
        status=JobStatus.RUNNING,
        run_owner="scheduler-1",
        run_attempt_id="attempt-healthy",
    )
    get_info = AsyncMock(
        side_effect=[RuntimeError("storage unavailable"), healthy_info]
    )
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.list_jobs = AsyncMock(
        return_value={failed.id: failed, healthy.id: healthy}
    )
    scheduler.job_manager = SimpleNamespace(
        job_info_client=lambda: SimpleNamespace(get_info=get_info)
    )
    scheduler._extend_run_lease_db = AsyncMock(return_value={healthy.id})
    scheduler._semaphore_store = MagicMock()
    scheduler._scheduler_counter = MagicMock()
    scheduler.logger = MagicMock()
    scheduler.gateway_instance_id = "gateway-1"
    scheduler.lease_owner = "scheduler-1"

    await scheduler._renew_active_run_leases()

    scheduler._extend_run_lease_db.assert_awaited_once_with(
        [healthy.id],
        run_owner="scheduler-1",
        run_attempt_id="attempt-healthy",
    )
    scheduler.logger.error.assert_called_once()
