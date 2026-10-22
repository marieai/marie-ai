from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.job.common import JobStatus
from marie.scheduler.models import RecoveredRunLease
from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.state import WorkState


def build_scheduler(queue="foreign"):
    item = SimpleNamespace(id="job", dag_id="dag", name=queue, state=WorkState.ACTIVE)
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.subscribed_queues = frozenset({"extract"})
    scheduler.repository = SimpleNamespace(get_job_by_id=AsyncMock(return_value=item))
    scheduler._job_cache = {}
    scheduler.frontier = SimpleNamespace(
        jobs_by_id={}, on_job_failed=AsyncMock(), on_job_retry=AsyncMock()
    )
    scheduler.dag_service = SimpleNamespace(
        consumption_rejection_reason=AsyncMock(return_value=None),
        resolve_dag_status_with_retry=AsyncMock(),
    )
    scheduler.hydrate_single_dag_from_db = AsyncMock(return_value=False)
    scheduler.attempt_lifecycle_service = SimpleNamespace(
        transition_terminal=AsyncMock(return_value=True)
    )
    scheduler.logger = MagicMock()
    scheduler.gateway_instance_id = "gateway"
    scheduler.lease_owner = "owner"
    scheduler._scheduler_counter = MagicMock()
    return scheduler, item


@pytest.mark.parametrize("state", ["retry", "failed"])
async def test_foreign_recovery_callback_has_no_side_effects(state):
    scheduler, _ = build_scheduler()
    recovery = RecoveredRunLease(
        id="job",
        dag_id="dag",
        name="foreign",
        previous_state="active",
        recovered_state=state,
        retry_count=1,
        retry_limit=3,
        start_after=None,
        previous_run_owner="old",
        previous_run_attempt_id="attempt",
    )
    await scheduler._reconcile_recovered_run_leases([recovery])
    assert scheduler._job_cache == {}
    scheduler.hydrate_single_dag_from_db.assert_not_awaited()
    scheduler.frontier.on_job_failed.assert_not_awaited()
    scheduler.dag_service.resolve_dag_status_with_retry.assert_not_awaited()


async def test_foreign_storage_repair_has_no_terminal_effect():
    scheduler, item = build_scheduler()
    evidence = SimpleNamespace(
        status=JobStatus.SUCCEEDED,
        end_time=(datetime.now(timezone.utc).timestamp() - 600) * 1000,
        run_owner="old",
        run_attempt_id="attempt",
    )
    assert not await scheduler._sync_terminal_job_state(
        "job", item, evidence, min_sync_interval_seconds=300
    )
    scheduler.attempt_lifecycle_service.transition_terminal.assert_not_awaited()
