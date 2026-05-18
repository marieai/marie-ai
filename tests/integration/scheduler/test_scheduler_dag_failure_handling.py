import asyncio
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import marie.scheduler.psql as scheduler_psql
from marie.job.common import JobInfo, JobStatus
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import RecoveredRunLease, WorkInfo
from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.repository.job_repository import JobRepository
from marie.scheduler.state import WorkState


class FakeLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class RecordingRepository:
    def __init__(self, dag_state: str):
        self.dag_state = dag_state
        self.resolve_calls: list[str] = []
        self.cancel_calls: list[dict] = []

    async def resolve_dag_state(self, dag_id: str) -> str:
        self.resolve_calls.append(dag_id)
        return self.dag_state

    async def cancel_pending_jobs_for_dag(
        self,
        dag_id: str,
        output_metadata: dict | None = None,
        schema: str = "marie_scheduler",
    ) -> int:
        self.cancel_calls.append(
            {
                "dag_id": dag_id,
                "output_metadata": output_metadata or {},
                "schema": schema,
            }
        )
        return 2


class FlakyResolveRepository(RecordingRepository):
    def __init__(self, outcomes: list[object]):
        super().__init__(dag_state="active")
        self.outcomes = list(outcomes)

    async def resolve_dag_state(self, dag_id: str) -> str:
        self.resolve_calls.append(dag_id)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RecordingFrontier(MemoryFrontier):
    def __init__(self):
        super().__init__(higher_priority_wins=True, default_lease_ttl=0.25)
        self.finalize_calls: list[str] = []

    async def finalize_dag(self, dag_id: str) -> dict[str, int]:
        self.finalize_calls.append(dag_id)
        return await super().finalize_dag(dag_id)


class RecordingSemaphoreStore:
    def __init__(self):
        self.release_calls: list[tuple[str, str, str]] = []

    def release_owned(self, executor: str, ticket_id: str, owner: str):
        self.release_calls.append((executor, ticket_id, owner))
        return True


def build_work_item(job_id: str, dag_id: str, name: str = "extract") -> WorkInfo:
    now = datetime.now(timezone.utc)
    return WorkInfo(
        id=job_id,
        dag_id=dag_id,
        name=name,
        priority=0,
        data={
            "name": "GEN5_EXTRACT",
            "api_key": "test-api-key",
            "metadata": {"ref_type": "document"},
        },
        state=WorkState.CREATED,
        retry_limit=1,
        retry_delay=0,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=3600,
        keep_until=now + timedelta(days=1),
        dependencies=[],
        job_level=0,
    )


def build_scheduler(
    repository: RecordingRepository, frontier: RecordingFrontier
) -> PostgreSQLJobScheduler:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = FakeLogger()
    scheduler.repository = repository
    scheduler.frontier = frontier
    scheduler.active_dags = {}
    scheduler.max_concurrent_dags = 16
    scheduler._dag_admission_lock = asyncio.Lock()
    scheduler._dag_resolution_lock = AsyncJobLock()
    scheduler._terminal_dag_states = {}
    scheduler._job_cache = {}
    scheduler._scheduler_counters = defaultdict(int)
    scheduler.notify_calls: list[bool] = []
    scheduler.hydrated_dag_ids: list[str] = []
    scheduler._semaphore_store = RecordingSemaphoreStore()
    scheduler._dag_resolution_retry_limit = 2
    scheduler._dag_resolution_retry_delay = 0.0
    scheduler._dag_resolution_retry_backoff = False
    scheduler._dag_resolution_retry_max_delay = 0.0
    scheduler.lease_owner = "test-scheduler"

    async def notify_event() -> bool:
        scheduler.notify_calls.append(True)
        return True

    async def hydrate_single_dag_from_db(dag_id: str) -> bool:
        scheduler.hydrated_dag_ids.append(dag_id)
        return True

    scheduler.notify_event = notify_event
    scheduler.hydrate_single_dag_from_db = hydrate_single_dag_from_db
    return scheduler


@pytest.mark.asyncio
async def test_resolve_dag_status_failed_finalizes_frontier_and_cancels_pending_jobs(
    monkeypatch,
):
    dag_id = "dag-1"
    failed_job = build_work_item("job-1", dag_id)
    sibling_job = build_work_item("job-2", dag_id)
    frontier = RecordingFrontier()
    await frontier.add_dag(None, [failed_job, sibling_job])

    repository = RecordingRepository(dag_state="failed")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()
    scheduler._job_cache = {failed_job.id: failed_job, sibling_job.id: sibling_job}

    failed_toasts: list[dict] = []
    complete_toasts: list[dict] = []

    async def record_failed_toast(**kwargs):
        failed_toasts.append(kwargs)
        return True

    async def record_complete_toast(**kwargs):
        complete_toasts.append(kwargs)
        return True

    monkeypatch.setattr(scheduler_psql, "mark_as_failed_toast", record_failed_toast)
    monkeypatch.setattr(
        scheduler_psql, "mark_as_complete_toast", record_complete_toast
    )

    handled = await scheduler.resolve_dag_status(failed_job.id, failed_job)

    assert handled is True
    assert repository.resolve_calls == [dag_id]
    assert len(repository.cancel_calls) == 1
    assert repository.cancel_calls[0]["dag_id"] == dag_id
    assert (
        repository.cancel_calls[0]["output_metadata"]["resolved_by_job_id"]
        == failed_job.id
    )
    assert frontier.finalize_calls == [dag_id]
    assert dag_id not in scheduler.active_dags
    assert scheduler._terminal_dag_states == {dag_id: "failed"}
    assert failed_job.id not in scheduler._job_cache
    assert sibling_job.id not in scheduler._job_cache
    assert await frontier.get_jobs_by_dag_id(dag_id) == []
    assert len(failed_toasts) == 1
    assert failed_toasts[0]["job_id"] == dag_id
    assert failed_toasts[0]["status"] == "FAILED"
    assert complete_toasts == []


@pytest.mark.asyncio
async def test_resolve_dag_status_failed_is_idempotent(monkeypatch):
    dag_id = "dag-2"
    failed_job = build_work_item("job-3", dag_id)
    sibling_job = build_work_item("job-4", dag_id)
    frontier = RecordingFrontier()
    await frontier.add_dag(None, [failed_job, sibling_job])

    repository = RecordingRepository(dag_state="failed")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()
    scheduler._job_cache = {failed_job.id: failed_job, sibling_job.id: sibling_job}

    failed_toasts: list[dict] = []

    async def record_failed_toast(**kwargs):
        failed_toasts.append(kwargs)
        return True

    monkeypatch.setattr(scheduler_psql, "mark_as_failed_toast", record_failed_toast)
    monkeypatch.setattr(
        scheduler_psql,
        "mark_as_complete_toast",
        lambda **kwargs: pytest.fail("unexpected completion toast"),
    )

    first = await scheduler.resolve_dag_status(failed_job.id, failed_job)
    second = await scheduler.resolve_dag_status(sibling_job.id, sibling_job)

    assert first is True
    assert second is False
    assert repository.resolve_calls == [dag_id, dag_id]
    assert len(repository.cancel_calls) == 1
    assert frontier.finalize_calls == [dag_id]
    assert len(failed_toasts) == 1


@pytest.mark.asyncio
async def test_resolve_dag_status_retries_after_transient_error_and_succeeds(
    monkeypatch,
):
    dag_id = "dag-retry-success"
    failed_job = build_work_item("job-retry-success-1", dag_id)
    sibling_job = build_work_item("job-retry-success-2", dag_id)
    frontier = RecordingFrontier()
    await frontier.add_dag(None, [failed_job, sibling_job])

    repository = FlakyResolveRepository([RuntimeError("db busy"), "failed"])
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()
    scheduler._job_cache = {failed_job.id: failed_job, sibling_job.id: sibling_job}

    failed_toasts: list[dict] = []

    async def record_failed_toast(**kwargs):
        failed_toasts.append(kwargs)
        return True

    monkeypatch.setattr(scheduler_psql, "mark_as_failed_toast", record_failed_toast)
    monkeypatch.setattr(
        scheduler_psql,
        "mark_as_complete_toast",
        lambda **kwargs: pytest.fail("unexpected completion toast"),
    )

    handled = await scheduler._resolve_dag_status_with_retry(
        failed_job.id,
        failed_job,
        source="test",
    )

    assert handled is True
    assert repository.resolve_calls == [dag_id, dag_id]
    assert len(repository.cancel_calls) == 1
    assert frontier.finalize_calls == [dag_id]
    assert dag_id not in scheduler.active_dags
    assert scheduler._terminal_dag_states == {dag_id: "failed"}
    assert failed_job.id not in scheduler._job_cache
    assert sibling_job.id not in scheduler._job_cache
    assert len(failed_toasts) == 1


@pytest.mark.asyncio
async def test_resolve_dag_status_retry_exhaustion_returns_false():
    dag_id = "dag-retry-fail"
    failed_job = build_work_item("job-retry-fail-1", dag_id)
    sibling_job = build_work_item("job-retry-fail-2", dag_id)
    frontier = RecordingFrontier()
    await frontier.add_dag(None, [failed_job, sibling_job])

    repository = FlakyResolveRepository(
        [RuntimeError("db busy"), RuntimeError("still busy"), RuntimeError("timeout")]
    )
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()
    scheduler._job_cache = {failed_job.id: failed_job, sibling_job.id: sibling_job}

    handled = await scheduler._resolve_dag_status_with_retry(
        failed_job.id,
        failed_job,
        source="test",
    )

    assert handled is False
    assert repository.resolve_calls == [dag_id, dag_id, dag_id]
    assert repository.cancel_calls == []
    assert frontier.finalize_calls == []
    assert dag_id in scheduler.active_dags
    assert scheduler._terminal_dag_states == {}


@pytest.mark.asyncio
async def test_dag_state_notification_created_clears_terminal_guard():
    dag_id = "dag-3"
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="failed")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()
    scheduler._terminal_dag_states[dag_id] = "failed"

    await scheduler._handle_dag_state_notification(
        {"op": "UPDATE", "dag_id": dag_id, "state": "created"}
    )

    assert dag_id not in scheduler._terminal_dag_states
    assert scheduler.hydrated_dag_ids == [dag_id]
    assert dag_id not in scheduler.active_dags
    assert frontier.finalize_calls == [dag_id]
    assert scheduler.notify_calls == [True]


@pytest.mark.asyncio
async def test_control_flow_node_requeues_when_active_dag_limit_is_full():
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)
    scheduler.max_concurrent_dags = 1
    scheduler.active_dags["existing-dag"] = object()

    work_item = build_work_item("job-control", "new-dag")
    work_item.data["metadata"]["on"] = "noop://control"

    released_db: list[list[str]] = []
    released_local: list[str] = []
    activated: list[str] = []

    async def get_dag_by_id(dag_id: str):
        return object()

    async def mark_as_active_dag(wi: WorkInfo) -> bool:
        activated.append(wi.dag_id)
        return True

    async def release_lease_db(job_ids: list[str]) -> None:
        released_db.append(job_ids)

    async def release_lease_local(job_id: str) -> None:
        released_local.append(job_id)

    scheduler.get_dag_by_id = get_dag_by_id
    scheduler.mark_as_active_dag = mark_as_active_dag
    scheduler._release_lease_db = release_lease_db
    frontier.release_lease_local = release_lease_local

    await scheduler._process_control_flow_node(work_item)

    assert activated == []
    assert released_db == [[work_item.id]]
    assert released_local == [work_item.id]
    assert work_item.dag_id not in scheduler.active_dags


@pytest.mark.asyncio
async def test_control_flow_activation_marks_job_active_before_local_completion():
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)

    work_item = build_work_item("job-control", "dag-control")
    await frontier.add_dag(None, [work_item])

    activated_ids: list[list[str]] = []

    async def activate_from_lease(ids: list[str]) -> dict[str, str]:
        activated_ids.append(ids)
        return {work_item.id: "06a0ac88-5326-7f90-8000-0274669de089"}

    scheduler._activate_from_lease_db = activate_from_lease

    activated = await scheduler._activate_control_flow_job(work_item)

    assert activated is True
    assert activated_ids == [[work_item.id]]
    assert work_item.run_owner == "test-scheduler"
    assert work_item.run_attempt_id == "06a0ac88-5326-7f90-8000-0274669de089"
    assert frontier.jobs_by_id[work_item.id].state == WorkState.ACTIVE


@pytest.mark.asyncio
async def test_dag_state_notification_terminal_marks_dag_as_terminal():
    dag_id = "dag-4"
    work_item = build_work_item("job-5", dag_id)
    frontier = RecordingFrontier()
    await frontier.add_dag(None, [work_item])

    repository = RecordingRepository(dag_state="failed")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()

    await scheduler._handle_dag_state_notification(
        {"op": "UPDATE", "dag_id": dag_id, "state": "failed"}
    )

    assert scheduler._terminal_dag_states == {dag_id: "failed"}
    assert dag_id not in scheduler.active_dags
    assert frontier.finalize_calls == [dag_id]
    assert await frontier.get_jobs_by_dag_id(dag_id) == []
    assert scheduler.notify_calls == [True]


@pytest.mark.asyncio
async def test_handle_dispatch_failure_marks_job_for_retry_and_releases_semaphore():
    dag_id = "dag-5"
    work_item = build_work_item("job-6", dag_id)
    frontier = RecordingFrontier()
    await frontier.add_dag(None, [work_item])
    await frontier.take([work_item.id])

    repository = RecordingRepository(dag_state="failed")
    scheduler = build_scheduler(repository, frontier)
    fail_calls: list[dict] = []

    async def fake_fail(
        job_id: str,
        wi: WorkInfo,
        output_metadata: dict | None = None,
        **_kwargs,
    ):
        fail_calls.append(
            {
                "job_id": job_id,
                "output_metadata": output_metadata or {},
            }
        )
        return WorkState.RETRY.value

    async def fake_resolve_dag_status(*args, **kwargs):
        pytest.fail("retry path should not resolve DAG status")

    scheduler.fail = fake_fail
    scheduler.resolve_dag_status = fake_resolve_dag_status

    await scheduler._handle_dispatch_failure(
        work_item,
        "annotator_llm",
        work_item.id,
        RuntimeError("duplicate key"),
    )

    assert fail_calls == [
        {
            "job_id": work_item.id,
            "output_metadata": {
                "dispatch_failed": True,
                "dispatch_error": "duplicate key",
                "failure_stage": "enqueue",
            },
        }
    ]
    assert frontier.jobs_by_id[work_item.id].state == WorkState.RETRY
    assert scheduler._semaphore_store.release_calls == [
        ("annotator_llm", work_item.id, work_item.id)
    ]
    assert scheduler.notify_calls == [True]


@pytest.mark.asyncio
async def test_handle_dispatch_failure_marks_job_failed_and_resolves_dag():
    dag_id = "dag-6"
    work_item = build_work_item("job-7", dag_id)
    frontier = RecordingFrontier()
    await frontier.add_dag(None, [work_item])
    await frontier.take([work_item.id])

    repository = RecordingRepository(dag_state="failed")
    scheduler = build_scheduler(repository, frontier)
    fail_calls: list[dict] = []
    resolve_calls: list[tuple[str, str]] = []

    async def fake_fail(
        job_id: str,
        wi: WorkInfo,
        output_metadata: dict | None = None,
        **_kwargs,
    ):
        fail_calls.append(
            {
                "job_id": job_id,
                "output_metadata": output_metadata or {},
            }
        )
        return WorkState.FAILED.value

    async def fake_resolve_dag_status(job_id: str, wi: WorkInfo, *args, **kwargs):
        resolve_calls.append((job_id, wi.dag_id))
        return True

    scheduler.fail = fake_fail
    scheduler.resolve_dag_status = fake_resolve_dag_status

    await scheduler._handle_dispatch_failure(
        work_item,
        "annotator_llm",
        work_item.id,
        RuntimeError("dispatch failed"),
    )

    assert fail_calls == [
        {
            "job_id": work_item.id,
            "output_metadata": {
                "dispatch_failed": True,
                "dispatch_error": "dispatch failed",
                "failure_stage": "enqueue",
            },
        }
    ]
    assert frontier.jobs_by_id[work_item.id].state == WorkState.FAILED
    assert resolve_calls == [(work_item.id, dag_id)]
    assert scheduler._semaphore_store.release_calls == [
        ("annotator_llm", work_item.id, work_item.id)
    ]
    assert scheduler.notify_calls == [True]


@pytest.mark.asyncio
async def test_sync_terminal_job_state_succeeded_unblocks_children_and_notifies():
    dag_id = "dag-sync"
    parent = build_work_item("job-parent", dag_id)
    child = build_work_item("job-child", dag_id)
    child.dependencies = [parent.id]
    child.job_level = 1

    frontier = RecordingFrontier()
    await frontier.add_dag(None, [parent, child])

    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)
    scheduler._status_update_lock = AsyncJobLock()

    complete_calls: list[dict] = []
    resolve_calls: list[tuple[str, str]] = []

    async def fake_complete(
        job_id: str,
        wi: WorkInfo,
        output_metadata: dict | None = None,
        force: bool = False,
        **_kwargs,
    ) -> int:
        complete_calls.append(
            {
                "job_id": job_id,
                "output_metadata": output_metadata or {},
                "force": force,
            }
        )
        return 1

    async def fake_resolve_dag_status(job_id: str, wi: WorkInfo, *args, **kwargs):
        resolve_calls.append((job_id, wi.dag_id))
        return False

    async def fake_get_dag_by_id(dag_id: str):
        return None

    scheduler.complete = fake_complete
    scheduler.resolve_dag_status = fake_resolve_dag_status
    scheduler.get_dag_by_id = fake_get_dag_by_id

    old_end = int(
        (datetime.now(timezone.utc) - timedelta(minutes=10)).timestamp() * 1000
    )
    job_info = JobInfo(
        status=JobStatus.SUCCEEDED,
        entrypoint="test-entrypoint",
        end_time=old_end,
        run_owner="test-scheduler",
        run_attempt_id="06a0ac88-5326-7f90-8000-0274669de089",
    )

    synced = await scheduler._sync_terminal_job_state(
        parent.id,
        parent,
        job_info,
        min_sync_interval_seconds=300,
    )

    assert synced is True
    assert complete_calls == [
        {
            "job_id": parent.id,
            "output_metadata": {"synced": True},
            "force": False,
        }
    ]
    assert resolve_calls == [(parent.id, dag_id)]
    assert scheduler.notify_calls == [True]

    ready = await frontier.peek_ready(10)
    assert [wi.id for wi in ready] == [child.id]


@pytest.mark.asyncio
async def test_recovered_failure_updates_frontier_and_dag_failure_path(monkeypatch):
    dag_id = "dag-recovered-failed"
    failed_job = build_work_item("job-recovered-failed", dag_id)
    sibling_job = build_work_item("job-recovered-sibling", dag_id)

    frontier = RecordingFrontier()
    await frontier.add_dag(None, [failed_job, sibling_job])

    repository = RecordingRepository(dag_state="failed")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()
    scheduler._job_cache = {failed_job.id: failed_job, sibling_job.id: sibling_job}

    db_failed_job = failed_job.model_copy(update={"state": WorkState.FAILED})

    async def get_job_by_id(job_id: str):
        if job_id == failed_job.id:
            return db_failed_job
        return None

    repository.get_job_by_id = get_job_by_id

    failed_toasts: list[dict] = []

    async def record_failed_toast(**kwargs):
        failed_toasts.append(kwargs)
        return True

    monkeypatch.setattr(scheduler_psql, "mark_as_failed_toast", record_failed_toast)
    monkeypatch.setattr(
        scheduler_psql,
        "mark_as_complete_toast",
        lambda **kwargs: pytest.fail("unexpected completion toast"),
    )

    await scheduler._reconcile_recovered_run_leases(
        [
            RecoveredRunLease(
                id=failed_job.id,
                name=failed_job.name,
                previous_state=WorkState.ACTIVE.value,
                recovered_state="failed",
                dag_id=dag_id,
                retry_count=1,
                retry_limit=1,
                start_after=None,
                previous_run_owner="dead-scheduler",
                previous_run_attempt_id="06a0ac88-5326-7f90-8000-0274669de089",
            )
        ]
    )

    assert repository.resolve_calls == [dag_id]
    assert len(repository.cancel_calls) == 1
    assert frontier.finalize_calls == [dag_id]
    assert dag_id not in scheduler.active_dags
    assert failed_job.id not in scheduler._job_cache
    assert sibling_job.id not in scheduler._job_cache
    assert await frontier.get_jobs_by_dag_id(dag_id) == []
    assert len(failed_toasts) == 1
    assert scheduler._scheduler_counters[
        scheduler_psql.RUN_LEASE_RECOVERED_FAILED_TOTAL
    ] == 1


@pytest.mark.asyncio
async def test_late_success_for_old_run_attempt_updates_zero_rows(monkeypatch):
    work_item = build_work_item("job-late-success", "dag-late-success")
    work_item.state = WorkState.ACTIVE
    work_item.run_owner = "current-owner"
    work_item.run_attempt_id = "06a0ac88-5326-7f90-8000-0274669de089"

    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)
    scheduler._job_cache = {work_item.id: work_item}

    complete_calls: list[dict] = []
    completed_calls: list[str] = []
    trace_events: list[tuple[str, dict]] = []

    async def complete(job_id: str, wi: WorkInfo, *_args, **kwargs) -> int:
        complete_calls.append(
            {
                "job_id": job_id,
                "work_item": wi.id,
                "run_owner": kwargs.get("run_owner"),
                "run_attempt_id": kwargs.get("run_attempt_id"),
            }
        )
        return 0

    async def on_job_completed(job_id: str):
        completed_calls.append(job_id)

    scheduler.complete = complete
    scheduler.frontier.on_job_completed = on_job_completed
    monkeypatch.setattr(
        scheduler_psql,
        "scheduler_trace",
        lambda event, **fields: trace_events.append((event, fields)),
    )

    await scheduler.handle_job_event(
        JobStatus.SUCCEEDED.value,
        {
            "job_id": work_item.id,
            "run_owner": "old-owner",
            "run_attempt_id": "06a0ac88-5326-7f90-8000-111111111111",
        },
    )

    assert complete_calls == [
        {
            "job_id": work_item.id,
            "work_item": work_item.id,
            "run_owner": "old-owner",
            "run_attempt_id": "06a0ac88-5326-7f90-8000-111111111111",
        }
    ]
    assert completed_calls == []
    assert scheduler.notify_calls == []
    assert scheduler._scheduler_counters[
        scheduler_psql.TERMINAL_EVENT_STALE_ATTEMPT_TOTAL
    ] == 1
    assert [event for event, _fields in trace_events].count(
        scheduler_psql.TERMINAL_EVENT_STALE_ATTEMPT_TOTAL
    ) == 1


@pytest.mark.asyncio
async def test_late_failure_for_old_run_attempt_updates_zero_rows(monkeypatch):
    work_item = build_work_item("job-late-failure", "dag-late-failure")
    work_item.state = WorkState.ACTIVE
    work_item.run_owner = "current-owner"
    work_item.run_attempt_id = "06a0ac88-5326-7f90-8000-0274669de089"

    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)
    scheduler._job_cache = {work_item.id: work_item}

    fail_calls: list[dict] = []
    failed_calls: list[str] = []
    trace_events: list[tuple[str, dict]] = []

    async def fail(job_id: str, wi: WorkInfo, *_args, **kwargs) -> str | None:
        fail_calls.append(
            {
                "job_id": job_id,
                "work_item": wi.id,
                "run_owner": kwargs.get("run_owner"),
                "run_attempt_id": kwargs.get("run_attempt_id"),
            }
        )
        return None

    async def on_job_failed(job_id: str):
        failed_calls.append(job_id)

    scheduler.fail = fail
    scheduler.frontier.on_job_failed = on_job_failed
    monkeypatch.setattr(
        scheduler_psql,
        "scheduler_trace",
        lambda event, **fields: trace_events.append((event, fields)),
    )

    await scheduler.handle_job_event(
        JobStatus.FAILED.value,
        {
            "job_id": work_item.id,
            "run_owner": "old-owner",
            "run_attempt_id": "06a0ac88-5326-7f90-8000-222222222222",
        },
    )

    assert fail_calls == [
        {
            "job_id": work_item.id,
            "work_item": work_item.id,
            "run_owner": "old-owner",
            "run_attempt_id": "06a0ac88-5326-7f90-8000-222222222222",
        }
    ]
    assert failed_calls == []
    assert scheduler.notify_calls == []
    assert scheduler._scheduler_counters[
        scheduler_psql.TERMINAL_EVENT_STALE_ATTEMPT_TOTAL
    ] == 1
    assert [event for event, _fields in trace_events].count(
        scheduler_psql.TERMINAL_EVENT_STALE_ATTEMPT_TOTAL
    ) == 1


@pytest.mark.asyncio
async def test_stale_running_heartbeat_exposes_run_lease_counter(monkeypatch):
    work_item = build_work_item("job-stale-heartbeat", "dag-stale-heartbeat")
    work_item.state = WorkState.ACTIVE

    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)
    scheduler._job_cache = {work_item.id: work_item}

    extend_calls: list[dict] = []
    trace_events: list[tuple[str, dict]] = []

    async def extend_run_lease(ids: list[str], *, run_owner: str, run_attempt_id: str):
        extend_calls.append(
            {
                "ids": ids,
                "run_owner": run_owner,
                "run_attempt_id": run_attempt_id,
            }
        )
        return set()

    scheduler._extend_run_lease_db = extend_run_lease
    monkeypatch.setattr(
        scheduler_psql,
        "scheduler_trace",
        lambda event, **fields: trace_events.append((event, fields)),
    )

    await scheduler.handle_job_event(
        JobStatus.RUNNING.value,
        {
            "job_id": work_item.id,
            "run_owner": "old-owner",
            "run_attempt_id": "06a0ac88-5326-7f90-8000-333333333333",
        },
    )

    assert extend_calls == [
        {
            "ids": [work_item.id],
            "run_owner": "old-owner",
            "run_attempt_id": "06a0ac88-5326-7f90-8000-333333333333",
        }
    ]
    assert scheduler._scheduler_counters[
        scheduler_psql.RUN_LEASE_EXTEND_STALE_ATTEMPT_TOTAL
    ] == 1
    assert [event for event, _fields in trace_events].count(
        scheduler_psql.RUN_LEASE_EXTEND_STALE_ATTEMPT_TOTAL
    ) == 1


@pytest.mark.asyncio
async def test_admit_dag_requires_db_activation_success():
    dag_id = "dag-admit"
    work_item = build_work_item("job-admit", dag_id)
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)

    async def mark_as_active_dag(_work_info: WorkInfo) -> bool:
        return False

    scheduler.mark_as_active_dag = mark_as_active_dag

    admitted = await scheduler._admit_dag(work_item, object(), source="test")

    assert admitted is False
    assert dag_id not in scheduler.active_dags


@pytest.mark.asyncio
async def test_scheduler_start_initializes_notification_listener_before_hydration(
    monkeypatch,
):
    order: list[str] = []

    class Repo:
        async def is_installed(self, _schema):
            order.append("is_installed")
            return True

        async def get_defined_queues(self, _schema):
            order.append("get_defined_queues")
            return set()

        def create_tables(self, _schema):
            order.append("create_tables")

        async def validate_durable_scheduler_schema(self, _schema):
            order.append("validate_durable_scheduler_schema")

        async def create_queue(self, _queue):
            order.append("create_queue")

    class NotificationService:
        async def start(self):
            order.append("notification_start")

    class MaintenanceService:
        maintenance_interval = 30

        async def start(self):
            order.append("maintenance_start")

    class DummyTask:
        def __init__(self, coro):
            self._coro = coro

        def done(self):
            return False

        def cancel(self):
            self._coro.close()

    def fake_create_task(coro):
        coro.close()
        return DummyTask(coro)

    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = FakeLogger()
    scheduler.repository = Repo()
    scheduler.known_queues = set()
    scheduler.notification_service = NotificationService()
    scheduler.maintenance_service = MaintenanceService()
    scheduler.max_workers = 0

    async def hydrate_from_db():
        order.append("hydrate_from_db")

    async def notify_event():
        order.append("notify_event")
        return True

    async def noop():
        return None

    scheduler.hydrate_from_db = hydrate_from_db
    scheduler.notify_event = notify_event
    scheduler._sync = noop
    scheduler._poll = noop
    scheduler._sync_dag = noop
    scheduler._process_submission_queue = lambda _worker_id: noop()
    scheduler._PostgreSQLJobScheduler__monitor_deployment_updates = noop

    monkeypatch.setattr(scheduler_psql.asyncio, "create_task", fake_create_task)

    await scheduler.start()

    assert order.index("notification_start") < order.index("hydrate_from_db")
    assert order[-1] == "notify_event"


@pytest.mark.asyncio
async def test_evict_dag_from_memory_finalizes_frontier_and_clears_terminal_state():
    dag_id = "dag-evict"
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()
    scheduler._terminal_dag_states[dag_id] = "completed"
    scheduler.dag_service = SimpleNamespace(
        remove_dag=lambda dag_id, _reason: scheduler.active_dags.pop(dag_id, None)
        is not None
    )

    work_item = build_work_item("job-evict", dag_id)
    await frontier.add_dag(None, [work_item])

    removed = await scheduler._evict_dag_from_memory(
        dag_id, "no longer active or deleted in database"
    )

    assert removed is True
    assert frontier.finalize_calls == [dag_id]
    assert dag_id not in scheduler.active_dags
    assert dag_id not in scheduler._terminal_dag_states
    assert await frontier.get_jobs_by_dag_id(dag_id) == []


@pytest.mark.asyncio
async def test_sync_dag_once_reaps_stale_memory_dags_and_notifies():
    repository = RecordingRepository(dag_state="active")
    frontier = RecordingFrontier()
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags = {
        "dag-valid": object(),
        "dag-stale": object(),
    }
    scheduler._terminal_dag_states = {"dag-stale": "completed"}

    removed: list[tuple[str, str]] = []
    resolved: list[str] = []

    def remove_dag(dag_id: str, reason: str) -> bool:
        removed.append((dag_id, reason))
        scheduler.active_dags.pop(dag_id, None)
        return True

    scheduler.dag_service = SimpleNamespace(remove_dag=remove_dag)

    async def get_active_dag_ids(_dag_ids: list[str]) -> set[str]:
        return {"dag-valid"}

    scheduler.repository.get_active_dag_ids = get_active_dag_ids
    original_resolve_dag_state = scheduler.repository.resolve_dag_state

    async def resolve_dag_state(dag_id: str) -> str:
        resolved.append(dag_id)
        if dag_id == "dag-stale":
            return "completed"
        return await original_resolve_dag_state(dag_id)

    scheduler.repository.resolve_dag_state = resolve_dag_state

    await scheduler._sync_dag_once()

    assert removed == [
        ("dag-stale", "no longer active or deleted in database")
    ]
    assert resolved == ["dag-valid", "dag-stale"]
    assert "dag-stale" not in scheduler.active_dags
    assert "dag-stale" not in scheduler._terminal_dag_states
    assert scheduler.notify_calls == [True]


@pytest.mark.asyncio
async def test_submission_priority_refresh_request_does_not_refresh_inline():
    repository = RecordingRepository(dag_state="active")
    frontier = RecordingFrontier()
    scheduler = build_scheduler(repository, frontier)
    scheduler.priority_refresh_interval = 10
    scheduler._submission_count = 10
    scheduler._request_queue = asyncio.Queue()
    scheduler._pending_requests = {}
    scheduler._next_priority_refresh_at = time.monotonic() + 60.0

    refresh_calls: list[str] = []

    async def refresh_job_priorities(source: str = "unknown") -> int:
        refresh_calls.append(source)
        return 1

    scheduler._refresh_job_priorities = refresh_job_priorities

    await scheduler._handle_priority_refresh()

    assert refresh_calls == []
    assert scheduler._next_priority_refresh_at > time.monotonic()
    assert scheduler.notify_calls == []


@pytest.mark.asyncio
async def test_submission_priority_refresh_request_wakes_when_due():
    repository = RecordingRepository(dag_state="active")
    frontier = RecordingFrontier()
    scheduler = build_scheduler(repository, frontier)
    scheduler.priority_refresh_interval = 10
    scheduler._submission_count = 10
    scheduler._request_queue = asyncio.Queue()
    scheduler._pending_requests = {}
    scheduler._next_priority_refresh_at = time.monotonic() - 1.0

    refresh_calls: list[str] = []

    async def refresh_job_priorities(source: str = "unknown") -> int:
        refresh_calls.append(source)
        return 1

    scheduler._refresh_job_priorities = refresh_job_priorities

    await scheduler._handle_priority_refresh()

    assert refresh_calls == []
    assert scheduler.notify_calls == [True]


def test_limit_planned_jobs_to_available_slots_keeps_order_and_caps_per_executor():
    dag_id = "dag-limit"
    now = datetime.now(timezone.utc)

    def make_job(job_id: str, endpoint: str) -> tuple[str, WorkInfo]:
        return (
            endpoint,
            WorkInfo(
                id=job_id,
                dag_id=dag_id,
                name="extract",
                priority=0,
                data={"metadata": {"on": endpoint}},
                state=WorkState.CREATED,
                retry_limit=1,
                retry_delay=0,
                retry_backoff=False,
                start_after=now,
                expire_in_seconds=3600,
                keep_until=now + timedelta(days=1),
                dependencies=[],
                job_level=1,
            ),
        )

    planned = [
        make_job("extract-1", "extract_executor://document/extract"),
        make_job("extract-2", "extract_executor://document/extract"),
        make_job("parser-1", "annotator_parser://document/parse"),
        make_job("parser-2", "annotator_parser://document/parse"),
        make_job("parser-3", "annotator_parser://document/parse"),
    ]

    limited = scheduler_psql.limit_planned_jobs_to_available_slots(
        planned,
        {"extract_executor": 1, "annotator_parser": 2},
    )

    assert [wi.id for _, wi in limited] == [
        "extract-1",
        "parser-1",
        "parser-2",
    ]


def test_regular_candidates_cover_available_slots_requires_enough_per_executor():
    dag_id = "dag-coverage"
    now = datetime.now(timezone.utc)

    def make_job(job_id: str, endpoint: str) -> WorkInfo:
        return WorkInfo(
            id=job_id,
            dag_id=dag_id,
            name="extract",
            priority=0,
            data={"metadata": {"on": endpoint}},
            state=WorkState.CREATED,
            retry_limit=1,
            retry_delay=0,
            retry_backoff=False,
            start_after=now,
            expire_in_seconds=3600,
            keep_until=now + timedelta(days=1),
            dependencies=[],
            job_level=1,
        )

    assert (
        scheduler_psql.regular_candidates_cover_available_slots(
            [
                make_job("extract-1", "extract_executor://document/extract"),
                make_job("extract-2", "extract_executor://document/extract"),
            ],
            {"extract_executor": 3},
        )
        is False
    )
    assert (
        scheduler_psql.regular_candidates_cover_available_slots(
            [
                make_job("extract-1", "extract_executor://document/extract"),
                make_job("extract-2", "extract_executor://document/extract"),
                make_job("extract-3", "extract_executor://document/extract"),
            ],
            {"extract_executor": 3},
        )
        is True
    )
    assert (
        scheduler_psql.regular_candidates_cover_available_slots(
            [
                make_job("extract-1", "extract_executor://document/extract"),
                make_job("extract-2", "extract_executor://document/extract"),
                make_job("parser-1", "annotator_parser://document/parse"),
            ],
            {"extract_executor": 2, "annotator_parser": 2},
        )
        is False
    )


def test_repository_record_to_work_info_normalizes_uuid_fields():
    repository = object.__new__(JobRepository)
    now = datetime.now(timezone.utc)
    job_id = uuid.uuid4()
    dag_id = uuid.uuid4()

    work_info = repository._record_to_work_info(
        (
            job_id,
            "extract",
            0,
            WorkState.CREATED.value,
            1,
            now,
            timedelta(seconds=60),
            {},
            0,
            False,
            now + timedelta(days=1),
            dag_id,
            0,
            None,
            None,
            "test-scheduler",
            uuid.uuid4(),
        )
    )

    assert work_info.id == str(job_id)
    assert work_info.dag_id == str(dag_id)


@pytest.mark.asyncio
async def test_process_control_flow_node_notifies_after_unblocking_children(monkeypatch):
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = FakeLogger()
    scheduler.distributed_scheduler = False
    scheduler.active_dags = {"dag-cf": object()}
    scheduler._topology_cache = SimpleNamespace(
        get_sorted_nodes_and_levels=lambda _dag, _dag_id: ([], {"noop-root": 2, "child": 1})
    )
    scheduler.frontier = SimpleNamespace(
        leased_until={},
        on_job_completed=None,
    )

    order: list[str] = []

    async def complete(job_id: str, _wi: WorkInfo, *_args, **_kwargs):
        order.append(f"complete:{job_id}")
        return 1

    async def on_job_completed(job_id: str):
        order.append(f"frontier_completed:{job_id}")

    async def notify_event() -> bool:
        order.append("notify_event")
        return True

    async def resolve_dag_status_with_retry(*_args, **_kwargs):
        order.append("resolve_dag_status")
        return True

    scheduler.complete = complete
    scheduler.notify_event = notify_event
    scheduler._resolve_dag_status_with_retry = resolve_dag_status_with_retry
    scheduler.frontier.on_job_completed = on_job_completed
    
    async def activate_control_flow_job(_wi: WorkInfo) -> bool:
        _wi.run_owner = "test-scheduler"
        _wi.run_attempt_id = "06a0ac88-5326-7f90-8000-0274669de089"
        return True

    scheduler._activate_control_flow_job = activate_control_flow_job

    monkeypatch.setattr(scheduler_psql, "get_node_from_dag", lambda *_args, **_kwargs: object())

    started_calls: list[dict] = []

    async def mark_started_toast(**kwargs):
        started_calls.append(kwargs)
        return True

    monkeypatch.setattr(scheduler_psql, "mark_as_started_toast", mark_started_toast)

    work_item = WorkInfo(
        id="noop-root",
        dag_id="dag-cf",
        name="extract",
        priority=0,
        data={
            "name": "extract",
            "api_key": "api-key",
            "metadata": {"on": "noop://noop", "ref_type": "extract"},
        },
        state=WorkState.CREATED,
        retry_limit=1,
        retry_delay=0,
        retry_backoff=False,
        start_after=datetime.now(timezone.utc),
        expire_in_seconds=3600,
        keep_until=datetime.now(timezone.utc) + timedelta(days=1),
        dependencies=[],
        job_level=2,
    )

    await scheduler._process_control_flow_node(work_item)

    assert order == [
        "complete:noop-root",
        "frontier_completed:noop-root",
        "notify_event",
    ]
    assert len(started_calls) == 1


@pytest.mark.asyncio
async def test_wait_for_dispatch_wake_returns_early_on_notify():
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler._event_queue = asyncio.Queue()
    scheduler._debounced_notify = False

    await scheduler.notify_event()

    loop = asyncio.get_running_loop()
    started = loop.time()
    woke = await scheduler._wait_for_dispatch_wake(1.0)
    elapsed = loop.time() - started

    assert woke is True
    assert elapsed < 0.2
    assert scheduler._debounced_notify is False
