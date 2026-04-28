import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import marie.scheduler.psql as scheduler_psql
from marie.job.common import JobInfo, JobStatus
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.psql import PostgreSQLJobScheduler
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
    scheduler.notify_calls: list[bool] = []
    scheduler.hydrated_dag_ids: list[str] = []
    scheduler._semaphore_store = RecordingSemaphoreStore()

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

    async def fake_fail(job_id: str, wi: WorkInfo, output_metadata: dict | None = None):
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

    async def fake_fail(job_id: str, wi: WorkInfo, output_metadata: dict | None = None):
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
    ) -> None:
        complete_calls.append(
            {
                "job_id": job_id,
                "output_metadata": output_metadata or {},
                "force": force,
            }
        )

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
            "force": True,
        }
    ]
    assert resolve_calls == [(parent.id, dag_id)]
    assert scheduler.notify_calls == [True]

    ready = await frontier.peek_ready(10)
    assert [wi.id for wi in ready] == [child.id]


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
async def test_blocking_sync_dag_reaps_stale_memory_dags_and_notifies(monkeypatch):
    repository = RecordingRepository(dag_state="active")
    frontier = RecordingFrontier()
    scheduler = build_scheduler(repository, frontier)
    scheduler.running = True
    scheduler._loop = object()
    scheduler.active_dags = {
        "dag-valid": object(),
        "dag-stale": object(),
    }
    scheduler._terminal_dag_states = {"dag-stale": "completed"}

    removed: list[tuple[str, str]] = []

    def remove_dag(dag_id: str, reason: str) -> bool:
        removed.append((dag_id, reason))
        scheduler.active_dags.pop(dag_id, None)
        return True

    scheduler.dag_service = SimpleNamespace(remove_dag=remove_dag)

    async def get_active_dag_ids(_dag_ids: list[str]) -> set[str]:
        return {"dag-valid"}

    scheduler.repository.get_active_dag_ids = get_active_dag_ids

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):
            return self._value

    def fake_run_coroutine_threadsafe(coro, _loop):
        try:
            coro_name = coro.cr_code.co_name
            if coro_name == "get_active_dag_ids":
                return FakeFuture({"dag-valid"})
            if coro_name == "notify_event":
                scheduler.notify_calls.append(True)
                return FakeFuture(True)
            raise AssertionError(f"Unexpected coroutine submitted: {coro_name}")
        finally:
            coro.close()

    def fake_sleep(_interval: float) -> None:
        scheduler.running = False

    monkeypatch.setattr(scheduler_psql.asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)
    monkeypatch.setattr(scheduler_psql.time, "sleep", fake_sleep)

    scheduler._blocking_sync_dag(interval=0)

    assert removed == [
        ("dag-stale", "no longer active or deleted in database")
    ]
    assert "dag-stale" not in scheduler.active_dags
    assert scheduler.notify_calls == [True]
