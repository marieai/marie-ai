import asyncio
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import marie.scheduler.psql as scheduler_psql
import marie.scheduler.services.control_flow_execution_service as control_flow_module
from marie.job.common import JobInfo, JobStatus
from marie.query_planner.base import Query, QueryPlan
from marie.query_planner.branching import SkipReason
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import RecoveredRunLease, WorkInfo
from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.repository.job_repository import JobRepository
from marie.scheduler.services.attempt_lifecycle_service import (
    AttemptLifecycleService,
)
from marie.scheduler.services.control_flow_execution_service import (
    ControlFlowExecutionService,
)
from marie.scheduler.services.dag_management_service import DAGManagementService
from marie.scheduler.services.scheduler_runtime import SchedulerRuntime
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
    def __init__(self, dag_state: str, *, mark_active_result: bool = True):
        self.dag_state = dag_state
        self.mark_active_result = mark_active_result
        self.resolve_calls: list[str] = []
        self.cancel_calls: list[dict] = []
        self.mark_active_calls: list[str] = []
        self.released_lease_calls: list[list[str]] = []

    async def mark_dag_as_active(self, dag_id: str) -> bool:
        self.mark_active_calls.append(dag_id)
        return self.mark_active_result

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

    async def record_job_attempt_terminal(self, **_kwargs) -> None:
        return None

    async def get_dag_by_id(self, _dag_id: str):
        return None

    async def release_lease(self, job_ids: list[str]) -> set[str]:
        self.released_lease_calls.append(job_ids)
        return set(job_ids)


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

    def release_owned(
        self,
        executor: str,
        ticket_id: str,
        owner: str,
        run_attempt_id: str | None = None,
    ) -> bool:
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
    scheduler.running = True
    scheduler.repository = repository
    scheduler.frontier = frontier
    scheduler.active_dags = {}
    scheduler.max_concurrent_dags = 16
    scheduler._dag_admission_lock = asyncio.Lock()
    scheduler._status_update_lock = AsyncJobLock()
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
    scheduler.gateway_instance_id = "test-gateway"
    scheduler.run_ttl_seconds = 60
    scheduler.priority_refresh_enabled = True
    scheduler._priority_refresh_event = asyncio.Event()
    scheduler._priority_refresh_source = "test"
    scheduler._priority_refresh_running = False
    scheduler.priority_refresh_interval_seconds = 5.0
    scheduler.submission_service = SimpleNamespace(
        submission_count=0,
        queue_size=0,
        pending_count=0,
    )

    async def notify_event() -> bool:
        scheduler.notify_calls.append(True)
        return True

    async def hydrate_single_dag_from_db(dag_id: str) -> bool:
        scheduler.hydrated_dag_ids.append(dag_id)
        return True

    scheduler.notify_event = notify_event
    scheduler.hydrate_single_dag_from_db = hydrate_single_dag_from_db
    scheduler.dag_service = DAGManagementService(
        repository=repository,
        frontier=frontier,
        active_dags=scheduler.active_dags,
        notify_callback=notify_event,
        max_active_dags=scheduler.max_concurrent_dags,
        admission_lock=scheduler._dag_admission_lock,
        job_cache=scheduler._job_cache,
        terminal_event_callback=scheduler._emit_dag_terminal_event,
        resolution_retry_limit=scheduler._dag_resolution_retry_limit,
        resolution_retry_delay=scheduler._dag_resolution_retry_delay,
        resolution_retry_backoff=scheduler._dag_resolution_retry_backoff,
        resolution_retry_max_delay=scheduler._dag_resolution_retry_max_delay,
    )
    scheduler.dag_service.hydrate_single_dag = hydrate_single_dag_from_db
    scheduler.control_flow_service = ControlFlowExecutionService(
        repository=repository,
        frontier=frontier,
        dag_service=scheduler.dag_service,
        status_update_lock=scheduler._status_update_lock,
        topology_cache=DagTopologyCache(),
        job_cache=scheduler._job_cache,
        lease_owner=scheduler.lease_owner,
        run_ttl_seconds=scheduler.run_ttl_seconds,
        gateway_instance_id=scheduler.gateway_instance_id,
        notify_callback=notify_event,
    )
    scheduler.attempt_lifecycle_service = AttemptLifecycleService(
        repository=repository,
        frontier=frontier,
        dag_service=scheduler.dag_service,
        control_flow_service=scheduler.control_flow_service,
        status_update_lock=scheduler._status_update_lock,
        job_cache=scheduler._job_cache,
        scheduler_lease_owner=scheduler.lease_owner,
        gateway_instance_id=scheduler.gateway_instance_id,
        notify_callback=notify_event,
        counter_callback=scheduler._scheduler_counter,
    )
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
    scheduler._job_cache.update(
        {failed_job.id: failed_job, sibling_job.id: sibling_job}
    )

    failed_toasts: list[dict] = []
    complete_toasts: list[dict] = []

    async def record_failed_toast(**kwargs):
        failed_toasts.append(kwargs)
        return True

    async def record_complete_toast(**kwargs):
        complete_toasts.append(kwargs)
        return True

    monkeypatch.setattr(scheduler_psql, "mark_as_failed_toast", record_failed_toast)
    monkeypatch.setattr(scheduler_psql, "mark_as_complete_toast", record_complete_toast)

    handled = await scheduler.dag_service.resolve_dag_status(failed_job.id, failed_job)

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
    assert scheduler.dag_service._terminal_dag_states == {dag_id: "failed"}
    assert scheduler.dag_service._admission_event.is_set()
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
    scheduler._job_cache.update(
        {failed_job.id: failed_job, sibling_job.id: sibling_job}
    )

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

    first = await scheduler.dag_service.resolve_dag_status(failed_job.id, failed_job)
    second = await scheduler.dag_service.resolve_dag_status(sibling_job.id, sibling_job)

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
    scheduler._job_cache.update(
        {failed_job.id: failed_job, sibling_job.id: sibling_job}
    )

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

    handled = await scheduler.dag_service.resolve_dag_status_with_retry(
        failed_job.id,
        failed_job,
        source="test",
    )

    assert handled is True
    assert repository.resolve_calls == [dag_id, dag_id]
    assert len(repository.cancel_calls) == 1
    assert frontier.finalize_calls == [dag_id]
    assert dag_id not in scheduler.active_dags
    assert scheduler.dag_service._terminal_dag_states == {dag_id: "failed"}
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
    scheduler._job_cache.update(
        {failed_job.id: failed_job, sibling_job.id: sibling_job}
    )

    handled = await scheduler.dag_service.resolve_dag_status_with_retry(
        failed_job.id,
        failed_job,
        source="test",
    )

    assert handled is False
    assert repository.resolve_calls == [dag_id, dag_id, dag_id]
    assert repository.cancel_calls == []
    assert frontier.finalize_calls == []
    assert dag_id in scheduler.active_dags
    assert scheduler.dag_service._terminal_dag_states == {}


@pytest.mark.asyncio
async def test_dag_state_notification_created_clears_terminal_guard():
    dag_id = "dag-3"
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="failed")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()
    scheduler.dag_service._terminal_dag_states[dag_id] = "failed"

    await scheduler.dag_service.handle_state_change(
        {"op": "UPDATE", "dag_id": dag_id, "state": "created"}
    )

    assert dag_id not in scheduler.dag_service._terminal_dag_states
    assert scheduler.hydrated_dag_ids == []
    assert scheduler.dag_service._admission_event.is_set()
    assert dag_id not in scheduler.active_dags
    assert frontier.finalize_calls == [dag_id]
    assert scheduler.notify_calls == [True]


@pytest.mark.asyncio
async def test_control_flow_node_requeues_when_active_dag_limit_is_full():
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)
    scheduler.max_concurrent_dags = 1
    scheduler.dag_service.max_active_dags = 1
    scheduler.active_dags["existing-dag"] = object()

    work_item = build_work_item("job-control", "new-dag")
    work_item.data["metadata"]["on"] = "noop://control"

    released_local: list[str] = []

    async def get_dag_by_id(dag_id: str):
        return object()

    async def release_lease_local(job_id: str) -> None:
        released_local.append(job_id)

    scheduler.dag_service.get_dag = get_dag_by_id
    frontier.release_lease_local = release_lease_local

    await scheduler.control_flow_service.process_node(work_item)

    assert repository.mark_active_calls == []
    assert repository.released_lease_calls == [[work_item.id]]
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

    async def activate_from_lease(
        *,
        job_ids: list[str],
        owner: str,
        run_ttl_seconds: int,
        gateway_instance_id: str,
    ) -> dict[str, str]:
        activated_ids.append(job_ids)
        assert owner == "test-scheduler"
        assert run_ttl_seconds == 60
        assert gateway_instance_id == "test-gateway"
        return {work_item.id: "06a0ac88-5326-7f90-8000-0274669de089"}

    repository.activate_from_lease = activate_from_lease

    activated = await scheduler.control_flow_service._activate(work_item)

    assert activated is True
    assert activated_ids == [[work_item.id]]
    assert work_item.run_owner == "test-scheduler"
    assert work_item.run_attempt_id == "06a0ac88-5326-7f90-8000-0274669de089"
    assert frontier.jobs_by_id[work_item.id].state == WorkState.ACTIVE


@pytest.mark.asyncio
async def test_dag_state_notification_terminal_evicts_dag():
    dag_id = "dag-4"
    work_item = build_work_item("job-5", dag_id)
    frontier = RecordingFrontier()
    await frontier.add_dag(None, [work_item])

    repository = RecordingRepository(dag_state="failed")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()

    await scheduler.dag_service.handle_state_change(
        {"op": "UPDATE", "dag_id": dag_id, "state": "failed"}
    )

    assert scheduler.dag_service._terminal_dag_states == {}
    assert dag_id not in scheduler.active_dags
    assert frontier.finalize_calls == [dag_id]
    assert await frontier.get_jobs_by_dag_id(dag_id) == []
    assert scheduler.notify_calls == [True]


@pytest.mark.asyncio
async def test_terminal_notification_does_not_suppress_resolution_side_effects(
    monkeypatch,
):
    dag_id = "dag-notify-resolve"
    work_item = build_work_item("job-notify-resolve", dag_id)
    frontier = RecordingFrontier()
    await frontier.add_dag(None, [work_item])
    repository = RecordingRepository(dag_state="failed")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()

    failed_toasts: list[dict] = []

    async def record_failed_toast(**kwargs):
        failed_toasts.append(kwargs)
        return True

    monkeypatch.setattr(scheduler_psql, "mark_as_failed_toast", record_failed_toast)

    await scheduler.dag_service.handle_state_change(
        {"op": "UPDATE", "dag_id": dag_id, "state": "failed"}
    )
    handled = await scheduler.dag_service.resolve_dag_status(work_item.id, work_item)

    assert handled is True
    assert repository.cancel_calls[0]["dag_id"] == dag_id
    assert scheduler.dag_service._terminal_dag_states == {dag_id: "failed"}
    assert failed_toasts[0]["job_id"] == dag_id


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
        queue_name: str,
        output_metadata: dict | None = None,
        **_kwargs,
    ):
        fail_calls.append(
            {
                "job_id": job_id,
                "output_metadata": output_metadata or {},
            }
        )
        return 1, WorkState.RETRY.value

    async def fake_resolve_dag_status(*args, **kwargs):
        pytest.fail("retry path should not resolve DAG status")

    repository.fail_job = fake_fail
    scheduler.dag_service.resolve_dag_status_with_retry = fake_resolve_dag_status

    await scheduler._handle_dispatch_failure(
        work_item,
        "annotator_llm",
        work_item.id,
        RuntimeError("duplicate key"),
        run_owner="test-scheduler",
        run_attempt_id="06a0ac88-5326-7f90-8000-0274669de089",
    )

    assert fail_calls == [
        {
            "job_id": work_item.id,
            "output_metadata": {
                "dispatch_failed": True,
                "dispatch_error": "duplicate key",
                "failure_stage": "enqueue",
                "failure_source": "dispatch_failure",
                "error_message": "duplicate key",
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
        queue_name: str,
        output_metadata: dict | None = None,
        **_kwargs,
    ):
        fail_calls.append(
            {
                "job_id": job_id,
                "output_metadata": output_metadata or {},
            }
        )
        return 1, WorkState.FAILED.value

    async def fake_resolve_dag_status(job_id: str, wi: WorkInfo, *args, **kwargs):
        resolve_calls.append((job_id, wi.dag_id))
        return True

    repository.fail_job = fake_fail
    scheduler.dag_service.resolve_dag_status_with_retry = fake_resolve_dag_status

    await scheduler._handle_dispatch_failure(
        work_item,
        "annotator_llm",
        work_item.id,
        RuntimeError("dispatch failed"),
        run_owner="test-scheduler",
        run_attempt_id="06a0ac88-5326-7f90-8000-0274669de089",
    )

    assert fail_calls == [
        {
            "job_id": work_item.id,
            "output_metadata": {
                "dispatch_failed": True,
                "dispatch_error": "dispatch failed",
                "failure_stage": "enqueue",
                "failure_source": "dispatch_failure",
                "error_message": "dispatch failed",
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
    parent.state = WorkState.ACTIVE
    parent.run_owner = "test-scheduler"
    parent.run_attempt_id = "06a0ac88-5326-7f90-8000-0274669de089"
    child = build_work_item("job-child", dag_id)
    child.dependencies = [parent.id]
    child.job_level = 1

    frontier = RecordingFrontier()
    await frontier.add_dag(None, [parent, child])

    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)

    complete_calls: list[dict] = []
    resolve_calls: list[tuple[str, str]] = []

    async def get_job_by_id(job_id: str) -> WorkInfo | None:
        return parent if job_id == parent.id else None

    async def fake_complete(
        job_id: str,
        queue_name: str,
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

    repository.get_job_by_id = get_job_by_id
    repository.complete_job = fake_complete
    scheduler.dag_service.resolve_dag_status_with_retry = fake_resolve_dag_status
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
    scheduler._job_cache.update(
        {failed_job.id: failed_job, sibling_job.id: sibling_job}
    )

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
    assert (
        scheduler._scheduler_counters[scheduler_psql.RUN_LEASE_RECOVERED_FAILED_TOTAL]
        == 1
    )


@pytest.mark.asyncio
async def test_recovered_retry_hydrates_missing_dag_before_frontier_retry() -> None:
    dag_id = "dag-recovered-retry"
    retry_job = build_work_item("job-recovered-retry", dag_id)
    retry_job.state = WorkState.RETRY
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    repository.get_job_by_id = AsyncMock(return_value=retry_job)
    scheduler = build_scheduler(repository, frontier)
    hydrated: list[str] = []

    async def hydrate(dag_to_hydrate: str) -> bool:
        hydrated.append(dag_to_hydrate)
        scheduler.active_dags[dag_to_hydrate] = object()
        await frontier.add_dag(None, [retry_job])
        return True

    scheduler.hydrate_single_dag_from_db = hydrate

    await scheduler._reconcile_recovered_run_leases(
        [
            RecoveredRunLease(
                id=retry_job.id,
                name=retry_job.name,
                previous_state=WorkState.ACTIVE.value,
                recovered_state="retry",
                dag_id=dag_id,
                retry_count=1,
                retry_limit=retry_job.retry_limit,
                start_after=retry_job.start_after,
                previous_run_owner="dead-scheduler",
                previous_run_attempt_id="06a0ac88-5326-7f90-8000-111111111111",
            )
        ]
    )

    assert hydrated == [dag_id]
    assert retry_job.id in frontier.jobs_by_id
    assert frontier.jobs_by_id[retry_job.id].state == WorkState.RETRY
    assert not scheduler.dag_service._admission_event.is_set()


@pytest.mark.asyncio
async def test_late_success_for_old_run_attempt_updates_zero_rows(monkeypatch):
    work_item = build_work_item("job-late-success", "dag-late-success")
    work_item.state = WorkState.ACTIVE
    work_item.run_owner = "current-owner"
    work_item.run_attempt_id = "06a0ac88-5326-7f90-8000-0274669de089"

    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)
    scheduler._job_cache[work_item.id] = work_item

    complete_calls: list[dict] = []
    completed_calls: list[str] = []
    trace_events: list[tuple[str, dict]] = []

    async def complete(
        job_id: str,
        queue_name: str,
        output_metadata: dict | None = None,
        **kwargs,
    ) -> int:
        complete_calls.append(
            {
                "job_id": job_id,
                "queue_name": queue_name,
                "run_owner": kwargs.get("run_owner"),
                "run_attempt_id": kwargs.get("run_attempt_id"),
            }
        )
        return 0

    async def on_job_completed(job_id: str):
        completed_calls.append(job_id)

    repository.complete_job = complete
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
            "queue_name": work_item.name,
            "run_owner": "old-owner",
            "run_attempt_id": "06a0ac88-5326-7f90-8000-111111111111",
        }
    ]
    assert completed_calls == []
    assert scheduler.notify_calls == []
    assert (
        scheduler._scheduler_counters[scheduler_psql.TERMINAL_EVENT_STALE_ATTEMPT_TOTAL]
        == 1
    )
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
    scheduler._job_cache[work_item.id] = work_item

    fail_calls: list[dict] = []
    failed_calls: list[str] = []
    trace_events: list[tuple[str, dict]] = []

    async def fail(
        job_id: str,
        queue_name: str,
        output_metadata: dict | None = None,
        **kwargs,
    ) -> tuple[int, str | None]:
        fail_calls.append(
            {
                "job_id": job_id,
                "queue_name": queue_name,
                "output_metadata": output_metadata,
                "run_owner": kwargs.get("run_owner"),
                "run_attempt_id": kwargs.get("run_attempt_id"),
            }
        )
        return 0, None

    async def on_job_failed(job_id: str):
        failed_calls.append(job_id)

    repository.fail_job = fail
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
            "message": "processor crashed",
            "jobinfo_replace_kwargs": {
                "runtime_env": {
                    "attributes": {"document_id": "not-persisted"},
                    "error": {
                        "type": "RuntimeError",
                        "message": "processor crashed",
                        "filename": "executor.py",
                        "name": "process",
                        "line_no": 42,
                    },
                }
            },
        },
    )

    assert fail_calls == [
        {
            "job_id": work_item.id,
            "queue_name": work_item.name,
            "output_metadata": {
                "failure_source": "job_event",
                "error_message": "processor crashed",
                "error": {
                    "type": "RuntimeError",
                    "message": "processor crashed",
                    "filename": "executor.py",
                    "name": "process",
                    "line_no": 42,
                },
            },
            "run_owner": "old-owner",
            "run_attempt_id": "06a0ac88-5326-7f90-8000-222222222222",
        }
    ]
    assert failed_calls == []
    assert scheduler.notify_calls == []
    assert (
        scheduler._scheduler_counters[scheduler_psql.TERMINAL_EVENT_STALE_ATTEMPT_TOTAL]
        == 1
    )
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
    scheduler._job_cache[work_item.id] = work_item

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
    assert (
        scheduler._scheduler_counters[
            scheduler_psql.RUN_LEASE_EXTEND_STALE_ATTEMPT_TOTAL
        ]
        == 1
    )
    assert [event for event, _fields in trace_events].count(
        scheduler_psql.RUN_LEASE_EXTEND_STALE_ATTEMPT_TOTAL
    ) == 1


@pytest.mark.asyncio
async def test_admit_dag_requires_db_activation_success():
    dag_id = "dag-admit"
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active", mark_active_result=False)
    scheduler = build_scheduler(repository, frontier)

    admitted = await scheduler.dag_service.admit_dag(dag_id, object(), source="test")

    assert admitted is False
    assert repository.mark_active_calls == [dag_id]
    assert dag_id not in scheduler.active_dags


@pytest.mark.asyncio
async def test_scheduler_start_initializes_notifications_before_admission(
    monkeypatch,
):
    order: list[str] = []

    class Repo:
        async def initialize(self):
            order.append("initialize")

        async def is_installed(self, _schema):
            order.append("is_installed")
            return True

        async def get_defined_queues(self, _schema):
            order.append("get_defined_queues")
            return set()

        async def create_tables(self, _schema):
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

    class DagService:
        async def start_admission(self):
            order.append("admission_start")

        async def start_sync(self):
            order.append("dag_sync_start")

    class SemaphoreStore:
        def reconcile_all(self, *, delete_orphan_holders, fix_counters):
            assert delete_orphan_holders is True
            assert fix_counters is True
            order.append("semaphore_reconcile")
            return {}

    class DummyTask:
        def __init__(self, coro):
            self._coro = coro

        def done(self):
            return False

        def cancel(self):
            self._coro.close()

    def fake_create_task(coro, *, name=None):
        coro.close()
        return DummyTask(coro)

    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = FakeLogger()
    scheduler.repository = Repo()
    scheduler.known_queues = set()
    scheduler.notification_service = NotificationService()
    scheduler.maintenance_service = MaintenanceService()
    scheduler.dag_service = DagService()
    scheduler._semaphore_store = SemaphoreStore()
    scheduler._lifecycle_lock = asyncio.Lock()
    scheduler.running = False
    scheduler._resources_closed = False
    scheduler._priority_refresh_event = asyncio.Event()
    scheduler.priority_refresh_enabled = False
    scheduler._setup_event_subscriptions = lambda: None
    scheduler.runtime = SchedulerRuntime(scheduler.logger)

    async def notify_event():
        order.append("notify_event")
        return True

    async def noop():
        return None

    async def initial_run_lease_renewal():
        order.append("initial_run_lease_renewal")

    scheduler.notify_event = notify_event
    scheduler._renew_active_run_leases = initial_run_lease_renewal
    scheduler._renew_run_leases = noop
    scheduler._sync = noop
    scheduler._poll = noop
    scheduler._PostgreSQLJobScheduler__monitor_deployment_updates = noop

    monkeypatch.setattr(scheduler_psql.asyncio, "create_task", fake_create_task)

    await scheduler.start()

    assert order.index("notification_start") < order.index("admission_start")
    assert order.index("semaphore_reconcile") < order.index("admission_start")
    assert order.index("initial_run_lease_renewal") < order.index("maintenance_start")
    assert order[-1] == "notify_event"


@pytest.mark.asyncio
async def test_evict_dag_from_memory_finalizes_frontier_and_clears_terminal_state():
    dag_id = "dag-evict"
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()
    scheduler.dag_service._terminal_dag_states[dag_id] = "completed"

    work_item = build_work_item("job-evict", dag_id)
    await frontier.add_dag(None, [work_item])
    scheduler._job_cache[work_item.id] = work_item

    removed = await scheduler.dag_service.evict_dag(
        dag_id, "no longer active or deleted in database"
    )

    assert removed is True
    assert frontier.finalize_calls == [dag_id]
    assert dag_id not in scheduler.active_dags
    assert dag_id not in scheduler.dag_service._terminal_dag_states
    assert await frontier.get_jobs_by_dag_id(dag_id) == []
    assert work_item.id not in scheduler._job_cache


@pytest.mark.asyncio
async def test_control_flow_lease_miss_evicts_missing_db_job_without_requeue():
    dag_id = "dag-missing-control"
    frontier = RecordingFrontier()
    repository = RecordingRepository(dag_state="active")
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()

    work_item = build_work_item("job-missing-control", dag_id, name="noop")
    await frontier.add_dag(None, [work_item])
    await frontier.take([work_item.id])
    scheduler._job_cache[work_item.id] = work_item

    released_local: list[str] = []

    async def release_lease_local(job_id: str) -> None:
        released_local.append(job_id)

    frontier.release_lease_local = release_lease_local

    reconciled = await scheduler._reconcile_control_flow_lease_miss(work_item, None)

    assert reconciled is True
    assert released_local == []
    assert frontier.finalize_calls == [dag_id]
    assert dag_id not in scheduler.active_dags
    assert work_item.id not in scheduler._job_cache
    assert scheduler.notify_calls == [True]


@pytest.mark.asyncio
async def test_dispatch_lease_shortfall_evicts_missing_db_job_without_requeue():
    dag_id = "dag-missing-dispatch"
    frontier = RecordingFrontier()

    class MissingJobRepository(RecordingRepository):
        def __init__(self):
            super().__init__(dag_state="active")
            self.lookup_calls: list[str] = []

        async def get_job_by_id(self, job_id: str):
            self.lookup_calls.append(job_id)
            return None

    repository = MissingJobRepository()
    scheduler = build_scheduler(repository, frontier)
    scheduler.active_dags[dag_id] = object()

    work_item = build_work_item("job-missing-dispatch", dag_id)
    leased_item = build_work_item("job-leased-dispatch", dag_id)
    await frontier.add_dag(None, [work_item, leased_item])
    await frontier.take([work_item.id, leased_item.id])

    released_local: list[str] = []
    released_db: list[list[str]] = []

    async def release_lease_local(job_id: str) -> None:
        released_local.append(job_id)

    async def release_lease_db(job_ids: list[str]) -> set[str]:
        released_db.append(job_ids)
        return set(job_ids)

    frontier.release_lease_local = release_lease_local
    scheduler._release_lease_db = release_lease_db

    leased_ids = {leased_item.id}

    reconciled = await scheduler._reconcile_db_lease_shortfall(
        [work_item, leased_item], leased_ids
    )

    assert reconciled == 1
    assert repository.lookup_calls == [work_item.id]
    assert released_local == []
    assert released_db == [[leased_item.id]]
    assert leased_ids == set()
    assert frontier.finalize_calls == [dag_id]
    assert dag_id not in scheduler.active_dags
    assert scheduler.notify_calls == [True]


@pytest.mark.asyncio
async def test_sync_dag_once_reaps_stale_memory_dags_and_notifies():
    repository = RecordingRepository(dag_state="active")
    frontier = RecordingFrontier()
    scheduler = build_scheduler(repository, frontier)
    scheduler.dag_service.max_active_dags = 2
    scheduler.active_dags.update({"dag-valid": object(), "dag-stale": object()})
    scheduler.dag_service._terminal_dag_states = {"dag-stale": "completed"}
    resolved: list[str] = []

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

    await scheduler.dag_service.sync_once()

    assert resolved == ["dag-valid", "dag-stale"]
    assert "dag-stale" not in scheduler.active_dags
    assert scheduler.dag_service._terminal_dag_states == {"dag-stale": "completed"}
    assert frontier.finalize_calls == ["dag-stale"]
    assert scheduler.notify_calls == [True]
    assert scheduler.dag_service._admission_event.is_set()


@pytest.mark.asyncio
async def test_submission_priority_refresh_request_does_not_refresh_inline():
    repository = RecordingRepository(dag_state="active")
    frontier = RecordingFrontier()
    scheduler = build_scheduler(repository, frontier)
    scheduler.priority_refresh_interval = 10
    scheduler._next_priority_refresh_at = time.monotonic() + 60.0

    refresh_calls: list[str] = []

    async def refresh_job_priorities(source: str = "unknown") -> int:
        refresh_calls.append(source)
        return 1

    scheduler._refresh_job_priorities = refresh_job_priorities

    await scheduler._handle_priority_refresh(10)

    assert refresh_calls == []
    assert scheduler._next_priority_refresh_at > time.monotonic()
    assert scheduler.notify_calls == []


@pytest.mark.asyncio
async def test_submission_priority_refresh_request_wakes_when_due():
    repository = RecordingRepository(dag_state="active")
    frontier = RecordingFrontier()
    scheduler = build_scheduler(repository, frontier)
    scheduler.priority_refresh_interval = 10
    scheduler._next_priority_refresh_at = time.monotonic() - 1.0

    refresh_calls: list[str] = []

    async def refresh_job_priorities(source: str = "unknown") -> int:
        refresh_calls.append(source)
        return 1

    scheduler._refresh_job_priorities = refresh_job_priorities

    await scheduler._handle_priority_refresh(10)

    assert refresh_calls == []
    assert scheduler._priority_refresh_event.is_set()
    assert scheduler.notify_calls == []


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
async def test_mark_nodes_skipped_reconciles_from_committed_ids_only():
    skip_calls: list[dict[str, object]] = []
    metadata_calls: list[str] = []
    frontier_skip_calls: list[list[str]] = []

    async def mark_jobs_as_skipped(**kwargs: object) -> set[str]:
        skip_calls.append(kwargs)
        return {"job-committed"}

    async def update_job_metadata(job_id: str, **_kwargs: object) -> bool:
        metadata_calls.append(job_id)
        return True

    async def on_jobs_skipped(job_ids: list[str]) -> None:
        frontier_skip_calls.append(job_ids)

    service = object.__new__(ControlFlowExecutionService)
    service.logger = FakeLogger()
    service.repository = SimpleNamespace(
        mark_jobs_as_skipped=mark_jobs_as_skipped,
        update_job_metadata=update_job_metadata,
    )
    service.frontier = SimpleNamespace(on_jobs_skipped=on_jobs_skipped)

    dag_plan = QueryPlan(
        nodes=[
            Query(task_id="active", query_str="active"),
            Query(task_id="job-uncommitted", query_str="uncommitted"),
            Query(task_id="job-committed", query_str="committed"),
            Query(
                task_id="committed-child",
                query_str="committed child",
                dependencies=["job-committed"],
            ),
            Query(
                task_id="shared-merger",
                query_str="shared merger",
                dependencies=["active", "committed-child"],
            ),
        ]
    )

    skip_reason = SkipReason(branch_node_id="branch", reason="not selected")
    await service._mark_nodes_skipped(
        ["job-uncommitted", "job-committed"],
        "extract",
        skip_reason,
        dag_plan,
    )

    skip_metadata = skip_calls[0]["output_metadata"]
    assert (
        skip_metadata["skip_reason"]["timestamp"]
        == skip_reason.model_dump(mode="json")["timestamp"]
    )
    assert skip_calls[0]["job_ids"] == [
        "job-uncommitted",
        "job-committed",
        "committed-child",
    ]
    assert metadata_calls == ["job-committed"]
    assert frontier_skip_calls == [["job-committed"]]


@pytest.mark.asyncio
async def test_process_control_flow_node_notifies_after_unblocking_children(
    monkeypatch,
):
    service = object.__new__(ControlFlowExecutionService)
    service.logger = FakeLogger()
    service.dag_service = SimpleNamespace(
        active_dags={"dag-cf": object()},
        resolve_dag_status_with_retry=None,
    )
    service._topology_cache = SimpleNamespace(
        get_sorted_nodes_and_levels=lambda _dag, _dag_id: (
            [],
            {"noop-root": 2, "child": 1},
        )
    )
    service.frontier = SimpleNamespace(
        leased_until={},
        on_job_completed=None,
    )

    order: list[str] = []

    async def complete_attempt(work_item: WorkInfo) -> bool:
        order.append(f"complete:{work_item.id}")
        return 1

    async def on_job_completed(job_id: str):
        order.append(f"frontier_completed:{job_id}")

    async def notify_event() -> bool:
        order.append("notify_event")
        return True

    async def resolve_dag_status_with_retry(*_args, **_kwargs):
        order.append("resolve_dag_status")
        return True

    service._complete_attempt = complete_attempt
    service._notify_callback = notify_event
    service.dag_service.resolve_dag_status_with_retry = resolve_dag_status_with_retry
    service.frontier.on_job_completed = on_job_completed

    async def activate_control_flow_job(_wi: WorkInfo) -> bool:
        _wi.run_owner = "test-scheduler"
        _wi.run_attempt_id = "06a0ac88-5326-7f90-8000-0274669de089"
        return True

    service._activate = activate_control_flow_job

    monkeypatch.setattr(
        control_flow_module, "get_node_from_dag", lambda *_args, **_kwargs: object()
    )

    started_calls: list[dict] = []

    async def mark_started_toast(**kwargs):
        started_calls.append(kwargs)
        return True

    monkeypatch.setattr(
        control_flow_module, "mark_as_started_toast", mark_started_toast
    )

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

    await service.process_node(work_item)

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
