from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.job.common import JobStatus
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.services.attempt_lifecycle_service import (
    TERMINAL_EVENT_STALE_ATTEMPT_TOTAL,
    AttemptLifecycleService,
)
from marie.scheduler.state import WorkState

JOB_ID = "00000000-0000-0000-0000-000000000001"
DAG_ID = "00000000-0000-0000-0000-000000000010"
ATTEMPT_ID = "00000000-0000-0000-0000-00000000000a"


def build_service(*, work_state: WorkState = WorkState.ACTIVE) -> SimpleNamespace:
    work_item = SimpleNamespace(
        id=JOB_ID,
        dag_id=DAG_ID,
        name="extract",
        state=work_state,
        run_owner="worker-1",
        run_attempt_id=ATTEMPT_ID,
    )
    repository = SimpleNamespace(
        complete_job=AsyncMock(return_value=1),
        fail_job=AsyncMock(return_value=(1, WorkState.FAILED.value)),
        cancel_job_attempt=AsyncMock(return_value={JOB_ID}),
        record_job_attempt_terminal=AsyncMock(),
    )
    frontier = SimpleNamespace(
        on_job_retry=AsyncMock(),
        on_job_failed=AsyncMock(),
        on_job_cancelled=AsyncMock(),
    )
    dag_service = SimpleNamespace(
        resolve_dag_status_with_retry=AsyncMock(),
        request_admission=AsyncMock(),
    )
    control_flow_service = SimpleNamespace(
        commit_guardrail_route_if_needed=AsyncMock(return_value=None),
        handle_successful_job_completion=AsyncMock(),
    )
    notify = AsyncMock(return_value=True)
    counter = MagicMock()
    job_cache = {}
    service = AttemptLifecycleService(
        repository=repository,
        frontier=frontier,
        dag_service=dag_service,
        control_flow_service=control_flow_service,
        status_update_lock=AsyncJobLock(),
        job_cache=job_cache,
        scheduler_lease_owner="scheduler-1",
        gateway_instance_id="gateway-1",
        notify_callback=notify,
        counter_callback=counter,
    )
    return SimpleNamespace(
        service=service,
        work_item=work_item,
        repository=repository,
        frontier=frontier,
        dag_service=dag_service,
        control_flow_service=control_flow_service,
        notify=notify,
        counter=counter,
        job_cache=job_cache,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["job_event", "storage_sync"])
async def test_success_uses_the_same_fenced_transition_for_each_source(
    source: str,
) -> None:
    context = build_service()

    accepted = await context.service.transition_terminal(
        JOB_ID,
        context.work_item,
        JobStatus.SUCCEEDED,
        run_owner="worker-1",
        run_attempt_id=ATTEMPT_ID,
        source=source,
        output_metadata={"synced": source == "storage_sync"},
    )

    assert accepted is True
    context.repository.complete_job.assert_awaited_once_with(
        job_id=JOB_ID,
        queue_name="extract",
        output_metadata={"synced": source == "storage_sync"},
        schema="marie_scheduler",
        run_owner="worker-1",
        run_attempt_id=ATTEMPT_ID,
    )
    assert context.work_item.state == WorkState.COMPLETED
    assert context.job_cache == {JOB_ID: context.work_item}
    context.control_flow_service.handle_successful_job_completion.assert_awaited_once_with(
        JOB_ID, context.work_item
    )
    context.dag_service.resolve_dag_status_with_retry.assert_awaited_once_with(
        JOB_ID,
        context.work_item,
        source=source,
    )
    context.dag_service.request_admission.assert_awaited_once_with(
        "executor_capacity_released"
    )
    context.notify.assert_awaited_once_with()
    audit = context.repository.record_job_attempt_terminal.await_args.kwargs
    assert audit["source"] == source
    assert audit["accepted"] is True
    assert audit["terminal_work_state"] == WorkState.COMPLETED.value


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["job_event", "storage_sync"])
async def test_retry_reconciles_memory_and_wakes_without_resolving_dag(
    source: str,
) -> None:
    context = build_service()
    context.repository.fail_job.return_value = (1, WorkState.RETRY.value)

    accepted = await context.service.transition_terminal(
        JOB_ID,
        context.work_item,
        JobStatus.FAILED,
        run_owner="worker-1",
        run_attempt_id=ATTEMPT_ID,
        source=source,
        message="processor crashed",
        runtime_env={
            "attributes": {"document_id": "not-persisted"},
            "error": {"type": "RuntimeError", "message": "processor crashed"},
        },
    )

    assert accepted is True
    failure = context.repository.fail_job.await_args.kwargs
    assert failure["output_metadata"] == {
        "failure_source": source,
        "error_message": "processor crashed",
        "error": {"type": "RuntimeError", "message": "processor crashed"},
    }
    assert context.work_item.state == WorkState.RETRY
    context.frontier.on_job_retry.assert_awaited_once_with(JOB_ID, context.work_item)
    context.frontier.on_job_failed.assert_not_awaited()
    context.dag_service.resolve_dag_status_with_retry.assert_not_awaited()
    context.dag_service.request_admission.assert_awaited_once_with(
        "executor_capacity_released"
    )
    context.notify.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["job_event", "storage_sync"])
async def test_stale_attempt_is_audited_and_does_not_change_memory(
    source: str,
) -> None:
    context = build_service()
    context.repository.complete_job.return_value = 0

    accepted = await context.service.transition_terminal(
        JOB_ID,
        context.work_item,
        JobStatus.SUCCEEDED,
        run_owner="old-worker",
        run_attempt_id=ATTEMPT_ID,
        source=source,
    )

    assert accepted is False
    assert context.work_item.state == WorkState.ACTIVE
    assert context.job_cache == {}
    context.control_flow_service.handle_successful_job_completion.assert_not_awaited()
    context.dag_service.resolve_dag_status_with_retry.assert_not_awaited()
    context.dag_service.request_admission.assert_not_awaited()
    context.notify.assert_not_awaited()
    audit = context.repository.record_job_attempt_terminal.await_args.kwargs
    assert audit["accepted"] is False
    assert audit["reject_reason"] == "db_update_zero_rows"
    assert audit["source"] == source
    context.counter.assert_called_once_with(
        TERMINAL_EVENT_STALE_ATTEMPT_TOTAL,
        job_id=JOB_ID,
        dag_id=DAG_ID,
        status=JobStatus.SUCCEEDED.value,
        run_owner="old-worker",
        run_attempt_id=ATTEMPT_ID,
        source=source,
    )


@pytest.mark.asyncio
async def test_stopped_attempt_clears_attempt_identity_after_commit() -> None:
    context = build_service()

    accepted = await context.service.transition_terminal(
        JOB_ID,
        context.work_item,
        JobStatus.STOPPED,
        run_owner="worker-1",
        run_attempt_id=ATTEMPT_ID,
        source="job_event",
    )

    assert accepted is True
    assert context.work_item.state == WorkState.CANCELLED
    assert context.work_item.run_owner is None
    assert context.work_item.run_attempt_id is None
    context.frontier.on_job_cancelled.assert_awaited_once_with(JOB_ID)
    context.dag_service.resolve_dag_status_with_retry.assert_awaited_once()
    context.dag_service.request_admission.assert_awaited_once_with(
        "executor_capacity_released"
    )
    context.notify.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_terminal_transition_traces_resolution_before_scheduler_wake(
    monkeypatch,
) -> None:
    context = build_service()
    context.dag_service.resolve_dag_status_with_retry.return_value = True
    trace = MagicMock()
    monkeypatch.setattr(
        "marie.scheduler.services.attempt_lifecycle_service.scheduler_trace",
        trace,
    )

    accepted = await context.service.transition_terminal(
        JOB_ID,
        context.work_item,
        JobStatus.SUCCEEDED,
        run_owner="worker-1",
        run_attempt_id=ATTEMPT_ID,
        source="job_event",
    )

    assert accepted is True
    events = [item.args[0] for item in trace.call_args_list]
    assert events == [
        "job_terminal_attempt_accepted",
        "terminal_dag_resolution_started",
        "terminal_dag_resolution_completed",
        "terminal_scheduler_wake_completed",
    ]
    resolution = trace.call_args_list[2].kwargs
    assert resolution["dag_resolved"] is True
    assert resolution["elapsed_ms"] >= 0
    wake = trace.call_args_list[3].kwargs
    assert wake["wake_queued"] is True
    assert wake["terminal"] is True
