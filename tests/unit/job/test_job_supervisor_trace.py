import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from marie.job.common import JobInfo, JobStatus
from marie.job.job_supervisor import JobSupervisor
from marie.proto import jina_pb2
from marie.types_core.request.data import DataRequest


def make_supervisor(confirmation_event: asyncio.Event) -> JobSupervisor:
    return JobSupervisor(
        job_id="test-job-id",
        job_info_client=Mock(),
        job_distributor=Mock(),
        event_publisher=Mock(),
        etcd_client=Mock(),
        confirmation_event=confirmation_event,
    )


@pytest.mark.asyncio
async def test_pre_send_traces_admission_and_desired_state(monkeypatch) -> None:
    confirmation_event = asyncio.Event()
    supervisor = make_supervisor(confirmation_event)
    supervisor._loop = asyncio.get_running_loop()
    supervisor._desired_store.schedule_new_epoch = Mock(
        return_value=SimpleNamespace(epoch=7)
    )
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "marie.job.job_supervisor.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    await supervisor.pre_send_callback(
        requests=[],
        ctx={"address": "127.0.0.1:12345", "deployment": "mock_executor_a"},
    )

    await asyncio.wait_for(confirmation_event.wait(), timeout=1)
    assert [event for event, _ in events] == [
        "job_supervisor_pre_send_started",
        "job_supervisor_dispatch_admitted",
        "job_supervisor_desired_state_written",
    ]
    assert events[-1][1]["epoch"] == 7
    assert events[-1][1]["deployment"] == "mock_executor_a"


@pytest.mark.asyncio
async def test_after_send_traces_response_and_worker_ack(monkeypatch) -> None:
    confirmation_event = asyncio.Event()
    supervisor = make_supervisor(confirmation_event)
    supervisor._loop = asyncio.get_running_loop()
    supervisor._current_job_epoch = 7
    supervisor._await_worker_ack = Mock(return_value=True)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "marie.job.job_supervisor.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    await supervisor.after_send_callback(
        requests=[],
        ctx={"address": "127.0.0.1:12345", "deployment": "mock_executor_a"},
        response=DataRequest(),
    )

    assert [event for event, _ in events] == [
        "job_supervisor_response_received",
        "job_supervisor_worker_ack_wait_completed",
    ]
    assert events[-1][1]["acknowledged"] is True
    assert events[-1][1]["skipped"] is False


@pytest.mark.asyncio
async def test_finalize_traces_send_completion_before_terminal_event_enqueue(
    monkeypatch,
) -> None:
    confirmation_event = asyncio.Event()
    send_task = asyncio.get_running_loop().create_future()
    send_task.set_result(
        SimpleNamespace(status=SimpleNamespace(code=jina_pb2.StatusProto.SUCCESS))
    )
    job_info_client = SimpleNamespace(
        get_status=AsyncMock(return_value=JobStatus.SUCCEEDED),
        get_info=AsyncMock(
            return_value=SimpleNamespace(
                run_owner="owner-1", run_attempt_id="attempt-1"
            )
        ),
    )
    job_distributor = SimpleNamespace(send_nowait=AsyncMock(return_value=send_task))
    event_publisher = SimpleNamespace(publish=AsyncMock())
    supervisor = JobSupervisor(
        job_id="test-job-id",
        job_info_client=job_info_client,
        job_distributor=job_distributor,
        event_publisher=event_publisher,
        etcd_client=Mock(),
        confirmation_event=confirmation_event,
    )
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "marie.job.job_supervisor.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    await supervisor._submit_job_in_background(
        JobInfo(
            status=JobStatus.PENDING,
            entrypoint="mock_executor_a:///document/extract",
        )
    )
    for _ in range(10):
        if event_publisher.publish.await_count:
            break
        await asyncio.sleep(0)

    assert [event for event, _ in events] == [
        "job_supervisor_send_task_completed",
        "job_supervisor_terminal_status_read",
    ]
    assert events[1][1]["status"] == "SUCCEEDED"
    assert events[1][1]["terminal"] is True
    event_publisher.publish.assert_awaited_once()
