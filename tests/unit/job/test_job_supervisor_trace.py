import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from marie.job.common import JobInfo, JobStatus
from marie.job.job_supervisor import JobSupervisor
from marie.proto import jina_pb2
from marie.types_core.request.data import DataRequest


def make_supervisor(confirmation_event: asyncio.Event) -> JobSupervisor:
    desired_state_executor = SimpleNamespace(
        schedule_new_epoch=AsyncMock(return_value=SimpleNamespace(epoch=1))
    )
    return JobSupervisor(
        job_id="test-job-id",
        job_info_client=Mock(),
        job_distributor=Mock(),
        event_publisher=Mock(),
        etcd_client=Mock(),
        desired_state_executor=desired_state_executor,
        confirmation_event=confirmation_event,
    )


@pytest.mark.asyncio
async def test_confirmation_is_signaled_immediately_on_owning_loop() -> None:
    confirmation_event = asyncio.Event()
    supervisor = make_supervisor(confirmation_event)
    supervisor._loop = asyncio.get_running_loop()

    supervisor._signal_confirmation_threadsafe()

    assert confirmation_event.is_set()


@pytest.mark.asyncio
async def test_confirmation_uses_threadsafe_signal_for_foreign_loop() -> None:
    confirmation_event = asyncio.Event()
    supervisor = make_supervisor(confirmation_event)
    foreign_loop = Mock()
    foreign_loop.is_running.return_value = True
    supervisor._loop = foreign_loop

    supervisor._signal_confirmation_threadsafe()

    assert not confirmation_event.is_set()
    foreign_loop.call_soon_threadsafe.assert_called_once_with(confirmation_event.set)


@pytest.mark.asyncio
async def test_pre_send_traces_admission_and_desired_state(monkeypatch) -> None:
    confirmation_event = asyncio.Event()
    supervisor = make_supervisor(confirmation_event)
    supervisor._loop = asyncio.get_running_loop()
    supervisor._desired_state_executor.schedule_new_epoch = AsyncMock(
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
        get_info=AsyncMock(
            return_value=SimpleNamespace(
                status=JobStatus.SUCCEEDED,
                run_owner="owner-1",
                run_attempt_id="attempt-1",
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
        desired_state_executor=SimpleNamespace(schedule_new_epoch=AsyncMock()),
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
        "job_supervisor_terminal_info_read",
    ]
    assert events[1][1]["status"] == "SUCCEEDED"
    assert events[1][1]["terminal"] is True
    assert events[2][1]["found"] is True
    assert events[2][1]["elapsed_ms"] >= 0
    job_info_client.get_info.assert_awaited_once_with("test-job-id")
    event_publisher.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_uses_terminal_event_callback() -> None:
    send_task = asyncio.get_running_loop().create_future()
    send_task.set_result(
        SimpleNamespace(status=SimpleNamespace(code=jina_pb2.StatusProto.SUCCESS))
    )
    job_info_client = SimpleNamespace(
        get_info=AsyncMock(
            return_value=SimpleNamespace(
                status=JobStatus.SUCCEEDED,
                run_owner="owner-1",
                run_attempt_id="attempt-1",
            )
        ),
    )
    job_distributor = SimpleNamespace(send_nowait=AsyncMock(return_value=send_task))
    event_publisher = SimpleNamespace(publish=AsyncMock())
    terminal_event_callback = AsyncMock(return_value=False)
    supervisor = JobSupervisor(
        job_id="test-job-id",
        job_info_client=job_info_client,
        job_distributor=job_distributor,
        event_publisher=event_publisher,
        etcd_client=Mock(),
        desired_state_executor=SimpleNamespace(schedule_new_epoch=AsyncMock()),
        confirmation_event=asyncio.Event(),
        terminal_event_callback=terminal_event_callback,
    )

    await supervisor._submit_job_in_background(
        JobInfo(
            status=JobStatus.PENDING,
            entrypoint="mock_executor_a:///document/extract",
        )
    )
    for _ in range(10):
        if terminal_event_callback.await_count:
            break
        await asyncio.sleep(0)

    terminal_event_callback.assert_awaited_once_with(
        "test-job-id",
        JobStatus.SUCCEEDED,
        "owner-1",
        "attempt-1",
        "supervisor_finalize",
    )
    job_info_client.get_info.assert_awaited_once_with("test-job-id")
    event_publisher.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_reuses_committed_terminal_status(monkeypatch) -> None:
    send_task = asyncio.get_running_loop().create_future()
    send_task.set_result(
        SimpleNamespace(status=SimpleNamespace(code=jina_pb2.StatusProto.SUCCESS))
    )
    job_info_client = SimpleNamespace(get_info=AsyncMock())
    job_distributor = SimpleNamespace(send_nowait=AsyncMock(return_value=send_task))
    event_publisher = SimpleNamespace(publish=AsyncMock())
    terminal_event_callback = AsyncMock()
    committed_terminal_lookup = Mock(return_value=JobStatus.SUCCEEDED)
    supervisor = JobSupervisor(
        job_id="test-job-id",
        job_info_client=job_info_client,
        job_distributor=job_distributor,
        event_publisher=event_publisher,
        etcd_client=Mock(),
        desired_state_executor=SimpleNamespace(schedule_new_epoch=AsyncMock()),
        confirmation_event=asyncio.Event(),
        terminal_event_callback=terminal_event_callback,
        committed_terminal_lookup=committed_terminal_lookup,
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
            run_owner="owner-1",
            run_attempt_id="attempt-1",
        )
    )
    for _ in range(10):
        if any(
            event == "job_supervisor_terminal_status_cache_hit" for event, _ in events
        ):
            break
        await asyncio.sleep(0)

    committed_terminal_lookup.assert_called_once_with("test-job-id", "attempt-1")
    job_info_client.get_info.assert_not_awaited()
    terminal_event_callback.assert_not_awaited()
    event_publisher.publish.assert_not_awaited()
    assert [event for event, _ in events] == [
        "job_supervisor_send_task_completed",
        "job_supervisor_terminal_status_cache_hit",
    ]
