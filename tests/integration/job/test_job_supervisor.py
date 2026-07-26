import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor
from marie.job.job_supervisor import JobSupervisor
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
async def test_pre_send_callback_confirms_before_desired_store_write():
    confirmation_event = asyncio.Event()
    supervisor = make_supervisor(confirmation_event)
    supervisor._loop = asyncio.get_running_loop()
    supervisor._desired_state_executor.schedule_new_epoch = AsyncMock(
        side_effect=RuntimeError("desired store unavailable")
    )

    with pytest.raises(RuntimeError):
        await supervisor.pre_send_callback(
            requests=[],
            ctx={"address": "127.0.0.1:12345", "deployment": "test-deployment"},
        )

    await asyncio.wait_for(confirmation_event.wait(), timeout=1)


@pytest.mark.asyncio
async def test_after_send_callback_confirms_without_epoch():
    confirmation_event = asyncio.Event()
    supervisor = make_supervisor(confirmation_event)
    supervisor._loop = asyncio.get_running_loop()
    supervisor._current_job_epoch = None

    await supervisor.after_send_callback(
        requests=[],
        ctx={"address": "127.0.0.1:12345", "deployment": "test-deployment"},
        response=DataRequest(),
    )

    await asyncio.wait_for(confirmation_event.wait(), timeout=1)


@pytest.mark.asyncio
async def test_after_send_callback_confirms_before_waiting_for_ack():
    confirmation_event = asyncio.Event()
    supervisor = make_supervisor(confirmation_event)
    supervisor._loop = asyncio.get_running_loop()
    supervisor._current_job_epoch = 7

    with patch.object(supervisor, "_await_worker_ack", return_value=True) as ack:
        await supervisor.after_send_callback(
            requests=[],
            ctx={"address": "127.0.0.1:12345", "deployment": "test-deployment"},
            response=DataRequest(),
        )

    await asyncio.wait_for(confirmation_event.wait(), timeout=1)
    ack.assert_called_once_with("127.0.0.1:12345", "test-deployment", 7)


@pytest.mark.asyncio
async def test_send_nowait_cancelled_task_does_not_raise_from_done_callback():
    distributor = GatewayJobDistributor()

    async def never_send(*args, **kwargs):
        await asyncio.Event().wait()

    distributor.send = AsyncMock(side_effect=never_send)

    task = await distributor.send_nowait(
        submission_id="test-job-id",
        job_info=JobInfo(
            status=JobStatus.PENDING,
            entrypoint="/default",
            metadata={"metadata": {}},
        ),
        send_callback=Mock(),
    )

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert "test-job-id" not in distributor._inflight
