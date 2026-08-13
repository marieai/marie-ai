import asyncio
from unittest.mock import MagicMock

import pytest

from marie.executor.marie_executor import MarieExecutor
from marie.serve.executors import close_executor


@pytest.mark.asyncio
async def test_aclose_awaits_gpu_monitor_shutdown() -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def monitor() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    executor = object.__new__(MarieExecutor)
    executor._gpu_monitor_task = asyncio.create_task(monitor(), name="gpu-monitor")
    executor._nvml_shutdown = MagicMock()
    await started.wait()

    await executor.aclose()

    assert stopped.is_set()
    assert executor._gpu_monitor_task is None
    executor._nvml_shutdown.assert_called_once_with()


@pytest.mark.asyncio
async def test_close_executor_awaits_async_close() -> None:
    closed = False

    class AsyncExecutor:
        async def close(self) -> None:
            nonlocal closed
            await asyncio.sleep(0)
            closed = True

    await close_executor(AsyncExecutor())

    assert closed is True


@pytest.mark.asyncio
async def test_aclose_awaits_overridden_async_close() -> None:
    closed = False

    class AsyncMarieExecutor(MarieExecutor):
        async def close(self) -> None:
            nonlocal closed
            await asyncio.sleep(0)
            closed = True

    executor = object.__new__(AsyncMarieExecutor)
    executor._gpu_monitor_task = None
    executor._nvml_shutdown = MagicMock()

    await close_executor(executor)

    assert closed is True
    executor._nvml_shutdown.assert_called_once_with()
