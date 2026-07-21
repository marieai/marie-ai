import asyncio
from unittest.mock import MagicMock

import pytest

from marie.scheduler.psql import PostgreSQLJobScheduler


def build_scheduler() -> PostgreSQLJobScheduler:
    scheduler = PostgreSQLJobScheduler.__new__(PostgreSQLJobScheduler)
    scheduler.running = True
    scheduler._priority_refresh_event = asyncio.Event()
    scheduler._priority_refresh_source = "startup"
    scheduler._priority_refresh_running = False
    scheduler.priority_refresh_interval_seconds = 5.0
    scheduler.priority_refresh_timeout_seconds = 1.0
    scheduler._next_priority_refresh_at = 0.0
    scheduler._submission_count = 0
    scheduler._request_queue = asyncio.Queue()
    scheduler._pending_requests = {}
    scheduler.logger = MagicMock()
    return scheduler


async def wait_until(predicate, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_priority_refresh_worker_coalesces_pending_requests() -> None:
    scheduler = build_scheduler()
    first_refresh_release = asyncio.Event()
    sources: list[str] = []

    async def refresh(source: str) -> int:
        sources.append(source)
        if len(sources) == 1:
            await first_refresh_release.wait()
        return len(sources)

    scheduler._refresh_job_priorities = refresh
    worker = asyncio.create_task(scheduler._priority_refresh_loop())

    scheduler._request_priority_refresh("first")
    await wait_until(lambda: sources == ["first"])
    scheduler._request_priority_refresh("second")
    scheduler._request_priority_refresh("latest")
    first_refresh_release.set()
    await wait_until(lambda: len(sources) == 2)

    assert sources == ["first", "latest"]

    scheduler.running = False
    worker.cancel()
    await worker


@pytest.mark.asyncio
async def test_priority_refresh_worker_enforces_timeout() -> None:
    scheduler = build_scheduler()
    scheduler.priority_refresh_timeout_seconds = 0.01

    async def refresh(source: str) -> int:
        await asyncio.Event().wait()
        return 1

    scheduler._refresh_job_priorities = refresh
    worker = asyncio.create_task(scheduler._priority_refresh_loop())

    scheduler._request_priority_refresh("timeout-test")
    await wait_until(lambda: scheduler.logger.warning.called)

    scheduler.logger.warning.assert_called_once()
    assert scheduler._priority_refresh_running is False

    scheduler.running = False
    worker.cancel()
    await worker
