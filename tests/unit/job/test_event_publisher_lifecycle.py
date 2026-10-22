import asyncio
import subprocess
import sys
import threading

import pytest

from marie.job.event_publisher import EventPublisher


async def test_sync_timeouts_preserve_capacity_until_the_call_finishes() -> None:
    release = threading.Event()
    finished = threading.Event()
    calls: list[int] = []
    threads: list[threading.Thread] = []
    received: list[int] = []

    def subscriber(_event_type: str, message: int) -> None:
        calls.append(message)
        threads.append(threading.current_thread())
        try:
            if message == 0:
                release.wait(5)
        finally:
            finished.set()

    async def async_subscriber(_event_type: str, message: int) -> None:
        received.append(message)

    publisher = EventPublisher(
        max_queue_size=16,
        worker_count=1,
        max_thread_workers=1,
        subscriber_timeout_s=0.02,
    )
    publisher.subscribe("event", subscriber)
    publisher.subscribe("event", async_subscriber)
    try:
        for message in range(10):
            await publisher.publish("event", message)
        await asyncio.wait_for(publisher.join(), timeout=2)
        assert calls == [0]
        assert received == list(range(10))
        assert all(thread.daemon for thread in threads)

        release.set()
        assert await asyncio.to_thread(finished.wait, 1)
        await publisher.publish("event", 10)
        await asyncio.wait_for(publisher.join(), timeout=1)
        assert calls == [0, 10]
        assert received == list(range(11))
    finally:
        release.set()
        await publisher.stop()


def test_timed_out_sync_subscriber_does_not_block_process_exit() -> None:
    code = """
import asyncio
import threading
from marie.job.event_publisher import EventPublisher

def subscriber(event_type, message):
    threading.Event().wait()

async def main():
    publisher = EventPublisher(
        worker_count=1, max_queue_size=1, subscriber_timeout_s=0.02
    )
    publisher.subscribe("event", subscriber)
    await publisher.publish("event", "message")
    await publisher.join()
    await publisher.stop()

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stderr


async def test_zero_async_timeout_keeps_a_positive_sync_timeout() -> None:
    release = threading.Event()
    received = asyncio.Event()

    def subscriber(_event_type: str, _message: str) -> None:
        release.wait(5)

    async def async_subscriber(_event_type: str, _message: str) -> None:
        received.set()

    publisher = EventPublisher(
        worker_count=1,
        subscriber_timeout_s=0,
        sync_subscriber_timeout_s=0.02,
    )
    publisher.subscribe("event", subscriber)
    publisher.subscribe("event", async_subscriber)
    try:
        assert publisher.subscriber_timeout_s == 0
        assert publisher.sync_subscriber_timeout_s == 0.02
        await publisher.publish("event", "message")
        await asyncio.wait_for(publisher.join(), timeout=1)
        assert received.is_set()
    finally:
        release.set()
        await publisher.stop()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_sync_timeout_must_be_positive_and_finite(timeout: float) -> None:
    with pytest.raises(ValueError, match="sync_subscriber_timeout_s"):
        EventPublisher(sync_subscriber_timeout_s=timeout)


async def test_sync_subscriber_exception_releases_capacity() -> None:
    received: list[int] = []

    def subscriber(_event_type: str, message: int) -> None:
        if message == 1:
            raise ValueError("subscriber failed")
        received.append(message)

    publisher = EventPublisher(worker_count=1, max_thread_workers=1)
    publisher.subscribe("event", subscriber)
    try:
        await publisher.publish("event", 1)
        await publisher.publish("event", 2)
        await asyncio.wait_for(publisher.join(), timeout=1)
        assert received == [2]
    finally:
        await publisher.stop()


async def test_stop_bounds_drain_and_unblocks_pending_publish(
    caplog, monkeypatch
) -> None:
    from marie.job.event_publisher import logger

    started = asyncio.Event()
    release = asyncio.Event()

    async def subscriber(_event_type: str, _message: int) -> None:
        started.set()
        await release.wait()

    publisher = EventPublisher(
        worker_count=1,
        max_queue_size=1,
        publish_blocking=True,
        subscriber_timeout_s=0,
    )
    publisher.subscribe("event", subscriber)
    monkeypatch.setattr(logger.logger, "propagate", True)
    blocked = None
    try:
        await publisher.publish("event", 0)
        await asyncio.wait_for(started.wait(), timeout=1)
        await publisher.publish("event", 1)
        blocked = asyncio.create_task(publisher.publish("event", 2))
        await asyncio.sleep(0)
        assert not blocked.done()
        await asyncio.wait_for(publisher.stop(timeout_s=0.05), timeout=0.5)
        with pytest.raises(RuntimeError, match="stopped"):
            await asyncio.wait_for(blocked, timeout=0.5)
        await asyncio.wait_for(publisher.join(), timeout=0.5)
        assert publisher.queue_size == 0
        assert publisher._active_publishes == 0
        assert "pending_publishes=1" in caplog.text
        assert "queued_events=1" in caplog.text
        await publisher.stop(timeout_s=0.05)
        with pytest.raises(RuntimeError, match="stopped"):
            await publisher.publish("event", 3)
    finally:
        release.set()
        if blocked is not None:
            await asyncio.gather(blocked, return_exceptions=True)
        await publisher.stop()


async def test_stop_does_not_await_a_subscriber_that_defers_cancellation() -> None:
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def subscriber(_event_type: str, _message: str) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()

    publisher = EventPublisher(worker_count=1, subscriber_timeout_s=0)
    publisher.subscribe("event", subscriber)
    workers = []
    try:
        await publisher.publish("event", "message")
        await asyncio.wait_for(started.wait(), timeout=1)
        workers = list(publisher._worker_tasks)
        await asyncio.wait_for(publisher.stop(timeout_s=0.05), timeout=0.5)
        await asyncio.wait_for(cancelling.wait(), timeout=0.5)
        assert any(not worker.done() for worker in workers)
        with pytest.raises(RuntimeError, match="cannot restart"):
            publisher.start()
    finally:
        release.set()
        if workers:
            for worker in workers:
                worker.cancel()
            await asyncio.wait_for(
                asyncio.gather(*workers, return_exceptions=True), timeout=1
            )
        await publisher.stop()


async def test_stop_cancellation_still_cleans_up_workers() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def subscriber(_event_type: str, _message: str) -> None:
        started.set()
        await release.wait()

    publisher = EventPublisher(worker_count=1, subscriber_timeout_s=0)
    publisher.subscribe("event", subscriber)
    stopping = None
    try:
        await publisher.publish("event", "message")
        await asyncio.wait_for(started.wait(), timeout=1)
        stopping = asyncio.create_task(publisher.stop())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert publisher._stopped.is_set()
        await asyncio.wait_for(publisher.join(), timeout=0.5)
    finally:
        release.set()
        if stopping is not None:
            await asyncio.gather(stopping, return_exceptions=True)
        await publisher.stop()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
async def test_stop_timeout_must_be_positive_and_finite(timeout: float) -> None:
    publisher = EventPublisher()
    try:
        with pytest.raises(ValueError, match="stop timeout_s"):
            await publisher.stop(timeout_s=timeout)
        assert publisher._accepting
    finally:
        await publisher.stop()


async def test_concurrent_stop_respects_its_own_deadline() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def subscriber(_event_type: str, _message: str) -> None:
        started.set()
        await release.wait()

    publisher = EventPublisher(worker_count=1, subscriber_timeout_s=0)
    publisher.subscribe("event", subscriber)
    stopping = None
    try:
        await publisher.publish("event", "message")
        await asyncio.wait_for(started.wait(), timeout=1)
        stopping = asyncio.create_task(publisher.stop(timeout_s=1))
        while publisher._accepting:
            await asyncio.sleep(0)
        await asyncio.wait_for(publisher.stop(timeout_s=0.02), timeout=0.5)
        assert not stopping.done()
        release.set()
        await asyncio.wait_for(stopping, timeout=1)
    finally:
        release.set()
        if stopping is not None:
            await asyncio.gather(stopping, return_exceptions=True)
        await publisher.stop()
