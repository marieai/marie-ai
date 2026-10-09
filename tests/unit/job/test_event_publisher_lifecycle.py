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
