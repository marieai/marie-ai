import asyncio

import pytest

from marie.job.event_publisher_async import EventPublisher


def test_construction_does_not_require_a_running_loop() -> None:
    publisher = EventPublisher()
    assert publisher._dispatcher_task is None


def test_start_requires_a_running_loop() -> None:
    publisher = EventPublisher()
    with pytest.raises(RuntimeError, match="no running event loop"):
        publisher.start()
    assert publisher._dispatcher_task is None


async def test_dispatch_starts_explicitly() -> None:
    publisher = EventPublisher()
    delivered = asyncio.Event()
    received: list[tuple[str, str]] = []

    async def subscriber(event_type: str, message: str) -> None:
        received.append((event_type, message))
        delivered.set()

    publisher.subscribe("event", subscriber)
    try:
        assert publisher._dispatcher_task is None
        publisher.start()
        dispatcher = publisher._dispatcher_task
        publisher.start()
        assert publisher._dispatcher_task is dispatcher
        await publisher.publish("event", "message")
        await asyncio.wait_for(delivered.wait(), timeout=1)
        assert received == [("event", "message")]
    finally:
        await publisher.stop()
