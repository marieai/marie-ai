import asyncio
import logging

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


async def test_dispatch_resumes_after_restart() -> None:
    publisher = EventPublisher()
    delivered = asyncio.Event()
    received: list[int] = []

    async def subscriber(_event_type: str, message: int) -> None:
        received.append(message)
        delivered.set()

    publisher.subscribe("event", subscriber)
    try:
        publisher.start()
        await publisher.publish("event", 1)
        await asyncio.wait_for(delivered.wait(), timeout=1)
        await publisher.stop()

        delivered.clear()
        publisher.start()
        await publisher.publish("event", 2)
        await asyncio.wait_for(delivered.wait(), timeout=1)
        assert received == [1, 2]
    finally:
        await publisher.stop()


@pytest.mark.parametrize("stopped", [False, True])
async def test_publish_requires_a_running_dispatcher(stopped: bool) -> None:
    publisher = EventPublisher()
    if stopped:
        publisher.start()
        await publisher.stop()
    with pytest.raises(RuntimeError, match="not running"):
        await publisher.publish("event", "message")
    assert publisher._queue.empty()


async def test_start_rejects_restart_until_stop_finishes() -> None:
    publisher = EventPublisher()
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def subscriber(_event_type: str, _message: str) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelling.set()
            await release.wait()

    publisher.subscribe("event", subscriber)
    publisher.start()
    stopping = None
    try:
        await publisher.publish("event", "message")
        await asyncio.wait_for(started.wait(), timeout=1)
        stopping = asyncio.create_task(publisher.stop())
        await asyncio.wait_for(cancelling.wait(), timeout=1)
        with pytest.raises(RuntimeError, match="stopping"):
            publisher.start()
        with pytest.raises(RuntimeError, match="not running"):
            await publisher.publish("event", "another")
    finally:
        release.set()
        if stopping is not None:
            await asyncio.wait_for(stopping, timeout=1)
        await publisher.stop()


def test_unsubscribe_is_safe_for_missing_and_removed_subscribers() -> None:
    publisher = EventPublisher()

    def subscriber(_event_type: str, _message: str) -> None:
        pass

    def other_subscriber(_event_type: str, _message: str) -> None:
        pass

    publisher.unsubscribe("missing", subscriber)
    publisher.subscribe("event", subscriber)
    publisher.unsubscribe("event", other_subscriber)
    assert publisher._subscribers["event"] == [subscriber]

    publisher.subscribe("event", other_subscriber)
    publisher.unsubscribe("event", subscriber)
    publisher.unsubscribe("event", subscriber)
    assert publisher._subscribers["event"] == [other_subscriber]

    publisher.unsubscribe("event", other_subscriber)
    publisher.unsubscribe("event", other_subscriber)
    assert "event" not in publisher._subscribers


@pytest.mark.parametrize("asynchronous", [True, False])
async def test_subscriber_failure_logs_context_and_preserves_delivery(
    caplog, monkeypatch, asynchronous: bool
) -> None:
    publisher = EventPublisher()
    delivered = asyncio.Event()
    received: list[int] = []
    failure = ValueError("subscriber failed")

    def failing_subscriber(_event_type: str, _message: int) -> None:
        raise failure

    async def failing_async_subscriber(event_type: str, message: int) -> None:
        failing_subscriber(event_type, message)

    async def successful_subscriber(_event_type: str, message: int) -> None:
        received.append(message)
        if len(received) == 2:
            delivered.set()

    subscriber = failing_async_subscriber if asynchronous else failing_subscriber
    publisher.subscribe("test-event", subscriber)
    publisher.subscribe("test-event", successful_subscriber)
    log = logging.getLogger("marie.job.event_publisher_async")
    monkeypatch.setattr(log, "propagate", True)
    try:
        with caplog.at_level(logging.ERROR, logger=log.name):
            publisher.start()
            await publisher.publish("test-event", 1)
            await publisher.publish("test-event", 2)
            await asyncio.wait_for(delivered.wait(), timeout=1)
        assert received == [1, 2]
        records = [record for record in caplog.records if record.name == log.name]
        assert len(records) == 2
        for record in records:
            assert "test-event" in record.getMessage()
            assert subscriber.__name__ in record.getMessage()
            assert record.exc_info is not None
            assert record.exc_info[1] is failure
            assert record.exc_info[2] is not None
    finally:
        await publisher.stop()
