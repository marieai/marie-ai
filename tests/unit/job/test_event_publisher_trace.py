import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from marie.job.common import JobStatus
from marie.job.event_publisher import EventPublisher
from marie.job.job_manager import JobManager


async def test_job_status_event_trace_covers_enqueue_dispatch_and_completion(
    monkeypatch,
) -> None:
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "marie.job.event_publisher.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )
    received = asyncio.Event()

    async def subscriber(event_type: str, message: dict) -> None:
        assert event_type == JobStatus.SUCCEEDED
        assert message["job_id"] == "job-1"
        received.set()

    publisher = EventPublisher()
    publisher.subscribe(JobStatus.SUCCEEDED, subscriber)
    try:
        await publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-1"})
        await asyncio.wait_for(received.wait(), timeout=1)
        await asyncio.wait_for(publisher.join(), timeout=1)
    finally:
        await publisher.stop()

    assert [event for event, _ in events] == [
        "job_status_event_enqueued",
        "job_status_event_dequeued",
        "job_status_event_dispatch_completed",
    ]
    assert all(fields["job_id"] == "job-1" for _, fields in events)
    assert all(fields["status"] == "SUCCEEDED" for _, fields in events)
    assert events[1][1]["subscriber_count"] == 1
    assert events[1][1]["queue_wait_ms"] >= 0
    assert events[1][1]["dequeue_rate_per_second"] >= 1
    assert events[1][1]["queue_capacity"] == 1024
    assert events[1][1]["worker_queue_size"] >= 0
    assert events[2][1]["error_count"] == 0
    assert events[2][1]["timeout_count"] == 0
    assert events[2][1]["subscriber_delivery_ms"] >= 0


async def test_event_without_job_id_is_not_added_to_scheduler_trace(
    monkeypatch,
) -> None:
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "marie.job.event_publisher.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    publisher = EventPublisher()
    try:
        await publisher.publish("generic", {"value": "not-a-job"})
        await asyncio.wait_for(publisher.join(), timeout=1)
    finally:
        await publisher.stop()

    assert events == []


async def test_unrelated_jobs_dispatch_concurrently_with_per_job_order() -> None:
    active = 0
    peak_active = 0
    received: dict[str, list[int]] = {}

    async def subscriber(_event_type: str, message: dict) -> None:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.0005)
        received.setdefault(message["job_id"], []).append(message["sequence"])
        active -= 1

    publisher = EventPublisher(
        max_queue_size=64,
        publish_blocking=True,
        subscriber_timeout_s=0,
        worker_count=8,
    )
    publisher.subscribe(
        [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED], subscriber
    )
    try:
        await publisher.publish(
            JobStatus.PENDING, {"job_id": "ordered-job", "sequence": 1}
        )
        await publisher.publish(
            JobStatus.RUNNING, {"job_id": "ordered-job", "sequence": 2}
        )
        for index in range(1200):
            await publisher.publish(
                JobStatus.SUCCEEDED,
                {"job_id": f"job-{index}", "sequence": index},
            )
        await publisher.publish(
            JobStatus.SUCCEEDED, {"job_id": "ordered-job", "sequence": 3}
        )
        await asyncio.wait_for(publisher.join(), timeout=5)
    finally:
        await publisher.stop()

    assert peak_active > 1
    assert received["ordered-job"] == [1, 2, 3]
    assert sum(job_id.startswith("job-") for job_id in received) == 1200


async def test_blocking_publish_waits_for_a_full_worker_queue() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    received: list[int] = []

    async def subscriber(_event_type: str, message: dict) -> None:
        if message["sequence"] == 0:
            first_started.set()
            await release_first.wait()
        received.append(message["sequence"])

    publisher = EventPublisher(
        max_queue_size=1,
        publish_blocking=True,
        subscriber_timeout_s=0,
        worker_count=1,
    )
    publisher.subscribe(JobStatus.RUNNING, subscriber)
    try:
        await publisher.publish(JobStatus.RUNNING, {"job_id": "job-1", "sequence": 0})
        await asyncio.wait_for(first_started.wait(), timeout=1)
        await publisher.publish(JobStatus.RUNNING, {"job_id": "job-1", "sequence": 1})
        blocked = asyncio.create_task(
            publisher.publish(JobStatus.RUNNING, {"job_id": "job-1", "sequence": 2})
        )
        await asyncio.sleep(0)

        assert not blocked.done()
        assert publisher.queue_size == publisher.queue_capacity == 1

        release_first.set()
        await asyncio.wait_for(blocked, timeout=1)
        await asyncio.wait_for(publisher.join(), timeout=1)
    finally:
        release_first.set()
        await publisher.stop()

    assert received == [0, 1, 2]


async def test_subscriber_failure_isolated_from_later_same_job_event(
    monkeypatch,
) -> None:
    events: list[tuple[str, dict]] = []
    processed = asyncio.Event()

    async def subscriber(_event_type: str, message: dict) -> None:
        if message["sequence"] == 1:
            raise RuntimeError("expected failure")
        processed.set()

    monkeypatch.setattr(
        "marie.job.event_publisher.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )
    publisher = EventPublisher(
        max_queue_size=2,
        publish_blocking=True,
        subscriber_timeout_s=0,
        worker_count=1,
    )
    publisher.subscribe(JobStatus.RUNNING, subscriber)
    try:
        await publisher.publish(JobStatus.RUNNING, {"job_id": "job-1", "sequence": 1})
        await publisher.publish(JobStatus.RUNNING, {"job_id": "job-1", "sequence": 2})
        await asyncio.wait_for(processed.wait(), timeout=1)
        await asyncio.wait_for(publisher.join(), timeout=1)
    finally:
        await publisher.stop()

    completed = [
        fields
        for event, fields in events
        if event == "job_status_event_dispatch_completed"
    ]
    assert [fields["error_count"] for fields in completed] == [1, 0]


async def test_stop_drains_pending_events_and_rejects_late_publish() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    received: list[int] = []

    async def subscriber(_event_type: str, message: dict) -> None:
        started.set()
        await release.wait()
        received.append(message["sequence"])

    publisher = EventPublisher(
        max_queue_size=1,
        publish_blocking=True,
        subscriber_timeout_s=0,
        worker_count=1,
    )
    publisher.subscribe(JobStatus.SUCCEEDED, subscriber)
    await publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-1", "sequence": 1})
    await asyncio.wait_for(started.wait(), timeout=1)
    await publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-1", "sequence": 2})
    blocked_publish = asyncio.create_task(
        publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-1", "sequence": 3})
    )
    await asyncio.sleep(0)
    assert not blocked_publish.done()

    stop_task = asyncio.create_task(publisher.stop())
    await asyncio.sleep(0)
    assert not stop_task.done()

    release.set()
    await asyncio.wait_for(stop_task, timeout=1)

    await blocked_publish
    assert received == [1, 2, 3]
    with pytest.raises(RuntimeError, match="stopped"):
        await publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-late"})


async def test_lossless_publisher_settings_are_observable() -> None:
    publisher = EventPublisher(
        publish_blocking=True,
        subscriber_timeout_s=0,
    )
    try:
        assert publisher.publish_blocking is True
        assert publisher.subscriber_timeout_s == 0
    finally:
        await publisher.stop()


async def test_job_manager_blocks_publish_without_subscriber_timeout() -> None:
    storage = SimpleNamespace(close=AsyncMock())
    manager = JobManager(
        storage=storage,
        job_distributor=object(),
        etcd_client=object(),
        job_event_worker_count=3,
        job_event_queue_size=12,
    )
    try:
        assert manager.event_publisher.worker_count == 3
        assert manager.event_publisher.queue_capacity == 12
        assert manager.event_publisher.publish_blocking is True
        assert manager.event_publisher.subscriber_timeout_s == 0
    finally:
        await manager.shutdown()
    storage.close.assert_awaited_once()


async def test_nonblocking_publish_drops_when_worker_queue_is_full(
    monkeypatch,
) -> None:
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "marie.job.event_publisher.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )
    received: list[str] = []

    async def subscriber(_event_type: str, message: dict) -> None:
        received.append(message["job_id"])

    publisher = EventPublisher(max_queue_size=1, worker_count=1)
    publisher.subscribe(JobStatus.SUCCEEDED, subscriber)
    try:
        await publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-1"})
        await publisher.publish(JobStatus.SUCCEEDED, {"job_id": "job-2"})
        await asyncio.wait_for(publisher.join(), timeout=1)
    finally:
        await publisher.stop()

    assert received == ["job-1"]
    dropped = [
        fields for event, fields in events if event == "job_status_event_dropped"
    ]
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "queue_full"
    assert dropped[0]["queue_capacity"] == 1
