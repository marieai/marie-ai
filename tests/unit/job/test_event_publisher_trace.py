import asyncio

from marie.job.common import JobStatus
from marie.job.event_publisher import EventPublisher


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
        await asyncio.wait_for(publisher._queue.join(), timeout=1)
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
    assert events[2][1]["error_count"] == 0
    assert events[2][1]["timeout_count"] == 0


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
        await asyncio.wait_for(publisher._queue.join(), timeout=1)
    finally:
        await publisher.stop()

    assert events == []
