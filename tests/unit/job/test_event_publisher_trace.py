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
        await asyncio.sleep(0.005)
        received.setdefault(message["job_id"], []).append(message["sequence"])
        active -= 1

    publisher = EventPublisher(
        max_queue_size=512,
        subscriber_timeout_s=0,
        worker_count=8,
    )
    publisher.subscribe(JobStatus.SUCCEEDED, subscriber)
    try:
        await publisher.publish(
            JobStatus.SUCCEEDED, {"job_id": "ordered-job", "sequence": 1}
        )
        for index in range(200):
            await publisher.publish(
                JobStatus.SUCCEEDED,
                {"job_id": f"job-{index}", "sequence": index},
            )
        await publisher.publish(
            JobStatus.SUCCEEDED, {"job_id": "ordered-job", "sequence": 2}
        )
        await asyncio.wait_for(publisher.join(), timeout=5)
    finally:
        await publisher.stop()

    assert peak_active > 1
    assert received["ordered-job"] == [1, 2]
    assert sum(job_id.startswith("job-") for job_id in received) == 200


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
