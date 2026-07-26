import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from marie.job.common import JobStatus
from marie.job.job_manager import JobManager


async def test_monitor_trace_records_poll_sleep_and_terminal_observation(
    monkeypatch,
) -> None:
    manager = object.__new__(JobManager)
    manager.logger = Mock()
    manager._terminal_notifications = {}
    manager._terminal_wake_events = {}
    manager._job_info_client = SimpleNamespace(
        get_status=AsyncMock(side_effect=[JobStatus.RUNNING, JobStatus.SUCCEEDED])
    )
    supervisor = SimpleNamespace(ping=AsyncMock())
    sleep = AsyncMock()
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr("marie.job.job_manager.random.uniform", lambda _a, _b: 1.1)
    monkeypatch.setattr("marie.job.job_manager.asyncio.sleep", sleep)
    monkeypatch.setattr(
        "marie.job.job_manager.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    await manager._monitor_job_internal("job-1", supervisor)

    assert [event for event, _ in events] == [
        "job_monitor_status_observed",
        "job_monitor_sleep_started",
        "job_monitor_status_observed",
        "job_monitor_terminal_observed",
    ]
    assert events[1][1]["wait_ms"] == 1100.0
    assert events[-1][1]["status"] == "SUCCEEDED"
    supervisor.ping.assert_awaited_once()
    sleep.assert_awaited_once_with(1.1)


async def test_terminal_notification_publishes_and_wakes_monitor(monkeypatch) -> None:
    manager = object.__new__(JobManager)
    manager.logger = Mock()
    manager.event_publisher = SimpleNamespace(publish=AsyncMock())
    manager._terminal_notifications = {}
    manager._terminal_wake_events = {}
    manager._published_terminal_events = {}
    manager._active_run_attempts = {"job-1": "attempt-1"}
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "marie.job.job_manager.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    await manager.handle_job_status_notification(
        {
            "job_id": "job-1",
            "status": "SUCCEEDED",
            "run_owner": "owner-1",
            "run_attempt_id": "attempt-1",
        }
    )

    assert manager._terminal_notifications == {"job-1": JobStatus.SUCCEEDED}
    assert manager._terminal_wake_events["job-1"].is_set()
    manager.event_publisher.publish.assert_awaited_once()
    assert [event for event, _ in events] == [
        "job_terminal_notification_received",
        "job_terminal_event_published",
    ]

    published = await manager._publish_terminal_event(
        "job-1",
        JobStatus.SUCCEEDED,
        "owner-1",
        "attempt-1",
        "supervisor_finalize",
    )

    assert published is False
    manager.event_publisher.publish.assert_awaited_once()
    assert events[-1][0] == "job_terminal_event_publish_skipped"


async def test_terminal_notification_rejects_stale_run_attempt(monkeypatch) -> None:
    manager = object.__new__(JobManager)
    manager.logger = Mock()
    manager.event_publisher = SimpleNamespace(publish=AsyncMock())
    manager._terminal_notifications = {}
    manager._terminal_wake_events = {}
    manager._published_terminal_events = {}
    manager._active_run_attempts = {"job-1": "attempt-2"}
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "marie.job.job_manager.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    await manager.handle_job_status_notification(
        {
            "job_id": "job-1",
            "status": "SUCCEEDED",
            "run_attempt_id": "attempt-1",
        }
    )

    assert manager._terminal_notifications == {}
    assert manager._terminal_wake_events == {}
    manager.event_publisher.publish.assert_not_awaited()
    assert events == [
        (
            "job_terminal_event_publish_skipped",
            {
                "job_id": "job-1",
                "status": "SUCCEEDED",
                "run_attempt_id": "attempt-1",
                "source": "postgres_notify",
                "reason": "stale_run_attempt",
                "expected_run_attempt_id": "attempt-2",
            },
        )
    ]


async def test_monitor_consumes_terminal_notification_without_database_read(
    monkeypatch,
) -> None:
    manager = object.__new__(JobManager)
    manager.logger = Mock()
    manager._terminal_notifications = {"job-1": JobStatus.SUCCEEDED}
    manager._terminal_wake_events = {"job-1": asyncio.Event()}
    manager._job_info_client = SimpleNamespace(get_status=AsyncMock())
    supervisor = SimpleNamespace(ping=AsyncMock())
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "marie.job.job_manager.scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    await manager._monitor_job_internal("job-1", supervisor)

    manager._job_info_client.get_status.assert_not_awaited()
    supervisor.ping.assert_not_awaited()
    assert events[-1] == (
        "job_monitor_terminal_observed",
        {
            "job_id": "job-1",
            "status": "SUCCEEDED",
            "source": "postgres_notify",
        },
    )
