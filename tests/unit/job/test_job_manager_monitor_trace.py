from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from marie.job.common import JobStatus
from marie.job.job_manager import JobManager


async def test_monitor_trace_records_poll_sleep_and_terminal_observation(
    monkeypatch,
) -> None:
    manager = object.__new__(JobManager)
    manager.logger = Mock()
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
