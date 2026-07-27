import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import marie.scheduler.psql as scheduler_psql
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.psql import PostgreSQLJobScheduler, PriorityRefreshResult
from marie.scheduler.state import WorkState


def build_scheduler() -> PostgreSQLJobScheduler:
    scheduler = PostgreSQLJobScheduler.__new__(PostgreSQLJobScheduler)
    scheduler.running = True
    scheduler.priority_refresh_enabled = True
    scheduler._priority_refresh_event = asyncio.Event()
    scheduler._priority_refresh_source = "startup"
    scheduler._priority_refresh_running = False
    scheduler.priority_refresh_interval_seconds = 5.0
    scheduler.priority_refresh_timeout_seconds = 1.0
    scheduler._next_priority_refresh_at = 0.0
    scheduler.submission_service = SimpleNamespace(
        submission_count=0,
        queue_size=0,
        pending_count=0,
    )
    scheduler.logger = MagicMock()
    return scheduler


def build_frontier(job_count: int = 600) -> MemoryFrontier:
    frontier = MemoryFrontier()
    start_after = datetime.now(timezone.utc) - timedelta(seconds=1)
    for index in range(job_count):
        work_item = WorkInfo.model_construct(
            id=f"job-{index}",
            dag_id=f"dag-{index // 10}",
            name="extract",
            priority=0,
            data={"metadata": {"on": "extract://default"}},
            state=WorkState.CREATED,
            retry_limit=0,
            retry_delay=0,
            retry_backoff=False,
            start_after=start_after,
            expire_in_seconds=0,
            keep_until=start_after,
            job_level=0,
            soft_sla=None,
            hard_sla=None,
        )
        frontier.jobs_by_id[work_item.id] = work_item
        frontier.dag_nodes[work_item.dag_id].add(work_item.id)
        frontier.unmet_count[work_item.id] = 0
        frontier._push_ready(work_item)
    return frontier


async def wait_until(predicate, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_priority_refresh_worker_coalesces_pending_requests() -> None:
    scheduler = build_scheduler()
    first_refresh_release = asyncio.Event()
    sources: list[str] = []

    async def refresh(source: str) -> PriorityRefreshResult:
        sources.append(source)
        if len(sources) == 1:
            await first_refresh_release.wait()
        return PriorityRefreshResult(refresh_id=len(sources))

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

    async def refresh(source: str) -> PriorityRefreshResult:
        await asyncio.Event().wait()
        return PriorityRefreshResult(refresh_id=1)

    scheduler._refresh_job_priorities = refresh
    worker = asyncio.create_task(scheduler._priority_refresh_loop())

    scheduler._request_priority_refresh("timeout-test")
    await wait_until(lambda: scheduler.logger.warning.called)

    scheduler.logger.warning.assert_called_once()
    assert scheduler._priority_refresh_running is False

    scheduler.running = False
    worker.cancel()
    await worker


@pytest.mark.asyncio
async def test_priority_refresh_worker_emits_only_failure_for_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_scheduler()
    trace_events: list[str] = []
    monkeypatch.setattr(
        scheduler_psql,
        "scheduler_trace",
        lambda event, **_fields: trace_events.append(event),
    )
    scheduler._refresh_job_priorities = AsyncMock(
        return_value=PriorityRefreshResult(refresh_id=7, error="database failed")
    )
    worker = asyncio.create_task(scheduler._priority_refresh_loop())

    scheduler._request_priority_refresh("failure-test")
    await wait_until(lambda: "scheduler_priority_refresh_failed" in trace_events)

    assert trace_events.count("scheduler_priority_refresh_failed") == 1
    assert "scheduler_priority_refresh_completed" not in trace_events
    assert "scheduler_priority_refresh_done" not in trace_events
    assert "scheduler_priority_refresh_returned" not in trace_events

    scheduler.running = False
    worker.cancel()
    await worker


@pytest.mark.asyncio
async def test_priority_refresh_worker_emits_only_completed_for_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_scheduler()
    trace_events: list[str] = []
    monkeypatch.setattr(
        scheduler_psql,
        "scheduler_trace",
        lambda event, **_fields: trace_events.append(event),
    )
    scheduler._refresh_job_priorities = AsyncMock(
        return_value=PriorityRefreshResult(refresh_id=8)
    )
    worker = asyncio.create_task(scheduler._priority_refresh_loop())

    scheduler._request_priority_refresh("success-test")
    await wait_until(lambda: "scheduler_priority_refresh_completed" in trace_events)

    assert trace_events.count("scheduler_priority_refresh_completed") == 1
    assert "scheduler_priority_refresh_failed" not in trace_events
    assert "scheduler_priority_refresh_done" not in trace_events
    assert "scheduler_priority_refresh_returned" not in trace_events

    scheduler.running = False
    worker.cancel()
    await worker


@pytest.mark.asyncio
async def test_refresh_job_priorities_returns_typed_failure() -> None:
    scheduler = build_scheduler()
    scheduler._priority_refresh_seq = 0
    scheduler.priority_refresh_hydrate_limit = 100
    scheduler.dag_service = SimpleNamespace(
        refresh_frontier_priorities=AsyncMock(
            side_effect=RuntimeError("database failed")
        )
    )

    result = await scheduler._refresh_job_priorities(source="test")

    assert result == PriorityRefreshResult(refresh_id=1, error="database failed")


@pytest.mark.asyncio
async def test_priority_refresh_reports_hard_sla_misses() -> None:
    scheduler = build_scheduler()
    scheduler._priority_refresh_seq = 0
    scheduler.priority_refresh_hydrate_limit = 100
    scheduler.sla_warning_top_n = 5
    scheduler.dag_service = SimpleNamespace(
        refresh_frontier_priorities=AsyncMock(
            return_value={"tracked": 1, "fetched": 1, "changed": 0}
        )
    )
    scheduler.frontier = SimpleNamespace(
        refresh_ready_ordering=AsyncMock(),
        priority_refresh_summary=AsyncMock(
            return_value={
                "totals": {"jobs": 1, "dags": 1, "ready": 1, "blocked": 0},
                "sla": {"tracked": 1, "hard_missed": 1},
            }
        ),
    )

    result = await scheduler._refresh_job_priorities(source="test")

    assert result == PriorityRefreshResult(refresh_id=1)
    scheduler.logger.warning.assert_called_once_with(
        "[SLA] 1 jobs have missed hard SLA; planner ranking continues to prefer them"
    )


@pytest.mark.asyncio
async def test_ready_order_refresh_cancellation_preserves_heap() -> None:
    frontier = build_frontier()
    original_heap = list(frontier._ready_heap)
    original_versions = dict(frontier._ver)
    original_sequence = frontier._seq
    task = asyncio.create_task(frontier.refresh_ready_ordering())

    await asyncio.sleep(0)

    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert frontier._ready_heap == original_heap
    assert dict(frontier._ver) == original_versions
    assert frontier._seq == original_sequence


@pytest.mark.asyncio
async def test_priority_update_cancellation_preserves_priorities_and_heap() -> None:
    frontier = build_frontier()
    original_heap = list(frontier._ready_heap)
    original_versions = dict(frontier._ver)
    priorities = {job_id: 10 for job_id in frontier.jobs_by_id}
    task = asyncio.create_task(frontier.refresh_priorities(priorities))

    await asyncio.sleep(0)

    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(work_item.priority == 0 for work_item in frontier.jobs_by_id.values())
    assert frontier._ready_heap == original_heap
    assert dict(frontier._ver) == original_versions


@pytest.mark.asyncio
async def test_priority_refresh_summary_is_cancellable() -> None:
    frontier = build_frontier()
    task = asyncio.create_task(frontier.priority_refresh_summary(top_n=5))

    await asyncio.sleep(0)

    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_priority_refresh_summary_matches_frontier_summary() -> None:
    frontier = build_frontier(job_count=10)

    refresh_summary = await frontier.priority_refresh_summary(top_n=5)
    full_summary = frontier.summary(detail=True, top_n=5)

    assert refresh_summary["totals"] == full_summary["totals"]
    assert refresh_summary["sla"] == full_summary["sla"]
