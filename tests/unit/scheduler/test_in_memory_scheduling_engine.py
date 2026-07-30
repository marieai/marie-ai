from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.scheduler.in_memory_scheduling_engine import InMemorySchedulingEngine
from marie.scheduler.memory_frontier import ReadyCapture
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState


def make_job(
    job_id: str,
    endpoint: str,
    *,
    priority: int,
    dag_id: str = "dag-1",
) -> WorkInfo:
    now = datetime.now(timezone.utc)
    return WorkInfo(
        id=job_id,
        dag_id=dag_id,
        name="extract",
        priority=priority,
        data={"metadata": {"on": endpoint}},
        state=WorkState.CREATED,
        retry_limit=1,
        retry_delay=0,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=3600,
        keep_until=now + timedelta(days=1),
        dependencies=[],
        job_level=1,
    )


@pytest.mark.asyncio
async def test_select_ready_owns_snapshot_rank_cap_and_take() -> None:
    jobs = [
        make_job("extract-1", "extract_executor://document/extract", priority=5),
        make_job("extract-2", "extract_executor://document/extract", priority=4),
        make_job("parser-1", "annotator_parser://document/parse", priority=3),
        make_job("parser-2", "annotator_parser://document/parse", priority=2),
        make_job("parser-3", "annotator_parser://document/parse", priority=1),
    ]
    capture = ReadyCapture(
        jobs=jobs,
        dag_remaining={"dag-1": 5},
        eligible_by_executor={"extract_executor": 2, "annotator_parser": 3},
        captured_by_executor={"extract_executor": 2, "annotator_parser": 3},
        eligible_by_dag={"dag-1": 5},
        captured_by_dag={"dag-1": 5},
        ready_heap_entries=8,
        ready_set_entries=5,
        stale_heap_entries=3,
    )
    frontier = SimpleNamespace(
        capture_ready=AsyncMock(return_value=capture),
        take=AsyncMock(return_value=[jobs[0], jobs[2], jobs[3]]),
    )
    engine = InMemorySchedulingEngine(
        frontier,
        sla_priority_interval_seconds=60,
    )

    result = await engine.select_ready(
        slots_by_executor={"extract_executor": 1, "annotator_parser": 2},
        batch_size=32,
        dispatch_capacity=3,
        lease_ttl=5,
        resident_dag_ids={"dag-1"},
        max_resident_dags=1,
    )

    assert result.candidate_ids == tuple(job.id for job in jobs)
    assert result.ranked_ids == tuple(job.id for job in jobs)
    assert result.requested_ids == ("extract-1", "parser-1", "parser-2")
    assert tuple(job.id for job in result.selected) == result.requested_ids
    assert result.candidate_window == 76
    frontier.take.assert_awaited_once_with(
        ["extract-1", "parser-1", "parser-2"],
        lease_ttl=5,
    )

    capture_call = frontier.capture_ready.await_args
    assert capture_call.args[:2] == (
        76,
        {"extract_executor": 1, "annotator_parser": 2},
    )
    eligible = capture_call.kwargs["filter_fn"]
    assert eligible(jobs[0]) is True
    assert eligible(make_job("noop", "noop://default", priority=100)) is False
    assert (
        eligible(
            make_job(
                "new-dag",
                "extract_executor://document/extract",
                priority=100,
                dag_id="dag-2",
            )
        )
        is False
    )


@pytest.mark.asyncio
async def test_select_ready_traces_actual_phase_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job("job-1", "extract://default", priority=1)
    frontier = SimpleNamespace(
        capture_ready=AsyncMock(
            return_value=ReadyCapture(
                [job],
                {"dag-1": 1},
                {"extract": 1},
                {"extract": 1},
                {"dag-1": 1},
                {"dag-1": 1},
                7,
                4,
                3,
            )
        ),
        take=AsyncMock(return_value=[job]),
    )
    trace = MagicMock()
    monkeypatch.setattr(
        "marie.scheduler.in_memory_scheduling_engine.scheduler_trace",
        trace,
    )
    engine = InMemorySchedulingEngine(
        frontier,
        sla_priority_interval_seconds=60,
    )

    await engine.select_ready(
        slots_by_executor={"extract": 1},
        batch_size=32,
        dispatch_capacity=1,
        lease_ttl=5,
        resident_dag_ids={"dag-1"},
        max_resident_dags=16,
    )

    events = [call.args[0] for call in trace.call_args_list]
    assert events == [
        "scheduler_selection_started",
        "scheduler_selection_capture_completed",
        "scheduler_selection_rank_completed",
        "scheduler_selection_cap_completed",
        "scheduler_selection_take_completed",
        "scheduler_selection_completed",
    ]
    capture_trace = trace.call_args_list[1].kwargs
    assert capture_trace["ready_heap_entries"] == 7
    assert capture_trace["ready_set_entries"] == 4
    assert capture_trace["stale_heap_entries"] == 3
    assert capture_trace["elapsed_ms"] >= 0

    completed_trace = trace.call_args_list[-1].kwargs
    assert completed_trace["capture_ms"] >= 0
    assert completed_trace["rank_ms"] >= 0
    assert completed_trace["cap_ms"] >= 0
    assert completed_trace["take_ms"] >= 0
    assert completed_trace["job_ids"] == ["job-1"]
    diagnostics = engine.diagnostics()
    assert diagnostics["sample_count"] == 1
    assert diagnostics["totals"] == {
        "candidates": 1,
        "requested": 1,
        "selected": 1,
    }
    assert diagnostics["latency_ms"]["p95"] >= 0
    assert (
        diagnostics["last"]["candidate_window"] == completed_trace["candidate_window"]
    )


@pytest.mark.asyncio
async def test_select_ready_reports_requested_and_selected_after_take_loss() -> None:
    first = make_job("job-1", "extract://default", priority=2)
    second = make_job("job-2", "extract://default", priority=1)
    frontier = SimpleNamespace(
        capture_ready=AsyncMock(
            return_value=ReadyCapture(
                [first, second],
                {"dag-1": 2},
                {"extract": 2},
                {"extract": 2},
                {"dag-1": 2},
                {"dag-1": 2},
                2,
                2,
                0,
            )
        ),
        take=AsyncMock(return_value=[second]),
    )
    engine = InMemorySchedulingEngine(
        frontier,
        sla_priority_interval_seconds=60,
    )

    result = await engine.select_ready(
        slots_by_executor={"extract": 2},
        batch_size=32,
        dispatch_capacity=2,
        lease_ttl=5,
        resident_dag_ids={"dag-1"},
        max_resident_dags=16,
    )

    assert result.requested_ids == ("job-1", "job-2")
    assert tuple(job.id for job in result.selected) == ("job-2",)


@pytest.mark.asyncio
async def test_select_ready_skips_regular_capture_without_executor_capacity() -> None:
    frontier = SimpleNamespace(
        capture_ready=AsyncMock(),
        take=AsyncMock(),
    )
    engine = InMemorySchedulingEngine(
        frontier,
        sla_priority_interval_seconds=60,
    )

    result = await engine.select_ready(
        slots_by_executor={"extract": 0},
        batch_size=32,
        dispatch_capacity=10,
        lease_ttl=5,
        resident_dag_ids=set(),
        max_resident_dags=16,
    )

    assert result.candidate_ids == ()
    assert result.selected == ()
    frontier.capture_ready.assert_not_awaited()
    frontier.take.assert_not_awaited()
