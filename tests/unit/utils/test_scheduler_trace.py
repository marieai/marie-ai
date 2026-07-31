import json

import marie.utils.scheduler_trace as scheduler_trace_module
from marie.utils.scheduler_trace import flush_scheduler_trace, scheduler_trace


def flush_trace() -> None:
    flush_scheduler_trace(close=True)


def test_scheduler_trace_disabled_does_not_create_file(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.delenv("MARIE_SCHEDULER_TRACE_ENABLED", raising=False)
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))

    scheduler_trace("ignored", job_id="job-1")
    flush_trace()

    assert not trace_path.exists()


def test_scheduler_trace_writes_jsonl_when_enabled(monkeypatch, tmp_path):
    trace_path = tmp_path / "nested" / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "full")

    scheduler_trace("job_started", job_id="job-1", elapsed_ms=12.5)
    flush_trace()

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "job_started"
    assert rows[0]["job_id"] == "job-1"
    assert rows[0]["elapsed_ms"] == 12.5
    assert "ts" in rows[0]
    assert "ts_unix" in rows[0]
    assert "pid" in rows[0]


def test_scheduler_trace_full_profile_drops_sensitive_fields(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "full")

    scheduler_trace(
        "gateway_submit_received",
        job_id="job-1",
        api_key="secret-api-key",
        project_id="secret-project-id",
        ref_id="document-1",
    )
    flush_trace()

    row = json.loads(trace_path.read_text().strip())
    assert row["job_id"] == "job-1"
    assert row["ref_id"] == "document-1"
    assert "api_key" not in row
    assert "project_id" not in row


def test_scheduler_trace_bad_path_is_best_effort(monkeypatch, tmp_path):
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(tmp_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "full")

    scheduler_trace("bad_path_is_ignored", job_id="job-1")
    flush_trace()


def test_scheduler_trace_compact_writes_allowed_events(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "compact")

    scheduler_trace(
        "candidate_built",
        candidates=2,
        job_ids=["job-1", "job-2"],
        project_id="project-1",
        ref_id="ref-1",
        planner="extract",
    )
    flush_trace()

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "candidate_built"
    assert rows[0]["candidates"] == 2
    assert rows[0]["job_ids"] == ["job-1", "job-2"]
    assert "project_id" not in rows[0]
    assert "ref_id" not in rows[0]
    assert "planner" not in rows[0]


def test_scheduler_trace_compact_drops_noisy_events(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "compact")

    scheduler_trace("control_flow_started", job_id="job-1")
    flush_trace()

    assert not trace_path.exists()


def test_scheduler_trace_compact_drops_raw_postgres_operations(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "compact")

    scheduler_trace("postgres_operation", operation="execute", elapsed_ms=1.0)
    flush_trace()

    assert not trace_path.exists()


def test_scheduler_trace_compact_keeps_full_batch_job_ids(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "compact")

    scheduler_trace(
        "dispatch_batch_start",
        count=2,
        job_ids=["job-1", "job-2"],
    )
    flush_trace()

    row = json.loads(trace_path.read_text().strip())
    assert row["event"] == "dispatch_batch_start"
    assert row["count"] == 2
    assert row["job_ids"] == ["job-1", "job-2"]


def test_scheduler_trace_compact_keeps_selection_timings(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "compact")

    scheduler_trace(
        "scheduler_selection_completed",
        outcome="completed",
        elapsed_ms=12.5,
        capture_ms=8.0,
        rank_ms=2.0,
        cap_ms=1.0,
        take_ms=1.5,
        ready_heap_entries=20,
        stale_heap_entries=3,
        job_ids=["job-1"],
    )
    flush_trace()

    row = json.loads(trace_path.read_text().strip())
    assert row["event"] == "scheduler_selection_completed"
    assert row["capture_ms"] == 8.0
    assert row["ready_heap_entries"] == 20
    assert row["stale_heap_entries"] == 3


def test_scheduler_trace_compact_writes_scheduler_counters(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "compact")

    scheduler_trace(
        "terminal_event_stale_attempt_total",
        count=1,
        job_id="job-1",
        run_attempt_id="attempt-1",
    )
    flush_trace()

    row = json.loads(trace_path.read_text().strip())
    assert row["event"] == "terminal_event_stale_attempt_total"
    assert row["count"] == 1
    assert row["job_id"] == "job-1"


def test_scheduler_trace_compact_writes_priority_refresh_completion(
    monkeypatch, tmp_path
):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "compact")

    scheduler_trace(
        "scheduler_priority_refresh_completed",
        refresh_id=8,
        elapsed_ms=12.5,
    )
    flush_trace()

    row = json.loads(trace_path.read_text().strip())
    assert row["event"] == "scheduler_priority_refresh_completed"
    assert row["refresh_id"] == 8
    assert row["elapsed_ms"] == 12.5


def test_scheduler_trace_compact_keeps_terminal_diagnostics(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "compact")
    events = [
        "terminal_job_lock_acquired",
        "terminal_db_operation_completed",
        "terminal_dag_lock_acquired",
        "postgres_notification_handler_completed",
        "job_supervisor_terminal_info_read",
        "gateway_event_loop_lag",
        "postgres_kv_operation_completed",
        "scheduler_dispatch_wait_completed",
        "scheduler_dispatch_cycle_started",
        "scheduler_dispatch_capacity_snapshot",
        "scheduler_dispatch_candidate_capture_completed",
    ]

    for event in events:
        scheduler_trace(event, job_id="job-1", elapsed_ms=1.0)
    flush_trace()

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == events


def test_scheduler_trace_reuses_writer_file_descriptor(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "full")
    real_open = scheduler_trace_module.os.open
    opened_paths: list[str] = []

    def recording_open(path, flags, mode=0o777):
        opened_paths.append(str(path))
        return real_open(path, flags, mode)

    monkeypatch.setattr(scheduler_trace_module.os, "open", recording_open)

    scheduler_trace("first", job_id="job-1")
    scheduler_trace("second", job_id="job-2")
    flush_trace()

    assert opened_paths == [str(trace_path)]
    assert len(trace_path.read_text().splitlines()) == 2
