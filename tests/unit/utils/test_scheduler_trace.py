import json

from marie.utils.scheduler_trace import scheduler_trace


def test_scheduler_trace_disabled_does_not_create_file(monkeypatch, tmp_path):
    trace_path = tmp_path / "scheduler-trace.jsonl"
    monkeypatch.delenv("MARIE_SCHEDULER_TRACE_ENABLED", raising=False)
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))

    scheduler_trace("ignored", job_id="job-1")

    assert not trace_path.exists()


def test_scheduler_trace_writes_jsonl_when_enabled(monkeypatch, tmp_path):
    trace_path = tmp_path / "nested" / "scheduler-trace.jsonl"
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PROFILE", "full")

    scheduler_trace("job_started", job_id="job-1", elapsed_ms=12.5)

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

    row = json.loads(trace_path.read_text().strip())
    assert row["event"] == "dispatch_batch_start"
    assert row["count"] == 2
    assert row["job_ids"] == ["job-1", "job-2"]


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

    row = json.loads(trace_path.read_text().strip())
    assert row["event"] == "scheduler_priority_refresh_completed"
    assert row["refresh_id"] == 8
    assert row["elapsed_ms"] == 12.5
