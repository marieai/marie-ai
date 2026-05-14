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

    scheduler_trace("job_started", job_id="job-1", elapsed_ms=12.5)

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "job_started"
    assert rows[0]["job_id"] == "job-1"
    assert rows[0]["elapsed_ms"] == 12.5
    assert "ts" in rows[0]
    assert "ts_unix" in rows[0]
    assert "pid" in rows[0]


def test_scheduler_trace_bad_path_is_best_effort(monkeypatch, tmp_path):
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_ENABLED", "true")
    monkeypatch.setenv("MARIE_SCHEDULER_TRACE_PATH", str(tmp_path))

    scheduler_trace("bad_path_is_ignored", job_id="job-1")
