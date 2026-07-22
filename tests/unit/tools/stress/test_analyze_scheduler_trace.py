from collections import Counter

from tools.stress.analyze_scheduler_trace import (
    _print_priority_refresh_report,
    _print_terminal_feedback_report,
    _print_terminal_handoff_report,
    _print_trace_coverage,
    _priority_refresh_success_count,
    _refresh_attempt_status,
    summarize_job,
)


def trace_row(event: str, ts: float, **fields) -> dict:
    return {"event": event, "ts_unix": ts, **fields}


def test_priority_refresh_completed_is_a_terminal_event(capsys) -> None:
    rows = [
        trace_row("scheduler_priority_refresh_due", 1.0, refresh_id=8),
        trace_row("scheduler_priority_refresh_start", 1.1, refresh_id=8),
        trace_row(
            "scheduler_priority_refresh_completed",
            1.3,
            refresh_id=8,
            elapsed_ms=200.0,
        ),
    ]

    assert _refresh_attempt_status(rows) == "completed"
    assert _priority_refresh_success_count(Counter(row["event"] for row in rows)) == 1

    _print_priority_refresh_report(rows)

    output = capsys.readouterr().out
    assert "total 1" in output
    assert "incomplete:" not in output
    assert "incomplete attempts:" not in output


def test_trace_coverage_exposes_missing_executor_process_events(capsys) -> None:
    rows = [
        trace_row("gateway_dispatch_start", 1.0, job_id="job-1"),
        trace_row("gateway_dispatch_confirmed", 1.1, job_id="job-1"),
        trace_row("job_terminal_attempt_accepted", 2.0, job_id="job-1"),
    ]

    _print_trace_coverage(rows)

    output = capsys.readouterr().out
    assert (
        "scheduler: dispatch=1 admission=1 terminal_handler=0 durable_terminal=1"
        in output
    )
    assert "executor: received=0" in output


def test_terminal_feedback_reports_resolution_and_next_candidate(capsys) -> None:
    rows = [
        trace_row("job_terminal_attempt_accepted", 1.0, job_id="job-1"),
        trace_row("terminal_dag_resolution_started", 1.1, job_id="job-1"),
        trace_row("terminal_dag_resolution_completed", 1.3, job_id="job-1"),
        trace_row("terminal_scheduler_wake_completed", 1.4, job_id="job-1"),
        trace_row("candidate_built", 1.6, job_ids=["job-2"]),
    ]
    summaries = [
        summarize_job(
            "job-1",
            rows[:-1],
            {},
        )
    ]

    _print_terminal_feedback_report(rows, summaries)

    output = capsys.readouterr().out
    assert "terminal to DAG-resolution start: count=1" in output
    assert "DAG resolution: count=1" in output
    assert "scheduler wake to next global candidate snapshot: count=1" in output


def test_terminal_handoff_uses_gateway_process_and_terminal_status(capsys) -> None:
    rows = [
        trace_row(
            "job_status_event_enqueued",
            0.9,
            job_id="job-1",
            pid=11,
            status="RUNNING",
        ),
        trace_row(
            "executor_success_recorded",
            1.0,
            job_id="job-1",
            pid=22,
        ),
        trace_row(
            "job_status_event_enqueued",
            1.01,
            job_id="job-1",
            pid=22,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_monitor_sleep_started",
            1.1,
            job_id="job-1",
            pid=11,
            status="RUNNING",
            wait_ms=1000.0,
        ),
        trace_row(
            "job_monitor_terminal_observed",
            1.4,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_supervisor_send_task_completed",
            1.7,
            job_id="job-1",
            pid=11,
        ),
        trace_row(
            "job_supervisor_terminal_status_read",
            1.71,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_status_event_enqueued",
            1.72,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_status_event_dequeued",
            1.82,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "scheduler_job_event_received",
            1.83,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_terminal_attempt_accepted",
            2.0,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_status_event_dispatch_completed",
            2.1,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
    ]

    summary = summarize_job("job-1", rows, {})

    assert round(summary["executor_terminal_to_supervisor_send_complete"]) == 700
    assert round(summary["executor_terminal_to_monitor_observed"]) == 400
    assert round(summary["supervisor_send_complete_to_status_read"]) == 10
    assert round(summary["status_read_to_event_enqueue"]) == 10
    assert round(summary["event_queue_wait"]) == 100
    assert round(summary["event_dequeue_to_scheduler_handler"]) == 10
    assert round(summary["scheduler_handler_to_terminal"]) == 170
    assert round(summary["terminal_to_event_dispatch_complete"]) == 100

    _print_terminal_handoff_report(rows, [summary])

    output = capsys.readouterr().out
    assert "executor terminal to supervisor send-task completion: count=1" in output
    assert "configured monitor poll sleep: count=1" in output
