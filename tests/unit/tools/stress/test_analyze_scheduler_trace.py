from collections import Counter

from tools.stress.analyze_scheduler_trace import (
    _candidate_slot_capacity,
    _max_observed_free_slots,
    _print_dispatch_efficiency_report,
    _print_findings,
    _print_priority_refresh_report,
    _print_terminal_feedback_report,
    _print_terminal_handoff_report,
    _print_trace_coverage,
    _priority_refresh_success_count,
    _refresh_attempt_status,
    _workload_executors,
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
    assert "slot_release_failed=0" in output


def test_trace_coverage_reports_deferred_frontier_admission(capsys) -> None:
    rows = [
        trace_row("gateway_dispatch_start", 1.0, job_id="job-1"),
        trace_row("dag_frontier_added", 1.1, dag_id="dag-1"),
        trace_row(
            "dag_frontier_deferred",
            1.2,
            dag_id="dag-2",
            reason="active_limit",
        ),
        trace_row(
            "dag_frontier_deferred",
            1.3,
            dag_id="dag-3",
            reason="executor_capacity",
        ),
    ]

    _print_trace_coverage(rows)

    output = capsys.readouterr().out
    assert "submission frontier: admitted=1 deferred=2" in output
    assert "active_limit=1 executor_capacity=1" in output


def test_findings_report_executor_slot_release_failures(capsys) -> None:
    _print_findings(
        {"gateway_dispatch_start": (0, None, None)},
        [],
        Counter({"executor_slot_release_failed": 2}),
        [],
    )

    output = capsys.readouterr().out
    assert "Executor slot release failed for 2 terminal jobs" in output


def test_terminal_feedback_reports_resolution_and_next_candidate(capsys) -> None:
    rows = [
        trace_row("job_terminal_attempt_accepted", 1.0, job_id="job-1"),
        trace_row("terminal_dag_resolution_started", 1.1, job_id="job-1"),
        trace_row("terminal_dag_resolution_completed", 1.3, job_id="job-1"),
        trace_row(
            "terminal_scheduler_wake_completed",
            1.4,
            job_id="job-1",
            wake_queued=True,
        ),
        trace_row(
            "terminal_scheduler_wake_completed",
            1.5,
            job_id="job-2",
            wake_queued=False,
        ),
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
    assert "scheduler wake to next global candidate snapshot: count=2" in output
    assert "terminal scheduler wakes: queued=1 coalesced=1 coalesced_pct=50.0%" in output


def test_dispatch_capacity_excludes_unrelated_executors(capsys) -> None:
    rows = [
        trace_row(
            "candidate_built",
            1.0,
            job_ids=["job-a", "job-b"],
            slots_by_executor={
                "mock_executor_a": 2,
                "mock_executor_b": 2,
                "annotator_llm": 1,
                "plugin_daemon_executor": 1,
            },
        ),
        trace_row(
            "slot_reserved",
            1.1,
            job_id="job-a",
            executor="mock_executor_a",
            slots_before=2,
        ),
        trace_row(
            "slot_reserved",
            1.1,
            job_id="job-b",
            executor="mock_executor_b",
            slots_before=2,
        ),
        trace_row(
            "gateway_dispatch_start",
            1.2,
            job_id="job-a",
            entrypoint="mock_executor_a://document/process",
        ),
        trace_row(
            "gateway_dispatch_start",
            1.2,
            job_id="job-b",
            entrypoint="mock_executor_b://document/process",
        ),
        trace_row(
            "executor_slot_released",
            1.6,
            job_id="job-a",
            deployment="mock_executor_a",
        ),
        trace_row(
            "executor_slot_released",
            1.6,
            job_id="job-b",
            deployment="mock_executor_b",
        ),
    ]
    workload_executors = _workload_executors(rows)

    assert workload_executors == {"mock_executor_a", "mock_executor_b"}
    assert _max_observed_free_slots(rows, workload_executors) == 4
    assert _candidate_slot_capacity(rows, workload_executors) == ([4.0], [2.0])

    rate_stats = {
        "gateway_submit_received": (2, 10.0, 0.2),
        "gateway_dispatch_start": (2, 4.0, 0.5),
        "executor_success_recorded": (2, 4.0, 0.5),
        "executor_failed_recorded": (0, None, None),
    }
    _print_dispatch_efficiency_report(rows, [], rate_stats)

    output = capsys.readouterr().out
    assert "dag_submit=10.000/s" in output
    assert "executor_dispatch=4.000/s" in output
    assert "rate units: dag_submit counts DAGs" in output
    assert "submit-dispatch drift" not in output
    assert "workload executors: 2 (mock_executor_a, mock_executor_b)" in output
    assert "max_observed_slots=4" in output
    assert "ready-compatible share of workload free slots: 50.0%" in output


def test_terminal_handoff_uses_gateway_process_and_terminal_status(capsys) -> None:
    rows = [
        trace_row(
            "executor_slot_released",
            0.99,
            job_id="job-1",
            pid=22,
            deployment="mock_executor_a",
        ),
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
            "scheduler_job_event_enqueued",
            1.83,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_status_event_dispatch_completed",
            1.84,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "scheduler_job_event_dequeued",
            1.9,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "scheduler_job_event_received",
            1.91,
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
            "scheduler_job_event_processed",
            2.05,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
    ]

    summary = summarize_job("job-1", rows, {})

    assert round(summary["slot_release_to_terminal"]) == 1010
    assert round(summary["executor_terminal_to_supervisor_send_complete"]) == 700
    assert round(summary["executor_terminal_to_monitor_observed"]) == 400
    assert round(summary["supervisor_send_complete_to_status_read"]) == 10
    assert round(summary["status_read_to_event_enqueue"]) == 10
    assert round(summary["event_queue_wait"]) == 100
    assert round(summary["event_dequeue_to_scheduler_enqueue"]) == 10
    assert round(summary["scheduler_event_queue_wait"]) == 70
    assert round(summary["scheduler_event_dequeue_to_handler"]) == 10
    assert round(summary["scheduler_handler_to_terminal"]) == 90
    assert round(summary["terminal_to_scheduler_event_processed"]) == 50

    _print_terminal_handoff_report(rows, [summary])

    output = capsys.readouterr().out
    assert "slot release to durable terminal acceptance: count=1" in output
    assert "executor terminal to supervisor send-task completion: count=1" in output
    assert "configured monitor poll sleep: count=1" in output
