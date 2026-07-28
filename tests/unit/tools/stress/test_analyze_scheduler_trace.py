from collections import Counter

from tools.stress.analyze_scheduler_trace import (
    _candidate_slot_capacity,
    _max_observed_free_slots,
    _print_dispatch_cycle_report,
    _print_dispatch_efficiency_report,
    _print_findings,
    _print_gateway_runtime_diagnostics,
    _print_latency_report,
    _print_priority_refresh_report,
    _print_selection_report,
    _print_terminal_critical_path_report,
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


def test_summary_reports_direct_durable_submission_latency() -> None:
    dag_events = {
        "gateway_submit_received": 1.0,
        "dag_persist_start": 1.02,
        "dag_persisted": 1.05,
    }
    summary = summarize_job(
        "job-1",
        [trace_row("dag_persisted", 1.05, job_id="job-1", dag_id="dag-1")],
        {"dag-1": dag_events},
    )

    assert round(summary["gateway_to_persist_start"]) == 20
    assert round(summary["gateway_to_persisted"]) == 50
    assert summary["submit_queue_wait"] is None


def test_summary_prefers_measured_selection_phases() -> None:
    rows = [
        trace_row(
            "scheduler_selection_capture_completed",
            1.01,
            job_ids=["job-1"],
        ),
        trace_row(
            "scheduler_selection_take_completed",
            1.04,
            job_ids=["job-1"],
        ),
        trace_row(
            "scheduler_selection_completed",
            1.05,
            job_ids=["job-1"],
            outcome="completed",
            elapsed_ms=50.0,
            capture_ms=20.0,
            rank_ms=15.0,
            cap_ms=5.0,
            take_ms=10.0,
        ),
        trace_row("candidate_built", 1.06, job_ids=["job-1"]),
        trace_row("planner_selected", 1.07, job_ids=["job-1"]),
        trace_row("frontier_taken", 1.08, job_ids=["job-1"]),
        trace_row("db_leased", 1.14, job_id="job-1"),
    ]

    summary = summarize_job("job-1", rows, {})

    assert summary["in_memory_selection"] == 50.0
    assert summary["selection_capture"] == 20.0
    assert summary["selection_rank"] == 15.0
    assert summary["selection_cap"] == 5.0
    assert summary["selection_take"] == 10.0
    assert summary["candidate_to_planned"] is None
    assert summary["planned_to_taken"] is None
    assert round(summary["taken_to_db_lease"]) == 100


def test_selection_report_exposes_phase_and_heap_measurements(capsys) -> None:
    rows = [
        trace_row(
            "scheduler_control_flow_peek_completed",
            1.0,
            elapsed_ms=4.0,
        ),
        trace_row(
            "scheduler_selection_completed",
            1.1,
            outcome="completed",
            elapsed_ms=10.0,
            capture_ms=6.0,
            rank_ms=2.0,
            cap_ms=1.0,
            take_ms=1.0,
            ready_heap_entries=20,
            ready_set_entries=15,
            stale_heap_entries=5,
        ),
    ]

    _print_selection_report(rows)

    output = capsys.readouterr().out
    assert "In-Memory Selection" in output
    assert "frontier capture: count=1" in output
    assert "control-flow frontier peek: count=1" in output
    assert "ready heap entries: count=1" in output
    assert "stale heap share: count=1 avg=25.0%" in output
    assert "frontier full scans: control_flow_peek=1 selection_capture=1" in output


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


def test_dispatch_confirmation_pipeline_is_reported(capsys) -> None:
    rows = [
        trace_row("gateway_dispatch_start", 1.0, job_id="job-1"),
        trace_row(
            "dispatch_batch_start",
            1.0,
            count=1,
            job_ids=["job-1"],
        ),
        trace_row(
            "dispatch_batch_launched",
            1.01,
            count=1,
            pending=3,
            limit=256,
        ),
        trace_row(
            "dispatch_confirmation_settled",
            1.2,
            job_id="job-1",
            outcome="confirmed",
            elapsed_ms=190.0,
        ),
    ]
    rate_stats = {
        "gateway_submit_received": (0, None, None),
        "gateway_dispatch_start": (1, None, None),
        "executor_success_recorded": (0, None, None),
        "executor_failed_recorded": (0, None, None),
    }

    _print_trace_coverage(rows)
    _print_dispatch_efficiency_report(rows, [], rate_stats)

    output = capsys.readouterr().out
    assert "dispatch confirmations: launched=1 settled=1 backpressure=0" in output
    assert "pending dispatch confirmations: count=1" in output
    assert "dispatch confirmation settlement: count=1" in output
    assert "dispatch confirmation outcomes: confirmed=1" in output


def test_findings_report_executor_slot_release_failures(capsys) -> None:
    _print_findings(
        {"gateway_dispatch_start": (0, None, None)},
        [],
        Counter({"executor_slot_release_failed": 2}),
        [],
    )

    output = capsys.readouterr().out
    assert "Executor slot release remains pending for 2 terminal jobs" in output


def test_findings_distinguish_recovered_slot_releases(capsys) -> None:
    rows = [
        trace_row(
            "executor_slot_release_failed",
            1.0,
            job_id="job-1",
            release_reason="counter_contention",
        ),
        trace_row(
            "executor_slot_release_retry_succeeded",
            2.0,
            job_id="job-1",
            release_reason="released",
        ),
    ]
    _print_findings(
        {"gateway_dispatch_start": (0, None, None)},
        [],
        Counter(row["event"] for row in rows),
        rows,
    )

    output = capsys.readouterr().out
    assert "All 1 transient executor slot release failures recovered" in output


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
    assert (
        "terminal scheduler wakes: queued=1 coalesced=1 coalesced_pct=50.0%" in output
    )


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
            queue_size=3,
            worker_queue_size=2,
        ),
        trace_row(
            "job_status_event_dequeued",
            1.82,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
            worker_id=5,
            dequeue_rate_per_second=42.0,
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
            subscriber_delivery_ms=12.0,
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
    assert round(summary["event_to_durable_terminal"]) == 280
    assert round(summary["event_dequeue_to_scheduler_enqueue"]) == 10
    assert round(summary["scheduler_event_queue_wait"]) == 70
    assert round(summary["scheduler_event_dequeue_to_handler"]) == 10
    assert round(summary["scheduler_handler_to_terminal"]) == 90
    assert round(summary["terminal_to_scheduler_event_processed"]) == 50

    _print_terminal_handoff_report(rows, [summary])

    output = capsys.readouterr().out
    assert "slot release to durable terminal acceptance: count=1" in output
    assert "executor terminal to supervisor send-task completion: count=1" in output
    assert "status publisher workers observed: 1 (5=1)" in output
    assert "status publisher total queue depth: count=1" in output
    assert "status publisher worker queue depth: count=1" in output
    assert "status publisher dequeue rate per second: count=1" in output
    assert "status publisher subscriber delivery: count=1" in output
    assert "configured monitor poll sleep: count=1" in output


def test_consolidated_event_path_reports_queue_handler_and_durable_latency() -> None:
    rows = [
        trace_row(
            "job_status_event_enqueued",
            1.0,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_status_event_dequeued",
            1.1,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "scheduler_job_event_received",
            1.11,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_terminal_attempt_accepted",
            1.3,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_status_event_dispatch_completed",
            1.35,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
    ]

    summary = summarize_job("job-1", rows, {})

    assert round(summary["event_queue_wait"]) == 100
    assert round(summary["event_dequeue_to_scheduler_enqueue"]) == 10
    assert summary["scheduler_event_queue_wait"] is None
    assert round(summary["scheduler_event_dequeue_to_handler"]) == 10
    assert round(summary["event_to_durable_terminal"]) == 300
    assert round(summary["scheduler_handler_to_terminal"]) == 190
    assert round(summary["terminal_to_scheduler_event_processed"]) == 50


def test_gateway_runtime_diagnostics_report_loop_and_kv_queues(capsys) -> None:
    rows = [
        trace_row(
            "gateway_event_loop_lag",
            1.0,
            pid=11,
            lag_ms=25.0,
            task_count=80,
            task_names={"supervisor:*": 60, "event-publisher-0": 1},
            task_names_other=19,
        ),
        trace_row(
            "gateway_event_loop_lag",
            2.0,
            pid=11,
            lag_ms=30.0,
            task_count=120,
            task_names={"supervisor:*": 90, "event-publisher-0": 1},
            task_names_other=29,
        ),
        trace_row(
            "postgres_kv_operation_completed",
            1.1,
            pid=11,
            operation="get",
            executor_queue_wait_ms=40.0,
            blocking_operation_ms=2.0,
            event_loop_resume_ms=15.0,
            total_ms=57.0,
        ),
        trace_row(
            "postgres_kv_operation_completed",
            1.2,
            pid=22,
            mode="async",
            operation="get",
            pool_wait_ms=3.0,
            database_operation_ms=4.0,
            total_ms=7.0,
        ),
    ]

    _print_gateway_runtime_diagnostics(rows)

    output = capsys.readouterr().out
    assert "Gateway Runtime Diagnostics" in output
    assert "event loop lag: count=2" in output
    assert "pending asyncio tasks: count=2" in output
    assert "asyncio task trend: first=80 last=120 delta=+40" in output
    assert "latest asyncio task groups: supervisor:*=90" in output
    assert "peak asyncio task groups: supervisor:*=90" in output
    assert "postgres KV gateway get (threaded): count=1 processes=1" in output
    assert "executor queue wait: count=1" in output
    assert "blocking connection/SQL: count=1" in output
    assert "event loop resume: count=1" in output
    assert "postgres KV executor get (async): count=1 processes=1" in output
    assert "pool wait: count=1" in output
    assert "database/transaction: count=1" in output


def test_dispatch_cycle_report_exposes_wake_to_capture_phases(capsys) -> None:
    rows = [
        trace_row(
            "scheduler_dispatch_wait_completed",
            1.0,
            cycle_index=4,
            outcome="wake",
            elapsed_ms=20.0,
        ),
        trace_row(
            "scheduler_dispatch_cycle_started",
            1.01,
            cycle_index=4,
            trigger="wake",
            wait_to_cycle_ms=1.0,
        ),
        trace_row(
            "scheduler_dispatch_capacity_snapshot",
            1.02,
            cycle_index=4,
            elapsed_ms=2.0,
            cycle_elapsed_ms=10.0,
        ),
        trace_row(
            "scheduler_dispatch_candidate_capture_completed",
            1.05,
            cycle_index=4,
            elapsed_ms=5.0,
            capacity_to_capture_ms=30.0,
            cycle_elapsed_ms=40.0,
        ),
    ]

    _print_dispatch_cycle_report(rows)

    output = capsys.readouterr().out
    assert "Scheduler Dispatch Cycle" in output
    assert "phases: started=1 capacity=1 candidates=1" in output
    assert "triggers: wake=1" in output
    assert "cycle start to capacity snapshot: count=1" in output
    assert "capacity snapshot to candidate capture: count=1" in output


def test_event_to_durable_terminal_report_includes_tail_percentiles(capsys) -> None:
    summaries = [{"event_to_durable_terminal": float(value)} for value in range(1, 101)]

    _print_latency_report(summaries)

    row = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("event->durable-terminal ")
    )
    assert row.split() == [
        "event->durable-terminal",
        "100",
        "50.5ms",
        "51.0ms",
        "90.0ms",
        "95.0ms",
        "99.0ms",
        "100.0ms",
    ]


def test_terminal_handoff_omits_executor_latency_without_executor_events(
    capsys,
) -> None:
    rows = [
        trace_row(
            "job_supervisor_send_task_completed",
            1.0,
            job_id="job-1",
            pid=11,
        ),
        trace_row(
            "job_monitor_terminal_observed",
            1.1,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_terminal_attempt_accepted",
            1.2,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
    ]

    summary = summarize_job("job-1", rows, {})

    assert summary["executor_terminal_to_supervisor_send_complete"] is None
    assert summary["executor_terminal_to_monitor_observed"] is None
    assert summary["terminal"] == "-"

    _print_terminal_handoff_report(rows, [summary])

    output = capsys.readouterr().out
    assert "executor terminal to supervisor send-task completion" not in output
    assert "executor terminal to monitor observation" not in output


def test_terminal_notification_fast_path_latency(capsys) -> None:
    rows = [
        trace_row(
            "executor_terminal_status_write_started",
            1.0,
            job_id="job-1",
            pid=22,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_terminal_notification_emit_started",
            1.1,
            job_id="job-1",
            pid=22,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_terminal_notification_received",
            1.15,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_status_event_enqueued",
            1.16,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "job_monitor_terminal_observed",
            1.17,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
        trace_row(
            "executor_success_recorded",
            1.2,
            job_id="job-1",
            pid=22,
        ),
        trace_row(
            "job_terminal_attempt_accepted",
            1.5,
            job_id="job-1",
            pid=11,
            status="SUCCEEDED",
        ),
    ]

    summary = summarize_job("job-1", rows, {})

    assert round(summary["terminal_write_to_notification_received"]) == 150
    assert round(summary["notification_emit_to_received"]) == 50
    assert round(summary["notification_received_to_event_enqueue"]) == 10
    assert round(summary["notification_received_to_monitor_observed"]) == 20

    _print_terminal_handoff_report(rows, [summary])

    output = capsys.readouterr().out
    assert "terminal status write to notification receipt: count=1" in output
    assert "notification receipt to event enqueue: count=1" in output


def test_terminal_critical_path_report_exposes_new_boundaries(capsys) -> None:
    rows = [
        trace_row(
            "terminal_job_lock_acquired",
            1.0,
            job_id="job-1",
            contended=True,
            wait_ms=12.0,
        ),
        trace_row(
            "terminal_db_operation_completed",
            1.1,
            job_id="job-1",
            operation="job_terminal_transition",
            pool_wait_ms=10.0,
            sql_ms=20.0,
            commit_ms=5.0,
            total_ms=38.0,
        ),
        trace_row(
            "terminal_db_operation_completed",
            1.15,
            job_id="job-1",
            operation="job_complete",
            pool_wait_ms=20.0,
            sql_ms=30.0,
            commit_ms=None,
            total_ms=55.0,
        ),
        trace_row(
            "terminal_db_operation_completed",
            1.2,
            job_id="job-1",
            operation="attempt_terminal_audit",
            pool_wait_ms=10.0,
            sql_ms=15.0,
            commit_ms=5.0,
            total_ms=32.0,
        ),
        trace_row(
            "terminal_db_operation_completed",
            1.25,
            operation="control_flow_batch_complete",
            pool_wait_ms=8.0,
            sql_ms=12.0,
            commit_ms=None,
            total_ms=22.0,
        ),
        trace_row(
            "terminal_dag_lock_acquired",
            1.3,
            job_id="job-1",
            contended=False,
            wait_ms=2.0,
        ),
        trace_row(
            "postgres_notification_handler_completed",
            1.4,
            job_id="job-1",
            driver_to_dispatch_ms=40.0,
            handler_ms=3.0,
        ),
        trace_row(
            "job_supervisor_terminal_status_read",
            1.5,
            job_id="job-1",
            elapsed_ms=25.0,
        ),
        trace_row(
            "job_supervisor_terminal_info_read",
            1.6,
            job_id="job-1",
            elapsed_ms=18.0,
        ),
        trace_row(
            "job_supervisor_terminal_status_cache_hit",
            1.7,
            job_id="job-2",
            status="SUCCEEDED",
        ),
    ]

    _print_terminal_critical_path_report(rows)

    output = capsys.readouterr().out
    assert "Terminal Critical Path Detail" in output
    assert "terminal job lock: acquisitions=1 contended=1" in output
    assert "terminal DAG lock: acquisitions=1 contended=0" in output
    assert "terminal database job_terminal_transition: count=1" in output
    assert "terminal database job_complete: count=1" in output
    assert "terminal database attempt_terminal_audit: count=1" in output
    assert "terminal database control_flow_batch_complete: count=1" in output
    assert "notification driver receipt to handler dispatch: count=1" in output
    assert "supervisor terminal info KV read: count=1" in output
    assert (
        "supervisor terminal status resolution: cache_hits=1 kv_fallback_reads=1"
        in output
    )
