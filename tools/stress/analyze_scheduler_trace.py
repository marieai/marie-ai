#!/usr/bin/env python3
"""Summarize Marie scheduler JSONL trace timings by job."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EVENTS = (
    "gateway_submit_received",
    "gateway_submit_accepted",
    "gateway_submit_notified",
    "gateway_submit_persisted",
    "gateway_submit_scheduled_event",
    "scheduler_submission_enqueued",
    "scheduler_submission_dequeued",
    "dag_plan_built",
    "dag_persist_start",
    "dag_persisted",
    "dag_frontier_add_start",
    "dag_frontier_added",
    "dag_admission_decision",
    "candidate_built",
    "planner_selected",
    "frontier_taken",
    "db_leased",
    "semaphore_reserve_batch_start",
    "semaphore_reserve_batch_done",
    "slot_unavailable",
    "slot_reserve_failed",
    "slot_reserved",
    "job_activate_start",
    "job_active_marked",
    "gateway_dispatch_start",
    "gateway_dispatch_submitted",
    "gateway_dispatch_confirmed",
    "executor_request_received",
    "executor_running_recorded",
    "executor_callback_invoked",
    "executor_slot_released",
    "executor_success_recorded",
    "executor_failed_recorded",
    "scheduler_priority_refresh_requested",
    "scheduler_priority_refresh_due",
    "scheduler_priority_refresh_start",
    "scheduler_priority_refresh_frontier_start",
    "scheduler_priority_refresh_frontier_priority_load_start",
    "scheduler_priority_refresh_frontier_priority_load_done",
    "scheduler_priority_refresh_frontier_priority_apply_start",
    "scheduler_priority_refresh_frontier_priority_apply_done",
    "scheduler_priority_refresh_frontier_discover_start",
    "scheduler_priority_refresh_frontier_discover_connection_start",
    "scheduler_priority_refresh_frontier_discover_connection_done",
    "scheduler_priority_refresh_frontier_discover_query_start",
    "scheduler_priority_refresh_frontier_discover_query_done",
    "scheduler_priority_refresh_frontier_discover_fetch_start",
    "scheduler_priority_refresh_frontier_discover_fetch_done",
    "scheduler_priority_refresh_frontier_discover_commit_start",
    "scheduler_priority_refresh_frontier_discover_commit_done",
    "scheduler_priority_refresh_frontier_discover_failed",
    "scheduler_priority_refresh_frontier_discover_done",
    "scheduler_priority_refresh_frontier_hydrate_start",
    "scheduler_priority_refresh_frontier_hydrate_done",
    "scheduler_priority_refresh_frontier_hydrate_skip",
    "scheduler_priority_refresh_frontier_hydrate_stop",
    "scheduler_priority_refresh_frontier_done",
    "scheduler_priority_refresh_ready_ordering_start",
    "scheduler_priority_refresh_ready_ordering_done",
    "scheduler_priority_refresh_summary_start",
    "scheduler_priority_refresh_summary_done",
    "scheduler_priority_refresh_hard_sla_policy_start",
    "scheduler_priority_refresh_hard_sla_policy_done",
    "scheduler_priority_refresh_done",
    "scheduler_priority_refresh_failed",
    "scheduler_priority_refresh_returned",
    "scheduler_dag_sync_loop_start",
    "scheduler_dag_sync_cycle_skipped",
    "scheduler_dag_sync_cycle_start",
    "scheduler_dag_sync_cycle_done",
    "scheduler_dag_sync_cycle_failed",
    "scheduler_dag_sync_loop_stopped",
    "postgres_pool_acquire_wait_start",
    "postgres_pool_acquire_wait_done",
    "postgres_pool_acquire_timeout",
)

SORT_ALIASES = {
    "dag_scheduled_to_dispatch": "notified_to_dispatch",
    "dag_persisted_to_dispatch": "accepted_to_dispatch",
}

RATE_EVENTS = (
    "gateway_submit_received",
    "gateway_dispatch_start",
    "executor_success_recorded",
    "executor_failed_recorded",
)

REPORT_LATENCIES = (
    ("gateway->dispatch", "gateway_to_dispatch"),
    ("submit-queue", "submit_queue_wait"),
    ("dag-persist", "dag_persist"),
    ("frontier->candidate", "frontier_to_candidate"),
    ("candidate->planned", "candidate_to_planned"),
    ("planned->taken", "planned_to_taken"),
    ("taken->db-lease", "taken_to_db_lease"),
    ("db-lease->slot", "db_lease_to_slot"),
    ("slot->active", "slot_to_active"),
    ("active->dispatch", "activate_to_dispatch"),
    ("dispatch->confirm", "dispatch_to_confirm"),
    ("dispatch->executor", "dispatch_to_executor"),
    ("receive->running", "executor_start_record"),
    ("service", "executor_service"),
    ("callback->release", "callback_to_slot_release"),
    ("callback->terminal", "callback_to_terminal_status"),
)

SCHEDULER_DISPATCH_PATH = (
    "candidate_to_planned",
    "planned_to_taken",
    "taken_to_db_lease",
    "db_lease_to_slot",
    "slot_to_active",
    "activate_to_dispatch",
)

EXECUTOR_HANDOFF_PATH = (
    "dispatch_to_executor",
    "executor_start_record",
)

DISPATCH_BOTTLENECK_STAGES = (
    ("candidate->planned", "candidate_to_planned"),
    ("planned->taken", "planned_to_taken"),
    ("taken->db-lease", "taken_to_db_lease"),
    ("db-lease->slot", "db_lease_to_slot"),
    ("slot->active", "slot_to_active"),
    ("active->dispatch", "activate_to_dispatch"),
    ("dispatch->confirm", "dispatch_to_confirm"),
    ("dispatch->executor", "dispatch_to_executor"),
    ("receive->running", "executor_start_record"),
    ("callback->release", "callback_to_slot_release"),
    ("callback->terminal", "callback_to_terminal_status"),
)

DAG_LATENCIES = {
    "submit_queue_wait",
    "submit_worker_to_persist_start",
    "dag_persist",
    "persisted_to_frontier",
}

PRIORITY_REFRESH_EVENTS = (
    "scheduler_priority_refresh_due",
    "scheduler_priority_refresh_start",
    "scheduler_priority_refresh_frontier_start",
    "scheduler_priority_refresh_frontier_priority_load_start",
    "scheduler_priority_refresh_frontier_priority_load_done",
    "scheduler_priority_refresh_frontier_priority_apply_start",
    "scheduler_priority_refresh_frontier_priority_apply_done",
    "scheduler_priority_refresh_frontier_discover_start",
    "scheduler_priority_refresh_frontier_discover_connection_start",
    "scheduler_priority_refresh_frontier_discover_connection_done",
    "scheduler_priority_refresh_frontier_discover_query_start",
    "scheduler_priority_refresh_frontier_discover_query_done",
    "scheduler_priority_refresh_frontier_discover_fetch_start",
    "scheduler_priority_refresh_frontier_discover_fetch_done",
    "scheduler_priority_refresh_frontier_discover_commit_start",
    "scheduler_priority_refresh_frontier_discover_commit_done",
    "scheduler_priority_refresh_frontier_discover_failed",
    "scheduler_priority_refresh_frontier_discover_done",
    "scheduler_priority_refresh_frontier_hydrate_start",
    "scheduler_priority_refresh_frontier_hydrate_done",
    "scheduler_priority_refresh_frontier_hydrate_skip",
    "scheduler_priority_refresh_frontier_hydrate_stop",
    "scheduler_priority_refresh_frontier_done",
    "scheduler_priority_refresh_ready_ordering_start",
    "scheduler_priority_refresh_ready_ordering_done",
    "scheduler_priority_refresh_summary_start",
    "scheduler_priority_refresh_summary_done",
    "scheduler_priority_refresh_hard_sla_policy_start",
    "scheduler_priority_refresh_hard_sla_policy_done",
    "scheduler_priority_refresh_done",
    "scheduler_priority_refresh_failed",
    "scheduler_priority_refresh_returned",
)

PRIORITY_REFRESH_PHASES = (
    ("due->returned", "scheduler_priority_refresh_returned"),
    ("total", "scheduler_priority_refresh_done"),
    ("frontier", "scheduler_priority_refresh_frontier_done"),
    (
        "frontier-priority-load",
        "scheduler_priority_refresh_frontier_priority_load_done",
    ),
    (
        "frontier-priority-apply",
        "scheduler_priority_refresh_frontier_priority_apply_done",
    ),
    ("frontier-discover", "scheduler_priority_refresh_frontier_discover_done"),
    (
        "frontier-discover-connect",
        "scheduler_priority_refresh_frontier_discover_connection_done",
    ),
    (
        "frontier-discover-query",
        "scheduler_priority_refresh_frontier_discover_query_done",
    ),
    (
        "frontier-discover-fetch",
        "scheduler_priority_refresh_frontier_discover_fetch_done",
    ),
    (
        "frontier-discover-commit",
        "scheduler_priority_refresh_frontier_discover_commit_done",
    ),
    ("frontier-hydrate", "scheduler_priority_refresh_frontier_hydrate_done"),
    ("ready-ordering", "scheduler_priority_refresh_ready_ordering_done"),
    ("summary", "scheduler_priority_refresh_summary_done"),
    ("hard-sla-policy", "scheduler_priority_refresh_hard_sla_policy_done"),
)

DAG_SYNC_EVENTS = (
    "scheduler_dag_sync_loop_start",
    "scheduler_dag_sync_cycle_skipped",
    "scheduler_dag_sync_cycle_start",
    "scheduler_dag_sync_cycle_done",
    "scheduler_dag_sync_cycle_failed",
    "scheduler_dag_sync_loop_stopped",
)

POSTGRES_POOL_EVENTS = (
    "postgres_pool_acquire_wait_start",
    "postgres_pool_acquire_wait_done",
    "postgres_pool_acquire_timeout",
)


def _event_times(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_event: dict[str, float] = {}
    for row in sorted(rows, key=lambda item: item.get("ts_unix", 0.0)):
        event = row.get("event")
        ts = row.get("ts_unix")
        if event in EVENTS and isinstance(ts, (int, float)):
            by_event.setdefault(event, float(ts))
    return by_event


def _milliseconds(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def _first(by_event: dict[str, float], *events: str) -> float | None:
    for event in events:
        if event in by_event:
            return by_event[event]
    return None


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1000.0:
        return f"{value / 1000.0:.3f}s"
    return f"{value:.1f}ms"


def _fmt_s(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 3600.0:
        return f"{value / 3600.0:.2f}h"
    if value >= 60.0:
        return f"{value / 60.0:.2f}m"
    return f"{value:.1f}s"


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}/s"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _job_key(row: dict[str, Any]) -> str | None:
    job_id = row.get("job_id")
    if isinstance(job_id, str) and job_id:
        return job_id
    dag_id = row.get("dag_id")
    if isinstance(dag_id, str) and dag_id:
        return dag_id
    return None


def load_trace(
    path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
            rows.append(row)
            key = _job_key(row)
            if key:
                grouped[key].append(row)
                continue
            job_ids = row.get("job_ids")
            if isinstance(job_ids, list):
                for job_id in job_ids:
                    if isinstance(job_id, str) and job_id:
                        grouped[job_id].append(row)
    return grouped, rows


def load_events(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped, _ = load_trace(path)
    return grouped


def summarize_job(
    job_id: str,
    rows: list[dict[str, Any]],
    dag_events_by_id: dict[str, dict[str, float]],
) -> dict[str, Any]:
    dag_id = None
    for row in sorted(rows, key=lambda item: item.get("ts_unix", 0.0)):
        if dag_id is None and isinstance(row.get("dag_id"), str):
            dag_id = row["dag_id"]
    by_event = _event_times(rows)
    dag_events = dag_events_by_id.get(dag_id or "", {})

    terminal_event = None
    if "executor_success_recorded" in by_event:
        terminal_event = "executor_success_recorded"
    elif "executor_failed_recorded" in by_event:
        terminal_event = "executor_failed_recorded"

    dispatch_start = by_event.get("gateway_dispatch_start")
    accepted_at = _first(
        dag_events,
        "gateway_submit_accepted",
        "gateway_submit_persisted",
    )
    notified_at = _first(
        dag_events,
        "gateway_submit_notified",
        "gateway_submit_scheduled_event",
    )

    return {
        "job_id": job_id,
        "dag_id": dag_id or "-",
        "gateway_to_dispatch": _milliseconds(
            dag_events.get("gateway_submit_received"),
            dispatch_start,
        ),
        "accepted_to_dispatch": _milliseconds(
            accepted_at,
            dispatch_start,
        ),
        "notified_to_dispatch": _milliseconds(
            notified_at,
            dispatch_start,
        ),
        "submit_queue_wait": _milliseconds(
            dag_events.get("scheduler_submission_enqueued"),
            dag_events.get("scheduler_submission_dequeued"),
        ),
        "submit_worker_to_persist_start": _milliseconds(
            dag_events.get("scheduler_submission_dequeued"),
            dag_events.get("dag_persist_start"),
        ),
        "dag_persist": _milliseconds(
            dag_events.get("dag_persist_start"),
            dag_events.get("dag_persisted"),
        ),
        "persisted_to_frontier": _milliseconds(
            dag_events.get("dag_persisted"),
            dag_events.get("dag_frontier_added"),
        ),
        "frontier_to_dispatch": _milliseconds(
            dag_events.get("dag_frontier_added"),
            dispatch_start,
        ),
        "frontier_to_candidate": _milliseconds(
            dag_events.get("dag_frontier_added"),
            by_event.get("candidate_built"),
        ),
        "candidate_to_planned": _milliseconds(
            by_event.get("candidate_built"),
            by_event.get("planner_selected"),
        ),
        "planned_to_taken": _milliseconds(
            by_event.get("planner_selected"),
            by_event.get("frontier_taken"),
        ),
        "taken_to_db_lease": _milliseconds(
            by_event.get("frontier_taken"),
            by_event.get("db_leased"),
        ),
        "db_lease_to_slot": _milliseconds(
            by_event.get("db_leased"),
            by_event.get("slot_reserved"),
        ),
        "slot_to_active": _milliseconds(
            by_event.get("slot_reserved"),
            by_event.get("job_active_marked"),
        ),
        "activate_to_dispatch": _milliseconds(
            by_event.get("job_active_marked"),
            dispatch_start,
        ),
        "dispatch_to_confirm": _milliseconds(
            dispatch_start,
            by_event.get("gateway_dispatch_confirmed"),
        ),
        "dispatch_to_executor": _milliseconds(
            dispatch_start,
            by_event.get("executor_request_received"),
        ),
        "executor_start_record": _milliseconds(
            by_event.get("executor_request_received"),
            by_event.get("executor_running_recorded"),
        ),
        "executor_service": _milliseconds(
            by_event.get("executor_running_recorded"),
            by_event.get("executor_callback_invoked"),
        ),
        "callback_to_slot_release": _milliseconds(
            by_event.get("executor_callback_invoked"),
            by_event.get("executor_slot_released"),
        ),
        "callback_to_terminal_status": _milliseconds(
            by_event.get("executor_callback_invoked"),
            by_event.get(terminal_event) if terminal_event else None,
        ),
        "terminal": terminal_event or "-",
        "events": sum(1 for row in rows if row.get("event") in EVENTS),
    }


def _rate_stats(
    rows: list[dict[str, Any]],
) -> dict[str, tuple[int, float | None, float | None]]:
    by_event: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        event = row.get("event")
        ts = row.get("ts_unix")
        if event in RATE_EVENTS and isinstance(ts, (int, float)):
            by_event[event].append(float(ts))

    stats: dict[str, tuple[int, float | None, float | None]] = {}
    for event in RATE_EVENTS:
        values = by_event.get(event, [])
        if len(values) < 2:
            stats[event] = (len(values), None, None)
            continue
        span = max(values) - min(values)
        rate = len(values) / span if span > 0 else None
        stats[event] = (len(values), rate, span)
    return stats


def _numeric_values(
    summaries: list[dict[str, Any]],
    key: str,
    *,
    unique_dag: bool = False,
) -> list[float]:
    seen_dags: set[str] = set()
    values: list[float] = []
    for item in summaries:
        if unique_dag:
            dag_id = item.get("dag_id")
            if not isinstance(dag_id, str) or dag_id in seen_dags:
                continue
            seen_dags.add(dag_id)
        value = item.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _summed_latency_values(
    summaries: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    for item in summaries:
        total = 0.0
        for key in keys:
            value = item.get(key)
            if not isinstance(value, (int, float)):
                break
            total += float(value)
        else:
            values.append(total)
    return values


def _print_distribution(label: str, values: list[float]) -> None:
    if not values:
        return
    print(
        f"{label}: "
        f"count={len(values)} "
        f"avg={_fmt(_avg(values))} "
        f"p50={_fmt(_percentile(values, 0.50))} "
        f"p95={_fmt(_percentile(values, 0.95))} "
        f"max={_fmt(max(values))}"
    )


def _print_count_distribution(label: str, values: list[float]) -> None:
    if not values:
        return
    print(
        f"{label}: "
        f"count={len(values)} "
        f"avg={_avg(values):.1f} "
        f"p50={_percentile(values, 0.50):.0f} "
        f"p95={_percentile(values, 0.95):.0f} "
        f"max={max(values):.0f}"
    )


def _capacity_per_second(duration_ms: float | None) -> float | None:
    if duration_ms is None or duration_ms <= 0:
        return None
    return 1000.0 / duration_ms


def _event_intervals_ms(rows: list[dict[str, Any]], event_name: str) -> list[float]:
    times = sorted(
        float(row["ts_unix"])
        for row in rows
        if row.get("event") == event_name
        and isinstance(row.get("ts_unix"), (int, float))
    )
    return [(end - start) * 1000.0 for start, end in zip(times, times[1:])]


def _max_observed_free_slots(rows: list[dict[str, Any]]) -> int | None:
    max_slots: int | None = None
    for row in rows:
        event = row.get("event")
        if event == "candidate_built":
            slots = row.get("slots_by_executor")
            if not isinstance(slots, dict):
                continue
            total = sum(value for value in slots.values() if isinstance(value, int))
        elif event == "slot_reserved":
            slots_before = row.get("slots_before")
            if not isinstance(slots_before, int):
                continue
            total = slots_before
        else:
            continue
        max_slots = total if max_slots is None else max(max_slots, total)
    return max_slots


def _candidate_selection_attempts(
    rows: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    candidate_times: dict[str, list[float]] = defaultdict(list)
    selected_times: dict[str, float] = {}
    for row in sorted(rows, key=lambda item: item.get("ts_unix", 0.0)):
        ts = row.get("ts_unix")
        if not isinstance(ts, (int, float)):
            continue
        job_ids = row.get("job_ids")
        if not isinstance(job_ids, list):
            continue
        event = row.get("event")
        if event == "candidate_built":
            for job_id in job_ids:
                if isinstance(job_id, str):
                    candidate_times[job_id].append(float(ts))
        elif event == "planner_selected":
            for job_id in job_ids:
                if isinstance(job_id, str):
                    selected_times.setdefault(job_id, float(ts))

    appearances: list[float] = []
    first_candidate_wait: list[float] = []
    for job_id, selected_at in selected_times.items():
        before_selection = [
            candidate_at
            for candidate_at in candidate_times.get(job_id, [])
            if candidate_at <= selected_at
        ]
        if not before_selection:
            continue
        appearances.append(float(len(before_selection)))
        first_candidate_wait.append((selected_at - before_selection[0]) * 1000.0)
    return appearances, first_candidate_wait


def _event_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(row["event"] for row in rows if isinstance(row.get("event"), str))


def _executor_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    request_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    for row in rows:
        event = row.get("event")
        pid = row.get("pid")
        if pid is None:
            continue
        if event == "executor_request_received":
            request_counts[str(pid)] += 1
        elif event in {"executor_success_recorded", "executor_failed_recorded"}:
            terminal_counts[str(pid)] += 1
    return request_counts or terminal_counts


def _candidate_counts(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("event") != "candidate_built":
            continue
        count = row.get("candidates")
        if isinstance(count, (int, float)):
            values.append(float(count))
    return values


def _slot_counts(rows: list[dict[str, Any]]) -> dict[str, Counter[int]]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        if row.get("event") != "candidate_built":
            continue
        slots = row.get("slots_by_executor")
        if not isinstance(slots, dict):
            continue
        for executor, slot_count in slots.items():
            if isinstance(slot_count, int):
                counts[str(executor)][slot_count] += 1
    return counts


def _executor_slot_idle_ms(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    releases_by_executor: dict[str, list[float]] = defaultdict(list)
    reserves_by_executor: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        event = row.get("event")
        ts = row.get("ts_unix")
        if not isinstance(ts, (int, float)):
            continue
        if event == "executor_slot_released":
            executor = row.get("deployment") or row.get("executor")
            if isinstance(executor, str) and executor:
                releases_by_executor[executor].append(float(ts))
        elif event == "slot_reserved":
            executor = row.get("executor") or row.get("deployment")
            if isinstance(executor, str) and executor:
                reserves_by_executor[executor].append(float(ts))

    idle_by_executor: dict[str, list[float]] = {}
    for executor, releases in releases_by_executor.items():
        reserves = sorted(reserves_by_executor.get(executor, []))
        if not reserves:
            continue
        idle: list[float] = []
        index = 0
        for released_at in sorted(releases):
            while index < len(reserves) and reserves[index] <= released_at:
                index += 1
            if index >= len(reserves):
                break
            idle.append((reserves[index] - released_at) * 1000.0)
        if idle:
            idle_by_executor[executor] = idle
    return idle_by_executor


def _slot_hold_ms(rows: list[dict[str, Any]]) -> list[float]:
    by_job: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        job_id = row.get("job_id")
        event = row.get("event")
        ts = row.get("ts_unix")
        if not isinstance(job_id, str) or not isinstance(ts, (int, float)):
            continue
        if event in {"slot_reserved", "executor_slot_released"}:
            by_job[job_id][str(event)] = float(ts)

    values: list[float] = []
    for events in by_job.values():
        reserved_at = events.get("slot_reserved")
        released_at = events.get("executor_slot_released")
        value = _milliseconds(reserved_at, released_at)
        if value is not None:
            values.append(value)
    return values


def _slot_cycle_ms(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    by_job: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        job_id = row.get("job_id")
        event = row.get("event")
        ts = row.get("ts_unix")
        if not isinstance(job_id, str) or not isinstance(ts, (int, float)):
            continue
        if event == "slot_reserved":
            executor = row.get("executor") or row.get("deployment")
            if isinstance(executor, str) and executor:
                by_job[job_id]["executor"] = executor
            by_job[job_id]["reserved_at"] = float(ts)
        elif event == "executor_slot_released":
            executor = row.get("deployment") or row.get("executor")
            if isinstance(executor, str) and executor:
                by_job[job_id]["executor"] = executor
            by_job[job_id]["released_at"] = float(ts)

    intervals_by_executor: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for events in by_job.values():
        executor = events.get("executor")
        reserved_at = events.get("reserved_at")
        released_at = events.get("released_at")
        if (
            isinstance(executor, str)
            and isinstance(reserved_at, float)
            and isinstance(released_at, float)
        ):
            intervals_by_executor[executor].append((reserved_at, released_at))

    cycles_by_executor: dict[str, list[float]] = {}
    for executor, intervals in intervals_by_executor.items():
        ordered = sorted(intervals)
        cycles: list[float] = []
        for index, (reserved_at, released_at) in enumerate(ordered[:-1]):
            next_reserved_at = ordered[index + 1][0]
            if next_reserved_at >= released_at:
                cycles.append((next_reserved_at - reserved_at) * 1000.0)
        if cycles:
            cycles_by_executor[executor] = cycles
    return cycles_by_executor


def _print_slot_idle_report(rows: list[dict[str, Any]]) -> None:
    idle_by_executor = _executor_slot_idle_ms(rows)
    slot_hold = _slot_hold_ms(rows)
    if not idle_by_executor and not slot_hold:
        return

    print("\nExecutor Slot Idle")
    if slot_hold:
        print(
            "slot hold: "
            f"count={len(slot_hold)} "
            f"avg={_fmt(_avg(slot_hold))} "
            f"p50={_fmt(_percentile(slot_hold, 0.50))} "
            f"p95={_fmt(_percentile(slot_hold, 0.95))} "
            f"max={_fmt(max(slot_hold))}"
        )

    for executor in sorted(idle_by_executor):
        values = idle_by_executor[executor]
        print(
            f"idle after release {executor}: "
            f"count={len(values)} "
            f"avg={_fmt(_avg(values))} "
            f"p50={_fmt(_percentile(values, 0.50))} "
            f"p95={_fmt(_percentile(values, 0.95))} "
            f"max={_fmt(max(values))}"
        )


def _print_dispatch_efficiency_report(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    rate_stats: dict[str, tuple[int, float | None, float | None]],
) -> None:
    dispatch_count = rate_stats["gateway_dispatch_start"][0]
    success_count = rate_stats["executor_success_recorded"][0]
    if not dispatch_count and not success_count:
        return

    submit_rate = rate_stats["gateway_submit_received"][1]
    dispatch_rate = rate_stats["gateway_dispatch_start"][1]
    success_rate = rate_stats["executor_success_recorded"][1]
    scheduler_path = _summed_latency_values(summaries, SCHEDULER_DISPATCH_PATH)
    handoff_path = _summed_latency_values(summaries, EXECUTOR_HANDOFF_PATH)
    dispatch_confirm = _numeric_values(summaries, "dispatch_to_confirm")
    callback_release = _numeric_values(summaries, "callback_to_slot_release")
    callback_terminal = _numeric_values(summaries, "callback_to_terminal_status")
    service = _numeric_values(summaries, "executor_service")
    slot_hold = _slot_hold_ms(rows)
    idle_by_executor = _executor_slot_idle_ms(rows)
    slot_cycles_by_executor = _slot_cycle_ms(rows)
    candidate_appearances, first_candidate_wait = _candidate_selection_attempts(rows)
    max_observed_slots = _max_observed_free_slots(rows)

    print("\nDispatch Efficiency")
    print(
        "rates: "
        f"submit={_fmt_rate(submit_rate)} "
        f"dispatch={_fmt_rate(dispatch_rate)} "
        f"complete={_fmt_rate(success_rate)}"
    )
    if submit_rate is not None and dispatch_rate is not None:
        drift = submit_rate - dispatch_rate
        print("submit-dispatch drift: " f"{drift:.3f}/s ({drift * 3600.0:.0f}/hour)")

    _print_distribution(
        "frontier backlog wait",
        _numeric_values(summaries, "frontier_to_candidate"),
    )
    _print_distribution("scheduler dispatch path", scheduler_path)
    _print_distribution("executor handoff path", handoff_path)
    _print_distribution("dispatch confirmation", dispatch_confirm)
    _print_distribution("callback to slot release", callback_release)
    _print_distribution("callback to terminal", callback_terminal)
    _print_distribution("first candidate to selected", first_candidate_wait)
    _print_count_distribution(
        "planner limited batch size",
        _numeric_event_field(rows, "planner_selected", "limited"),
    )
    _print_count_distribution(
        "dispatch batch size",
        _numeric_event_field(rows, "dispatch_batch_start", "count"),
    )
    _print_count_distribution(
        "semaphore reserve requested",
        _numeric_event_field(rows, "semaphore_reserve_batch_done", "requested"),
    )
    _print_count_distribution(
        "semaphore reserve granted",
        _numeric_event_field(rows, "semaphore_reserve_batch_done", "reserved"),
    )
    _print_distribution(
        "semaphore reserve batch latency",
        _numeric_event_field(rows, "semaphore_reserve_batch_done", "elapsed_ms"),
    )
    _print_distribution(
        "dispatch batch interval",
        _event_intervals_ms(rows, "dispatch_batch_start"),
    )
    if candidate_appearances:
        print(
            "candidate snapshots before selection: "
            f"count={len(candidate_appearances)} "
            f"avg={_avg(candidate_appearances):.1f} "
            f"p50={_percentile(candidate_appearances, 0.50):.0f} "
            f"p95={_percentile(candidate_appearances, 0.95):.0f} "
            f"max={max(candidate_appearances):.0f}"
        )

    slot_hold_p50 = _percentile(slot_hold, 0.50)
    slot_hold_capacity = _capacity_per_second(slot_hold_p50)
    if slot_hold_capacity is not None:
        print(
            "slot-hold capacity estimate: "
            f"p50={_fmt(slot_hold_p50)} => {_fmt_rate(slot_hold_capacity)} "
            "per occupied slot"
        )
        if dispatch_rate is not None and max_observed_slots and max_observed_slots <= 1:
            print(
                "actual dispatch vs slot-hold estimate: "
                f"{(dispatch_rate / slot_hold_capacity) * 100.0:.1f}%"
            )
        elif max_observed_slots and max_observed_slots > 1:
            aggregate_capacity = slot_hold_capacity * max_observed_slots
            print(
                "aggregate slot-hold capacity estimate: "
                f"max_observed_slots={max_observed_slots} "
                f"=> {_fmt_rate(aggregate_capacity)}"
            )
            if dispatch_rate is not None:
                print(
                    "actual dispatch vs aggregate slot-hold estimate: "
                    f"{(dispatch_rate / aggregate_capacity) * 100.0:.1f}%"
                )

    service_p50 = _percentile(service, 0.50)
    service_capacity = _capacity_per_second(service_p50)
    if service_capacity is not None:
        print(
            "service-only capacity estimate: "
            f"p50={_fmt(service_p50)} => {_fmt_rate(service_capacity)} "
            "per executor"
        )

    for executor in sorted(idle_by_executor):
        values = idle_by_executor[executor]
        print(
            f"slot refill idle {executor}: "
            f"p50={_fmt(_percentile(values, 0.50))} "
            f"p95={_fmt(_percentile(values, 0.95))}"
        )
    if max_observed_slots is None or max_observed_slots <= 1:
        for executor in sorted(slot_cycles_by_executor):
            values = slot_cycles_by_executor[executor]
            p50 = _percentile(values, 0.50)
            capacity = _capacity_per_second(p50)
            print(
                f"slot cycle {executor}: "
                f"p50={_fmt(p50)} "
                f"p95={_fmt(_percentile(values, 0.95))} "
                f"rate@p50={_fmt_rate(capacity)}"
            )
    elif slot_cycles_by_executor:
        print(
            "slot cycle estimate: skipped for multi-slot executor "
            f"(max_observed_slots={max_observed_slots})"
        )

    stage_rows: list[tuple[float, str, float | None]] = []
    for label, key in DISPATCH_BOTTLENECK_STAGES:
        values = _numeric_values(summaries, key)
        p50 = _percentile(values, 0.50)
        if p50 is not None:
            stage_rows.append((p50, label, _percentile(values, 0.95)))
    if stage_rows:
        print("largest non-backlog p50 stages:")
        for p50, label, p95 in sorted(stage_rows, reverse=True)[:5]:
            print(f"  {label}: p50={_fmt(p50)} p95={_fmt(p95)}")


def _print_latency_report(summaries: list[dict[str, Any]]) -> None:
    print("\nLatency Percentiles")
    print("stage count avg p50 p90 p95 p99 max")
    for label, key in REPORT_LATENCIES:
        values = _numeric_values(summaries, key, unique_dag=key in DAG_LATENCIES)
        if not values:
            continue
        print(
            f"{label} "
            f"{len(values)} "
            f"{_fmt(_avg(values))} "
            f"{_fmt(_percentile(values, 0.50))} "
            f"{_fmt(_percentile(values, 0.90))} "
            f"{_fmt(_percentile(values, 0.95))} "
            f"{_fmt(_percentile(values, 0.99))} "
            f"{_fmt(max(values))}"
        )


def _print_pressure_report(rows: list[dict[str, Any]]) -> None:
    candidates = _candidate_counts(rows)
    if candidates:
        print("\nPlanner Pressure")
        print(
            "candidate snapshots: "
            f"count={len(candidates)} "
            f"avg={_avg(candidates):.1f} "
            f"p50={_percentile(candidates, 0.50):.0f} "
            f"p95={_percentile(candidates, 0.95):.0f} "
            f"p99={_percentile(candidates, 0.99):.0f} "
            f"max={max(candidates):.0f}"
        )

    slots_by_executor = _slot_counts(rows)
    if not slots_by_executor:
        return

    print("slot snapshots:")
    for executor in sorted(slots_by_executor):
        dist = slots_by_executor[executor]
        pieces = [f"{slot}={dist[slot]}" for slot in sorted(dist)]
        print(f"  {executor}: " + ", ".join(pieces))


def _print_admission_report(rows: list[dict[str, Any]]) -> None:
    decisions = [row for row in rows if row.get("event") == "dag_admission_decision"]
    if not decisions:
        return

    print("\nDAG Admission")
    print(f"decisions: {len(decisions)}")
    for field in ("mode", "source", "policy_reason", "legacy_reason"):
        counts = Counter(str(row.get(field, "-")) for row in decisions)
        pieces = [f"{key}={counts[key]}" for key in sorted(counts)]
        print(f"{field}: " + ", ".join(pieces))

    policy_denied = sum(1 for row in decisions if row.get("policy_decision") is False)
    legacy_denied = sum(1 for row in decisions if row.get("legacy_decision") is False)
    effective_denied = sum(
        1 for row in decisions if row.get("effective_decision") is False
    )
    print(
        "denials: "
        f"policy={policy_denied} "
        f"legacy={legacy_denied} "
        f"effective={effective_denied}"
    )

    pressure_by_executor: dict[str, list[float]] = defaultdict(list)
    for row in decisions:
        pressure = row.get("executor_pressure")
        if not isinstance(pressure, dict):
            continue
        for executor, payload in pressure.items():
            if not isinstance(payload, dict):
                continue
            value = payload.get("pressure")
            if isinstance(value, (int, float)):
                pressure_by_executor[str(executor)].append(float(value))

    if pressure_by_executor:
        print("pressure:")
        for executor in sorted(pressure_by_executor):
            values = pressure_by_executor[executor]
            print(
                f"  {executor}: "
                f"count={len(values)} "
                f"avg={_avg(values):.2f} "
                f"p95={_percentile(values, 0.95):.2f} "
                f"max={max(values):.2f}"
            )


def _numeric_event_field(
    rows: list[dict[str, Any]], event_name: str, field: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("event") != event_name:
            continue
        value = row.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _latest_priority_refresh_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for row in rows:
        if row.get("event") not in PRIORITY_REFRESH_EVENTS:
            continue
        if not isinstance(row.get("ts_unix"), (int, float)):
            continue
        if latest is None or float(row["ts_unix"]) > float(latest["ts_unix"]):
            latest = row
    return latest


def _priority_refresh_attempts(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("event") not in PRIORITY_REFRESH_EVENTS:
            continue
        refresh_id = row.get("refresh_id")
        if refresh_id is None:
            continue
        attempts[str(refresh_id)].append(row)
    return attempts


def _sort_refresh_id(refresh_id: str) -> tuple[int, str]:
    try:
        return int(refresh_id), refresh_id
    except ValueError:
        return 0, refresh_id


def _refresh_attempt_status(rows: list[dict[str, Any]]) -> str:
    events = Counter(row.get("event") for row in rows)
    if events.get("scheduler_priority_refresh_returned", 0):
        return "returned"
    if events.get("scheduler_priority_refresh_failed", 0):
        return "failed_no_return"
    if events.get("scheduler_priority_refresh_done", 0):
        return "done_no_return"
    if events.get("scheduler_priority_refresh_start", 0):
        return "in_progress"
    return "due_only"


def _print_priority_refresh_report(rows: list[dict[str, Any]]) -> None:
    events = _event_counts(rows)
    refresh_requested = events.get("scheduler_priority_refresh_requested", 0)
    if not refresh_requested and not any(
        events.get(event, 0) for event in PRIORITY_REFRESH_EVENTS
    ):
        return

    print("\nScheduler Priority Refresh")
    if refresh_requested:
        requested_rows = [
            row
            for row in rows
            if row.get("event") == "scheduler_priority_refresh_requested"
        ]
        wake_requests = sum(1 for row in requested_rows if row.get("wake_scheduler"))
        deferred_requests = len(requested_rows) - wake_requests
        print(
            f"requests: scheduler_priority_refresh_requested {refresh_requested} "
            f"wake={wake_requests} deferred={deferred_requests}"
        )
    print("event count")
    for event in PRIORITY_REFRESH_EVENTS:
        count = events.get(event, 0)
        if count:
            print(f"{event} {count}")

    due = events.get("scheduler_priority_refresh_due", 0)
    returned = events.get("scheduler_priority_refresh_returned", 0)
    started = events.get("scheduler_priority_refresh_start", 0)
    completed = events.get("scheduler_priority_refresh_done", 0)
    failed = events.get("scheduler_priority_refresh_failed", 0)
    if due != returned or started != completed + failed:
        print(
            "incomplete: "
            f"due_without_return={max(0, due - returned)} "
            f"started_without_terminal={max(0, started - completed - failed)}"
        )

    print("phase count avg p50 p95 max")
    for label, event_name in PRIORITY_REFRESH_PHASES:
        values = _numeric_event_field(rows, event_name, "elapsed_ms")
        if not values:
            continue
        print(
            f"{label} "
            f"{len(values)} "
            f"{_fmt(_avg(values))} "
            f"{_fmt(_percentile(values, 0.50))} "
            f"{_fmt(_percentile(values, 0.95))} "
            f"{_fmt(max(values))}"
        )

    latest = _latest_priority_refresh_row(rows)
    if latest is None:
        return
    details = [
        f"event={latest.get('event')}",
        f"ts={latest.get('ts', '-')}",
    ]
    for field in (
        "source",
        "refresh_id",
        "submission_count",
        "request_queue_size",
        "pending_requests",
        "elapsed_ms",
        "error",
    ):
        value = latest.get(field)
        if value is not None:
            details.append(f"{field}={_fmt(value) if field == 'elapsed_ms' else value}")
    print("latest: " + " ".join(details))

    attempts = _priority_refresh_attempts(rows)
    if not attempts:
        return

    incomplete_attempts = [
        (refresh_id, attempt_rows)
        for refresh_id, attempt_rows in attempts.items()
        if _refresh_attempt_status(attempt_rows) != "returned"
    ]
    if not incomplete_attempts:
        return

    print("incomplete attempts:")
    print("refresh_id source status latest")
    for refresh_id, attempt_rows in sorted(
        incomplete_attempts, key=lambda item: _sort_refresh_id(item[0])
    )[:10]:
        latest_attempt = _latest_priority_refresh_row(attempt_rows) or {}
        source = next(
            (
                row.get("source")
                for row in attempt_rows
                if row.get("source") is not None
            ),
            "-",
        )
        print(
            f"{refresh_id} "
            f"{source} "
            f"{_refresh_attempt_status(attempt_rows)} "
            f"{latest_attempt.get('event', '-')}"
        )


def _print_dag_sync_report(rows: list[dict[str, Any]]) -> None:
    events = _event_counts(rows)
    if not any(events.get(event, 0) for event in DAG_SYNC_EVENTS):
        return

    print("\nDAG Sync")
    print("event count")
    for event in DAG_SYNC_EVENTS:
        count = events.get(event, 0)
        if count:
            print(f"{event} {count}")

    completed = [
        row for row in rows if row.get("event") == "scheduler_dag_sync_cycle_done"
    ]
    if completed:
        invalid = sum(int(row.get("invalid_dags") or 0) for row in completed)
        terminal = sum(int(row.get("terminal_dags") or 0) for row in completed)
        print(
            f"summary cycles={len(completed)} "
            f"invalid_dags={invalid} "
            f"terminal_dags={terminal}"
        )


def _print_postgres_pool_report(rows: list[dict[str, Any]]) -> None:
    events = _event_counts(rows)
    if not any(events.get(event, 0) for event in POSTGRES_POOL_EVENTS):
        return

    print("\nPostgres Pool")
    print("event count")
    for event in POSTGRES_POOL_EVENTS:
        count = events.get(event, 0)
        if count:
            print(f"{event} {count}")

    waits = _numeric_event_field(rows, "postgres_pool_acquire_wait_done", "elapsed_ms")
    if waits:
        print(
            "acquire-wait "
            f"{len(waits)} "
            f"{_fmt(_avg(waits))} "
            f"{_fmt(_percentile(waits, 0.50))} "
            f"{_fmt(_percentile(waits, 0.95))} "
            f"{_fmt(max(waits))}"
        )


def _print_findings(
    rate_stats: dict[str, tuple[int, float | None, float | None]],
    summaries: list[dict[str, Any]],
    events: Counter[str],
    rows: list[dict[str, Any]],
) -> None:
    findings: list[str] = []

    submit_rate = rate_stats["gateway_submit_received"][1]
    success_rate = rate_stats["executor_success_recorded"][1]
    if submit_rate is not None and success_rate is not None:
        drift = submit_rate - success_rate
        if drift > 0.01:
            findings.append(
                f"Backlog is growing by about {drift:.3f} jobs/s "
                f"({drift * 3600:.0f} jobs/hour)."
            )
        elif drift < -0.01:
            findings.append(
                f"Completion is draining faster than submissions by about "
                f"{abs(drift):.3f} jobs/s."
            )
        else:
            findings.append("Submission and completion rates are roughly balanced.")

    started = events.get("control_flow_started", 0)
    completed = events.get("control_flow_completed", 0)
    if started != completed:
        findings.append(
            f"Control-flow imbalance detected: started={started}, completed={completed}."
        )

    candidate_to_planned = _numeric_values(summaries, "candidate_to_planned")
    dispatch_to_executor = _numeric_values(summaries, "dispatch_to_executor")
    service = _numeric_values(summaries, "executor_service")
    candidate_appearances, _ = _candidate_selection_attempts(rows)
    if candidate_to_planned and (_percentile(candidate_to_planned, 0.95) or 0) > 1000:
        findings.append(
            "Selection wait is the dominant scheduler-side tail "
            f"(candidate->planned p95={_fmt(_percentile(candidate_to_planned, 0.95))})."
        )
    if candidate_appearances and (_percentile(candidate_appearances, 0.50) or 0) > 1:
        findings.append(
            "candidate->planned includes repeated candidate snapshots before selection "
            f"(candidate appearances p50={_percentile(candidate_appearances, 0.50):.0f})."
        )
    if dispatch_to_executor and (_percentile(dispatch_to_executor, 0.95) or 0) > 500:
        findings.append(
            "Gateway-to-executor handoff is high "
            f"(dispatch->executor p95={_fmt(_percentile(dispatch_to_executor, 0.95))})."
        )
    dispatch_rate = rate_stats["gateway_dispatch_start"][1]
    slot_hold = _slot_hold_ms(rows)
    slot_hold_capacity = _capacity_per_second(_percentile(slot_hold, 0.50))
    max_observed_slots = _max_observed_free_slots(rows)
    aggregate_slot_hold_capacity = (
        slot_hold_capacity * max_observed_slots
        if slot_hold_capacity is not None and max_observed_slots
        else slot_hold_capacity
    )
    if (
        dispatch_rate is not None
        and aggregate_slot_hold_capacity is not None
        and dispatch_rate < aggregate_slot_hold_capacity * 0.75
    ):
        findings.append(
            "Dispatch rate is below the slot-hold capacity estimate "
            f"({_fmt_rate(dispatch_rate)} actual vs "
            f"{_fmt_rate(aggregate_slot_hold_capacity)} estimated aggregate)."
        )
    if service and (_percentile(service, 0.95) or 0.0) < 5.0:
        stage_rows: list[tuple[float, str, float | None]] = []
        for label, key in DISPATCH_BOTTLENECK_STAGES:
            values = _numeric_values(summaries, key)
            p50 = _percentile(values, 0.50)
            if p50 is not None:
                stage_rows.append((p50, label, _percentile(values, 0.95)))
        if stage_rows:
            p50, label, p95 = max(stage_rows)
            findings.append(
                "With near-zero executor service time, the largest non-backlog "
                f"dispatch interval is {label} "
                f"(p50={_fmt(p50)}, p95={_fmt(p95)})."
            )
    idle_by_executor = _executor_slot_idle_ms(rows)
    for executor, values in sorted(idle_by_executor.items()):
        p95 = _percentile(values, 0.95)
        if p95 is not None and p95 > 50.0:
            findings.append(
                f"Executor slots are idle after release for {executor} "
                f"(p95={_fmt(p95)})."
            )
    if service:
        findings.append(
            f"Executor service p95 is {_fmt(_percentile(service, 0.95))}; "
            "compare this with SLA budget before blaming dispatch overhead."
        )

    refresh_sources = sorted(
        {
            str(row.get("source"))
            for row in rows
            if row.get("event") in PRIORITY_REFRESH_EVENTS
            and row.get("source") is not None
        }
    )
    if refresh_sources:
        for source in refresh_sources:
            source_events = Counter(
                row.get("event")
                for row in rows
                if row.get("source") == source
                and row.get("event") in PRIORITY_REFRESH_EVENTS
            )
            source_due = source_events.get("scheduler_priority_refresh_due", 0)
            source_returned = source_events.get(
                "scheduler_priority_refresh_returned", 0
            )
            source_started = source_events.get("scheduler_priority_refresh_start", 0)
            source_terminal = source_events.get(
                "scheduler_priority_refresh_done", 0
            ) + source_events.get("scheduler_priority_refresh_failed", 0)
            if source_due > source_returned:
                findings.append(
                    f"Priority refresh source={source} did not return "
                    f"(due={source_due}, returned={source_returned})."
                )
            elif source_started > source_terminal:
                findings.append(
                    f"Priority refresh source={source} started without a terminal "
                    f"trace event (started={source_started}, terminal={source_terminal})."
                )
    else:
        refresh_due = events.get("scheduler_priority_refresh_due", 0)
        refresh_returned = events.get("scheduler_priority_refresh_returned", 0)
        refresh_started = events.get("scheduler_priority_refresh_start", 0)
        refresh_terminal = events.get(
            "scheduler_priority_refresh_done", 0
        ) + events.get("scheduler_priority_refresh_failed", 0)
        if refresh_due > refresh_returned:
            findings.append(
                "Priority refresh appears to block the submission worker "
                f"(due={refresh_due}, returned={refresh_returned})."
            )
        elif refresh_started > refresh_terminal:
            findings.append(
                "Priority refresh started without a terminal trace event "
                f"(started={refresh_started}, terminal={refresh_terminal})."
            )

    pool_timeouts = events.get("postgres_pool_acquire_timeout", 0)
    if pool_timeouts:
        findings.append(
            f"Postgres connection pool acquisition timed out {pool_timeouts} time(s)."
        )

    if not findings:
        return

    print("\nFindings")
    for finding in findings:
        print(f"- {finding}")


def print_report(
    trace_path: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    events = _event_counts(rows)
    rates = _rate_stats(rows)
    executors = _executor_counts(rows)
    service = _numeric_values(summaries, "executor_service")
    success_rate = rates["executor_success_recorded"][1]
    utilization = None
    if executors and success_rate is not None and service:
        utilization = success_rate * ((_avg(service) or 0.0) / 1000.0) / len(executors)

    first_ts = min(
        (
            float(row["ts_unix"])
            for row in rows
            if isinstance(row.get("ts_unix"), (int, float))
        ),
        default=None,
    )
    last_ts = max(
        (
            float(row["ts_unix"])
            for row in rows
            if isinstance(row.get("ts_unix"), (int, float))
        ),
        default=None,
    )
    window = (
        (last_ts - first_ts) if first_ts is not None and last_ts is not None else None
    )

    print("Scheduler Trace Report")
    print(f"trace: {trace_path}")
    print(f"events: {len(rows)}")
    print(f"trace groups: {len(summaries)}")
    print(f"dispatches: {rates['gateway_dispatch_start'][0]}")
    print(f"window: {_fmt_s(window)}")

    print("\nEvent Rates")
    print("event count rate span")
    for event in RATE_EVENTS:
        count, rate, span = rates[event]
        print(f"{event} {count} {_fmt_rate(rate)} {_fmt_s(span)}")

    print("\nExecution Capacity")
    if executors:
        parts = [f"{pid}={count}" for pid, count in sorted(executors.items())]
        print(f"executors: {len(executors)} (" + ", ".join(parts) + ")")
    else:
        print("executors: -")
    if utilization is not None:
        print(f"estimated utilization: {utilization * 100.0:.1f}%")
    print(
        "control flow: "
        f"started={events.get('control_flow_started', 0)} "
        f"completed={events.get('control_flow_completed', 0)}"
    )

    _print_dispatch_efficiency_report(rows, summaries, rates)
    _print_pressure_report(rows)
    _print_slot_idle_report(rows)
    _print_admission_report(rows)
    _print_priority_refresh_report(rows)
    _print_dag_sync_report(rows)
    _print_postgres_pool_report(rows)
    _print_latency_report(summaries)
    _print_findings(rates, summaries, events, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print aggregate rates, capacity, pressure, and latency percentiles.",
    )
    parser.add_argument(
        "--sort",
        choices=(
            "gateway_to_dispatch",
            "accepted_to_dispatch",
            "notified_to_dispatch",
            "submit_queue_wait",
            "submit_worker_to_persist_start",
            "dag_persist",
            "persisted_to_frontier",
            "frontier_to_dispatch",
            "frontier_to_candidate",
            "candidate_to_planned",
            "planned_to_taken",
            "taken_to_db_lease",
            "db_lease_to_slot",
            "slot_to_active",
            "dispatch_to_executor",
            "dag_scheduled_to_dispatch",
            "dag_persisted_to_dispatch",
            "executor_service",
            "callback_to_terminal_status",
            "events",
        ),
        default="dispatch_to_executor",
    )
    args = parser.parse_args()
    sort_key = SORT_ALIASES.get(args.sort, args.sort)

    grouped, rows = load_trace(args.trace_path)
    dag_events_by_id = {
        key: _event_times(rows)
        for key, rows in grouped.items()
        if any(row.get("event", "").startswith("gateway_submit_") for row in rows)
    }
    summaries = [
        summarize_job(job_id, rows, dag_events_by_id)
        for job_id, rows in grouped.items()
    ]
    if args.report:
        print_report(args.trace_path, rows, summaries)
        return 0

    summaries.sort(
        key=lambda item: (
            item[sort_key] is None,
            -(item[sort_key] or 0),
        )
    )

    print(
        "job_id dag_id gateway->dispatch accepted->dispatch notified->dispatch "
        "submit-queue dequeue->persist-start dag-persist persisted->frontier "
        "frontier->dispatch frontier->candidate candidate->planned "
        "planned->taken taken->db-lease db-lease->slot slot->active "
        "active->dispatch dispatch->confirm dispatch->executor "
        "receive->running service callback->release callback->terminal "
        "terminal events"
    )
    for item in summaries[: args.limit]:
        print(
            f"{item['job_id']} "
            f"{item['dag_id']} "
            f"{_fmt(item['gateway_to_dispatch'])} "
            f"{_fmt(item['accepted_to_dispatch'])} "
            f"{_fmt(item['notified_to_dispatch'])} "
            f"{_fmt(item['submit_queue_wait'])} "
            f"{_fmt(item['submit_worker_to_persist_start'])} "
            f"{_fmt(item['dag_persist'])} "
            f"{_fmt(item['persisted_to_frontier'])} "
            f"{_fmt(item['frontier_to_dispatch'])} "
            f"{_fmt(item['frontier_to_candidate'])} "
            f"{_fmt(item['candidate_to_planned'])} "
            f"{_fmt(item['planned_to_taken'])} "
            f"{_fmt(item['taken_to_db_lease'])} "
            f"{_fmt(item['db_lease_to_slot'])} "
            f"{_fmt(item['slot_to_active'])} "
            f"{_fmt(item['activate_to_dispatch'])} "
            f"{_fmt(item['dispatch_to_confirm'])} "
            f"{_fmt(item['dispatch_to_executor'])} "
            f"{_fmt(item['executor_start_record'])} "
            f"{_fmt(item['executor_service'])} "
            f"{_fmt(item['callback_to_slot_release'])} "
            f"{_fmt(item['callback_to_terminal_status'])} "
            f"{item['terminal']} "
            f"{item['events']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
