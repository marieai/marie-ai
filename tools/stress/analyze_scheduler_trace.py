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
    "candidate_built",
    "planner_selected",
    "frontier_taken",
    "db_leased",
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

DAG_LATENCIES = {
    "submit_queue_wait",
    "submit_worker_to_persist_start",
    "dag_persist",
    "persisted_to_frontier",
}


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


def _event_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(row["event"] for row in rows if isinstance(row.get("event"), str))


def _executor_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("event") != "executor_request_received":
            continue
        pid = row.get("pid")
        if pid is None:
            continue
        counts[str(pid)] += 1
    return counts


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


def _print_findings(
    rate_stats: dict[str, tuple[int, float | None, float | None]],
    summaries: list[dict[str, Any]],
    events: Counter[str],
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
    if candidate_to_planned and (_percentile(candidate_to_planned, 0.95) or 0) > 1000:
        findings.append(
            "Selection wait is the dominant scheduler-side tail "
            f"(candidate->planned p95={_fmt(_percentile(candidate_to_planned, 0.95))})."
        )
    if dispatch_to_executor and (_percentile(dispatch_to_executor, 0.95) or 0) > 500:
        findings.append(
            "Gateway-to-executor handoff is high "
            f"(dispatch->executor p95={_fmt(_percentile(dispatch_to_executor, 0.95))})."
        )
    if service:
        findings.append(
            f"Executor service p95 is {_fmt(_percentile(service, 0.95))}; "
            "compare this with SLA budget before blaming dispatch overhead."
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

    _print_pressure_report(rows)
    _print_latency_report(summaries)
    _print_findings(rates, summaries, events)


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
