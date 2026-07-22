from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from marie.utils.types import to_bool

_LOCK = threading.Lock()
_DEFAULT_PATH = "/tmp/marie-scheduler-trace.jsonl"
_DEFAULT_PROFILE = "compact"

_SENSITIVE_FIELDS = {
    "api_key",
    "project_id",
}

_COMPACT_EVENTS = {
    "gateway_submit_received",
    "gateway_dispatch_start",
    "gateway_dispatch_confirmed",
    "job_supervisor_dispatch_admitted",
    "job_supervisor_send_task_completed",
    "job_supervisor_worker_ack_wait_completed",
    "job_monitor_terminal_observed",
    "executor_success_recorded",
    "executor_failed_recorded",
    "candidate_built",
    "planner_selected",
    "dispatch_batch_start",
    "dispatch_batch_complete",
    "job_run_attempt_started",
    "job_attempt_audit_failed",
    "job_terminal_attempt_accepted",
    "job_terminal_attempt_rejected",
    "job_status_event_dropped",
    "scheduler_job_event_enqueued",
    "scheduler_job_event_dequeued",
    "scheduler_job_event_received",
    "scheduler_job_event_processed",
    "scheduler_job_event_failed",
    "terminal_dag_resolution_started",
    "terminal_dag_resolution_completed",
    "terminal_scheduler_wake_completed",
    "semaphore_reserve_batch_done",
    "slot_unavailable",
    "slot_reserve_failed",
    "job_db_activate_failed",
    "hydrated_dag_activation_failed",
    "postgres_pool_acquire_wait_done",
    "postgres_pool_acquire_timeout",
    "postgres_operation",
    "scheduler_dag_sync_cycle_done",
    "scheduler_dag_sync_cycle_failed",
    "scheduler_dag_sync_cycle_skipped",
    "scheduler_priority_refresh_requested",
    "scheduler_priority_refresh_due",
    "scheduler_priority_refresh_completed",
    "scheduler_priority_refresh_failed",
    "run_lease_extend_stale_attempt_total",
    "terminal_event_stale_attempt_total",
    "run_lease_recovered_retry_total",
    "run_lease_recovered_failed_total",
}

_COMPACT_DROP_FIELDS = {
    "event_name",
    "planner",
    "ref_id",
    "ref_type",
}


def _profile() -> str:
    return os.getenv("MARIE_SCHEDULER_TRACE_PROFILE", _DEFAULT_PROFILE).strip().lower()


def _compact_fields(event: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    if event not in _COMPACT_EVENTS:
        return None
    return {
        key: value for key, value in fields.items() if key not in _COMPACT_DROP_FIELDS
    }


def scheduler_trace(event: str, **fields: Any) -> None:
    if not to_bool(os.getenv("MARIE_SCHEDULER_TRACE_ENABLED"), default=False):
        return

    fields = {
        key: value for key, value in fields.items() if key not in _SENSITIVE_FIELDS
    }

    profile = _profile()
    if profile in {"compact", "endurance"}:
        compacted = _compact_fields(event, fields)
        if compacted is None:
            return
        fields = compacted
    elif profile not in {"full", "verbose"}:
        return

    path = os.getenv("MARIE_SCHEDULER_TRACE_PATH", _DEFAULT_PATH)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ts_unix": time.time(),
        "event": event,
        "pid": os.getpid(),
        **fields,
    }
    line = json.dumps(payload, default=str, separators=(",", ":")) + "\n"

    try:
        with _LOCK:
            trace_path = Path(path).expanduser()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as fp:
                fp.write(line)
    except OSError:
        # Debug tracing must never affect scheduler or executor progress.
        return
