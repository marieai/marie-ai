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

_COMPACT_EVENTS = {
    "gateway_submit_received",
    "gateway_dispatch_start",
    "gateway_dispatch_confirmed",
    "executor_success_recorded",
    "executor_failed_recorded",
    "candidate_built",
    "planner_selected",
    "dispatch_batch_start",
    "dispatch_batch_complete",
    "semaphore_reserve_batch_done",
    "slot_unavailable",
    "slot_reserve_failed",
    "job_db_activate_failed",
    "postgres_pool_acquire_wait_done",
    "postgres_pool_acquire_timeout",
    "scheduler_dag_sync_cycle_done",
    "scheduler_dag_sync_cycle_failed",
    "scheduler_dag_sync_cycle_skipped",
    "scheduler_priority_refresh_requested",
    "scheduler_priority_refresh_due",
    "scheduler_priority_refresh_done",
    "scheduler_priority_refresh_failed",
    "scheduler_priority_refresh_returned",
}

_COMPACT_DROP_FIELDS = {
    "api_key",
    "event_name",
    "planner",
    "project_id",
    "ref_id",
    "ref_type",
}

_COMPACT_EVENT_DROP_FIELDS = {
    "dispatch_batch_start": {"job_ids"},
    "semaphore_reserve_batch_done": {"job_ids"},
}


def _profile() -> str:
    return os.getenv("MARIE_SCHEDULER_TRACE_PROFILE", _DEFAULT_PROFILE).strip().lower()


def _compact_fields(event: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    if event not in _COMPACT_EVENTS:
        return None
    dropped_fields = _COMPACT_DROP_FIELDS | _COMPACT_EVENT_DROP_FIELDS.get(event, set())
    return {key: value for key, value in fields.items() if key not in dropped_fields}


def scheduler_trace(event: str, **fields: Any) -> None:
    if not to_bool(os.getenv("MARIE_SCHEDULER_TRACE_ENABLED"), default=False):
        return

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
            trace_path = Path(path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as fp:
                fp.write(line)
    except OSError:
        # Debug tracing must never affect scheduler or executor progress.
        return
