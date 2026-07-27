from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from marie.utils.types import to_bool

_DEFAULT_PATH = "/tmp/marie-scheduler-trace.jsonl"
_DEFAULT_PROFILE = "compact"
_TRACE_BATCH_SIZE = 256
_TRACE_FLUSH_SECONDS = 0.05
_TRACE_QUEUE: queue.Queue[tuple[str, str] | None] = queue.Queue()
_TRACE_START_LOCK = threading.Lock()
_TRACE_THREAD: threading.Thread | None = None
_TRACE_PID = os.getpid()
_TRACE_FD: int | None = None
_TRACE_FD_PATH: str | None = None

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
    "job_monitor_woken_by_notification",
    "job_terminal_notification_emit_started",
    "job_terminal_notification_emitted",
    "job_terminal_notification_received",
    "job_terminal_event_published",
    "job_terminal_event_publish_skipped",
    "executor_terminal_status_write_started",
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
    "job_status_event_enqueued",
    "job_status_event_dequeued",
    "job_status_event_dispatch_completed",
    "job_status_event_dropped",
    "scheduler_job_event_received",
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


def _close_trace_fd() -> None:
    global _TRACE_FD, _TRACE_FD_PATH

    trace_fd = _TRACE_FD
    _TRACE_FD = None
    _TRACE_FD_PATH = None
    if trace_fd is not None:
        try:
            os.close(trace_fd)
        except OSError:
            pass


def _write_trace_batch(batch: list[tuple[str, str]]) -> None:
    global _TRACE_FD, _TRACE_FD_PATH

    try:
        index = 0
        while index < len(batch):
            path = batch[index][0]
            lines: list[str] = []
            while index < len(batch) and batch[index][0] == path:
                lines.append(batch[index][1])
                index += 1
            try:
                if _TRACE_FD is None or _TRACE_FD_PATH != path:
                    _close_trace_fd()
                    trace_path = Path(path).expanduser()
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    _TRACE_FD = os.open(
                        trace_path,
                        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                        0o666,
                    )
                    _TRACE_FD_PATH = path
                payload = "".join(lines).encode("utf-8")
                view = memoryview(payload)
                while view:
                    written = os.write(_TRACE_FD, view)
                    view = view[written:]
            except OSError:
                _close_trace_fd()
    finally:
        for _ in batch:
            _TRACE_QUEUE.task_done()


def _trace_writer() -> None:
    stopping = False
    while not stopping:
        item = _TRACE_QUEUE.get()
        if item is None:
            _TRACE_QUEUE.task_done()
            break

        batch = [item]
        deadline = time.monotonic() + _TRACE_FLUSH_SECONDS
        while len(batch) < _TRACE_BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = _TRACE_QUEUE.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                _TRACE_QUEUE.task_done()
                stopping = True
                break
            batch.append(item)
        _write_trace_batch(batch)

    _close_trace_fd()


def _reset_trace_writer() -> None:
    global _TRACE_PID, _TRACE_QUEUE, _TRACE_START_LOCK, _TRACE_THREAD

    _close_trace_fd()
    _TRACE_PID = os.getpid()
    _TRACE_QUEUE = queue.Queue()
    _TRACE_START_LOCK = threading.Lock()
    _TRACE_THREAD = None


def _ensure_trace_writer() -> None:
    global _TRACE_THREAD

    if _TRACE_PID != os.getpid():
        _reset_trace_writer()
    if _TRACE_THREAD is not None and _TRACE_THREAD.is_alive():
        return
    with _TRACE_START_LOCK:
        if _TRACE_THREAD is not None and _TRACE_THREAD.is_alive():
            return
        _TRACE_THREAD = threading.Thread(
            target=_trace_writer,
            name="scheduler-trace-writer",
            daemon=True,
        )
        _TRACE_THREAD.start()


def flush_scheduler_trace(*, close: bool = False) -> None:
    """Flush queued scheduler trace records to disk."""
    if _TRACE_PID != os.getpid():
        _reset_trace_writer()
    if _TRACE_THREAD is not None:
        _TRACE_QUEUE.join()
    if close:
        _close_trace_fd()


def _shutdown_trace_writer() -> None:
    if _TRACE_PID != os.getpid():
        return
    thread = _TRACE_THREAD
    if thread is None or not thread.is_alive():
        return
    _TRACE_QUEUE.join()
    _TRACE_QUEUE.put(None)
    thread.join(timeout=1.0)


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
    _ensure_trace_writer()
    _TRACE_QUEUE.put((path, line))


atexit.register(_shutdown_trace_writer)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_trace_writer)
