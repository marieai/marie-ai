"""
Job log sink handler for the GLOBAL_LOG_BUS.

This handler writes logs to per-job files based on the request_id
context that is already added by MDCContextFilter.

Usage:
    The handler is registered as a sink with GLOBAL_LOG_BUS.
    When request_id is present on a log record, it writes to:
    {log_dir}/job-{request_id}.log

Log files use JSON Lines format for easy parsing and streaming.
"""

import atexit
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, TextIO


class JobLogSink(logging.Handler):
    """
    A logging handler that writes logs to job-specific files.

    This handler is designed to be used as a sink with GLOBAL_LOG_BUS.
    It checks each log record for 'request_id' (added by MDCContextFilter)
    and writes to a per-job log file if present.

    Features:
    - One log file per request_id (job)
    - JSON Lines format for easy parsing
    - Thread-safe file handle management
    - LRU eviction of file handles to prevent exhaustion
    - Automatic directory creation

    Args:
        log_dir: Directory where job log files are stored.
                 Defaults to MARIE_JOB_LOGS_DIR env var or /var/log/marie/jobs
        max_handles: Maximum number of open file handles.
    """

    DEFAULT_MAX_HANDLES = 100
    DEFAULT_IDLE_TIMEOUT = 300  # seconds before an idle handle is reaped

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_handles: int = DEFAULT_MAX_HANDLES,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
    ):
        super().__init__()
        self._log_dir = log_dir or os.getenv(
            "MARIE_JOB_LOGS_DIR", "/var/log/marie/jobs"
        )
        self._max_handles = max_handles
        self._idle_timeout = idle_timeout
        self._file_handles: OrderedDict[str, tuple[TextIO, float, int, int]] = (
            OrderedDict()
        )  # request_id -> (handle, last_access, dev, ino)
        self._lock = threading.RLock()
        self._closed = False

        os.makedirs(self._log_dir, exist_ok=True)
        atexit.register(self._cleanup)

    def get_log_file_path(self, request_id: str) -> str:
        """Get the file path for a job's log file."""
        safe_id = "".join(c for c in request_id if c.isalnum() or c in "-_")
        return os.path.join(self._log_dir, f"job-{safe_id}.log")

    def _open_handle(self, file_path: str) -> tuple[TextIO, int, int]:
        """Open a file handle and return (handle, device, inode)."""
        handle = open(file_path, "a", encoding="utf-8", buffering=1)
        st = os.fstat(handle.fileno())
        return handle, st.st_dev, st.st_ino

    def _is_handle_stale(self, file_path: str, dev: int, ino: int) -> bool:
        """Check if the file on disk still matches the open handle's inode/device.

        Returns True if the file was deleted, rotated, or replaced.
        """
        try:
            st = os.stat(file_path)
            return st.st_dev != dev or st.st_ino != ino
        except FileNotFoundError:
            return True

    def _get_or_create_handle(self, request_id: str) -> TextIO:
        """Get existing file handle or create new one with LRU eviction."""
        with self._lock:
            return self._get_or_create_handle_locked(request_id)

    def _get_or_create_handle_locked(self, request_id: str) -> TextIO:
        current_time = time.time()
        if request_id in self._file_handles:
            handle, _, dev, ino = self._file_handles.pop(request_id)
            file_path = self.get_log_file_path(request_id)
            if self._is_handle_stale(file_path, dev, ino):
                try:
                    handle.close()
                except Exception:
                    pass
                handle, dev, ino = self._open_handle(file_path)
            self._file_handles[request_id] = (handle, current_time, dev, ino)
            return handle

        # Reap idle handles before checking capacity
        self._reap_idle_locked(current_time)

        while len(self._file_handles) >= self._max_handles:
            _, (old_handle, _, _, _) = self._file_handles.popitem(last=False)
            try:
                old_handle.close()
            except Exception:
                pass

        file_path = self.get_log_file_path(request_id)
        handle, dev, ino = self._open_handle(file_path)
        self._file_handles[request_id] = (handle, current_time, dev, ino)
        return handle

    def _reap_idle_locked(self, now: float) -> None:
        """Close handles that have been idle longer than ``_idle_timeout``.

        Must be called while ``_lock`` is held.  Iterates the OrderedDict
        from oldest to newest and stops at the first non-expired entry
        (entries are ordered by last-access time).
        """
        stale = []
        for rid, (_, last_access, _, _) in self._file_handles.items():
            if now - last_access > self._idle_timeout:
                stale.append(rid)
            else:
                break  # remaining entries are newer
        for rid in stale:
            handle, _, _, _ = self._file_handles.pop(rid)
            try:
                handle.close()
            except Exception:
                pass

    def _format_json(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON line."""
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        log_entry: Dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        request_id = getattr(record, "request_id", None)
        if request_id and request_id not in ("-", ""):
            log_entry["request_id"] = request_id

        context = getattr(record, "context", None)
        if context:
            log_entry["ctx"] = context

        if record.exc_info:
            import traceback

            log_entry["exc"] = "".join(traceback.format_exception(*record.exc_info))

        return json.dumps(log_entry, default=str, ensure_ascii=False)

    def emit(self, record: logging.LogRecord) -> None:
        """Write log record to the appropriate job log file."""
        if self._closed:
            return

        request_id = getattr(record, "request_id", None)
        if not request_id or request_id in ("-", ""):
            return

        try:
            # Format outside the lock to minimize hold time and avoid
            # deadlock if formatting triggers recursive logging.
            log_line = self._format_json(record)
        except Exception:
            self.handleError(record)
            return

        try:
            with self._lock:
                if self._closed:
                    return
                handle = self._get_or_create_handle_locked(request_id)
                handle.write(log_line + "\n")
        except Exception:
            self.handleError(record)

    def handle_many(self, records: Iterable[logging.LogRecord]) -> None:
        """Write a batch of records grouped by request_id."""
        if self._closed:
            return

        grouped_lines: OrderedDict[str, list[str]] = OrderedDict()
        grouped_records: OrderedDict[str, list[logging.LogRecord]] = OrderedDict()

        for record in records:
            if not self.filter(record):
                continue

            request_id = getattr(record, "request_id", None)
            if not request_id or request_id in ("-", ""):
                continue

            try:
                log_line = self._format_json(record)
            except Exception:
                self.handleError(record)
                continue

            grouped_lines.setdefault(request_id, []).append(log_line)
            grouped_records.setdefault(request_id, []).append(record)

        if not grouped_lines:
            return

        with self._lock:
            if self._closed:
                return

            for request_id, lines in grouped_lines.items():
                try:
                    handle = self._get_or_create_handle_locked(request_id)
                    handle.write("\n".join(lines) + "\n")
                except Exception:
                    for record in grouped_records[request_id]:
                        self.handleError(record)

    def close_handle(self, request_id: str) -> None:
        """Close the file handle for a specific job."""
        with self._lock:
            if request_id in self._file_handles:
                handle, _, _, _ = self._file_handles.pop(request_id)
                try:
                    handle.close()
                except Exception:
                    pass

    def flush(self) -> None:
        """Flush all open file handles."""
        with self._lock:
            for handle, _, _, _ in self._file_handles.values():
                try:
                    handle.flush()
                except Exception:
                    pass

    def close(self) -> None:
        """Close all file handles."""
        self._cleanup()
        super().close()

    def _cleanup(self) -> None:
        """Close all open file handles."""
        if self._closed:
            return
        self._closed = True

        with self._lock:
            for _, (handle, _, _, _) in list(self._file_handles.items()):
                try:
                    handle.flush()
                    handle.close()
                except Exception:
                    pass
            self._file_handles.clear()
