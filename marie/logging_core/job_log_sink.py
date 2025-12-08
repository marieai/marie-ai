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
from typing import Any, Dict, Optional, TextIO


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

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_handles: int = DEFAULT_MAX_HANDLES,
    ):
        super().__init__()
        self._log_dir = log_dir or os.getenv(
            "MARIE_JOB_LOGS_DIR", "/var/log/marie/jobs"
        )
        self._max_handles = max_handles
        self._file_handles: OrderedDict[str, tuple[TextIO, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False

        os.makedirs(self._log_dir, exist_ok=True)
        atexit.register(self._cleanup)

    def get_log_file_path(self, request_id: str) -> str:
        """Get the file path for a job's log file."""
        safe_id = "".join(c for c in request_id if c.isalnum() or c in "-_")
        return os.path.join(self._log_dir, f"job-{safe_id}.log")

    def _get_or_create_handle(self, request_id: str) -> TextIO:
        """Get existing file handle or create new one with LRU eviction."""
        current_time = time.time()

        with self._lock:
            if request_id in self._file_handles:
                handle, _ = self._file_handles.pop(request_id)
                self._file_handles[request_id] = (handle, current_time)
                return handle

            while len(self._file_handles) >= self._max_handles:
                _, (old_handle, _) = self._file_handles.popitem(last=False)
                try:
                    old_handle.close()
                except Exception:
                    pass

            file_path = self.get_log_file_path(request_id)
            handle = open(file_path, "a", encoding="utf-8", buffering=1)
            self._file_handles[request_id] = (handle, current_time)
            return handle

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
            log_line = self._format_json(record)
            handle = self._get_or_create_handle(request_id)

            with self._lock:
                handle.write(log_line + "\n")
                handle.flush()

        except Exception:
            self.handleError(record)

    def close_handle(self, request_id: str) -> None:
        """Close the file handle for a specific job."""
        with self._lock:
            if request_id in self._file_handles:
                handle, _ = self._file_handles.pop(request_id)
                try:
                    handle.close()
                except Exception:
                    pass

    def flush(self) -> None:
        """Flush all open file handles."""
        with self._lock:
            for handle, _ in self._file_handles.values():
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
            for _, (handle, _) in list(self._file_handles.items()):
                try:
                    handle.flush()
                    handle.close()
                except Exception:
                    pass
            self._file_handles.clear()
