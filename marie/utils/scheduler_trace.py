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


def scheduler_trace(event: str, **fields: Any) -> None:
    if not to_bool(os.getenv("MARIE_SCHEDULER_TRACE_ENABLED"), default=False):
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
