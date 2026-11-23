from __future__ import annotations

import calendar
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from marie.serve.discovery.etcd_client import EtcdClient


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_iso(ts: Optional[str]) -> float:
    if not ts:
        return 0.0
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return calendar.timegm(dt.timetuple())
    except Exception:
        return 0.0


def is_stale(
    ts_iso: Optional[str], timeout_s: float, now_s: Optional[float] = None
) -> bool:
    if timeout_s <= 0:
        return False
    now_s = time.time() if now_s is None else now_s
    return (now_s - _parse_iso(ts_iso)) > timeout_s


class BaseStore:
    def __init__(self, etcd_client: EtcdClient):
        self.etcd = etcd_client

    def _get_raw(self, key: str) -> Optional[bytes]:
        # Force linearizable read (NOT serializable) so we see our own writes
        val, _meta = self.etcd.get(key, metadata=True, serializable=False)
        if val is None:
            return None
        return val if isinstance(val, (bytes, bytearray)) else str(val).encode()

    def _put_json(self, key: str, obj: Dict[str, Any]) -> None:
        self.etcd.put(key, json.dumps(obj))
