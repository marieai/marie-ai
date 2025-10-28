from __future__ import annotations

import calendar
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional, Tuple

from grpc_health.v1.health_pb2 import HealthCheckResponse

from marie.serve.discovery.etcd_client import EtcdClient


def _tx_succeeded(ok) -> bool:
    # etcd3-py often returns (succeeded, responses)
    return ok[0] if isinstance(ok, tuple) else bool(ok)


def _value_to_json_str(v: Any) -> str:
    if v is None:
        return "{}"
    if isinstance(v, (bytes, bytearray)):
        return v.decode()
    if isinstance(v, str):
        return v
    # etcd resolver sometimes surfaces parsed dicts; standardize
    return json.dumps(v)


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


def _is_missing_lease_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    try:
        import grpc

        if isinstance(exc, grpc.RpcError) and exc.code() == grpc.StatusCode.NOT_FOUND:
            return True
    except Exception:
        pass
    return ("lease not found" in msg) or ("requested lease not found" in msg)


class BaseStore:
    def __init__(self, etcd_client: EtcdClient):
        self.etcd = etcd_client
        try:
            from etcd3 import transactions as tx  # noqa

            # self._tx = tx
            # FIXME : We can't use TRANSACTIONS JUST YET
            # OUR ETCD_CLIENT WRAPPER add a namespace and we don't have the TX implmemented in there yet
            # /marie/deployments/192.168.1.21:54481/extract_executor/desired
            # {"phase": "SCHEDULED", "epoch": 1, "params": {}, "updated_at": "2025-10-01T04:46:49Z"}"
            # /marie/deployments/192.168.1.21:60714/extract_executor/desired
            # {"phase": "SCHEDULED", "epoch": 1, "params": {}, "updated_at": "2025-10-01T04:49:44Z"}"
            # /marie/deployments/192.168.1.21:65001/extract_executor/desired
            # {"phase": "SCHEDULED", "epoch": 1, "params": {}, "updated_at": "2025-10-01T04:51:05Z"}
            # /marie/deployments/192.168.1.21:65001/extract_executor/status
            # {"status_code": 1, "status_name": "SERVING", "owner": "extract_executor/rep-0@192.168.1.21:65001", "epoch": 1, "updated_at":
            #  "2025-10-01T04:51:05Z", "heartbeat_at": "2025-10-01T04:51:05Z", "details": null}
            # /marie/gateway/marie/192.168.1.21:65001
            # {"Op": 0, "Addr": "192.168.1.21:65001", "Metadata": "\"{\\\"extract_executor\\\": [\\\"grpc://192.168.1.21:65001\\\"]}\""}
            self._tx = None
        except Exception:
            self._tx = None

    def _get_raw(self, key: str) -> Optional[bytes]:
        # Force linearizable read (NOT serializable) so we see our own writes
        val, _meta = self.etcd.get(key, metadata=True, serializable=False)
        if val is None:
            return None
        return val if isinstance(val, (bytes, bytearray)) else str(val).encode()

    def _put_json(self, key: str, obj: Dict[str, Any]) -> None:
        self.etcd.put(key, json.dumps(obj))

    def _put_json_with_lease(self, key: str, obj: Dict[str, Any], lease_getter) -> None:
        # print('putting json with lease to', key, obj)
        payload = json.dumps(obj)
        lease = lease_getter()  # etcd3.Lease
        try:
            self.etcd.put(key, payload, lease=lease)
        except Exception as e:
            if _is_missing_lease_error(e):
                lease = lease_getter()
                self.etcd.put(key, payload, lease=lease)
            else:
                raise
