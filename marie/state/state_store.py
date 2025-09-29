from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

from grpc_health.v1.health_pb2 import HealthCheckResponse

# Key structure:
# marie/deployments/{node}/{deployment}/desired   # gateway-only writer
# marie/deployments/{node}/{deployment}/status    # worker-only writer


def _value_to_json_str(v: Any) -> str:
    if v is None:
        return "{}"
    if isinstance(v, (bytes, bytearray)):
        return v.decode()
    if isinstance(v, str):
        return v
    # etcd resolver sometimes surfaces parsed dicts; standardize
    import json

    return json.dumps(v)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_iso(ts: Optional[str]) -> float:
    if not ts:
        return 0.0
    try:
        import calendar
        from datetime import datetime, timezone

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


def _dkey(prefix: str, node: str, depl: str) -> str:
    return f"{prefix}/deployments/{node}/{depl}/desired"


def _skey(prefix: str, node: str, depl: str) -> str:
    return f"{prefix}/deployments/{node}/{depl}/status"


def _status_name(code: int) -> str:
    # Map known codes to canonical names
    m = {
        HealthCheckResponse.ServingStatus.UNKNOWN: "UNKNOWN",
        HealthCheckResponse.ServingStatus.SERVING: "SERVING",
        HealthCheckResponse.ServingStatus.NOT_SERVING: "NOT_SERVING",
        HealthCheckResponse.ServingStatus.SERVICE_UNKNOWN: "SERVICE_UNKNOWN",
    }
    return m.get(code, f"CODE_{code}")


def _status_code(name_or_code: Any) -> int:
    if isinstance(name_or_code, int):
        return name_or_code
    s = str(name_or_code).upper()
    if s == "UNKNOWN":
        return HealthCheckResponse.ServingStatus.UNKNOWN
    if s == "SERVING":
        return HealthCheckResponse.ServingStatus.SERVING
    if s == "NOT_SERVING":
        return HealthCheckResponse.ServingStatus.NOT_SERVING
    if s == "SERVICE_UNKNOWN":
        return HealthCheckResponse.ServingStatus.SERVICE_UNKNOWN
    if s.startswith("CODE_"):
        try:
            return int(s[5:])
        except Exception:
            pass
    return HealthCheckResponse.ServingStatus.UNKNOWN


def _is_missing_lease_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    try:
        import grpc

        if isinstance(exc, grpc.RpcError) and exc.code() == grpc.StatusCode.NOT_FOUND:
            return True
    except Exception:
        pass
    return ("lease not found" in msg) or ("requested lease not found" in msg)


@dataclass
class DesiredDoc:
    # Intent/spec (gateway-only)
    phase: str  # e.g., "SCHEDULED"
    epoch: int  # fencing counter
    params: Dict[str, Any]  # arbitrary scheduler params
    updated_at: str

    @classmethod
    def from_json(cls, raw: bytes | str) -> "DesiredDoc":
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        return cls(**data)


@dataclass
class StatusDoc:
    # Observation/status (worker-only)
    status_code: int  # HealthCheckResponse.ServingStatus (int)
    status_name: str  # same as string for readability (e.g., "SERVING")
    owner: str  # worker id
    epoch: int  # must match desired.epoch when claimed
    updated_at: str
    heartbeat_at: str
    details: Optional[Dict[str, Any]] = None

    @classmethod
    def from_json(cls, raw: bytes | str) -> "StatusDoc":
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        return cls(**data)


class BaseStore:
    def __init__(self, etcd_client, prefix: str = "marie"):
        self.etcd = etcd_client
        self.prefix = prefix.rstrip("/")
        try:
            from etcd3 import transactions as tx  # noqa

            self._tx = tx
        except Exception:
            self._tx = None

    # raw kv helpers
    def _get_raw(self, key: str) -> Optional[bytes]:
        raw, _ = self.etcd.get(key, metadata=True)
        return raw

    def _put_json(self, key: str, obj: Dict[str, Any]) -> None:
        self.etcd.put(key, json.dumps(obj))

    def _put_json_with_lease(self, key: str, obj: Dict[str, Any], lease_getter) -> None:
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

    # key helpers
    def _desired_key(self, node: str, depl: str) -> str:
        return _dkey(self.prefix, node, depl)

    def _status_key(self, node: str, depl: str) -> str:
        return _skey(self.prefix, node, depl)

    # scans
    def iter_desired_pairs(self) -> Iterator[Tuple[str, str]]:
        root = f"{self.prefix}/deployments/"
        for _, meta in self.etcd.get_prefix(root):
            key = meta.key.decode()
            if not key.endswith("/desired"):
                continue
            parts = key.split("/")
            try:
                i = parts.index("deployments")
                yield parts[i + 1], parts[i + 2]
            except Exception:
                continue


class DesiredStore(BaseStore):
    """Gateway-only: desired intent/spec."""

    def set(
        self, node: str, depl: str, params: Dict[str, Any], phase: str = "SCHEDULED"
    ) -> DesiredDoc:
        k = self._desired_key(node, depl)
        cur = self._get_raw(k)
        epoch = 0
        if cur:
            try:
                epoch = int(json.loads(cur.decode()).get("epoch", 0))
            except Exception:
                pass
        doc = DesiredDoc(
            phase=phase, epoch=epoch + 1, params=params or {}, updated_at=_now_iso()
        )
        self._put_json(k, asdict(doc))
        return doc

    def get(self, node: str, depl: str) -> Optional[DesiredDoc]:
        k = self._desired_key(node, depl)
        raw = self._get_raw(k)
        return DesiredDoc.from_json(raw) if raw else None

    def bump_epoch(self, node: str, depl: str) -> Optional[DesiredDoc]:
        k = self._desired_key(node, depl)
        raw = self._get_raw(k)
        if not raw:
            return None
        d = DesiredDoc.from_json(raw)
        d.epoch += 1
        d.updated_at = _now_iso()
        self._put_json(k, asdict(d))
        return d

    def list_pairs(self) -> Iterator[Tuple[str, str]]:
        yield from self.iter_desired_pairs()


class StatusStore(BaseStore):
    """Worker-only: observed serving status with heartbeats (HealthCheckResponse)."""

    def __init__(self, etcd_client, prefix: str = "marie", lease_getter=None):
        super().__init__(etcd_client, prefix)
        self._lease_getter = lease_getter  # callable -> etcd lease (optional)

    def claim(
        self,
        node: str,
        depl: str,
        worker_id: str,
        epoch: int,
        initial_status: int = HealthCheckResponse.ServingStatus.NOT_SERVING,
    ) -> bool:
        """
        First write of status for this epoch. Best-effort CAS if supported.
        Default initial status is NOT_SERVING (worker booting / not yet handling).
        """
        k = self._status_key(node, depl)
        existing = self._get_raw(k)
        doc = StatusDoc(
            status_code=initial_status,
            status_name=_status_name(initial_status),
            owner=worker_id,
            epoch=epoch,
            updated_at=_now_iso(),
            heartbeat_at=_now_iso(),
            details=None,
        )
        v = json.dumps(asdict(doc))

        if self._tx:
            cmp_list = (
                [self._tx.Version(k) == 0]
                if not existing
                else [self._tx.Value(k) == existing.decode()]
            )
            try:
                put = self._tx.Put(
                    k, v, lease=(self._lease_getter() if self._lease_getter else None)
                )
                ok = self.etcd.transaction(compare=cmp_list, success=[put], failure=[])
                return bool(ok)
            except Exception as e:
                if _is_missing_lease_error(e) and self._lease_getter:
                    put = self._tx.Put(k, v, lease=self._lease_getter())
                    ok = self.etcd.transaction(
                        compare=cmp_list, success=[put], failure=[]
                    )
                    return bool(ok)
                raise
        else:
            if existing:
                return False
            if self._lease_getter:
                self._put_json_with_lease(k, asdict(doc), self._lease_getter)
            else:
                self._put_json(k, asdict(doc))
            return True

    def set_status(
        self,
        node: str,
        depl: str,
        worker_id: str,
        status: int | str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update to a specific ServingStatus (e.g., SERVING, NOT_SERVING, UNKNOWN...).
        """
        k = self._status_key(node, depl)
        raw = self._get_raw(k)
        if not raw:
            return False
        st = StatusDoc.from_json(raw)
        if st.owner != worker_id:
            return False
        code = _status_code(status)
        st.status_code = code
        st.status_name = _status_name(code)
        st.updated_at = _now_iso()
        if details:
            st.details = {**(st.details or {}), **details}
        if self._lease_getter:
            self._put_json_with_lease(k, asdict(st), self._lease_getter)
        else:
            self._put_json(k, asdict(st))
        return True

    # Convenience wrappers
    def set_serving(
        self,
        node: str,
        depl: str,
        worker_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.set_status(
            node, depl, worker_id, HealthCheckResponse.ServingStatus.SERVING, details
        )

    def set_not_serving(
        self,
        node: str,
        depl: str,
        worker_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.set_status(
            node,
            depl,
            worker_id,
            HealthCheckResponse.ServingStatus.NOT_SERVING,
            details,
        )

    def set_unknown(
        self,
        node: str,
        depl: str,
        worker_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.set_status(
            node, depl, worker_id, HealthCheckResponse.ServingStatus.UNKNOWN, details
        )

    def set_service_unknown(
        self,
        node: str,
        depl: str,
        worker_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.set_status(
            node,
            depl,
            worker_id,
            HealthCheckResponse.ServingStatus.SERVICE_UNKNOWN,
            details,
        )

    def heartbeat(self, node: str, depl: str, worker_id: str) -> bool:
        k = self._status_key(node, depl)
        raw = self._get_raw(k)
        if not raw:
            return False
        st = StatusDoc.from_json(raw)
        if st.owner != worker_id:
            return False
        st.heartbeat_at = _now_iso()
        if self._lease_getter:
            self._put_json_with_lease(k, asdict(st), self._lease_getter)
        else:
            self._put_json(k, asdict(st))
        return True

    def read(self, node: str, depl: str) -> Optional[StatusDoc]:
        k = self._status_key(node, depl)
        raw = self._get_raw(k)
        return StatusDoc.from_json(raw) if raw else None
