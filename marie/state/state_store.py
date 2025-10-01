from __future__ import annotations

import calendar
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional, Tuple

from grpc_health.v1.health_pb2 import HealthCheckResponse

from marie.serve.discovery.etcd_client import EtcdClient

# Key structure:
# marie/deployments/{node}/{deployment}/desired   # gateway-only writer
# marie/deployments/{node}/{deployment}/status    # worker-only writer


def _tx_succeeded(ok) -> bool:
    # etcd3-py often returns (succeeded, responses)
    print("tx_succeeded : ", ok)
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


def _dkey(node: str, depl: str) -> str:
    # Removed 'prefix' argument as EtcdClient handles the namespace
    return f"deployments/{node}/{depl}/desired"


def _skey(node: str, depl: str) -> str:
    # Removed 'prefix' argument as EtcdClient handles the namespace
    return f"deployments/{node}/{depl}/status"


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
        # etcd3-py: serializable=True allows stale reads; set it to False.
        val, _meta = self.etcd.get(key, metadata=True, serializable=False)
        if val is None:
            return None
        return val if isinstance(val, (bytes, bytearray)) else str(val).encode()

    def _put_json(self, key: str, obj: Dict[str, Any]) -> None:
        print('putting json to', key, obj)
        self.etcd.put(key, json.dumps(obj))

    def _put_json_with_lease(self, key: str, obj: Dict[str, Any], lease_getter) -> None:
        print('putting json with lease to', key, obj)
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

    def _desired_key(self, node: str, depl: str) -> str:
        # Call _dkey without the explicit prefix
        return _dkey(node, depl)

    def _status_key(self, node: str, depl: str) -> str:
        # Call _skey without the explicit prefix
        return _skey(node, depl)

    def iter_desired_pairs(self) -> Iterator[Tuple[str, str]]:
        # The key prefix for etcd_client should *not* contain the `marie` namespace,
        # as etcd_client._mangle_key will add it.
        # So, we pass "deployments/" and EtcdClient will turn it into "/marie/deployments/"
        root_prefix_for_etcd_client = "deployments/"
        nested = self.etcd.get_prefix_dict(root_prefix_for_etcd_client)

        print(f'nested for : {root_prefix_for_etcd_client}')
        print(nested)

        def _walk(d: dict, base: str):
            for k, v in d.items():
                full = f"{base.rstrip('/')}/{k}" if k else base.rstrip("/")
                if isinstance(v, dict):
                    yield from _walk(v, full)
                else:
                    yield full

        # The keys yielded by _walk are already demangled and start from "deployments/"
        for key in _walk(nested, root_prefix_for_etcd_client.rstrip('/')):
            if not key.endswith("/desired"):
                continue
            parts = key.split("/")
            try:
                # After demangling and splitting, parts would be like ['deployments', 'node', 'depl', 'desired']
                # So index 1 and 2 will correctly extract node and depl
                i = parts.index("deployments")
                yield parts[i + 1], parts[i + 2]
            except Exception as e:
                print('Error parsing key:', key, e)
                continue


class DesiredStore(BaseStore):
    """Gateway-only: desired intent/spec."""

    def __init__(self, etcd_client):  # Removed 'prefix' argument
        super().__init__(etcd_client)

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

    def upsert_scheduled(self, node: str, deployment: str) -> DesiredDoc:
        print('upsert_scheduled for', node, deployment)

        d = self.get(node, deployment)
        if not d:
            # First time → create with epoch=1 (or 0 if you prefer)
            return self._create(node, deployment, phase="SCHEDULED", epoch=1, params={})
        if d.phase != "SCHEDULED":
            # Transition to SCHEDULED but keep epoch (important to avoid fencing mismatch)
            return self._update_phase(node, deployment, phase="SCHEDULED")
        return d

    def _create(
        self,
        node: str,
        depl: str,
        phase: str,
        epoch: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> DesiredDoc:
        doc = DesiredDoc(
            phase=phase,
            epoch=epoch,
            params=params or {},
            updated_at=_now_iso(),
        )
        self._put_json(self._desired_key(node, depl), asdict(doc))
        return doc

    def _update_phase(self, node: str, depl: str, phase: str) -> DesiredDoc:
        existing = self.get(node, depl)
        if not existing:
            # If called without existing doc, create a new one with epoch=1
            return self._create(node, depl, phase=phase, epoch=1, params={})
        existing.phase = phase
        existing.updated_at = _now_iso()
        # NOTE: keep epoch unchanged here (important!)
        self._put_json(self._desired_key(node, depl), asdict(existing))
        return existing


class StatusStore(BaseStore):
    """Worker-only: observed serving status with heartbeats (HealthCheckResponse)."""

    def __init__(self, etcd_client, lease_getter=None):  # Removed 'prefix' argument
        super().__init__(etcd_client)
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
        print('claiming status for', node)

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
                succeeded = _tx_succeeded(ok)
                print(f"TX claim result key={k} succeeded={succeeded}")
                if succeeded:
                    # optional sanity check
                    if not self._get_raw(k):
                        print(f"TX reported success but readback empty for {k}")
                        return False
                return succeeded

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

        print('K = ', k)
        print('raw = ', raw)

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
