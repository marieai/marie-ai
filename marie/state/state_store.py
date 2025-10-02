from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

from grpc_health.v1.health_pb2 import HealthCheckResponse

from marie.serve.discovery.etcd_client import EtcdClient

from .base import BaseStore, _now_iso, _tx_succeeded

# Key structure:
# marie/deployments/{node}/{deployment}/desired   # gateway-only writer
# marie/deployments/{node}/{deployment}/status    # worker-only writer


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


class DesiredStore(BaseStore):
    """Gateway-only: desired intent/spec."""

    def __init__(self, etcd_client):  # Removed 'prefix' argument
        super().__init__(etcd_client)

    def _desired_key(self, node: str, depl: str) -> str:
        return _dkey(node, depl)

    def setXXX(
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

    def set(
        self, node: str, depl: str, params: Dict[str, Any], phase: str = "SCHEDULED"
    ) -> DesiredDoc:
        """
        Create or update desired with an atomic epoch bump using CAS on mod_revision.
        """
        k = self._desired_key(node, depl)

        # Fast path: if missing -> create epoch=1
        val, meta = self.etcd.get(k, metadata=True, serializable=False)
        if val is None:
            doc = DesiredDoc(
                phase=phase, epoch=1, params=params or {}, updated_at=_now_iso()
            )
            if self.etcd.put_if_absent(k, json.dumps(asdict(doc))):
                return doc
            # someone raced us; fall through to CAS loop

        # CAS loop on mod_revision
        for _ in range(8):
            val, meta = self.etcd.get(k, metadata=True, serializable=False)
            if val is None:
                # try to create again (another race may have deleted)
                doc = DesiredDoc(
                    phase=phase, epoch=1, params=params or {}, updated_at=_now_iso()
                )
                if self.etcd.put_if_absent(k, json.dumps(asdict(doc))):
                    return doc
                continue

            cur = DesiredDoc.from_json(val)
            cur.epoch += 1
            cur.phase = phase
            cur.params = params or {}
            cur.updated_at = _now_iso()

            if self.etcd.update_if_unchanged(
                k, json.dumps(asdict(cur)), meta.mod_revision
            ):
                return cur

            # lost the race; retry
            time.sleep(0.01)
        raise RuntimeError(f"DesiredStore.set CAS failed repeatedly for {k}")

    def get(self, node: str, depl: str) -> Optional[DesiredDoc]:
        k = self._desired_key(node, depl)
        raw = self._get_raw(k)
        print(' get   k = ', k)
        print(' get raw = ', raw)
        return DesiredDoc.from_json(raw) if raw else None

    def bump_epochXXX(self, node: str, depl: str) -> Optional[DesiredDoc]:
        k = self._desired_key(node, depl)
        raw = self._get_raw(k)
        if not raw:
            return None
        d = DesiredDoc.from_json(raw)
        d.epoch += 1
        d.updated_at = _now_iso()
        self._put_json(k, asdict(d))
        return d

    def bump_epoch(self, node: str, depl: str) -> Optional[DesiredDoc]:
        """
        Strict atomic epoch++ without altering phase/params.
        """
        k = self._desired_key(node, depl)
        for _ in range(8):
            val, meta = self.etcd.get(k, metadata=True, serializable=False)
            if val is None:
                return None
            cur = DesiredDoc.from_json(val)
            cur.epoch += 1
            cur.updated_at = _now_iso()
            if self.etcd.update_if_unchanged(
                k, json.dumps(asdict(cur)), meta.mod_revision
            ):
                return cur
            time.sleep(0.01)
        return None

    def iter_desired_pairs(self) -> Iterator[Tuple[str, str]]:
        root_prefix_for_etcd_client = "deployments/"
        nested = self.etcd.get_prefix_dict(root_prefix_for_etcd_client)

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

    def list_pairs(self) -> Iterator[Tuple[str, str]]:
        yield from self.iter_desired_pairs()

    def schedule_new_epochXXX(
        self, node: str, depl: str, params: Optional[Dict[str, Any]] = None
    ) -> DesiredDoc:
        """
        Bump epoch and set phase=SCHEDULED every time we schedule a new job.
        Creates the doc with epoch=1 if it doesn't exist.
        """
        k = self._desired_key(node, depl)
        raw = self._get_raw(k)

        if not raw:
            # First intent for this (node,depl)
            return self._create(
                node, depl, phase="SCHEDULED", epoch=1, params=params or {}
            )

        doc = DesiredDoc.from_json(raw)
        doc.epoch += 1
        doc.phase = "SCHEDULED"
        if params:
            # allow caller to include job-specific params (optional)
            doc.params = {**(doc.params or {}), **params}
        doc.updated_at = _now_iso()
        self._put_json(k, asdict(doc))
        return doc

    def schedule_new_epoch(
        self, node: str, depl: str, params: Optional[Dict[str, Any]] = None
    ) -> DesiredDoc:
        """
        Atomic: if missing -> create epoch=1; else epoch++, phase=SCHEDULED, merge params.
        """
        k = self._desired_key(node, depl)
        val, meta = self.etcd.get(k, metadata=True, serializable=False)
        if val is None:
            doc = DesiredDoc(
                phase="SCHEDULED", epoch=1, params=params or {}, updated_at=_now_iso()
            )
            if self.etcd.put_if_absent(k, json.dumps(asdict(doc))):
                return doc
            # raced; fall through

        for _ in range(8):
            val, meta = self.etcd.get(k, metadata=True, serializable=False)
            if val is None:
                # create again in case it was deleted
                doc = DesiredDoc(
                    phase="SCHEDULED",
                    epoch=1,
                    params=params or {},
                    updated_at=_now_iso(),
                )
                if self.etcd.put_if_absent(k, json.dumps(asdict(doc))):
                    return doc
                time.sleep(0.01)
                continue

            cur = DesiredDoc.from_json(val)
            cur.epoch += 1
            cur.phase = "SCHEDULED"
            if params:
                cur.params = {**(cur.params or {}), **params}
            cur.updated_at = _now_iso()

            if self.etcd.update_if_unchanged(
                k, json.dumps(asdict(cur)), meta.mod_revision
            ):
                return cur
            time.sleep(0.01)
        raise RuntimeError(f"DesiredStore.schedule_new_epoch CAS failed for {k}")

    def _create(
        self,
        node: str,
        depl: str,
        phase: str,
        epoch: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> DesiredDoc:
        """
        Create desired doc atomically:
          - If key is absent -> create with epoch provided.
          - If someone else created concurrently -> return the current stored doc (no overwrite).
        """
        k = self._desired_key(node, depl)
        doc = DesiredDoc(
            phase=phase,
            epoch=int(epoch),
            params=params or {},
            updated_at=_now_iso(),
        )
        payload = json.dumps(asdict(doc))

        # Fast path: atomic create
        if self.etcd.put_if_absent(k, payload):
            return doc

        # Lost the race: read and return the winner (don’t clobber)
        raw = self._get_raw(k)
        return DesiredDoc.from_json(raw) if raw else doc

    def _update_phaseXXX(self, node: str, depl: str, phase: str) -> DesiredDoc:
        existing = self.get(node, depl)
        if not existing:
            # If called without existing doc, create a new one with epoch=1
            return self._create(node, depl, phase=phase, epoch=1, params={})
        existing.phase = phase
        existing.updated_at = _now_iso()
        # NOTE: keep epoch unchanged here (important!)
        self._put_json(self._desired_key(node, depl), asdict(existing))
        return existing

    def _update_phase(self, node: str, depl: str, phase: str) -> DesiredDoc:
        """
        Atomic phase update (epoch unchanged).
        """
        k = self._desired_key(node, depl)
        val, meta = self.etcd.get(k, metadata=True, serializable=False)
        if val is None:
            # Create with epoch=1 if absent (preserves your original semantics)
            doc = DesiredDoc(phase=phase, epoch=1, params={}, updated_at=_now_iso())
            if self.etcd.put_if_absent(k, json.dumps(asdict(doc))):
                return doc

        for _ in range(8):
            val, meta = self.etcd.get(k, metadata=True, serializable=False)
            if val is None:
                # try again
                doc = DesiredDoc(phase=phase, epoch=1, params={}, updated_at=_now_iso())
                if self.etcd.put_if_absent(k, json.dumps(asdict(doc))):
                    return doc
                time.sleep(0.01)
                continue

            cur = DesiredDoc.from_json(val)
            cur.phase = phase
            cur.updated_at = _now_iso()

            if self.etcd.update_if_unchanged(
                k, json.dumps(asdict(cur)), meta.mod_revision
            ):
                return cur
            time.sleep(0.01)
        raise RuntimeError(f"DesiredStore._update_phase CAS failed for {k}")


class StatusStore(BaseStore):
    """Worker-only: observed serving status with heartbeats (HealthCheckResponse)."""

    def __init__(self, etcd_client, lease_getter=None):  # Removed 'prefix' argument
        super().__init__(etcd_client)
        self._lease_getter = lease_getter  # callable -> etcd lease (optional)

    def _status_key(self, node: str, depl: str) -> str:
        return _skey(node, depl)

    def _new_status_doc(
        self, worker_id: str, epoch: int, initial_status: int
    ) -> StatusDoc:
        return StatusDoc(
            status_code=initial_status,
            status_name=_status_name(initial_status),
            owner=worker_id,
            epoch=epoch,
            updated_at=_now_iso(),
            heartbeat_at=_now_iso(),
            details=None,
        )

    def claimXXX(
        self,
        node: str,
        depl: str,
        worker_id: str,
        epoch: int,
        initial_status: int = HealthCheckResponse.ServingStatus.NOT_SERVING,
    ) -> bool:

        k = self._status_key(node, depl)
        existing = self._get_raw(k)

        new_doc = lambda e: StatusDoc(
            status_code=initial_status,
            status_name=_status_name(initial_status),
            owner=worker_id,
            epoch=e,
            updated_at=_now_iso(),
            heartbeat_at=_now_iso(),
            details=None,
        )

        if not existing:
            doc = new_doc(epoch)
            if self._lease_getter:
                self._put_json_with_lease(k, asdict(doc), self._lease_getter)
            else:
                self._put_json(k, asdict(doc))
            return True

        st = StatusDoc.from_json(existing)

        # Same owner + same epoch -> idempotent success
        if st.owner == worker_id and st.epoch == epoch:
            # Ensure lease is refreshed and timestamps updated a bit
            st.updated_at = _now_iso()
            if self._lease_getter:
                self._put_json_with_lease(k, asdict(st), self._lease_getter)
            else:
                self._put_json(k, asdict(st))
            return True

        # Same owner + lower epoch -> roll forward to new epoch
        if st.owner == worker_id and st.epoch < epoch:
            doc = new_doc(epoch)
            if self._lease_getter:
                self._put_json_with_lease(k, asdict(doc), self._lease_getter)
            else:
                self._put_json(k, asdict(doc))
            return True

        # Different owner or future epoch owned elsewhere -> fail (fencing)
        return False

    def claim(
        self,
        node: str,
        depl: str,
        worker_id: str,
        epoch: int,
        initial_status: int = HealthCheckResponse.ServingStatus.NOT_SERVING,
    ) -> bool:
        """
        Claim ownership with fencing by epoch.
        Algorithm:
          A) If key missing: create (with lease if provided) atomically.
          B) Else read current -> if (same owner & same epoch) idempotent refresh.
             Else if (same owner & lower epoch) roll-forward to new epoch.
             Else (different owner or future epoch) -> reject.
        All writes use CAS on mod_revision; lease refresh is applied if provided.
        """
        k = self._status_key(node, depl)
        # A) attempt to create if absent
        doc = self._new_status_doc(worker_id, epoch, initial_status)
        payload = json.dumps(asdict(doc))
        lease_id = self._lease_getter().id if self._lease_getter else None
        if self.etcd.put_if_absent(k, payload, lease=lease_id):
            return True

        # B) read+CAS path
        for _ in range(8):
            val, meta = self.etcd.get(k, metadata=True, serializable=False)
            if val is None:
                # deleted between steps, retry create
                lease_id = self._lease_getter().id if self._lease_getter else None
                if self.etcd.put_if_absent(k, payload, lease=lease_id):
                    return True
                time.sleep(0.01)
                continue

            st = StatusDoc.from_json(val)

            # Idempotent refresh (same owner & same epoch)
            if st.owner == worker_id and st.epoch == epoch:
                st.updated_at = _now_iso()
                st.heartbeat_at = st.heartbeat_at or st.updated_at
                lease_id = self._lease_getter().id if self._lease_getter else None
                if self.etcd.update_if_unchanged(
                    k, json.dumps(asdict(st)), meta.mod_revision, lease=lease_id
                ):
                    return True
                time.sleep(0.01)
                continue

            # Roll-forward (same owner, lower epoch)
            if st.owner == worker_id and st.epoch < epoch:
                newdoc = self._new_status_doc(worker_id, epoch, initial_status)
                lease_id = self._lease_getter().id if self._lease_getter else None
                if self.etcd.update_if_unchanged(
                    k, json.dumps(asdict(newdoc)), meta.mod_revision, lease=lease_id
                ):
                    return True
                time.sleep(0.01)
                continue

            # Different owner or someone is ahead -> fencing reject
            return False

        return False

    def set_statusXXXX(
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

    def set_status(
        self,
        node: str,
        depl: str,
        worker_id: str,
        status: int | str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        CAS update: only the current owner can change status. Protect against concurrent owners by mod_revision CAS.
        """
        k = self._status_key(node, depl)
        for _ in range(8):
            val, meta = self.etcd.get(k, metadata=True, serializable=False)
            if val is None:
                return False
            st = StatusDoc.from_json(val)
            if st.owner != worker_id:
                return False
            code = _status_code(status)
            st.status_code = code
            st.status_name = _status_name(code)
            st.updated_at = _now_iso()
            if details:
                st.details = {**(st.details or {}), **details}

            try:
                lease_id = self._lease_getter().id if self._lease_getter else None
            except Exception:
                lease_id = None

            if self.etcd.update_if_unchanged(
                k, json.dumps(asdict(st)), meta.mod_revision, lease=lease_id
            ):
                return True
            time.sleep(0.01)
        return False

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

    def heartbeatXXX(self, node: str, depl: str, worker_id: str) -> bool:
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

    def heartbeat(self, node: str, depl: str, worker_id: str) -> bool:
        """
        CAS heartbeat: only the current owner may update heartbeat_at.
        If the lease vanished, surface False; caller can reacquire/claim.
        """
        k = self._status_key(node, depl)
        for _ in range(8):
            val, meta = self.etcd.get(k, metadata=True, serializable=False)
            if val is None:
                return False
            st = StatusDoc.from_json(val)
            if st.owner != worker_id:
                return False
            st.heartbeat_at = _now_iso()

            lease_id = None
            try:
                lease_id = self._lease_getter().id if self._lease_getter else None
            except Exception:
                pass

            try:
                if self.etcd.update_if_unchanged(
                    k, json.dumps(asdict(st)), meta.mod_revision, lease=lease_id
                ):
                    return True
            except Exception as e:
                # Handle missing/expired lease gracefully
                if _is_missing_lease_error(e) and self._lease_getter:
                    # refresh lease and retry once more in this iteration
                    try:
                        lease_id = self._lease_getter().id
                    except Exception:
                        return False
                else:
                    raise
            time.sleep(0.01)
        return False

    def read(self, node: str, depl: str) -> Optional[StatusDoc]:
        k = self._status_key(node, depl)
        raw = self._get_raw(k)
        print('read : k ', k)
        print('read : raw ', raw)

        return StatusDoc.from_json(raw) if raw else None
