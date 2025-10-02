from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from .base import BaseStore, _now_iso


@dataclass
class CapacityDoc:
    limit: int

    @classmethod
    def from_json(cls, raw: bytes | str) -> "CapacityDoc":
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        return cls(**data)


@dataclass
class SemaphoreHolder:
    job_id: str
    node: str
    ttl: int
    created_at: str

    @classmethod
    def from_json(cls, raw: bytes | str) -> "SemaphoreHolder":
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        return cls(**data)


def _cap_key(slot_type: str) -> str:
    return f"capacity/{slot_type}"


def _count_key(slot_type: str) -> str:
    return f"semaphores/{slot_type}/count"


def _holders_prefix(slot_type: str) -> str:
    return f"semaphores/{slot_type}/holders/"


def _holder_key(slot_type: str, ticket_id: str) -> str:
    return f"{_holders_prefix(slot_type)}{ticket_id}"


class SemaphoreStore(BaseStore):
    """
    Lock-free semaphore backed by etcd, implemented with CAS transactions.

    Layout:
      capacity/<slot_type>                 -> {"limit": N}
      semaphores/<slot_type>/count         -> "current_int"
      semaphores/<slot_type>/holders/<id>  -> {"job_id","node","ttl","created_at"} (with lease)

    reserve():
      1) read cap & count; if count >= cap -> fast-fail
      2) TX compare: cap unchanged, count unchanged (or missing), holder missing
         then: put count+1 and put holder (with lease)
    release():
      TX compare: count unchanged AND holder exists
         then: delete holder and put count-1
    reconcile():
      recompute count from holders and CAS update the counter
    """

    def __init__(self, etcd_client, default_lease_ttl: int = 30):
        super().__init__(etcd_client)
        self.default_lease_ttl = max(1, int(default_lease_ttl))

    def get_capacity(self, slot_type: str) -> Optional[int]:
        raw = self._get_raw(_cap_key(slot_type))
        if not raw:
            return None
        try:
            return CapacityDoc.from_json(raw).limit
        except Exception:
            return None

    def set_capacity(self, slot_type: str, limit: int) -> None:
        doc = CapacityDoc(limit=int(limit))
        self._put_json(_cap_key(slot_type), asdict(doc))

    def read_count(self, slot_type: str) -> int:
        raw = self._get_raw(_count_key(slot_type))
        if not raw:
            return 0
        try:
            return int(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        except Exception:
            return 0

    def available_slot_count(self, slot_type: str) -> int:
        cap = self.get_capacity(slot_type)
        if cap is None or cap < 0:
            return 0
        used = self.read_count(slot_type)
        return max(0, cap - used)

    def list_holders(self, slot_type: str) -> Dict[str, SemaphoreHolder]:
        prefix = _holders_prefix(slot_type)
        # use underlying etcd client to pull flat KVs
        results = self.etcd.client.get_prefix(self.etcd._mangle_key(prefix))
        holders: Dict[str, SemaphoreHolder] = {}
        for value_bytes, meta in results:
            key = self.etcd._demangle_key(meta.key)
            ticket_id = key.split('/')[-1]
            try:
                holders[ticket_id] = SemaphoreHolder.from_json(value_bytes)
            except Exception:
                # skip malformed
                pass
        return holders

    def reserve(
        self,
        slot_type: str,
        ticket_id: str,
        *,
        job_id: str,
        node: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Atomically try to reserve one slot. Returns True on success, False on capacity/race.
        """
        cap_k = _cap_key(slot_type)
        cnt_k = _count_key(slot_type)
        h_k = _holder_key(slot_type, ticket_id)

        # optimistic reads (linearizable because BaseStore._get_raw does serializable=False)
        cap_raw = self._get_raw(cap_k)
        if not cap_raw:
            return False  # capacity unknown -> treat as full/unavailable
        try:
            limit = CapacityDoc.from_json(cap_raw).limit
        except Exception:
            return False
        if limit <= 0:
            return False

        cnt_raw = self._get_raw(cnt_k)
        old_count = int(cnt_raw.decode()) if cnt_raw else 0
        if old_count >= limit:
            return False

        # CAS txn
        t = self.etcd.txn()

        # capacity unchanged
        t.if_value(cap_k, "==", cap_raw)

        # count unchanged (or missing)
        if cnt_raw is None:
            t.if_missing(cnt_k)
        else:
            t.if_value(cnt_k, "==", cnt_raw)

        # holder must not exist
        t.if_missing(h_k)

        # success writes
        new_count = old_count + 1
        lease = self.etcd.lease(ttl or self.default_lease_ttl)
        holder_doc = SemaphoreHolder(
            job_id=job_id,
            node=node,
            ttl=int(ttl or self.default_lease_ttl),
            created_at=_now_iso(),
        )

        t.put(cnt_k, str(new_count))
        # etcd3 tx.put expects lease id; your EtcdClient.Txn.put forwards lease as-is.
        # Pass lease.id for safety with etcd3-py.
        t.put(h_k, json.dumps(asdict(holder_doc)), lease=lease.id)

        ok, _resp = t.commit()
        return bool(ok)

    def release(self, slot_type: str, ticket_id: str) -> bool:
        """
        Atomically release a slot. Returns True if released, False otherwise.
        """
        cnt_k = _count_key(slot_type)
        h_k = _holder_key(slot_type, ticket_id)

        cnt_raw = self._get_raw(cnt_k)
        if cnt_raw is None:
            # We require a present counter to avoid underflow; caller can reconcile later.
            return False

        # ensure holder exists (read once to build compares)
        h_raw = self._get_raw(h_k)
        if not h_raw:
            return False

        old_count = int(cnt_raw.decode())
        new_count = max(0, old_count - 1)

        t = self.etcd.txn()
        t.if_value(cnt_k, "==", cnt_raw)  # count unchanged
        t.if_exists(h_k)  # holder must exist
        t.put(cnt_k, str(new_count))
        t.delete(h_k)

        ok, _resp = t.commit()
        return bool(ok)

    # ---- maintenance ----

    def reconcile(self, slot_type: str) -> int:
        """
        Recompute count from holders and CAS-update the counter.
        Returns the new count written (or current if CAS lost).
        """
        prefix = _holders_prefix(slot_type)
        results = self.etcd.client.get_prefix(self.etcd._mangle_key(prefix))
        live = sum(1 for _v, _m in results)

        cnt_k = _count_key(slot_type)
        cnt_raw = self._get_raw(cnt_k)
        cur = int(cnt_raw.decode()) if cnt_raw else 0
        if cur == live:
            return cur

        t = self.etcd.txn()
        if cnt_raw is None:
            t.if_missing(cnt_k)
        else:
            t.if_value(cnt_k, "==", cnt_raw)
        t.put(cnt_k, str(live))
        ok, _resp = t.commit()
        return live if ok else cur
