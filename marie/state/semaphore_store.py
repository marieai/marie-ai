from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Set

from .base import BaseStore, _now_iso


@dataclass
class CapacityDoc:
    limit: int
    owner: Optional[str] = None
    source: Optional[str] = None
    updated_at: str = ""

    @classmethod
    def from_json(cls, raw: bytes | str) -> "CapacityDoc":
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        limit = int(data.get("limit", 0))
        owner = data.get("owner")
        source = data.get("source")
        updated_at = data.get("updated_at") or _now_iso()
        return cls(limit=limit, owner=owner, source=source, updated_at=str(updated_at))


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
      capacity/<slot_type>                 -> {"limit": N, "owner": "...", "source": "...", "updated_at": "..."}
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

    # ---------- Capacity (single) ----------

    def get_capacity(self, slot_type: str) -> Optional[int]:
        raw = self._get_raw(_cap_key(slot_type))
        if not raw:
            return None
        try:
            return CapacityDoc.from_json(raw).limit
        except Exception:
            return None

    def set_capacity(self, slot_type: str, limit: int) -> None:
        doc = CapacityDoc(limit=max(0, int(limit)))
        self._put_json(_cap_key(slot_type), asdict(doc))

    def _get_capacity_raw(self, slot_type: str):
        """Return (val_bytes, meta) for capacity key (or (None, meta) if missing)."""
        return self.etcd.get(_cap_key(slot_type), metadata=True, serializable=False)

    def _write_capacity_doc(self, slot_type: str, doc: CapacityDoc) -> None:
        self._put_json(_cap_key(slot_type), asdict(doc))

    def set_capacity_safe(
        self,
        slot_type: str,
        requested_limit: int,
        *,
        never_below_used: bool = True,
        owner: Optional[str] = "capacity-manager",
        source: Optional[str] = "auto",
    ) -> int:
        """
        Safely set capacity with invariants:
          - If never_below_used=True, clamp to >= current used holders.
          - Uses CAS on mod_revision when updating existing docs.
          - Returns the effective limit written (or current if unchanged/race).
        """
        requested = max(0, int(requested_limit))
        used = max(0, int(self.read_count(slot_type))) if never_below_used else 0
        target = max(used, requested) if never_below_used else requested

        # Try create-if-absent first
        val_bytes, meta = self._get_capacity_raw(slot_type)
        now = _now_iso()

        if val_bytes is None:
            doc = CapacityDoc(limit=target, owner=owner, source=source, updated_at=now)
            payload = json.dumps(asdict(doc))
            if self.etcd.put_if_absent(_cap_key(slot_type), payload):
                return target
            # Lost race; fall through to CAS path.

        val_bytes, meta = self._get_capacity_raw(slot_type)
        if val_bytes is None:
            # Deleted between steps; just write best-effort.
            self._write_capacity_doc(
                slot_type,
                CapacityDoc(limit=target, owner=owner, source=source, updated_at=now),
            )
            return target

        cur = CapacityDoc.from_json(val_bytes)
        if int(cur.limit) == int(target):
            return cur.limit  # already desired

        new_doc = CapacityDoc(limit=target, owner=owner, source=source, updated_at=now)
        swapped = self.etcd.update_if_unchanged(
            _cap_key(slot_type), json.dumps(asdict(new_doc)), meta.mod_revision
        )
        if swapped:
            return target

        latest = self.get_capacity(slot_type)
        return int(latest) if latest is not None else target

    # ---------- Counters & holders (single) ----------

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
        results = self.etcd.client.get_prefix(self.etcd._mangle_key(prefix))
        holders: Dict[str, SemaphoreHolder] = {}
        for value_bytes, meta in results:
            key = self.etcd._demangle_key(meta.key)
            ticket_id = key.split("/")[-1]
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
            return False  # avoid underflow; caller can reconcile later

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

    # ---------- Multi-slot helpers (for summaries & audits) ----------

    def list_slot_types(self) -> Set[str]:
        """
        Union of slot types discovered under:
          - capacity/<slot_type>
          - semaphores/<slot_type>/count
          - semaphores/<slot_type>/holders/<id>
        """
        slots: Set[str] = set()

        # From capacity/*
        cap_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("capacity/"))
        for _val, meta in cap_iter:
            key = self.etcd._demangle_key(meta.key)
            parts = key.split("/")
            if len(parts) == 2 and parts[0] == "capacity":
                slots.add(parts[1])

        # From semaphores/*
        sem_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("semaphores/"))
        for _val, meta in sem_iter:
            key = self.etcd._demangle_key(meta.key)
            parts = key.split("/")
            if len(parts) >= 3 and parts[0] == "semaphores":
                slots.add(parts[1])

        return slots

    def read_count_all(self) -> Dict[str, int]:
        """
        Used counts for all slot types by scanning only /semaphores/*/count keys.
        Falls back to 0 for slot types that exist only in capacity/* but lack a counter.
        """
        used: Dict[str, int] = {s: 0 for s in self.list_slot_types()}

        sem_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("semaphores/"))
        for val_bytes, meta in sem_iter:
            key = self.etcd._demangle_key(meta.key)
            if key.endswith("/count"):
                parts = key.split("/")
                if len(parts) == 3 and parts[0] == "semaphores":
                    st = parts[1]
                    try:
                        used[st] = int(
                            val_bytes.decode()
                            if isinstance(val_bytes, (bytes, bytearray))
                            else val_bytes
                        )
                    except Exception:
                        used[st] = 0
        return used

    def holder_counts_all(self) -> Dict[str, int]:
        """
        Number of holder records by slot type (quick scan).
        """
        counts: Dict[str, int] = {}
        sem_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("semaphores/"))
        for _val, meta in sem_iter:
            key = self.etcd._demangle_key(meta.key)
            parts = key.split("/")
            if len(parts) == 4 and parts[0] == "semaphores" and parts[2] == "holders":
                st = parts[1]
                counts[st] = counts.get(st, 0) + 1

        for st in self.list_slot_types():
            counts.setdefault(st, 0)
        return counts

    def available_count_all(self) -> Dict[str, int]:
        """
        Available = capacity - used, for every discovered slot type (clamped to 0).
        """
        caps: Dict[str, int] = {}
        cap_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("capacity/"))
        for val_bytes, meta in cap_iter:
            key = self.etcd._demangle_key(meta.key)
            parts = key.split("/")
            if len(parts) == 2 and parts[0] == "capacity":
                try:
                    limit = CapacityDoc.from_json(val_bytes).limit
                except Exception:
                    limit = 0
                caps[parts[1]] = max(0, int(limit))

        used = self.read_count_all()
        all_slots = set(caps.keys()) | set(used.keys()) | self.list_slot_types()
        out: Dict[str, int] = {}
        for st in all_slots:
            cap = caps.get(st, 0)
            u = used.get(st, 0)
            out[st] = max(0, cap - max(0, u))
        return out

    def list_holders_all(self) -> Dict[str, Dict[str, SemaphoreHolder]]:
        """
        Return all holders grouped by slot type:
          { slot_type: { ticket_id: SemaphoreHolder, ... }, ... }
        """
        result: Dict[str, Dict[str, SemaphoreHolder]] = {}
        sem_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("semaphores/"))
        for val_bytes, meta in sem_iter:
            key = self.etcd._demangle_key(meta.key)
            parts = key.split("/")
            if len(parts) == 4 and parts[0] == "semaphores" and parts[2] == "holders":
                st = parts[1]
                ticket_id = parts[3]
                try:
                    holder = SemaphoreHolder.from_json(val_bytes)
                except Exception:
                    continue
                result.setdefault(st, {})[ticket_id] = holder

        for st in self.list_slot_types():
            result.setdefault(st, {})
        return result

    def capacities_all(self) -> Dict[str, int]:
        """
        capacity/<slot_type> -> {"limit": N}
        """
        caps: Dict[str, int] = {}
        cap_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("capacity/"))
        for val_bytes, meta in cap_iter:
            key = self.etcd._demangle_key(meta.key)
            parts = key.split("/")
            if len(parts) == 2 and parts[0] == "capacity":
                try:
                    caps[parts[1]] = max(0, int(CapacityDoc.from_json(val_bytes).limit))
                except Exception:
                    caps[parts[1]] = 0
        return caps

    def snapshot_all(self, include_holders: bool = False) -> Dict[str, dict]:
        """
        Single-pass snapshot for all slot types:
          {
            "<slot_type>": {
              "capacity": int,
              "used": int,
              "available": int,
              "holder_count": int,
              # "holders": { "<ticket_id>": SemaphoreHolder, ... }   (if include_holders=True)
            },
            ...
          }
        """
        # 1) Gather capacities
        caps = self.capacities_all()

        # 2) One scan over semaphores/ to collect used counts and holders
        used: Dict[str, int] = {}
        holders_map: Dict[str, Dict[str, SemaphoreHolder]] = {}
        holder_counts: Dict[str, int] = {}

        sem_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("semaphores/"))
        for val_bytes, meta in sem_iter:
            key = self.etcd._demangle_key(meta.key)
            parts = key.split("/")
            # semaphores/<slot_type>/count
            if len(parts) == 3 and parts[0] == "semaphores" and parts[2] == "count":
                st = parts[1]
                try:
                    used[st] = int(
                        val_bytes.decode()
                        if isinstance(val_bytes, (bytes, bytearray))
                        else val_bytes
                    )
                except Exception:
                    used[st] = 0
                continue

            # semaphores/<slot_type>/holders/<id>
            if len(parts) == 4 and parts[0] == "semaphores" and parts[2] == "holders":
                st = parts[1]
                ticket_id = parts[3]
                holder_counts[st] = holder_counts.get(st, 0) + 1
                if include_holders:
                    try:
                        holder = SemaphoreHolder.from_json(val_bytes)
                        holders_map.setdefault(st, {})[ticket_id] = holder
                    except Exception:
                        pass

        # 3) Union of all seen slot types
        all_slots: Set[str] = (
            set(caps.keys())
            | set(used.keys())
            | set(holder_counts.keys())
            | self.list_slot_types()
        )

        # 4) Assemble snapshot
        out: Dict[str, dict] = {}
        for st in sorted(all_slots):
            cap = caps.get(st, 0)
            u = max(0, used.get(st, 0))
            avail = max(0, cap - u)
            hc = holder_counts.get(st, 0)
            entry = {
                "capacity": cap,
                "used": u,
                "available": avail,
                "holder_count": hc,
            }
            if include_holders:
                entry["holders"] = holders_map.get(st, {})
            out[st] = entry

        return out

    # ---------- Maintenance ----------

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
