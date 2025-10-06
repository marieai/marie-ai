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
    ticket_id: str
    node: str
    ttl: int
    created_at: str
    # optional owner (worker identity) and renewal timestamp for observability
    owner: Optional[str] = None
    renewed_at: Optional[str] = None

    @classmethod
    def from_json(cls, raw: bytes | str) -> "SemaphoreHolder":
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        # tolerate unknown/missing fields and legacy job_id
        return cls(
            ticket_id=data.get("ticket_id") or data.get("job_id", ""),
            node=data.get("node", ""),
            ttl=int(data.get("ttl", 0) or 0),
            created_at=data.get("created_at") or _now_iso(),
            owner=data.get("owner"),
            renewed_at=data.get("renewed_at"),
        )


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
      semaphores/<slot_type>/holders/<id>  -> {"ticket_id","node","ttl","created_at","owner","renewed_at"} (with lease)

    reserve():
      1) read cap & count; if count >= cap -> fast-fail
      2) TX compare: cap unchanged, count unchanged (or missing), holder missing
         then: put count+1 and put holder (with lease)

    renew():
      TX compare: holder value unchanged (CAS), then re-put same holder (updated renewed_at/ttl)
      with a *new* lease so TTL is extended. Optional owner check enforced.

    release():
      TX compare: count unchanged AND holder exists -> delete holder, put count-1

    release_owned():
      Like release(), but only if holder.owner == owner (prevents hijack).

    reconcile():
      recompute count from holders and CAS update counter.
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
        requested = max(0, int(requested_limit))
        used = max(0, int(self.read_count(slot_type))) if never_below_used else 0
        target = max(used, requested) if never_below_used else requested

        val_bytes, meta = self._get_capacity_raw(slot_type)
        now = _now_iso()

        if val_bytes is None:
            doc = CapacityDoc(limit=target, owner=owner, source=source, updated_at=now)
            payload = json.dumps(asdict(doc))
            if self.etcd.put_if_absent(_cap_key(slot_type), payload):
                return target

        val_bytes, meta = self._get_capacity_raw(slot_type)
        if val_bytes is None:
            self._write_capacity_doc(
                slot_type,
                CapacityDoc(limit=target, owner=owner, source=source, updated_at=now),
            )
            return target

        cur = CapacityDoc.from_json(val_bytes)
        if int(cur.limit) == int(target):
            return cur.limit

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
                pass
        return holders

    def _get_holder_raw(self, slot_type: str, ticket_id: str):
        """Return (val_bytes, meta) for a holder key (or (None, meta) if missing)."""
        return self.etcd.get(
            _holder_key(slot_type, ticket_id), metadata=True, serializable=False
        )

    def reserve(
        self,
        slot_type: str,
        ticket_id: str,
        *,
        node: str,
        ttl: Optional[int] = None,
        owner: Optional[str] = None,
    ) -> bool:
        """
        Atomically try to reserve one slot. Returns True on success, False otherwise.
        ticket_id is the single identifier used for the holder key and stored in the holder.
        """
        cap_k = _cap_key(slot_type)
        cnt_k = _count_key(slot_type)
        h_k = _holder_key(slot_type, ticket_id)

        cap_raw = self._get_raw(cap_k)
        if not cap_raw:
            return False
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
        t.if_value(cap_k, "==", cap_raw)
        if cnt_raw is None:
            t.if_missing(cnt_k)
        else:
            t.if_value(cnt_k, "==", cnt_raw)
        t.if_missing(h_k)

        new_count = old_count + 1
        lease = self.etcd.lease(ttl or self.default_lease_ttl)

        holder_doc = SemaphoreHolder(
            ticket_id=ticket_id,
            node=node,
            ttl=int(ttl or self.default_lease_ttl),
            created_at=_now_iso(),
            owner=owner or f"{node}:{ticket_id}",  # default identity
            renewed_at=None,
        )

        t.put(cnt_k, str(new_count))
        t.put(h_k, json.dumps(asdict(holder_doc)), lease=lease.id)

        ok, _resp = t.commit()
        return bool(ok)

    def renew(
        self,
        slot_type: str,
        ticket_id: str,
        *,
        owner: Optional[str] = None,
        ttl: Optional[int] = None,
        update_ttl_field: bool = True,
    ) -> bool:
        """
        Renew a holder by re-binding it under a fresh lease.

        Safety:
          - CAS on current holder value (prevents hijack).
          - Optional owner check (recommended): if provided, must match stored owner.
          - Does NOT touch the counter (used stays the same).
        """
        h_k = _holder_key(slot_type, ticket_id)
        raw, meta = self._get_holder_raw(slot_type, ticket_id)
        if raw is None:
            return False

        try:
            holder = SemaphoreHolder.from_json(raw)
        except Exception:
            return False

        if owner is not None and holder.owner and holder.owner != owner:
            # Do not renew someone else's ticket
            return False

        # refresh timestamps / ttl metadata we store (lease TTL is enforced by etcd)
        if update_ttl_field and ttl:
            holder.ttl = int(ttl)
        holder.renewed_at = _now_iso()

        # Re-put with CAS on the exact previous value and a *new* lease
        lease = self.etcd.lease(ttl or holder.ttl or self.default_lease_ttl)
        t = self.etcd.txn()
        # Using value equality CAS (safer across servers than mod_revision for some etcd/python clients)
        t.if_value(h_k, "==", raw)
        t.put(h_k, json.dumps(asdict(holder)), lease=lease.id)
        ok, _resp = t.commit()
        return bool(ok)

    def release(
        self,
        slot_type: str,
        ticket_id: str,
        *,
        owner: Optional[str] = None,
    ) -> bool:
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
        t.if_value(h_k, "==", h_raw)  # holder unchanged & exists (CAS)
        t.put(cnt_k, str(new_count))
        t.delete(h_k)

        ok, _resp = t.commit()
        return bool(ok)

    def release_owned(self, slot_type: str, ticket_id: str, *, owner: str) -> bool:
        """
        Release only if the holder is owned by `owner`. Prevents accidental releases.
        """
        h_k = _holder_key(slot_type, ticket_id)
        cnt_k = _count_key(slot_type)

        h_raw, h_meta = self._get_holder_raw(slot_type, ticket_id)
        if h_raw is None:
            return False

        try:
            holder = SemaphoreHolder.from_json(h_raw)
        except Exception:
            return False

        if holder.owner and holder.owner != owner:
            return False

        cnt_raw = self._get_raw(cnt_k)
        if cnt_raw is None:
            return False

        old_count = int(cnt_raw.decode())
        new_count = max(0, old_count - 1)

        # CAS on both: holder value and count value
        t = self.etcd.txn()
        t.if_value(cnt_k, "==", cnt_raw)
        t.if_value(h_k, "==", h_raw)
        t.put(cnt_k, str(new_count))
        t.delete(h_k)

        ok, _resp = t.commit()
        return bool(ok)

    # ---------- Multi-slot helpers (for summaries & audits) ----------

    def list_slot_types(self) -> Set[str]:
        slots: Set[str] = set()

        cap_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("capacity/"))
        for _val, meta in cap_iter:
            key = self.etcd._demangle_key(meta.key)
            parts = key.split("/")
            if len(parts) == 2 and parts[0] == "capacity":
                slots.add(parts[1])

        sem_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("semaphores/"))
        for _val, meta in sem_iter:
            key = self.etcd._demangle_key(meta.key)
            parts = key.split("/")
            if len(parts) >= 3 and parts[0] == "semaphores":
                slots.add(parts[1])

        return slots

    def read_count_all(self) -> Dict[str, int]:
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
        caps = self.capacities_all()

        used: Dict[str, int] = {}
        holders_map: Dict[str, Dict[str, SemaphoreHolder]] = {}
        holder_counts: Dict[str, int] = {}

        sem_iter = self.etcd.client.get_prefix(self.etcd._mangle_key("semaphores/"))
        for val_bytes, meta in sem_iter:
            key = self.etcd._demangle_key(meta.key)
            parts = key.split("/")
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

        all_slots: Set[str] = (
            set(caps.keys())
            | set(used.keys())
            | set(holder_counts.keys())
            | self.list_slot_types()
        )

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

    def reconcile(self, slot_type: str) -> int:
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

    def get_holder(self, slot_type: str, ticket_id: str) -> Optional[SemaphoreHolder]:
        raw, _meta = self._get_holder_raw(slot_type, ticket_id)
        if raw is None:
            return None
        try:
            return SemaphoreHolder.from_json(raw)
        except Exception:
            return None

    def reconcile_all(
        self,
        *,
        delete_orphan_holders: bool = True,
        fix_counters: bool = True,
    ) -> Dict[str, dict]:
        """
        Server-startup reconciliation & cleanup across ALL slot types.

        Steps per slot_type:
          1) Find holder keys and identify "orphans" (lease_id missing/0).
          2) Optionally delete orphans (best-effort).
          3) Recompute live holder count.
          4) Optionally CAS-update the /count key to match live holders.

        Returns:
          {slot_type: {
              "before_count": int,
              "after_count": int,
              "deleted_orphans": int,
              "malformed_holders": int
          }, ...}
        """
        summary: Dict[str, dict] = {}
        slot_types = self.list_slot_types()

        for st in sorted(slot_types):
            holders_prefix = _holders_prefix(st)
            mangled_prefix = self.etcd._mangle_key(holders_prefix)

            # 1) Scan holders and detect orphans
            orphan_keys: list[str] = []
            live_keys: list[str] = []
            malformed = 0

            scan_iter = self.etcd.client.get_prefix(mangled_prefix)
            for val_bytes, meta in scan_iter:
                key = self.etcd._demangle_key(meta.key)
                # Try to parse holder for sanity; count malformed but don't delete by default
                try:
                    _ = SemaphoreHolder.from_json(val_bytes)
                except Exception:
                    malformed += 1

                # etcd-python metadata usually exposes lease_id (0 or None if none)
                lease_id = getattr(meta, "lease_id", 0) or getattr(meta, "lease", 0)
                if not lease_id:
                    orphan_keys.append(key)
                else:
                    live_keys.append(key)

            # 2) Delete orphan holder keys (best-effort) — no counter math here;
            # we'll recompute and CAS the counter in step (4)
            deleted_orphans = 0
            if delete_orphan_holders and orphan_keys:
                for k in orphan_keys:
                    try:
                        # Direct delete; even if already gone, it's fine
                        self.etcd.delete(k)
                        deleted_orphans += 1
                    except Exception:
                        # ignore individual delete failures
                        pass

            # 3) Recompute live holder count after orphan cleanup
            #    If we deleted orphans, we can avoid a second scan by using live_keys
            #    but live_keys might include keys that concurrently disappeared; a cheap
            #    re-scan ensures correctness.
            live_count = 0
            rescan_iter = self.etcd.client.get_prefix(mangled_prefix)
            for _v, _m in rescan_iter:
                live_count += 1

            # Read current counter
            cnt_k = _count_key(st)
            cnt_raw = self._get_raw(cnt_k)
            before_count = int(cnt_raw.decode()) if cnt_raw else 0
            after_count = before_count

            # 4) CAS-fix /count to the live holders
            if fix_counters and before_count != live_count:
                t = self.etcd.txn()
                if cnt_raw is None:
                    t.if_missing(cnt_k)
                else:
                    t.if_value(cnt_k, "==", cnt_raw)
                t.put(cnt_k, str(live_count))
                ok, _ = t.commit()
                after_count = live_count if ok else before_count

            summary[st] = {
                "before_count": before_count,
                "after_count": after_count,
                "deleted_orphans": deleted_orphans,
                "malformed_holders": malformed,
            }

        return summary
