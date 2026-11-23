from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Set

import etcd3
import grpc

from .base import BaseStore, _now_iso


class EtcdStoreUnavailable(RuntimeError):
    """Raised when a semaphore-store scan cannot reach etcd (outage/reconnect)."""


@dataclass
class CapacityDoc:
    limit: int
    owner: Optional[str] = None
    source: Optional[str] = None
    updated_at: str = ""
    requested_limit: Optional[int] = None

    @property
    def admission_limit(self) -> int:
        if self.requested_limit is None:
            return self.limit
        return min(self.limit, self.requested_limit)

    @classmethod
    def from_json(cls, raw: bytes | str) -> "CapacityDoc":
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        limit = int(data.get("limit", 0))
        owner = data.get("owner")
        source = data.get("source")
        updated_at = data.get("updated_at") or _now_iso()
        requested = data.get("requested_limit")
        return cls(
            limit=limit,
            owner=owner,
            source=source,
            updated_at=str(updated_at),
            requested_limit=int(requested) if requested is not None else None,
        )


def _decode_capacity(
    raw: bytes | str | None,
    *,
    errors: tuple[type[Exception], ...] = (
        ValueError,
        TypeError,
        AttributeError,
        OverflowError,
    ),
) -> CapacityDoc | None:
    if not raw:
        return None
    try:
        return CapacityDoc.from_json(raw)
    except errors:
        return None


@dataclass
class SemaphoreHolder:
    ticket_id: str
    node: str
    ttl: int
    created_at: str
    # optional owner (worker identity) and renewal timestamp for observability
    owner: Optional[str] = None
    renewed_at: Optional[str] = None
    run_attempt_id: Optional[str] = None

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
            run_attempt_id=data.get("run_attempt_id"),
        )


@dataclass(frozen=True)
class SemaphoreReleaseResult:
    """Outcome of an ownership-fenced semaphore release."""

    success: bool
    reason: str
    attempts: int
    retryable: bool = False
    holder_absent: bool = False
    counter_reconciled: bool | None = None

    def __bool__(self) -> bool:
        return self.success


def _cap_key(slot_type: str) -> str:
    return f"capacity/{slot_type}"


def _count_key(slot_type: str) -> str:
    return f"semaphores/{slot_type}/count"


def _holders_prefix(slot_type: str) -> str:
    return f"semaphores/{slot_type}/holders/"


def _holder_key(slot_type: str, ticket_id: str) -> str:
    return f"{_holders_prefix(slot_type)}{ticket_id}"


def _decode_count(raw) -> Optional[int]:
    """Decode a counter for point reads and transactions.

    None/empty -> 0 (missing counter means zero holders).
    Malformed  -> None (caller decides: repair via reconcile, or treat as 0
    while CASing on the raw value so the write itself is the repair).
    """
    if not raw:
        return 0
    try:
        s = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
        return int(s.strip())
    except Exception:
        return None


def _reported_count(raw: bytes | str) -> int:
    # Aggregate reads use int directly. Point reads also strip control characters.
    try:
        return int(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
    except (ValueError, TypeError, OverflowError):
        return 0


def _retry_after_contention(attempt: int) -> None:
    ceiling = min(0.05, 0.002 * (2**attempt))
    time.sleep(random.uniform(ceiling * 0.5, ceiling))


class SemaphoreStore(BaseStore):
    """
    Lock-free semaphore backed by etcd, implemented with CAS transactions.

    Layout:
      capacity/<slot_type>                 -> {"limit": N, "owner": "...", "source": "...", "updated_at": "..."}
      semaphores/<slot_type>/count         -> "current_int"
      semaphores/<slot_type>/holders/<id>  -> holder identity and lease metadata

    reserve():
      1) read cap & count; if count >= cap -> fast-fail
      2) TX compare: cap unchanged, count unchanged (or missing), holder missing
         then: put count+1 and put holder (with lease)

    renew():
      validate the holder identity and keep its existing lease alive.

    release():
      Check the stored owner and run attempt, then atomically delete the holder
      and decrement the counter. release_owned_result() also reports the outcome.

    reconcile():
      recompute count from holders and CAS update counter.
    """

    def __init__(self, etcd_client, default_lease_ttl: int = 30):
        super().__init__(etcd_client)
        self.default_lease_ttl = max(1, int(default_lease_ttl))

    def _scan_prefix(self, mangled_prefix):
        """All prefix scans go through here so an etcd outage surfaces as one
        typed error instead of raw grpc/channel exceptions in callers."""
        try:
            return self.etcd.client.get_prefix(mangled_prefix)
        except (grpc.RpcError, etcd3.exceptions.Etcd3Exception) as e:
            raise EtcdStoreUnavailable(f"etcd scan failed: {e}") from e
        except ValueError as e:
            msg = str(e).lower()
            if "closed channel" in msg or "channel closed" in msg:
                raise EtcdStoreUnavailable(f"etcd channel closed: {e}") from e
            raise

    # ---------- Capacity (single) ----------

    def get_capacity(self, slot_type: str) -> Optional[int]:
        doc = _decode_capacity(self._get_raw(_cap_key(slot_type)), errors=(Exception,))
        return doc.limit if doc is not None else None

    def set_capacity(self, slot_type: str, limit: int) -> None:
        doc = CapacityDoc(limit=max(0, int(limit)))
        self._put_json(_cap_key(slot_type), asdict(doc))

    def _get_capacity_raw(self, slot_type: str):
        """Return (val_bytes, meta) for capacity key (or (None, meta) if missing)."""
        return self.etcd.get(_cap_key(slot_type), metadata=True, serializable=False)

    def set_capacity_safe(
        self,
        slot_type: str,
        requested_limit: int,
        *,
        never_below_used: bool = True,
        owner: Optional[str] = "capacity-manager",
        source: Optional[str] = "auto",
    ) -> int:
        """Persist the admission target while optionally clamping limit to usage.

        Invariant: every WRITE this method performs is committed in a txn that
        proves the count key is unchanged since `target` was computed from it.
        No-write paths (already-equal, and the raise on exhaustion) satisfied
        the invariant at read time — "still true at return time" is not
        promisable by any implementation.
        """
        requested = max(0, int(requested_limit))
        cap_k = _cap_key(slot_type)
        cnt_k = _count_key(slot_type)

        for _attempt in range(10):
            cnt_raw = self._get_raw(cnt_k)
            if cnt_raw is None:
                used = 0
            else:
                try:
                    used = max(0, int(cnt_raw.decode()))
                except Exception as exc:
                    if never_below_used:
                        # computing from 0 could lower capacity below real
                        # usage; refuse — reconcile/repair owns malformed keys
                        raise RuntimeError(
                            f"Refusing to set capacity for {slot_type}: "
                            f"malformed count value {cnt_raw!r}"
                        ) from exc
                    used = 0
            target = max(used, requested) if never_below_used else requested

            val_bytes, meta = self._get_capacity_raw(slot_type)
            if val_bytes is not None:
                cur = CapacityDoc.from_json(val_bytes)
                if cur.limit == target and cur.requested_limit == requested:
                    # nothing to write; avoids one etcd revision per slot per
                    # gateway capacity pass
                    return cur.limit

            doc = CapacityDoc(
                limit=target,
                owner=owner,
                source=source,
                updated_at=_now_iso(),
                requested_limit=requested,
            )
            t = self.etcd.txn()
            if val_bytes is None:
                # create path must ALSO prove the count didn't move — a bare
                # put_if_absent would let `used` go stale between read & write
                t.if_missing(cap_k)
            else:
                t.if_mod_revision(cap_k, "==", meta.mod_revision)
            if never_below_used:
                if cnt_raw is None:
                    t.if_missing(cnt_k)
                else:
                    t.if_value(cnt_k, "==", cnt_raw)
            t.put(cap_k, json.dumps(asdict(doc)))
            ok, _resp = t.commit()
            if ok:
                return target
            # capacity or count moved underneath us — recompute and retry

        # Only report success if both the target and the usage clamp persisted.
        latest_used = max(0, int(self.read_count(slot_type)))
        latest_raw, _meta = self._get_capacity_raw(slot_type)
        latest = CapacityDoc.from_json(latest_raw) if latest_raw is not None else None
        if (
            latest is not None
            and latest.requested_limit == requested
            and (not never_below_used or latest.limit >= latest_used)
        ):
            return latest.limit
        raise RuntimeError(
            f"Failed to safely set capacity for {slot_type}: "
            f"requested={requested}, latest={latest}, used={latest_used}"
        )

    # ---------- Counters & holders (single) ----------

    def read_count(self, slot_type: str) -> int:
        val = _decode_count(self._get_raw(_count_key(slot_type)))
        return val if val is not None else 0

    def available_slot_count(self, slot_type: str) -> int:
        doc = _decode_capacity(self._get_raw(_cap_key(slot_type)))
        if doc is None:
            return 0
        used = self.read_count(slot_type)
        return max(0, doc.admission_limit - used)

    def list_holders(self, slot_type: str) -> Dict[str, SemaphoreHolder]:
        prefix = _holders_prefix(slot_type)
        results = self._scan_prefix(self.etcd._mangle_key(prefix))
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
        run_attempt_id: Optional[str] = None,
    ) -> bool:
        """
        Atomically try to reserve one slot. Returns True on success, False otherwise.
        ticket_id is the single identifier used for the holder key and stored in the holder.
        """
        cap_k = _cap_key(slot_type)
        cnt_k = _count_key(slot_type)
        h_k = _holder_key(slot_type, ticket_id)

        cap_raw = self._get_raw(cap_k)
        doc = _decode_capacity(cap_raw, errors=(Exception,))
        if doc is None or doc.admission_limit <= 0:
            return False
        limit = doc.admission_limit

        cnt_raw = self._get_raw(cnt_k)
        old_count = _decode_count(cnt_raw)
        if old_count is None:
            self.reconcile(slot_type)
            cnt_raw = self._get_raw(cnt_k)
            old_count = _decode_count(cnt_raw)
            if old_count is None:
                return False
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
            run_attempt_id=run_attempt_id,
        )

        t.put(cnt_k, str(new_count))
        t.put(h_k, json.dumps(asdict(holder_doc)), lease=lease.id)

        ok, _resp = t.commit()
        return bool(ok)

    def reserve_many(
        self,
        slot_type: str,
        ticket_ids: list[str],
        *,
        node: str,
        ttl: Optional[int] = None,
        owner_by_ticket: Optional[Dict[str, str]] = None,
        run_attempt_id_by_ticket: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        max_batch_size: int = 32,
    ) -> Set[str]:
        if not ticket_ids:
            return set()

        unique_ticket_ids = list(
            dict.fromkeys(str(ticket_id) for ticket_id in ticket_ids)
        )
        batch_size = max(1, int(max_batch_size))
        retries = max(1, int(max_retries))

        ttl_seconds = int(ttl or self.default_lease_ttl)
        reserved: Set[str] = set()
        for start in range(0, len(unique_ticket_ids), batch_size):
            batch = unique_ticket_ids[start : start + batch_size]
            reserved.update(
                self._reserve_many_batch(
                    slot_type,
                    batch,
                    node=node,
                    ttl_seconds=ttl_seconds,
                    owner_by_ticket=owner_by_ticket,
                    run_attempt_id_by_ticket=run_attempt_id_by_ticket,
                    max_retries=retries,
                )
            )
        return reserved

    def _reserve_many_batch(
        self,
        slot_type: str,
        ticket_ids: list[str],
        *,
        node: str,
        ttl_seconds: int,
        owner_by_ticket: Optional[Dict[str, str]],
        run_attempt_id_by_ticket: Optional[Dict[str, str]],
        max_retries: int,
    ) -> Set[str]:
        cap_k = _cap_key(slot_type)
        cnt_k = _count_key(slot_type)

        for attempt in range(max_retries):
            cap_raw = self._get_raw(cap_k)
            doc = _decode_capacity(cap_raw, errors=(Exception,))
            if doc is None or doc.admission_limit <= 0:
                return set()
            limit = doc.admission_limit

            cnt_raw = self._get_raw(cnt_k)
            old_count = _decode_count(cnt_raw)
            if old_count is None:
                self.reconcile(slot_type)
                continue
            available = max(0, int(limit) - max(0, old_count))
            if available <= 0:
                return set()

            selected = ticket_ids[:available]
            lease = self.etcd.lease(ttl_seconds)
            tx = self.etcd.txn()
            tx.if_value(cap_k, "==", cap_raw)
            if cnt_raw is None:
                tx.if_missing(cnt_k)
            else:
                tx.if_value(cnt_k, "==", cnt_raw)

            for ticket_id in selected:
                tx.if_missing(_holder_key(slot_type, ticket_id))

            tx.put(cnt_k, str(old_count + len(selected)))
            for ticket_id in selected:
                holder_doc = SemaphoreHolder(
                    ticket_id=ticket_id,
                    node=node,
                    ttl=ttl_seconds,
                    created_at=_now_iso(),
                    owner=(owner_by_ticket or {}).get(ticket_id)
                    or f"{node}:{ticket_id}",
                    renewed_at=None,
                    run_attempt_id=(run_attempt_id_by_ticket or {}).get(ticket_id),
                )
                tx.put(
                    _holder_key(slot_type, ticket_id),
                    json.dumps(asdict(holder_doc)),
                    lease=lease.id,
                )

            ok, _resp = tx.commit()
            if ok:
                return set(selected)
            if attempt < max_retries - 1:
                time.sleep(0.001 * (2**attempt))

        reserved: Set[str] = set()
        for ticket_id in ticket_ids:
            if self.reserve(
                slot_type,
                ticket_id,
                node=node,
                ttl=ttl_seconds,
                owner=(owner_by_ticket or {}).get(ticket_id),
                run_attempt_id=(run_attempt_id_by_ticket or {}).get(ticket_id),
            ):
                reserved.add(ticket_id)
        return reserved

    def renew(
        self,
        slot_type: str,
        ticket_id: str,
        *,
        owner: Optional[str] = None,
        run_attempt_id: Optional[str] = None,
        ttl: Optional[int] = None,
        update_ttl_field: bool = True,
    ) -> bool:
        """Keep the holder's attached lease alive.

        Safety:
          - The holder must still exist under an attached lease.
          - Optional owner and run-attempt checks prevent stale workers from renewing it.
          - The holder and counter are not rewritten.
        """
        raw, meta = self._get_holder_raw(slot_type, ticket_id)
        if raw is None or meta is None or not meta.lease_id:
            return False

        try:
            holder = SemaphoreHolder.from_json(raw)
        except Exception:
            return False

        if owner is not None and holder.owner and holder.owner != owner:
            # Do not renew someone else's ticket
            return False
        if (
            holder.run_attempt_id is not None
            and holder.run_attempt_id != run_attempt_id
        ):
            return False
        if update_ttl_field and ttl is not None and int(ttl) != holder.ttl:
            return False

        responses = self.etcd.refresh_lease(meta.lease_id)
        if not responses:
            return False
        response = responses[0]
        refreshed_ttl = getattr(response, "TTL", getattr(response, "ttl", 0))
        return int(refreshed_ttl or 0) > 0

    def validate_holder(
        self,
        slot_type: str,
        ticket_id: str,
        *,
        owner: str,
        run_attempt_id: str,
    ) -> bool:
        """Verify that a live holder belongs to the expected run attempt."""
        raw, meta = self._get_holder_raw(slot_type, ticket_id)
        if raw is None or meta is None or not meta.lease_id:
            return False
        try:
            holder = SemaphoreHolder.from_json(raw)
        except Exception:
            return False
        return (
            holder.ticket_id == ticket_id
            and holder.owner == owner
            and holder.run_attempt_id == run_attempt_id
        )

    def bind_run_attempt(
        self,
        slot_type: str,
        ticket_id: str,
        *,
        owner: str,
        run_attempt_id: str,
    ) -> bool:
        h_k = _holder_key(slot_type, ticket_id)
        raw, meta = self._get_holder_raw(slot_type, ticket_id)
        if raw is None or meta is None or not meta.lease_id:
            return False

        try:
            holder = SemaphoreHolder.from_json(raw)
        except Exception:
            return False
        if holder.owner and holder.owner != owner:
            return False
        if holder.run_attempt_id not in (None, run_attempt_id):
            return False

        holder.run_attempt_id = run_attempt_id
        transaction = self.etcd.txn()
        transaction.if_value(h_k, "==", raw)
        transaction.put(
            h_k,
            json.dumps(asdict(holder)),
            lease=meta.lease_id,
        )
        ok, _response = transaction.commit()
        return bool(ok)

    def release(
        self,
        slot_type: str,
        ticket_id: str,
        *,
        owner: str,
        run_attempt_id: Optional[str] = None,
        max_retries: int = 12,
    ) -> bool:
        """Return True after release or successful reconciliation of an absent holder."""

        return bool(
            self.release_owned_result(
                slot_type,
                ticket_id,
                owner=owner,
                run_attempt_id=run_attempt_id,
                max_retries=max_retries,
            )
        )

    def release_owned_result(
        self,
        slot_type: str,
        ticket_id: str,
        *,
        owner: str,
        run_attempt_id: Optional[str] = None,
        max_retries: int = 12,
    ) -> SemaphoreReleaseResult:
        """Release an owned ticket with retry and reconciliation details."""

        h_k = _holder_key(slot_type, ticket_id)
        cnt_k = _count_key(slot_type)
        retries = max(1, int(max_retries))

        for attempt in range(retries):
            attempts = attempt + 1
            h_raw, _h_meta = self._get_holder_raw(slot_type, ticket_id)
            if h_raw is None:
                _count, reconciled, reconcile_attempts = self._reconcile_count(
                    slot_type,
                    max_retries=retries,
                )
                if reconciled:
                    return SemaphoreReleaseResult(
                        success=True,
                        reason="already_absent",
                        attempts=attempts,
                        holder_absent=True,
                        counter_reconciled=True,
                    )
                return SemaphoreReleaseResult(
                    success=False,
                    reason="reconcile_contention",
                    attempts=attempts + reconcile_attempts,
                    retryable=True,
                    holder_absent=True,
                    counter_reconciled=False,
                )

            try:
                holder = SemaphoreHolder.from_json(h_raw)
            except Exception:
                return SemaphoreReleaseResult(
                    success=False,
                    reason="malformed_holder",
                    attempts=attempts,
                )

            if holder.owner and holder.owner != owner:
                return SemaphoreReleaseResult(
                    success=False,
                    reason="owner_mismatch",
                    attempts=attempts,
                )
            if (
                holder.run_attempt_id is not None
                and holder.run_attempt_id != run_attempt_id
            ):
                return SemaphoreReleaseResult(
                    success=False,
                    reason="attempt_mismatch",
                    attempts=attempts,
                )

            cnt_raw = self._get_raw(cnt_k)
            old_count = _decode_count(cnt_raw)
            if cnt_raw is None or old_count is None:
                _count, reconciled, reconcile_attempts = self._reconcile_count(
                    slot_type,
                    max_retries=retries,
                )
                if not reconciled:
                    return SemaphoreReleaseResult(
                        success=False,
                        reason="reconcile_contention",
                        attempts=attempts + reconcile_attempts,
                        retryable=True,
                        counter_reconciled=False,
                    )
                continue
            new_count = max(0, old_count - 1)

            t = self.etcd.txn()
            t.if_value(cnt_k, "==", cnt_raw)
            t.if_value(h_k, "==", h_raw)
            t.put(cnt_k, str(new_count))
            t.delete(h_k)

            ok, _resp = t.commit()
            if ok:
                return SemaphoreReleaseResult(
                    success=True,
                    reason="released",
                    attempts=attempts,
                )
            if attempt < retries - 1:
                _retry_after_contention(attempt)

        return SemaphoreReleaseResult(
            success=False,
            reason="counter_contention",
            attempts=retries,
            retryable=True,
        )

    # ---------- Multi-slot helpers (for summaries & audits) ----------

    def _capacity_items(self) -> Iterator[tuple[str, bytes]]:
        for raw, meta in self._scan_prefix(self.etcd._mangle_key("capacity/")):
            parts = self.etcd._demangle_key(meta.key).split("/")
            if len(parts) == 2 and parts[0] == "capacity":
                yield parts[1], raw

    def _semaphore_items(self) -> Iterator[tuple[list[str], bytes]]:
        for raw, meta in self._scan_prefix(self.etcd._mangle_key("semaphores/")):
            parts = self.etcd._demangle_key(meta.key).split("/")
            if len(parts) >= 3 and parts[0] == "semaphores":
                yield parts, raw

    def list_slot_types(self) -> Set[str]:
        slots = {slot for slot, _ in self._capacity_items()}
        slots.update(parts[1] for parts, _ in self._semaphore_items())
        return slots

    def read_count_all(self) -> Dict[str, int]:
        used = {slot: 0 for slot, _ in self._capacity_items()}
        for parts, raw in self._semaphore_items():
            slot = parts[1]
            if len(parts) == 3 and parts[2] == "count":
                used[slot] = _reported_count(raw)
            else:
                used.setdefault(slot, 0)
        return used

    def holder_counts_all(self) -> Dict[str, int]:
        counts = {slot: 0 for slot, _ in self._capacity_items()}
        for parts, _ in self._semaphore_items():
            slot = parts[1]
            counts.setdefault(slot, 0)
            if len(parts) == 4 and parts[2] == "holders":
                counts[slot] += 1
        return counts

    def available_count_all(self) -> Dict[str, int]:
        available = {
            slot: max(0, doc.admission_limit)
            for slot, doc in self._capacity_docs_all().items()
        }
        for parts, raw in self._semaphore_items():
            if len(parts) == 3 and parts[2] == "count":
                slot = parts[1]
                used = max(0, _reported_count(raw))
                available[slot] = max(0, available.get(slot, 0) - used)
        return available

    def list_holders_all(self) -> Dict[str, Dict[str, SemaphoreHolder]]:
        result: Dict[str, Dict[str, SemaphoreHolder]] = {
            slot: {} for slot, _ in self._capacity_items()
        }
        for parts, raw in self._semaphore_items():
            holders = result.setdefault(parts[1], {})
            if len(parts) == 4 and parts[2] == "holders":
                try:
                    holder = SemaphoreHolder.from_json(raw)
                except Exception:
                    continue
                holders[parts[3]] = holder
        return result

    def capacities_all(self) -> Dict[str, int]:
        return {
            slot: max(0, doc.limit) for slot, doc in self._capacity_docs_all().items()
        }

    def _capacity_docs_all(self) -> Dict[str, CapacityDoc]:
        return {
            slot: _decode_capacity(raw) or CapacityDoc(limit=0)
            for slot, raw in self._capacity_items()
        }

    def snapshot_all(self, include_holders: bool = False) -> Dict[str, dict]:
        caps = self._capacity_docs_all()
        all_slots = set(caps)

        used: Dict[str, int] = {}
        holders_map: Dict[str, Dict[str, SemaphoreHolder]] = {}
        holder_counts: Dict[str, int] = {}

        for parts, val_bytes in self._semaphore_items():
            st = parts[1]
            all_slots.add(st)
            if len(parts) == 3 and parts[2] == "count":
                used[st] = _reported_count(val_bytes)
                continue

            if len(parts) == 4 and parts[2] == "holders":
                ticket_id = parts[3]
                holder_counts[st] = holder_counts.get(st, 0) + 1
                if include_holders:
                    try:
                        holder = SemaphoreHolder.from_json(val_bytes)
                        holders_map.setdefault(st, {})[ticket_id] = holder
                    except Exception:
                        pass

        out: Dict[str, dict] = {}
        for st in sorted(all_slots):
            doc = caps.get(st, CapacityDoc(limit=0))
            cap = max(0, doc.limit)
            u = max(0, used.get(st, 0))
            avail = max(0, doc.admission_limit - u)
            hc = holder_counts.get(st, 0)
            entry = {
                "capacity": cap,
                "target": max(0, doc.admission_limit),
                "used": u,
                "available": avail,
                "holder_count": hc,
            }
            if include_holders:
                entry["holders"] = holders_map.get(st, {})
            out[st] = entry

        return out

    def reconcile(self, slot_type: str, *, max_retries: int = 8) -> int:
        count, _reconciled, _attempts = self._reconcile_count(
            slot_type,
            max_retries=max_retries,
        )
        return count

    def _reconcile_count(
        self,
        slot_type: str,
        *,
        max_retries: int,
    ) -> tuple[int, bool, int]:
        cnt_k = _count_key(slot_type)
        retries = max(1, int(max_retries))

        for attempt in range(retries):
            attempts = attempt + 1
            cnt_raw = self._get_raw(cnt_k)
            current = _decode_count(cnt_raw)
            results = self._scan_prefix(
                self.etcd._mangle_key(_holders_prefix(slot_type))
            )
            live = sum(1 for _value, _metadata in results)

            if current is not None and current == live:
                return current, True, attempts

            transaction = self.etcd.txn()
            if cnt_raw is None:
                transaction.if_missing(cnt_k)
            else:
                transaction.if_value(cnt_k, "==", cnt_raw)
            transaction.put(cnt_k, str(live))
            committed, _response = transaction.commit()
            if committed:
                return live, True, attempts
            if attempt < retries - 1:
                _retry_after_contention(attempt)

        current = _decode_count(self._get_raw(cnt_k))
        return (current if current is not None else 0), False, retries

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
              "malformed_holders": int,
              "counter_reconciled": bool,
              "reconcile_attempts": int
          }, ...}
        """
        summary: Dict[str, dict] = {}
        slot_types = self.list_slot_types()

        for st in sorted(slot_types):
            holders_prefix = _holders_prefix(st)
            mangled_prefix = self.etcd._mangle_key(holders_prefix)

            # 1) Scan holders and detect orphans
            orphans: list[tuple[str, int]] = []
            malformed = 0

            scan_iter = self._scan_prefix(mangled_prefix)
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
                    orphans.append((key, meta.mod_revision))

            # 2) Delete orphan holder keys (best-effort) — no counter math here;
            # we'll recompute and CAS the counter in step (4)
            deleted_orphans = 0
            if delete_orphan_holders:
                for key, revision in orphans:
                    try:
                        txn = self.etcd.txn()
                        txn.if_mod_revision(key, "==", revision)
                        txn.delete(key)
                        deleted, _ = txn.commit()
                        if deleted:
                            deleted_orphans += 1
                    except Exception:
                        # ignore individual delete failures
                        pass

            cnt_k = _count_key(st)
            cnt_raw = self._get_raw(cnt_k)
            before_count = _decode_count(cnt_raw)
            after_count = before_count if before_count is not None else -1

            if fix_counters:
                after_count, counter_reconciled, reconcile_attempts = (
                    self._reconcile_count(st, max_retries=8)
                )
            else:
                live_count = sum(
                    1 for _value, _metadata in self._scan_prefix(mangled_prefix)
                )
                counter_reconciled = before_count == live_count
                reconcile_attempts = 0

            summary[st] = {
                "before_count": before_count if before_count is not None else -1,
                "after_count": after_count,
                "deleted_orphans": deleted_orphans,
                "malformed_holders": malformed,
                "counter_reconciled": counter_reconciled,
                "reconcile_attempts": reconcile_attempts,
            }

        return summary
