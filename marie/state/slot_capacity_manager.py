from __future__ import annotations

import os
from typing import Callable, Dict, List, Mapping, Optional, Set

from marie.logging_core.logger import MarieLogger
from marie.state.semaphore_store import SemaphoreStore


class SlotCapacityManager:
    """
    Reconciles semaphore capacities with discovered nodes.

    - Derive capacity targets from {executor -> [nodes]} with per-node policy
    - Safely set capacities (CAS; never below current usage)
    - Print a compact summary
    - Optionally clamp absent executors down to current usage

    Env overrides:
      MARIE_SLOTS_PER_NODE=<int>                 (default=1)
      MARIE_SLOTS_<EXECUTOR>_PER_NODE=<int>      (per-executor)
      MARIE_CAPACITY_ZERO_ON_ABSENT=true|false   (default=false)
    """

    def __init__(
        self,
        semaphore_store: SemaphoreStore,
        logger: Optional[MarieLogger] = None,
        *,
        slot_type_resolver: Optional[Callable[[str], str]] = None,
    ):
        self.sem = semaphore_store
        if logger is None:
            logger = MarieLogger(SlotCapacityManager.__name__)
        self.log = logger
        self.slot_type_resolver = slot_type_resolver or (lambda exec_name: exec_name)
        self.default_slots_per_node_env = "MARIE_SLOTS_PER_NODE"
        self.zero_absent_env = "MARIE_CAPACITY_ZERO_ON_ABSENT"

    def refresh_from_nodes(
        self,
        deployment_nodes: Mapping[str, List[dict]],
    ) -> None:
        """
        Reconcile capacities for all executors present in `deployment_nodes`.
        Idempotent. Safe to call frequently.

        deployment_nodes: { executor: [ { "address": "...", "gateway": "...", ... }, ... ] }
        """
        targets = self._capacity_targets_from_nodes(deployment_nodes)

        # 1) ensure present executors meet target (clamped to used) via CAS
        for slot_type, limit in targets.items():
            self._safe_set_capacity(slot_type, limit)

        # 2) optionally clamp capacities for executors no longer present
        if self._zero_absent_enabled():
            known = self.sem.list_slot_types()
            missing = known - set(targets.keys())
            for slot_type in missing:
                used = max(0, int(self.sem.read_count(slot_type)))
                self._safe_set_capacity(slot_type, used)

        # 3) print a concise summary
        self.print_summary(targets=targets)

    def print_summary(self, targets: Optional[Dict[str, int]] = None) -> None:
        """
        Log a table of current slot state: SLOT | CAPACITY | TARGET | USED | AVAIL | HOLDERS | NOTES
        """
        targets = targets or {}
        snapshot = self.sem.snapshot_all(include_holders=False)

        if not snapshot:
            self.log.info("[capacity] No slots discovered.")
            return

        rows = []
        totals = {"capacity": 0, "used": 0, "available": 0, "holder_count": 0}
        for slot in sorted(snapshot.keys()):
            info = snapshot[slot]
            cap = int(info.get("capacity", 0))
            used = int(info.get("used", 0))
            avail = int(info.get("available", 0))
            holders = int(info.get("holder_count", 0))
            tgt = int(targets.get(slot, cap))

            notes = []
            if tgt < used and cap == used:
                notes.append("CLAMPED")
            if used != holders:
                notes.append("COUNT≠HOLDERS")

            totals["capacity"] += cap
            totals["used"] += used
            totals["available"] += avail
            totals["holder_count"] += holders

            rows.append((slot, cap, tgt, used, avail, holders, ", ".join(notes)))

        headers = ("SLOT", "CAPACITY", "TARGET", "USED", "AVAIL", "HOLDERS", "NOTES")
        widths = [
            max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))
        ]

        def fmt(row):
            return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row)))

        header = fmt(headers)
        sep = "-+-".join("-" * w for w in widths)
        body = "\n".join(fmt(r) for r in rows)
        total_row = fmt(
            (
                "TOTAL",
                totals["capacity"],
                "",
                totals["used"],
                totals["available"],
                totals["holder_count"],
                "",
            )
        )

        self.log.info(
            "\n".join(["[capacity] Slot summary:", header, sep, body, sep, total_row])
        )

    # ----------------- Internals -----------------

    def _zero_absent_enabled(self) -> bool:
        v = os.environ.get(self.zero_absent_env, "false").lower()
        return v in ("1", "true", "yes", "y")

    def _slots_per_node(self, executor: str) -> int:
        # global default
        try:
            base = int(os.environ.get(self.default_slots_per_node_env, "1"))
        except Exception:
            base = 1

        # per-executor override
        key = f"MARIE_SLOTS_{executor.upper()}_PER_NODE"
        try:
            specific = os.environ.get(key)
            if specific is not None:
                return max(0, int(specific))
        except Exception:
            pass
        return max(0, base)

    def _capacity_targets_from_nodes(
        self, deployment_nodes: Mapping[str, List[dict]]
    ) -> Dict[str, int]:
        """
        Capacity = (# unique addresses for that executor) * slots_per_node(executor),
        then mapped to slot_type via resolver.
        """
        targets: Dict[str, int] = {}
        for executor, nodes in (deployment_nodes or {}).items():
            unique_hosts: Set[str] = set()
            for n in nodes or []:
                addr = (n or {}).get("address") or ""
                hp = self._netloc(addr)
                if hp:
                    unique_hosts.add(hp)
            slots_per = self._slots_per_node(executor)
            slot_type = self.slot_type_resolver(executor)
            targets[slot_type] = max(0, len(unique_hosts) * slots_per)
        return targets

    def _safe_set_capacity(self, slot_type: str, new_limit: int) -> int:
        """
        Set capacity with clamp-to-used using SemaphoreStore.set_capacity_safe (CAS).
        """
        try:
            # CAS + invariant (never below used) handled in store
            return int(self.sem.set_capacity_safe(slot_type, int(new_limit)))
        except Exception as e:
            # This is considered non-fatal; we log and continue so other slots can update.
            self.log.warning(
                f"[capacity] set_capacity_safe failed for {slot_type}: {e}"
            )
            # As a last resort, do a local clamp and raw write to avoid stuck capacity.
            used = max(0, int(self.sem.read_count(slot_type)))
            target = max(used, int(new_limit))
            self.sem.set_capacity(slot_type, target)
            return target

    @staticmethod
    def _netloc(addr: str) -> str:
        """
        Accepts 'grpc://host:port' or 'host:port' and returns 'host:port'.
        """
        if "://" in addr:
            try:
                from urllib.parse import urlparse

                p = urlparse(addr)
                return p.netloc or addr
            except Exception:
                return addr
        return addr
