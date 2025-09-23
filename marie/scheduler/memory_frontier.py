import asyncio
import heapq
import time
from collections import defaultdict
from typing import Any, Callable, Iterable, Optional

from marie.query_planner.base import QueryPlan
from marie.scheduler.models import WorkInfo


class MemoryFrontier:
    """
    In-memory DAG frontier with:
      - Global ready pool (priority + age fairness) using a min-heap
      - Atomic peek/take (asyncio.Lock)
      - Soft-leases to avoid duplicate scheduling within this process
    """

    def __init__(
        self, *, higher_priority_wins: bool = True, default_lease_ttl: float = 5.0
    ):
        # Job graph / indices
        self.jobs_by_id: dict[str, WorkInfo] = {}
        self.dag_nodes: dict[str, set[str]] = defaultdict(set)
        self.dependents: dict[str, list[str]] = defaultdict(list)  # parent -> children
        self.unmet_count: dict[str, int] = defaultdict(int)

        # THIS HAS TO BE KEPT IN SYNC WITH THE GlobalPriorityExecutionPlanner
        # Heap entries: ((-job_level, -priority), added_at, seq, job_id)
        self._ready_heap: list[tuple[tuple[int, int], float, int, str]] = []

        self._ready_set: set[str] = set()  # fast membership / lazy deletion
        self._added_at: dict[str, float] = {}  # first time job became ready
        self._seq = 0  # tie breaker to keep heap stable

        # Soft leases: job_id -> unix expiry seconds
        self.leased_until: dict[str, float] = {}

        self.higher_priority_wins = higher_priority_wins
        self.default_lease_ttl = float(default_lease_ttl)

        self._lock = asyncio.Lock()

    def _priority_key(self, wi: WorkInfo) -> tuple[int, int]:
        lvl = int(wi.job_level)
        pri = int(wi.priority)
        return (-lvl, (-pri if self.higher_priority_wins else pri))

    def _push_ready(self, wi: WorkInfo) -> None:
        if wi.id in self._ready_set:
            return
        self._ready_set.add(wi.id)
        if wi.id not in self._added_at:
            self._added_at[wi.id] = time.time()
        self._seq += 1
        heapq.heappush(
            self._ready_heap,
            (self._priority_key(wi), self._added_at[wi.id], self._seq, wi.id),
        )

    def _remove_from_ready_set(self, job_id: str) -> None:
        self._ready_set.discard(job_id)

    def _still_ready(self, job_id: str) -> bool:
        # Ready if: exists, unmet deps == 0, not hard-leased, and in ready_set
        if job_id not in self.jobs_by_id:
            return False
        if self.unmet_count.get(job_id, 1) != 0:
            return False
        if job_id not in self._ready_set:
            return False
        # if soft-leased and not expired, don't expose it
        if job_id in self.leased_until and self.leased_until[job_id] > time.time():
            return False
        return True

    @staticmethod
    def _entrypoint(wi: 'WorkInfo') -> str:
        # Planner can read this if it needs to; we don't filter by executor here.
        return wi.data.get("metadata", {}).get("on", "")

    def add_dag(self, dag: 'QueryPlan', nodes: list['WorkInfo']) -> None:
        # Build graph + unmet dependency counts
        id_to_wi = {wi.id: wi for wi in nodes}
        for wi in nodes:
            self.jobs_by_id[wi.id] = wi
            self.dag_nodes[wi.dag_id].add(wi.id)
            deps = wi.dependencies or []
            self.unmet_count[wi.id] = len(deps)
            for d in deps:
                self.dependents[d].append(wi.id)

        # Seed roots into ready heap
        for wi in nodes:
            if self.unmet_count[wi.id] == 0:
                self._push_ready(wi)

    def on_job_completed(self, job_id: str) -> list['WorkInfo']:
        """Update children; return newly-ready WorkInfos (for tests/metrics)."""
        now_ready = []
        for child_id in self.dependents.get(job_id, []):
            if child_id not in self.jobs_by_id:
                continue
            # decrease unmet count
            self.unmet_count[child_id] = max(0, self.unmet_count[child_id] - 1)
            if self.unmet_count[child_id] == 0:
                wi = self.jobs_by_id[child_id]
                # if child wasn’t ready before, set added_at now for fair aging
                if child_id not in self._added_at:
                    self._added_at[child_id] = time.time()
                self._push_ready(wi)
                now_ready.append(wi)
        return now_ready

    def on_job_failed(self, job_id: str) -> list['WorkInfo']:
        # Conservative: don't enqueue descendants automatically
        return []

    def mark_leased(self, job_id: str, ttl_s: Optional[float] = None) -> None:
        ttl = self.default_lease_ttl if ttl_s is None else float(ttl_s)
        self.leased_until[job_id] = time.time() + ttl

    def release_lease_local(self, job_id: str) -> None:
        # Soft-lease ends; if deps still met, place back (at front by age)
        self.leased_until.pop(job_id, None)
        wi = self.jobs_by_id.get(job_id)
        if wi is None:
            return
        if self.unmet_count.get(job_id, 1) == 0:
            # keep original added_at to preserve aging
            self._push_ready(wi)

    async def peek_ready(
        self, max_n: int, filter_fn: Optional[Callable[['WorkInfo'], bool]] = None
    ) -> list['WorkInfo']:
        """Return up to max_n in true heap order without mutating state."""
        async with self._lock:
            if max_n <= 0 or not self._ready_heap:
                return []

            def eligible_items():
                for item in self._ready_heap:  # (key, added, seq, jid)
                    jid = item[3]
                    if not self._still_ready(jid):
                        continue
                    wi = self.jobs_by_id.get(jid)
                    if wi is None or (filter_fn and not filter_fn(wi)):
                        continue
                    yield item

            top = heapq.nsmallest(max_n, eligible_items())  # O(n log k)
            return [self.jobs_by_id[item[3]] for item in top]

    async def take(
        self,
        ids: Iterable[str],
        *,
        lease_ttl: Optional[float] = None,
    ) -> list['WorkInfo']:
        """
        Atomically mark the given ids as selected and return the corresponding WorkInfos.

        Behavior:
          - Only takes items that are *still ready* (deps met, not hard/soft leased, in ready_set).
          - Removes them from the ready_set so they won't be re-peeked.
          - Applies a soft lease (TTL) so parallel poll loops don’t double-schedule.
          - Returns the list of WorkInfos that were actually taken (in the same order as `ids`).
        """
        async with self._lock:
            taken: list['WorkInfo'] = []
            now = time.time()
            ttl = self.default_lease_ttl if lease_ttl is None else float(lease_ttl)

            for jid in ids:
                if not self._still_ready(jid):
                    continue  # skip stale/non-ready ids safely

                # remove from ready membership so future peeks skip it
                self._ready_set.discard(jid)

                # apply soft lease
                self.leased_until[jid] = now + ttl

                wi = self.jobs_by_id.get(jid)
                if wi is not None:
                    taken.append(wi)

            return taken

    async def select_ready(
        self,
        max_n: int,
        *,
        filter_fn: Optional[Callable[['WorkInfo'], bool]] = None,
        lease_ttl: Optional[float] = None,
        scan_budget: int = 4096,  # prevents pathological pops
    ) -> list['WorkInfo']:
        """Pop in heap order; skip & restore non-eligible; soft-lease selected."""
        async with self._lock:
            if max_n <= 0 or not self._ready_heap:
                return []
            now = time.time()
            ttl = self.default_lease_ttl if lease_ttl is None else float(lease_ttl)
            selected: list['WorkInfo'] = []
            skipped: list[tuple[tuple[int, int], float, int, str]] = []
            scans = 0
            while len(selected) < max_n and self._ready_heap and scans < scan_budget:
                scans += 1
                item = heapq.heappop(self._ready_heap)
                jid = item[3]
                if not self._still_ready(jid):
                    continue
                wi = self.jobs_by_id.get(jid)
                if wi is None or (filter_fn and not filter_fn(wi)):
                    skipped.append(item)
                    continue
                # select
                selected.append(wi)
                self._ready_set.discard(jid)
                self.leased_until[jid] = now + ttl
            # restore skipped
            for it in skipped:
                heapq.heappush(self._ready_heap, it)
            return selected

    async def reap_expired_soft_leases(self) -> int:
        """Optional: clear expired soft leases; return count re-added to ready."""
        async with self._lock:
            now = time.time()
            reap: list[str] = [
                jid for jid, until in self.leased_until.items() if until <= now
            ]
            for jid in reap:
                self.leased_until.pop(jid, None)
                wi = self.jobs_by_id.get(jid)
                if wi and self.unmet_count.get(jid, 1) == 0:
                    self._push_ready(wi)
            return len(reap)

    @staticmethod
    def _executor_of(wi: 'WorkInfo') -> str:
        ep = wi.data.get("metadata", {}).get("on", "")
        if "://" in ep:
            return ep.split("://", 1)[0]
        return ""

    def summary(self, detail: bool = False, top_n: int = 5) -> dict[str, Any]:
        """
        Lightweight snapshot of the frontier for logs/metrics.
        Returns a dict; safe to json-serialize.
        """
        now = time.time()

        # Totals
        total_jobs = len(self.jobs_by_id)
        total_dags = len(self.dag_nodes)
        total_edges = sum(len(v) for v in self.dependents.values())
        leased_total = len(self.leased_until)

        # Ready set (filter lazy-deleted / leased)
        ready_ids = [jid for jid in list(self._ready_set) if self._still_ready(jid)]
        ready_total = len(ready_ids)

        # Per-executor ready counts
        ready_by_exec: dict[str, int] = defaultdict(int)
        for jid in ready_ids:
            wi = self.jobs_by_id.get(jid)
            if not wi:
                continue
            ready_by_exec[self._executor_of(wi)] += 1

        # Unmet dependency stats
        unmet_vals = [self.unmet_count.get(jid, 0) for jid in self.jobs_by_id.keys()]
        unmet_nonzero = [v for v in unmet_vals if v > 0]

        # Ready age stats (seconds)
        ages = []
        for jid in ready_ids:
            at = self._added_at.get(jid)
            if at is not None:
                ages.append(max(0.0, now - at))
        ages.sort()

        def _quantiles(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"p50": 0.0, "p90": 0.0, "max": 0.0}

            def q(p: float) -> float:
                if len(vals) == 1:
                    return vals[0]
                idx = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
                return vals[idx]

            return {"p50": q(0.5), "p90": q(0.9), "max": vals[-1]}

        out: dict[str, Any] = {
            "totals": {
                "dags": total_dags,
                "jobs": total_jobs,
                "edges": total_edges,
                "ready": ready_total,
                "leased": leased_total,
                "blocked": len(unmet_nonzero),  # jobs with unmet deps > 0
            },
            "ready_by_executor": dict(ready_by_exec),
            "unmet_dependencies": {
                "count_nonzero": len(unmet_nonzero),
                "max": max(unmet_nonzero) if unmet_nonzero else 0,
                "avg": (
                    (sum(unmet_nonzero) / len(unmet_nonzero)) if unmet_nonzero else 0.0
                ),
            },
            "ready_age_seconds": _quantiles(ages),
        }

        if detail and ready_total:
            # Top-N stalest (oldest added_at) per executor
            per_exec_stale: dict[str, list[dict[str, Any]]] = {}
            ids_by_exec: dict[str, list[str]] = defaultdict(list)
            for jid in ready_ids:
                wi = self.jobs_by_id.get(jid)
                if not wi:
                    continue
                ids_by_exec[self._executor_of(wi)].append(jid)

            for exe, jids in ids_by_exec.items():
                jids.sort(key=lambda j: self._added_at.get(j, now))  # oldest first
                chosen = jids[: max(1, top_n)]
                entries = []
                for jid in chosen:
                    wi = self.jobs_by_id.get(jid)
                    if not wi:
                        continue
                    entries.append(
                        {
                            "id": wi.id,
                            "name": wi.name,
                            "priority": int(wi.priority),
                            "job_level": getattr(wi, "job_level", None),
                            "ready_for_s": max(0.0, now - self._added_at.get(jid, now)),
                        }
                    )
                per_exec_stale[exe] = entries
            out["stalest_ready"] = per_exec_stale

        return out

    def compact_ready_heap(self, max_scan: int = 5000) -> int:
        """
        Remove stale heap entries whose ids are no longer in _ready_set (lazy-deletion compaction).
        Returns number of entries removed.
        """
        removed = 0
        if not self._ready_heap:
            return removed

        tmp: list[tuple[tuple[int, int], float, int, str]] = []
        scans = 0
        while self._ready_heap and scans < max_scan:
            scans += 1
            key, added, seq, jid = heapq.heappop(self._ready_heap)
            if jid in self._ready_set:
                tmp.append((key, added, seq, jid))
            else:
                removed += 1

        for e in tmp:
            heapq.heappush(self._ready_heap, e)
        return removed
