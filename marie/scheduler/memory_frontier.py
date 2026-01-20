import asyncio
import heapq
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, NamedTuple, Optional

from marie.logging_core.predefined import default_logger as logger
from marie.query_planner.base import QueryPlan
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState


class ReadyEntry(NamedTuple):
    # Field order defines heap ordering
    key: tuple[int, int]  # (-job_level, ±priority)
    added_at: float  # monotonic timestamp when first ready
    seq: int  # FIFO tie-breaker
    ver: int  # version to invalidate stale entries
    jid: str  # job id


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
        self.parents: dict[str, list[str]] = defaultdict(list)  # child -> parents
        self.unmet_count: dict[str, int] = defaultdict(int)

        # THIS HAS TO BE KEPT IN SYNC WITH THE GlobalPriorityExecutionPlanner
        self._ready_heap: list[ReadyEntry] = []

        self._ready_set: set[str] = set()  # fast membership / lazy deletion
        self._added_at: dict[str, float] = {}  # first time job became ready
        self._seq = 0  # tie breaker to keep heap stable
        self._ver: dict[str, int] = defaultdict(int)

        # Soft leases: job_id -> unix expiry seconds
        self.leased_until: dict[str, float] = {}

        self.higher_priority_wins = higher_priority_wins
        self.default_lease_ttl = float(default_lease_ttl)

        self._lock = asyncio.Lock()
        self._now = time.monotonic

    def _priority_key(self, wi: WorkInfo) -> tuple[int, int]:
        lvl = int(wi.job_level)
        pri = int(wi.priority)
        return (-lvl, (-pri if self.higher_priority_wins else pri))

    def _push_ready(self, wi: WorkInfo) -> None:
        # Always bump version when (re)adding to ready
        self._ver[wi.id] += 1
        v = self._ver[wi.id]
        self._ready_set.add(wi.id)
        if wi.id not in self._added_at:
            self._added_at[wi.id] = self._now()  # was time.time()
        self._seq += 1
        heapq.heappush(
            self._ready_heap,
            ReadyEntry(
                self._priority_key(wi), self._added_at[wi.id], self._seq, v, wi.id
            ),
        )

    def _entry_is_current(self, entry: ReadyEntry) -> bool:
        return self._ver.get(entry.jid, 0) == entry.ver

    def _remove_from_ready_set(self, job_id: str) -> None:
        self._ready_set.discard(job_id)

    def _still_ready(self, job_id: str) -> bool:
        # Ready if: exists, unmet deps == 0, not soft-leased, in ready_set, and past start_after
        if job_id not in self.jobs_by_id:
            return False
        wi = self.jobs_by_id[job_id]
        if self.unmet_count.get(job_id, 1) != 0:
            return False
        if job_id not in self._ready_set:
            return False
        # Check start_after time (for retry delays)
        if wi.start_after and wi.start_after > datetime.now(timezone.utc):
            return False
        # if soft-leased and not expired, don't expose it
        if job_id in self.leased_until and self.leased_until[job_id] > self._now():
            return False
        return True

    @staticmethod
    def _entrypoint(wi: WorkInfo) -> str:
        md = {}
        if isinstance(wi.data, dict):
            md = (
                wi.data.get("metadata", {})
                if isinstance(wi.data.get("metadata", {}), dict)
                else {}
            )
        return md.get("on", "")

    async def add_dag(self, dag: 'QueryPlan', nodes: list[WorkInfo]) -> None:
        """Adds a new DAG to the frontier, building the dependency graph."""
        async with self._lock:
            # Build graph + unmet dependency counts
            for wi in nodes:
                self.jobs_by_id[wi.id] = wi
                self.dag_nodes[wi.dag_id].add(wi.id)
                deps = wi.dependencies or []
                self.unmet_count[wi.id] = len(deps)
                for d in deps:
                    self.dependents[d].append(wi.id)  # parent -> child
                    self.parents[wi.id].append(d)  # child  -> parent

            # Seed roots into ready heap
            for wi in nodes:
                if self.unmet_count[wi.id] == 0:
                    self._push_ready(wi)

    async def get_jobs_by_dag_id(self, dag_id: str) -> list[WorkInfo]:
        """
        Returns all WorkInfo objects associated with a given DAG ID.
        """
        async with self._lock:
            node_ids = self.dag_nodes.get(dag_id, set())
            return [
                self.jobs_by_id[job_id]
                for job_id in node_ids
                if job_id in self.jobs_by_id
            ]

    async def update_job_state(self, job_id: str, state: WorkState) -> bool:
        """Updates the state of a single job in the frontier."""
        async with self._lock:
            if job_id in self.jobs_by_id:
                self.jobs_by_id[job_id].state = state
                return True
            logger.warning(
                f"Job with id {job_id} not found in memory frontier for state update."
            )
            return False

    async def on_job_completed(self, job_id: str) -> list[WorkInfo]:
        """
        Marks a job as complete, decrements the dependency count of its children,
        and moves any newly ready children to the ready heap.
        """
        async with self._lock:
            # we do this to ensure that the job is marked appropriately when job completes
            # this will allow us to interrogate the state of the job later
            # Update the job's state to COMPLETED
            if job_id in self.jobs_by_id:
                self.jobs_by_id[job_id].state = WorkState.COMPLETED
            else:
                logger.warning(
                    f"Job with id {job_id} not found in memory frontier for completion."
                )

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
                        self._added_at[child_id] = self._now()
                    self._push_ready(wi)
                    now_ready.append(wi)
            return now_ready

    async def on_job_failed(self, job_id: str) -> list[WorkInfo]:
        """
        Handles permanent job failure. Updates state to FAILED and does not
        re-add to ready queue.
        """
        async with self._lock:
            # Conservative: don't enqueue descendants automatically
            # Update the job's state to FAILED
            if job_id in self.jobs_by_id:
                self.jobs_by_id[job_id].state = WorkState.FAILED
                self._remove_from_ready_set(job_id)
            else:
                logger.warning(
                    f"Job with id {job_id} not found in memory frontier for failure."
                )

            return []

    async def on_job_retry(self, job_id: str, work_item: WorkInfo) -> None:
        """
        Handles job marked for retry. Updates state to RETRY, calculates
        start_after from retry_delay, and re-adds to ready queue.

        :param job_id: The ID of the job to retry
        :param work_item: The WorkInfo containing retry configuration
        """
        async with self._lock:
            if job_id not in self.jobs_by_id:
                logger.warning(f"Job {job_id} not found in memory frontier for retry")
                return

            wi = self.jobs_by_id[job_id]
            wi.state = WorkState.RETRY

            # Calculate start_after based on retry_delay
            # Default to 2 seconds if retry_delay is not set
            retry_delay_seconds = work_item.retry_delay if work_item.retry_delay else 2
            wi.start_after = datetime.now(timezone.utc) + timedelta(
                seconds=retry_delay_seconds
            )

            # Re-add to ready queue
            # The _still_ready check will handle the start_after delay
            self._push_ready(wi)
            logger.info(
                f"Job {job_id} re-added to ready queue for retry "
                f"(retry_delay={retry_delay_seconds}s, start_after={wi.start_after})"
            )

    async def mark_leased(self, job_id: str, ttl_s: Optional[float] = None) -> None:
        """Applies a soft lease to a job to prevent re-scheduling."""
        async with self._lock:
            ttl = self.default_lease_ttl if ttl_s is None else float(ttl_s)
            now = self._now()
            self.leased_until[job_id] = now + ttl

    async def release_lease_local(self, job_id: str) -> None:
        """
        Releases a soft lease on a job, potentially re-adding it to the ready
        heap if its dependencies are still met.
        """
        async with self._lock:
            # Soft-lease ends; if deps still met, place back (at front by age)
            self.leased_until.pop(job_id, None)
            wi = self.jobs_by_id.get(job_id)
            if wi is None:
                return
            if self.unmet_count.get(job_id, 1) == 0:
                # keep original added_at to preserve aging
                self._push_ready(wi)

    async def peek_ready(self, max_n: int, filter_fn=None) -> list[WorkInfo]:
        """Return up to max_n in true heap order without mutating state."""
        async with self._lock:
            if max_n <= 0 or not self._ready_heap:
                return []

            stale_seen = 0

            def eligible_items():
                nonlocal stale_seen
                seen: set[str] = set()
                for entry in self._ready_heap:
                    if not self._entry_is_current(entry) or (
                        entry.jid not in self._ready_set
                    ):
                        stale_seen += 1
                        continue
                    if entry.jid in seen:
                        continue
                    if not self._still_ready(entry.jid):
                        continue
                    wi = self.jobs_by_id.get(entry.jid)
                    if wi is None or (filter_fn and not filter_fn(wi)):
                        continue
                    seen.add(entry.jid)
                    yield entry

            top = heapq.nsmallest(max_n, eligible_items())
            if stale_seen > 1024:
                await self.compact_ready_heap(max_scan=len(self._ready_heap))
            return [self.jobs_by_id[e.jid] for e in top]

    async def take(
        self,
        ids: Iterable[str],
        *,
        lease_ttl: Optional[float] = None,
    ) -> list[WorkInfo]:
        """
        Atomically mark the given ids as selected and return the corresponding WorkInfos.

        Behavior:
          - Only takes items that are *still ready* (deps met, not hard/soft leased, in ready_set).
          - Removes them from the ready_set so they won't be re-peeked.
          - Applies a soft lease (TTL) so parallel poll loops don’t double-schedule.
          - Returns the list of WorkInfos that were actually taken (in the same order as `ids`).
        """
        async with self._lock:
            taken: list[WorkInfo] = []
            now = self._now()  # was time.time()
            ttl = self.default_lease_ttl if lease_ttl is None else float(lease_ttl)
            for jid in ids:
                if not self._still_ready(jid):
                    continue
                self._ready_set.discard(jid)
                self.leased_until[jid] = now + ttl
                wi = self.jobs_by_id.get(jid)
                if wi is not None:
                    taken.append(wi)
            return taken

    async def select_ready(
        self,
        max_n: int,
        *,
        filter_fn: Optional[Callable[[WorkInfo], bool]] = None,
        lease_ttl: Optional[float] = None,
        scan_budget: int = 4096,  # prevents pathological pops
    ) -> list[WorkInfo]:
        """Pop in heap order; skip & restore non-eligible; soft-lease selected."""
        async with self._lock:
            if max_n <= 0 or not self._ready_heap:
                return []
            now = self._now()  # was time.time()
            ttl = self.default_lease_ttl if lease_ttl is None else float(lease_ttl)
            selected: list[WorkInfo] = []
            skipped: list[ReadyEntry] = []
            scans = 0

            while len(selected) < max_n and self._ready_heap and scans < scan_budget:
                scans += 1
                entry = heapq.heappop(self._ready_heap)
                if not self._entry_is_current(entry):
                    continue
                if not self._still_ready(entry.jid):
                    continue
                wi = self.jobs_by_id.get(entry.jid)
                if wi is None or (filter_fn and not filter_fn(wi)):
                    skipped.append(entry)
                    continue
                selected.append(wi)
                self._ready_set.discard(entry.jid)
                self.leased_until[entry.jid] = now + ttl

            for it in skipped:
                if self._entry_is_current(it) and (it.jid in self._ready_set):
                    heapq.heappush(self._ready_heap, it)
            return selected

    async def reap_expired_soft_leases(self) -> int:
        async with self._lock:
            now = self._now()  # was time.time()
            reap = [jid for jid, until in self.leased_until.items() if until <= now]
            for jid in reap:
                self.leased_until.pop(jid, None)
                wi = self.jobs_by_id.get(jid)
                if wi and self.unmet_count.get(jid, 1) == 0:
                    self._push_ready(wi)
            return len(reap)

    @staticmethod
    def _executor_of(wi: WorkInfo) -> str:
        ep = wi.data.get("metadata", {}).get("on", "")
        if "://" in ep:
            return ep.split("://", 1)[0]
        return ""

    def summary(self, detail: bool = False, top_n: int = 5) -> dict[str, Any]:
        """
        Lightweight snapshot of the frontier for logs/metrics.
        Returns a dict; safe to json-serialize.
        """
        now = self._now()

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

    async def finalize_dag(self, dag_id: str) -> dict[str, int]:
        """
        Permanently remove all in-memory state for a completed DAG.
        Safe to call multiple times; no-op if unknown.
        Returns counts of removed items for metrics.
        """

        async with self._lock:
            removed_jobs = 0
            removed_edges = 0

            node_ids = self.dag_nodes.pop(dag_id, None)
            if not node_ids:
                return {"removed_jobs": 0, "removed_edges": 0, "heap_compacted": 0}

            # for each job in DAG, remove graph edges and per-job state
            for jid in list(node_ids):
                # (a) remove as PARENT -> children
                children = self.dependents.pop(jid, [])
                removed_edges += len(children)
                # Each child had jid as a parent — remove that back-link
                for ch in children:
                    if ch in self.parents:
                        try:
                            lst = self.parents[ch]
                            # remove jid if present
                            idx = lst.index(jid)
                            lst.pop(idx)
                            removed_edges += 1
                            if not lst:
                                self.parents.pop(ch, None)
                        except ValueError:
                            pass

                # (b) remove as CHILD -> parents
                plist = self.parents.pop(jid, [])
                for p in plist:
                    cl = self.dependents.get(p)
                    if cl:
                        try:
                            idx = cl.index(jid)
                            cl.pop(idx)
                            removed_edges += 1
                            if not cl:
                                # optional: keep empty list or prune
                                self.dependents.pop(p, None)
                        except ValueError:
                            pass

                # (c) purge per-job state
                self.jobs_by_id.pop(jid, None)
                self._ready_set.discard(jid)
                self._added_at.pop(jid, None)
                self.leased_until.pop(jid, None)
                self.unmet_count.pop(jid, None)
                self._ver.pop(jid, None)
                removed_jobs += 1

            # heap rebuild
            compacted = await self.compact_ready_heap(full=True)

            return {
                "removed_jobs": removed_jobs,
                "removed_edges": removed_edges,
                "heap_compacted": compacted,
            }

    async def compact_ready_heap(
        self, max_scan: int = 5000, *, full: bool = False
    ) -> int:
        """
        Compact the ready heap by removing outdated or unnecessary entries.

        This method ensures the integrity of the ready heap by validating entries
        against current job IDs and versions. If `full` is set to True, all entries
        are scanned and only valid ones are retained. Otherwise, the scan is limited
        by the `max_scan` parameter.

        :param max_scan: The maximum number of entries to scan when `full` is False.
        :param full: If True, performs a full scan of the heap; otherwise, performs
            a partial scan limited by `max_scan`.
        :return: The number of entries removed from the heap.
        """

        if full:
            kept: list[ReadyEntry] = []
            for entry in self._ready_heap:
                if (entry.jid in self._ready_set) and (
                    self._ver.get(entry.jid, 0) == entry.ver
                ):
                    kept.append(entry)

            removed = len(self._ready_heap) - len(kept)
            heapq.heapify(kept)
            self._ready_heap = kept
            return removed

        removed = 0
        if not self._ready_heap:
            return 0
        tmp: list[ReadyEntry] = []
        scans = 0
        while self._ready_heap and scans < max_scan:
            scans += 1
            entry = heapq.heappop(self._ready_heap)
            if (entry.jid in self._ready_set) and self._entry_is_current(entry):
                tmp.append(entry)
            else:
                removed += 1
        for e in tmp:
            heapq.heappush(self._ready_heap, e)
        return removed
