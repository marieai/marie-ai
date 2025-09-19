import time
from collections import defaultdict, deque

from marie.query_planner.base import QueryPlan
from marie.scheduler.models import WorkInfo


class MemoryFrontier:
    def __init__(self):
        # job graph / indices
        self.jobs_by_id: dict[str, WorkInfo] = {}
        self.dag_nodes: dict[str, set[str]] = defaultdict(set)
        self.dependents: dict[str, list[str]] = defaultdict(list)  # parent -> children
        self.unmet_count: dict[str, int] = defaultdict(int)
        # ready queues per executor (protocol before ://)
        self.ready_by_executor: dict[str, deque[WorkInfo]] = defaultdict(deque)
        # local soft-leases (seconds since epoch)
        self.leased_until: dict[str, float] = {}

    @staticmethod
    def _executor_for(wi: WorkInfo) -> str:
        ep = wi.data.get("metadata", {}).get("on", "")
        return ep.split("://", 1)[0] if "://" in ep else ""

    def add_dag(self, dag: QueryPlan, nodes: list[WorkInfo]) -> None:
        # Build edges & unmet dep counts
        id_to_wi = {wi.id: wi for wi in nodes}
        for wi in nodes:
            self.jobs_by_id[wi.id] = wi
            self.dag_nodes[wi.dag_id].add(wi.id)
            deps = wi.dependencies if wi.dependencies else []
            self.unmet_count[wi.id] = len(deps)
            for d in deps:
                self.dependents[d].append(wi.id)

        # Seed roots into ready queues
        for wi in nodes:
            if self.unmet_count[wi.id] == 0:
                exe = self._executor_for(wi)
                if exe:
                    self.ready_by_executor[exe].append(wi)

    def on_job_completed(self, job_id: str) -> list[WorkInfo]:
        """Return newly-ready children to assist tests if needed."""
        now_ready = []
        for child_id in self.dependents.get(job_id, []):
            if child_id not in self.jobs_by_id:
                continue
            self.unmet_count[child_id] = max(0, self.unmet_count[child_id] - 1)
            if self.unmet_count[child_id] == 0 and child_id not in self.leased_until:
                wi = self.jobs_by_id[child_id]
                exe = self._executor_for(wi)
                if exe:
                    self.ready_by_executor[exe].append(wi)
                    now_ready.append(wi)
        return now_ready

    def on_job_failed(self, job_id: str) -> list[WorkInfo]:
        """Conservative policy: block descendants (don’t enqueue)."""
        # If we want "fail-fast" across the DAG, we can mark descendants as permanently blocked here.
        return []

    def mark_leased(self, job_id: str, ttl_s: int = 5) -> None:
        self.leased_until[job_id] = time.time() + ttl_s

    def release_lease_local(self, job_id: str) -> None:
        # Put the job back to the appropriate ready queue if it still exists
        if job_id in self.leased_until:
            del self.leased_until[job_id]
        wi = self.jobs_by_id.get(job_id)
        if wi is None:
            return
        # Only requeue if its deps are still satisfied
        if self.unmet_count.get(job_id, 1) == 0:
            exe = self._executor_for(wi)
            if exe:
                self.ready_by_executor[exe].appendleft(wi)  # prioritize requeue

    def _pop_one_for_executor(self, exe: str) -> tuple[str, WorkInfo] | None:
        dq = self.ready_by_executor.get(exe)
        if not dq:
            return None

        # Iterate through the current contents of the deque once (bounded iteration).
        for _ in range(len(dq)):
            wi = dq.popleft()

            #  If the job is still soft-leased, put it back at the end of the queue.
            if wi.id in self.leased_until and self.leased_until[wi.id] > time.time():
                dq.append(wi)
                continue

            # If not leased or lease expired:
            self.leased_until.pop(wi.id, None)

            # Check if the job has a valid execution endpoint.
            ep = wi.data.get("metadata", {}).get("on", "")
            if not ep:
                # If there's no endpoint, this WorkInfo is malformed or not meant for this kind of executor.
                # This indicates a data integrity issue. Raising an error highlights this immediately.
                raise ValueError(f'Job with no defined endpoint : {wi.id}')

            return ep, wi

        return None

    def pop_ready_batch(
        self, slots_by_executor: dict[str, int], max_n: int
    ) -> list[tuple[str, WorkInfo]]:
        picked: list[tuple[str, WorkInfo]] = []
        if max_n <= 0:
            return picked

        # simple round-robin across executors honoring available slots
        executors = [exe for exe, slots in slots_by_executor.items() if slots > 0]
        made_progress = True
        while len(picked) < max_n and executors and made_progress:
            made_progress = False
            for exe in list(executors):
                if slots_by_executor.get(exe, 0) <= 0:
                    continue
                item = self._pop_one_for_executor(exe)
                if item is None:
                    continue
                picked.append(item)
                slots_by_executor[exe] -= 1
                made_progress = True
                if len(picked) >= max_n:
                    break
        return picked
