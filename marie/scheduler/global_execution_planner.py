from math import inf
from typing import Any, Dict, List, Sequence, Tuple

from marie.scheduler.execution_planner import FlatJob


class GlobalPriorityExecutionPlanner:
    """
    Global ready-queue scheduler.
    Priority (within each runnable/blocked partition):
      1) deepest level first (critical path)
      2) higher user priority
      3) more free slots (still useful inside the runnable set)
      4) existing DAGs over new DAGs
      5) shorter estimated runtimes
      6) FIFO (original input order) as the final tiebreaker

    Notes:
      - We hard-gate jobs whose executor has no free slots so they don't block runnable work.
      - No getattr; direct field access only.
      - 'burst' logic removed.
    """

    def plan(
        self,
        jobs: Sequence[FlatJob],
        slots: Dict[str, int],
        active_dags: set[str],
    ) -> Sequence[FlatJob]:
        # (endpoint, wi, level, priority, free_slots, is_new, dag_id, est_runtime, fifo_idx)
        runnable: List[Tuple[str, FlatJob, int, int, int, bool, str, float, int]] = []
        blocked: List[Tuple[str, FlatJob, int, int, int, bool, str, float, int]] = []

        for idx, (endpoint, wi) in enumerate(jobs):
            executor = endpoint.split("://", 1)[0]
            free_slots = int(slots.get(executor, 0))

            level = wi.job_level
            priority = wi.priority
            dag_id = wi.dag_id
            is_new = dag_id not in active_dags

            meta = {}
            if wi.data is not None and isinstance(wi.data, dict):
                meta = wi.data.get("metadata", {}) or {}
            est = meta.get("estimated_runtime", None)
            rt = float(est) if est is not None else inf

            item = (endpoint, wi, level, priority, free_slots, is_new, dag_id, rt, idx)
            (runnable if free_slots > 0 else blocked).append(item)

        def sort_key(t):
            # Within partition (runnable first), prefer:
            # - deeper level, higher priority
            # - more free slots as a tie-breaker
            # - existing DAGs before new DAGs
            # - shorter runtimes
            # - FIFO by original order
            return (
                -t[2],  # level desc
                -t[3],  # priority desc
                -t[4],  # free_slots desc
                t[5],  # is_new (False < True)
                t[7],  # est_runtime asc
                t[8],  # fifo_idx asc
            )

        runnable.sort(key=sort_key)
        blocked.sort(key=sort_key)

        ordered = runnable + blocked
        return [(endpoint, wi) for endpoint, wi, *_ in ordered]
