from math import inf
from typing import Any, Dict, List, Sequence, Set, Tuple

from marie.scheduler.execution_planner import FlatJob


class GlobalPriorityExecutionPlanner:
    """
    Pure ranking (no filtering): returns *all* jobs, ordered so that:
      1) runnable (executor has free slots) before blocked
      2) existing DAGs before new DAGs
      3) deeper level (critical path) first
      4) higher user priority
      5) more executor free slots (tie-breaker)
      6) shorter estimated runtime
      7) FIFO (original input order)
    """

    def plan(
        self,
        jobs: Sequence[FlatJob],
        slots: Dict[str, int],
        active_dags: Set[str],
        *,
        exclude_blocked: bool = False,
    ) -> Sequence[FlatJob]:
        """
        Pure ordering by default (returns all jobs). If exclude_blocked=True,
        blocked jobs (executors with 0 free slots) are filtered out.
        Order among returned jobs:
          runnable(existing) → runnable(new) → blocked (if included)
          then: level ↓, priority ↓, free_slots ↓, est_runtime ↑, FIFO
        """
        # (endpoint, wi, is_blocked, is_new, level, priority, free_slots, est_rt, fifo_idx)
        annotated: List[Tuple[str, Any, bool, bool, int, int, int, float, int]] = []

        for idx, (endpoint, wi) in enumerate(jobs):
            executor = endpoint.split("://", 1)[0]
            # noop work is never blocked and does not consume a slot.
            if executor == "noop":
                free = inf
                is_blocked = False
            else:
                free = int(slots.get(executor, 0))
                is_blocked = free <= 0

            is_new = wi.dag_id not in active_dags
            level = wi.job_level
            priority = wi.priority

            meta = (
                wi.data.get("metadata", {})
                if (wi.data and isinstance(wi.data, dict))
                else {}
            )
            est = meta.get("estimated_runtime")
            est_rt = float(est) if est is not None else inf

            annotated.append(
                (endpoint, wi, is_blocked, is_new, level, priority, free, est_rt, idx)
            )

        # Optionally drop blocked jobs
        if exclude_blocked:
            annotated = [t for t in annotated if not t[2]]

        # Sort (if exclude_blocked=True, all is_blocked=False so first term is a no-op)
        annotated.sort(
            key=lambda t: (
                t[2],  # is_blocked: False < True  → runnable first
                t[3],  # is_new: False < True     → existing DAGs first
                -t[4],  # level desc
                -t[5],  # priority desc
                -t[6],  # free slots desc
                t[7],  # est runtime asc
                t[8],  # FIFO
            )
        )

        return [(t[0], t[1]) for t in annotated]
