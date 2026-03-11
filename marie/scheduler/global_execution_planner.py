from datetime import datetime, timezone
from math import inf
from typing import Any, Dict, List, Sequence, Set, Tuple

from marie.scheduler.execution_planner import FlatJob
from marie.scheduler.sla import compute_sla_priority_bucket

_DEFAULT_REMAINING = 2**31


class GlobalPriorityExecutionPlanner:
    """
    Pure ranking (no filtering): returns *all* jobs, ordered so that:
      1) runnable (executor has free slots) before blocked
      2) higher persisted priority (operator override) first
      3) higher SLA urgency
      4) existing DAGs before new DAGs
      5) deeper level (critical path) first
      6) more executor free slots (tie-breaker)
      7) shorter estimated runtime
      8) FIFO (original input order)
    """

    def plan(
        self,
        jobs: Sequence[FlatJob],
        slots: Dict[str, int],
        active_dags: Set[str],
        *,
        exclude_blocked: bool = False,
        now: datetime | None = None,
        dag_remaining: dict[str, int] | None = None,
    ) -> Sequence[FlatJob]:
        """
        Pure ordering by default (returns all jobs). If exclude_blocked=True,
        blocked jobs (executors with 0 free slots) are filtered out.
        Order among returned jobs:
          runnable → blocked (if included)
          then: priority ↓, SLA urgency ↓, existing DAGs, remaining ↑,
          level ↓, free_slots ↓, est_runtime ↑, FIFO
        """
        now_utc = now or datetime.now(timezone.utc)

        # (endpoint, wi, is_blocked, priority, sla_bucket, is_new, remaining, level, free_slots, est_rt, fifo_idx)
        annotated: List[
            Tuple[str, Any, bool, int, int, bool, int, int, int, float, int]
        ] = []

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
            remaining = (
                dag_remaining.get(wi.dag_id, _DEFAULT_REMAINING)
                if dag_remaining is not None
                else 0
            )
            level = wi.job_level
            priority = wi.priority
            sla_bucket = compute_sla_priority_bucket(
                now_utc,
                wi.soft_sla,
                wi.hard_sla,
            )

            meta = (
                wi.data.get("metadata", {})
                if (wi.data and isinstance(wi.data, dict))
                else {}
            )
            est = meta.get("estimated_runtime")
            est_rt = float(est) if est is not None else inf

            annotated.append(
                (
                    endpoint,
                    wi,
                    is_blocked,
                    int(priority),
                    sla_bucket,
                    is_new,
                    remaining,
                    level,
                    free,
                    est_rt,
                    idx,
                )
            )

        # Optionally drop blocked jobs
        if exclude_blocked:
            annotated = [t for t in annotated if not t[2]]

        # Sort (if exclude_blocked=True, all is_blocked=False so first term is a no-op)
        annotated.sort(
            key=lambda t: (
                t[2],  # is_blocked: False < True  → runnable first
                -t[3],  # persisted manual priority desc
                -t[4],  # SLA urgency desc
                t[5],  # is_new: False < True     → existing DAGs first
                t[6],  # remaining asc            → closer-to-done DAGs first
                -t[7],  # level desc
                -t[8],  # free slots desc
                t[9],  # est runtime asc
                t[10],  # FIFO
            )
        )

        return [(t[0], t[1]) for t in annotated]
