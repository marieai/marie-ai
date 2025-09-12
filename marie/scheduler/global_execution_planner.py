from typing import Any, Sequence, Tuple

from marie.scheduler.execution_planner import FlatJob


class GlobalPriorityExecutionPlanner:
    """
    Dynamic “global ready queue” scheduler with est.‐runtime as final tie-breaker.
    Prioritizes:
      1) deepest-level first (critical-path)
      2) higher user priority
      3) more free slots
      4) existing DAGs over new DAGs
      5) FIFO within same DAG
      6) shorter est. runtimes
      7) burst boost for recently activated DAGs

      # https://www.mdpi.com/2079-9292/10/16/1874
      # https://openreview.net/forum?id=km4omm25me
      # https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/mit6_042js15_session17.pdf
    """

    def plan(
        self,
        jobs: Sequence[FlatJob],
        slots: dict[str, int],
        active_dags: set[str],
        recently_activated_dags: set[str] = set(),
    ) -> Sequence[FlatJob]:
        annotated: list[Tuple[str, Any, int, int, int, bool, str, float, bool]] = []
        for endpoint, wi in jobs:
            level = wi.job_level
            priority = wi.priority
            dag_id = wi.dag_id
            executor = endpoint.split("://", 1)[0]
            free_slots = slots.get(executor, 0)
            is_new = wi.dag_id not in active_dags

            # extract estimated runtime (default to inf if missing)
            est = wi.data.get("metadata", {}).get("estimated_runtime")
            rt = float(est) if est is not None else float("inf")
            burst_boost = wi.dag_id in recently_activated_dags

            annotated.append(
                (
                    endpoint,
                    wi,
                    level,
                    priority,
                    free_slots,
                    is_new,
                    dag_id,
                    rt,
                    burst_boost,
                )
            )

        # sort by:
        #  - level descending
        #  - priority descending
        #  - free_slots descending
        #  - existing-DAG (False) before new-DAG (True)
        #  - dag_id (to ensure stable sort, i.e. FIFO within same dag)
        #  - dag insertion order, oldest first
        #  - runtime ascending
        annotated.sort(
            key=lambda t: (
                -t[2],  # level
                -t[3],  # priority
                -t[4],  # free_slots
                t[5],  # is_new (False < True)
                t[6],  # dag insertion order
                t[7],  # est. runtime
                not t[8],  # burst_boost: True < False
            )
        )

        return [(endpoint, wi) for endpoint, wi, *_ in annotated]
