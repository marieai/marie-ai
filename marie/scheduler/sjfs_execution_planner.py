from typing import Sequence

from marie.scheduler.execution_planner import FlatJob
from marie.scheduler.models import WorkInfo


class SJFSExecutionPlanner:
    """
    Shortest Job First Scheduler (SJFS) execution planner.
    """

    def plan(
        self,
        jobs: Sequence[FlatJob],
        slots: dict[str, int],
        active_dags: set[str],
        recently_activated_dags: set[str] = set(),
    ) -> Sequence[FlatJob]:
        # annotate each job with its estimated runtime (inf if missing)
        annotated: list[tuple[str, WorkInfo, float]] = []
        for ep, wi in jobs:
            # est = wi.data.get("metadata", {}).get("estimated_runtime")
            est = 1  # TODO: remove this line DEBUG
            rt = float(est) if est is not None else float("inf")
            annotated.append((ep, wi, rt))

        # sort by:
        # 1) shortest runtime
        # 2) most free slots on its executor (descending)
        # 3) job_level (asc)
        # 4) priority (desc)
        # 5) existing-DAG first
        annotated.sort(
            key=lambda tpl: (
                tpl[2],
                -slots.get(tpl[0].split("://")[0], 0),
                tpl[1].job_level,
                -tpl[1].priority,
                tpl[1].dag_id not in active_dags,
            )
        )

        # strip runtime back off
        return [(ep, wi) for ep, wi, _ in annotated]
