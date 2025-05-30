import time
from typing import Sequence

from marie.scheduler.execution_planner import FlatJob
from marie.scheduler.models import WorkInfo


class HRRNExecutionPlanner:
    """
    Highest Response Ratio Next
    """

    def plan(
        self,
        jobs: Sequence[FlatJob],
        slots: dict[str, int],
        active_dags: set[str],
        recently_activated_dags: set[str] = set(),
    ) -> Sequence[FlatJob]:
        now = time.time()

        def resp_ratio(wi: WorkInfo) -> float:
            submitted = wi.data["metadata"].get("submitted_at", now)
            wait_sec = now - submitted
            rt = wi.data["metadata"].get("estimated_runtime", float("inf"))
            return (wait_sec + rt) / rt

        annotated: list[tuple[str, WorkInfo, float]] = []
        for ep, wi in jobs:
            r = resp_ratio(wi)
            annotated.append((ep, wi, r))

        annotated.sort(
            key=lambda tpl: (
                -tpl[2],  # highest response ratio first
                -slots.get(tpl[0].split("://")[0], 0),
                tpl[1].job_level,
                -tpl[1].priority,
                tpl[1].dag_id not in active_dags,
            )
        )

        return [(ep, wi) for ep, wi, _ in annotated]
