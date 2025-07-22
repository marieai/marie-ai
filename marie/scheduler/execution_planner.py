from typing import Protocol, Sequence, Tuple

from marie.scheduler.models import WorkInfo

# a tuple of (entrypoint URI, WorkInfo)
FlatJob = Tuple[str, WorkInfo]


class ExecutionPlanner(Protocol):
    def plan(
        self,
        jobs: Sequence[FlatJob],
        slots: dict[str, int],
        active_dags: set[str],
        recently_activated_dags: set[str] = set(),
    ) -> Sequence[FlatJob]:
        """
        Given a list of ready-to-run jobs, return them in the order they
        should be dispatched.
        """
        ...
