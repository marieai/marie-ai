from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Optional


from marie.storage.marie_run import DagsterRun


class SubmitRunContext(NamedTuple):
    def get_request_header(self, key: str) -> Optional[str]:
        raise NotImplementedError()


class RunCoordinator(ABC):
    @abstractmethod
    def submit_run(self, context: SubmitRunContext) -> DagsterRun:
        """Submit a run to the run coordinator for execution.

        Args:
            context (SubmitRunContext): information about the submission - every run coordinator
            will need the PipelineRun, and some run coordinators may need information from the
            IWorkspace from which the run was launched.

        Returns:
            PipelineRun: The queued run
        """

    @abstractmethod
    def cancel_run(self, run_id: str) -> bool:
        """Cancels a run. The run may be queued in the coordinator, or it may have been launched.

        Returns False is the process was already canceled. Returns true if the cancellation was
        successful.
        """

    def dispose(self) -> None:
        """Do any resource cleanup that should happen when the DagsterInstance is
        cleaning itself up.
        """
