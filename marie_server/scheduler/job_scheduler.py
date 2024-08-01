import abc
from typing import Any, Dict, Optional

from marie_server.scheduler.models import WorkInfo
from marie_server.scheduler.state import WorkState


class JobScheduler(abc.ABC):
    """Abstract base class for a job scheduler. This component is responsible for interfacing with
    an external system such as cron to ensure scheduled repeated execution according to the schedule.

    JobScheduler API is similar to the JobManager API, but it is focused on scheduling jobs to be executed potentially at
    a later time. The JobScheduler is responsible for managing the scheduling of jobs, while the JobManager is responsible for
    managing the execution of jobs.
    """

    async def start(self) -> Any:
        """Starts the scheduler."""
        started_state = 1
        return started_state

    async def stop(
        self,
    ) -> Any:
        """Stops the scheduler."""
        stopped_state = 0
        return stopped_state

    @abc.abstractmethod
    def debug_info(self) -> str:
        """Returns debug information about the scheduler."""
        ...

    @abc.abstractmethod
    async def wipe(self) -> None:
        """Clears the schedule storage."""
        ...

    @abc.abstractmethod
    async def enqueue(self, work_info: WorkInfo) -> None:
        """Enqueues a job to be executed immediately on the next available worker
        Job will be executed according to the schedule defined in the WorkInfo object.
        """
        ...

    @abc.abstractmethod
    async def submit_job(self, work_info: WorkInfo, overwrite: bool = True) -> str:
        """Inserts a new work item into the scheduler.
        :param work_info: The work item to insert.
        :param overwrite: Whether to overwrite the work item if it already exists.
        :return: True if the work item was inserted, False if it was not.
        """
        ...

    @abc.abstractmethod
    async def get_job(self, job_id: str) -> Optional[WorkInfo]:
        """Retrieves the job with the given job_id."""
        ...

    @abc.abstractmethod
    async def list_jobs(self) -> Dict[str, WorkInfo]:
        """
        Lists all the jobs in the scheduler.
        """
        ...

    @abc.abstractmethod
    async def put_status(self, job_id, status: WorkState):
        """Updates the status of a job."""
        ...
