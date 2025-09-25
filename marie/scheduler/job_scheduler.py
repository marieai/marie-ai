import abc
import asyncio
from typing import Any, Dict, NamedTuple, Optional

from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState


class JobSubmissionRequest(NamedTuple):
    work_info: WorkInfo
    overwrite: bool
    request_id: str
    result_future: asyncio.Future


class JobScheduler(abc.ABC):
    """Abstract base class for a job scheduler. This component is responsible for interfacing with
    an external system such as cron,s3 etc... to ensure scheduled repeated execution according to the schedule.

    JobScheduler API is similar to the JobManager API, but it is focused on scheduling jobs to be executed potentially at
    a later time. The JobScheduler is responsible for managing the scheduling of jobs, while the JobManager is responsible for
    placement and managing the execution of jobs.
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
        """Enqueues a job to be executed immediately on the next available executor.
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
    def stop_job(self, job_id: str) -> bool:
        """Request a job to exit, fire and forget.
        Returns whether or not the job was running.
        """
        ...

    @abc.abstractmethod
    async def delete_job(self, job_id: str):
        """Deletes the job with the given job_id."""
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

    @abc.abstractmethod
    def get_available_slots(self) -> dict[str, int]:
        """
        Retrieve available slots in a structured format.

        :return: A dictionary with string keys representing slot types and integer
            values indicating the count of available slots for each type.
        """
        ...

    def reset_active_dags(self):
        """
        Reset the active DAGs dictionary, clearing all currently tracked DAGs.
        This can be useful for debugging or when you need to force a fresh state.

        Returns:
            dict: Information about the reset operation including count of cleared DAGs
        """
        return {
            "message": "Active DAGs reset",
            "cleared_dags_count": 0,
        }
