import asyncio
import time
from typing import Optional, Dict, Any, Iterator

from marie.logging.logger import MarieLogger
from marie_server.job.common import JobInfo, JobStatus, JobInfoStorageClient
from marie_server.storage.storage_client import StorageArea


def generate_job_id() -> str:
    # https://github.com/mdomke/python-ulid
    from ulid import ULID
    from datetime import datetime

    # A ULID is a universally unique lexicographically sortable identifier.
    #  01AN4Z07BY      79KA1307SR9X4MV3
    # |----------|    |----------------|
    #   Timestamp          Randomness
    #   48bits             80bits
    return str(ULID.from_datetime(datetime.now()))


class JobManager:
    """Provide python APIs for job submission and management.

    It does not provide persistence, all info will be lost if the cluster
    goes down.
    """

    def __init__(
        self,
        storage: StorageArea,
    ):
        self.logger = MarieLogger("JobManager")
        # self._log_client = JobLogStorageClient()
        self.monitored_jobs = set()
        self._job_info_client = JobInfoStorageClient(storage)
        self.monitored_jobs = set()

    async def submit_job(
        self,
        *,
        entrypoint: str,
        submission_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        runtime_env: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Job execution happens asynchronously.
        """

        self.logger.info(f"Starting job with submission_id: {submission_id}")
        entrypoint_num_cpus = 1
        entrypoint_num_gpus = 1
        entrypoint_resources = None

        job_info = JobInfo(
            entrypoint=entrypoint,
            status=JobStatus.PENDING,
            start_time=int(time.time() * 1000),
            metadata=metadata,
            runtime_env=runtime_env,
            entrypoint_num_cpus=entrypoint_num_cpus,
            entrypoint_num_gpus=entrypoint_num_gpus,
            entrypoint_resources=entrypoint_resources,
        )
        new_key_added = await self._job_info_client.put_info(
            submission_id, job_info, overwrite=False
        )
        if not new_key_added:
            raise ValueError(
                f"Job with submission_id {submission_id} already exists. "
                "Please use a different submission_id."
            )

        # Wait for the actor to start up asynchronously so this call always
        # returns immediately and we can catch errors with the actor starting
        # up.
        try:
            resources_specified = any(
                [
                    entrypoint_num_cpus is not None and entrypoint_num_cpus > 0,
                    entrypoint_num_gpus is not None and entrypoint_num_gpus > 0,
                    entrypoint_resources not in [None, {}],
                ]
            )
            scheduling_strategy = await self._get_scheduling_strategy(
                resources_specified
            )
            if self.event_logger:
                self.event_logger.info(
                    f"Started a  job {submission_id}.", submission_id=submission_id
                )

            # Monitor the job in the background so we can detect errors without
            # requiring a client to poll.
            # run_background_task(
            #     self._monitor_job(submission_id, job_supervisor=supervisor)
            # )
        except Exception as e:
            await self._job_info_client.put_status(
                submission_id,
                JobStatus.FAILED,
                message=f"Failed to start Job Supervisor actor: {e}.",
            )

        return submission_id

    def stop_job(self, job_id) -> bool:
        """Request a job to exit, fire and forget.

        Returns whether or not the job was running.
        """
        raise NotImplementedError

    async def delete_job(self, job_id):
        """Delete a job's info and metadata from the cluster."""
        job_status = await self._job_info_client.get_status(job_id)

        if job_status is None or not job_status.is_terminal():
            raise RuntimeError(
                f"Attempted to delete job '{job_id}', "
                f"but it is in a non-terminal state {job_status}."
            )

        await self._job_info_client.delete_info(job_id)
        return True

    def job_info_client(self) -> JobInfoStorageClient:
        return self._job_info_client

    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get latest status of a job."""
        return await self._job_info_client.get_status(job_id)

    async def get_job_info(self, job_id: str) -> Optional[JobInfo]:
        """Get latest info of a job."""
        return await self._job_info_client.get_info(job_id)

    async def list_jobs(self) -> Dict[str, JobInfo]:
        """Get info for all jobs."""
        return await self._job_info_client.get_all_jobs()

    def get_job_logs(self, job_id: str) -> str:
        """Get all logs produced by a job."""
        if True:
            raise NotImplementedError
        return self._log_client.get_logs(job_id)

    async def tail_job_logs(self, job_id: str) -> Iterator[str]:
        """Return an iterator following the logs of a job."""
        if True:
            raise NotImplementedError

        if await self.get_job_status(job_id) is None:
            raise RuntimeError(f"Job '{job_id}' does not exist.")

        for lines in self._log_client.tail_logs(job_id):
            if lines is None:
                # Return if the job has exited and there are no new log lines.
                status = await self.get_job_status(job_id)
                if status.is_terminal():
                    return

                await asyncio.sleep(self.LOG_TAIL_SLEEP_S)
            else:
                yield "".join(lines)
