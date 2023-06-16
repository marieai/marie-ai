import asyncio
import os
import time
from typing import Optional, Dict, Any, Iterator
from uuid_extensions import uuid7, uuid7str

from marie._core.utils import run_background_task
from marie.logging.logger import MarieLogger
from marie_server.job.common import JobInfo, JobStatus, JobInfoStorageClient
from marie_server.job.job_distributor import JobDistributor
from marie_server.storage.storage_client import StorageArea
from marie_server.job.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    SchedulingStrategyT,
)

ActorHandle = Any

# The max time to wait for the JobSupervisor to start before failing the job.
DEFAULT_JOB_START_TIMEOUT_SECONDS = 60 * 15
JOB_START_TIMEOUT_SECONDS_ENV_VAR = "JOB_START_TIMEOUT_SECONDS"


def generate_job_id() -> str:
    # A uuidv7 is a universally unique lexicographically sortable identifier
    # This will be in standard library in Python 3.11
    return uuid7str()


def get_event_logger():
    # TODO: Implement this
    return None


class JobSupervisor:
    """
    Supervise jobs and keep track of their status.
    """

    DEFAULT_JOB_STOP_WAIT_TIME_S = 3

    def __init__(
        self,
        job_id: str,
        job_info_client: JobInfoStorageClient,
        job_distributor: JobDistributor,
    ):
        self._job_id = job_id
        self._job_info_client = job_info_client
        self._job_distributor = job_distributor

    def ping(self):
        """Used to check the health of the actor/executor/deployment."""
        pass

    async def run(
        self,
        # Signal actor used in testing to capture PENDING -> RUNNING cases
        _start_signal_actor: Optional[ActorHandle] = None,
    ):
        """
        Stop and start both happen asynchronously, coordinated by asyncio event
        and coroutine, respectively.

        1) Sets job status as running
        2) Pass runtime env and metadata to subprocess as serialized env
            variables.
        3) Handle concurrent events of driver execution and
        """
        curr_info = await self._job_info_client.get_info(self._job_id)
        if curr_info is None:
            raise RuntimeError(f"Status could not be retrieved for job {self._job_id}.")
        curr_status = curr_info.status
        curr_message = curr_info.message
        if curr_status == JobStatus.RUNNING:
            raise RuntimeError(
                f"Job {self._job_id} is already in RUNNING state. "
                f"JobSupervisor.run() should only be called once. "
            )
        if curr_status != JobStatus.PENDING:
            raise RuntimeError(
                f"Job {self._job_id} is not in PENDING state. "
                f"Current status is {curr_status} with message {curr_message}."
            )
        if _start_signal_actor:
            # Block in PENDING state until start signal received.
            await _start_signal_actor.wait.remote()

        driver_agent_http_address = "grpc://127.0.0.1"
        driver_node_id = "GET_NODE_ID_FROM_CLUSTER"

        await self._job_info_client.put_status(
            self._job_id,
            JobStatus.RUNNING,
            jobinfo_replace_kwargs={
                "driver_agent_http_address": driver_agent_http_address,
                "driver_node_id": driver_node_id,
            },
        )

        response = await self._job_distributor.submit_job(curr_info)
        # format the response
        print("Response: ", response)
        print("Response type: ", type(response))
        print("Response data: ", response.data)
        print("Response status: ", response.status)


class JobManager:
    """Provide python APIs for job submission and management.

    It does not provide persistence, all info will be lost if the cluster
    goes down.
    """

    def __init__(
        self,
        storage: StorageArea,
        job_distributor: JobDistributor,
    ):
        self.logger = MarieLogger(self.__class__.__name__)
        self._job_distributor = job_distributor
        # self._log_client = JobLogStorageClient()
        self.monitored_jobs = set()
        self._job_info_client = JobInfoStorageClient(storage)

        try:
            self.event_logger = get_event_logger()
        except Exception:
            self.event_logger = None

        run_background_task(self._recover_running_jobs())

    async def _recover_running_jobs(self):
        """Recovers all running jobs from the status client.

        For each job, we will spawn a coroutine to monitor it.
        Each will be added to self._running_jobs and reconciled.
        """
        self.logger.debug("Recovering running jobs.")
        all_jobs = await self._job_info_client.get_all_jobs()
        for job_id, job_info in all_jobs.items():
            if not job_info.status.is_terminal():
                run_background_task(self._monitor_job(job_id))

    async def _monitor_job(
        self, job_id: str, job_supervisor: Optional[ActorHandle] = None
    ):
        """Monitors the specified job until it enters a terminal state.

        This is necessary because we need to handle the case where the
        JobSupervisor dies unexpectedly.
        """
        self.logger.info(f"Monitoring job : {job_id}.")
        if job_id in self.monitored_jobs:
            self.logger.debug(f"Job {job_id} is already being monitored.")
            return

        self.monitored_jobs.add(job_id)
        try:
            await self._monitor_job_internal(job_id, job_supervisor)
        finally:
            self.monitored_jobs.remove(job_id)

    async def _monitor_job_internal(
        self, job_id: str, job_supervisor: Optional[ActorHandle] = None
    ):
        """Monitors the specified job until it enters a terminal state.
        @param job_id: The id of the job to monitor.
        @param job_supervisor:
        """
        self.logger.info(f"Monitoring job internal : {job_id}.")
        timeout = float(
            os.environ.get(
                JOB_START_TIMEOUT_SECONDS_ENV_VAR,
                DEFAULT_JOB_START_TIMEOUT_SECONDS,
            )
        )

        is_alive = True

    async def _get_scheduling_strategy(
        self, resources_specified: bool
    ) -> SchedulingStrategyT:
        """Get the scheduling strategy for the job.
        Returns:
            The scheduling strategy to use for the job.
        """
        if resources_specified:
            return "DEFAULT"

        scheduling_strategy = NodeAffinitySchedulingStrategy()
        return scheduling_strategy

    async def submit_job(
        self,
        *,
        entrypoint: str,
        submission_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        runtime_env: Optional[Dict[str, Any]] = None,
        _start_signal_actor: Optional[ActorHandle] = None,
    ) -> str:
        """
        Job execution happens asynchronously.
        """
        if submission_id is None:
            submission_id = generate_job_id()

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

            supervisor = JobSupervisor(
                job_id=submission_id,
                job_info_client=self._job_info_client,
                job_distributor=self._job_distributor,
            )
            await supervisor.run(_start_signal_actor=_start_signal_actor)

            # Monitor the job in the background so we can detect errors without
            # requiring a client to poll.
            run_background_task(
                self._monitor_job(submission_id, job_supervisor=supervisor)
            )
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
