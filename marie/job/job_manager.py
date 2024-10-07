import asyncio
import os
import time
import traceback
from typing import Any, Dict, Iterator, Optional

from uuid_extensions import uuid7str

from marie._core.utils import run_background_task
from marie.job.common import ActorHandle, JobInfo, JobInfoStorageClient, JobStatus
from marie.job.event_publisher import EventPublisher
from marie.job.job_distributor import JobDistributor
from marie.job.job_log_storage_client import JobLogStorageClient
from marie.job.job_storage_client_proxy import JobInfoStorageClientProxy
from marie.job.job_supervisor import JobSupervisor
from marie.job.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    SchedulingStrategyT,
)
from marie.logging.logger import MarieLogger
from marie.storage.kv.storage_client import StorageArea

# The max time to wait for the JobSupervisor to start before failing the job.
DEFAULT_JOB_START_TIMEOUT_SECONDS = 60 * 15
JOB_START_TIMEOUT_SECONDS_ENV_VAR = "JOB_START_TIMEOUT_SECONDS"

ActorUnschedulableError = Exception


def generate_job_id() -> str:
    # A uuidv7 is a universally unique lexicographically sortable identifier
    # This will be in standard library in Python 3.11
    return uuid7str()


def get_event_logger():
    # TODO: Implement this
    return None


class JobManager:
    """Provide APIs for job submission and management to the cluster.

    It does not provide persistence, all info will be lost if the cluster
    goes down.
    """

    JOB_MONITOR_LOOP_PERIOD_S = 1
    # number of slots available for job submission (This will be set via service discovery and will change as nodes become available)
    SLOTS_AVAILABLE = 0

    def __init__(
        self,
        storage: StorageArea,
        job_distributor: JobDistributor,
    ):
        self.logger = MarieLogger(self.__class__.__name__)
        self._job_distributor = job_distributor
        self.event_publisher = EventPublisher()
        self._log_client = JobLogStorageClient()
        self.monitored_jobs = set()
        self._job_info_client = JobInfoStorageClientProxy(self.event_publisher, storage)

        try:
            self.event_logger = get_event_logger()
        except Exception:
            self.event_logger = None

        self._recover_running_jobs_event = asyncio.Event()
        run_background_task(self._recover_running_jobs())

    async def _recover_running_jobs(self):
        """Recovers all running jobs from the status client.

        For each job, we will spawn a coroutine to monitor it.
        Each will be added to self._running_jobs and reconciled.
        """
        self.logger.debug("Recovering running jobs.")
        try:
            all_jobs = await self._job_info_client.get_all_jobs()
            for job_id, job_info in all_jobs.items():
                if not job_info.status.is_terminal():
                    run_background_task(self._monitor_job(job_id))
        finally:
            # This event is awaited in `submit_job` to avoid race conditions between
            # recovery and new job submission, so it must always get set even if there
            # are exceptions.
            self._recover_running_jobs_event.set()

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
        @param job_supervisor: The actor handle for the job supervisor.
        """

        self.logger.info(f"Monitoring job internal : {job_id}.")

        timeout = float(
            os.environ.get(
                JOB_START_TIMEOUT_SECONDS_ENV_VAR,
                DEFAULT_JOB_START_TIMEOUT_SECONDS,
            )
        )
        is_alive = True

        while is_alive:
            try:
                job_status = await self._job_info_client.get_status(job_id)
                print(f"Job status: {job_id} : {job_status}")
                # print("len(self.monitored_jobs): ", len(self.monitored_jobs))
                # print("has_available_slot: ", self.has_available_slot())
                if job_status.is_terminal():
                    if job_status == JobStatus.SUCCEEDED:
                        is_alive = False
                        self.logger.info(f"Job succeeded : {job_id}")
                        break
                    elif job_status == JobStatus.FAILED:
                        is_alive = False
                        self.logger.warning(f"Job failed : {job_id}")
                        break

                if job_status == JobStatus.PENDING:
                    # Compare the current time with the job start time.
                    # If the job is still pending, we will set the status
                    # to FAILED.
                    job_info = await self._job_info_client.get_info(job_id)

                    if time.time() - job_info.start_time / 1000 > timeout:
                        err_msg = (
                            "Job supervisor actor failed to start within "
                            f"{timeout} seconds. This timeout can be "
                            f"configured by setting the environment "
                            f"variable {JOB_START_TIMEOUT_SECONDS_ENV_VAR}."
                        )
                        resources_specified = (
                            (
                                job_info.entrypoint_num_cpus is not None
                                and job_info.entrypoint_num_cpus > 0
                            )
                            or (
                                job_info.entrypoint_num_gpus is not None
                                and job_info.entrypoint_num_gpus > 0
                            )
                            or (
                                job_info.entrypoint_resources is not None
                                and len(job_info.entrypoint_resources) > 0
                            )
                        )
                        if resources_specified:
                            err_msg += (
                                " This may be because the job entrypoint's specified "
                                "resources (entrypoint_num_cpus, entrypoint_num_gpus, "
                                "entrypoint_resources, entrypoint_memory)"
                                "aren't available on the cluster."
                                " Try checking the cluster's available resources with "
                                "`marie nodes status` and specifying fewer resources for the "
                                "job entrypoint."
                            )
                        await self._job_info_client.put_status(
                            job_id,
                            JobStatus.FAILED,
                            message=err_msg,
                        )
                        is_alive = False
                        self.logger.error(err_msg)
                        continue

                if job_supervisor is None:
                    raise NotImplementedError

                if job_supervisor is None:
                    if job_status == JobStatus.PENDING:
                        # Maybe the job supervisor actor is not created yet.
                        # We will wait for the next loop.
                        continue
                    else:
                        # The job supervisor actor is not created, but the job
                        # status is not PENDING. This means the job supervisor
                        # actor is not created due to some unexpected errors.
                        # We will set the job status to FAILED.
                        self.logger.error(
                            f"Failed to get job supervisor for job {job_id}."
                        )
                        await self._job_info_client.put_status(
                            job_id,
                            JobStatus.FAILED,
                            message=(
                                "Unexpected error occurred: "
                                "failed to get job supervisor."
                            ),
                        )
                        is_alive = False
                        continue

                await job_supervisor.ping()

                await asyncio.sleep(self.JOB_MONITOR_LOOP_PERIOD_S)
            except Exception as e:
                is_alive = False
                job_status = await self._job_info_client.get_status(job_id)
                job_error_message = None
                if job_status == JobStatus.FAILED:
                    job_error_message = (
                        "See more details from the dashboard "
                        "`Job` page or the state API `marie list jobs`."
                    )

                job_error_message = ""
                if job_status.is_terminal():
                    # If the job is already in a terminal state, then the actor
                    # exiting is expected.
                    pass
                elif isinstance(e, ActorUnschedulableError):
                    self.logger.info(
                        f"Failed to schedule job {job_id} because the supervisor actor "
                        f"could not be scheduled: {e}"
                    )
                    job_error_message = (
                        f"Job supervisor actor could not be scheduled: {e}"
                    )
                    await self._job_info_client.put_status(
                        job_id,
                        JobStatus.FAILED,
                        message=job_error_message,
                    )
                else:
                    self.logger.warning(
                        f"Job supervisor for job {job_id} failed unexpectedly: {e}."
                    )
                    job_error_message = f"Unexpected error occurred: {e}"
                    job_status = JobStatus.FAILED
                    await self._job_info_client.put_status(
                        job_id,
                        job_status,
                        message=job_error_message,
                    )

                # Log error message to the job driver file for easy access.
                if job_error_message:
                    log_path = self._log_client.get_log_file_path(job_id)
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, "a") as log_file:
                        log_file.write(job_error_message)

                # Log events
                if self.event_logger:
                    event_log = (
                        f"Completed a ray job {job_id} with a status {job_status}."
                    )
                    if job_error_message:
                        event_log += f" {job_error_message}"
                        self.event_logger.error(event_log, submission_id=job_id)
                    else:
                        self.event_logger.info(event_log, submission_id=job_id)

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

        entrypoint_num_cpus = 1
        entrypoint_num_gpus = 1
        entrypoint_resources = None
        # Wait for `_recover_running_jobs` to run before accepting submissions to
        # avoid duplicate monitoring of the same job.
        await self._recover_running_jobs_event.wait()
        self.logger.info(f"Starting job with submission_id: {submission_id}")

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
                    f"Started a job {submission_id}.", submission_id=submission_id
                )

            supervisor = JobSupervisor(
                job_id=submission_id,
                job_info_client=self._job_info_client,
                job_distributor=self._job_distributor,
                event_publisher=self.event_publisher,
            )
            await supervisor.run(_start_signal_actor=_start_signal_actor)

            # Monitor the job in the background so we can detect errors without
            # requiring a client to poll.
            run_background_task(
                self._monitor_job(submission_id, job_supervisor=supervisor)
            )
        except Exception as e:
            tb_str = traceback.format_exc()

            self.logger.warning(
                f"Failed to start supervisor actor for job {submission_id}: '{e}'"
                f". Full traceback:\n{tb_str}"
            )
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

    def has_available_slot(self) -> bool:
        """
        Check if there are available slots for submitting a jobs.

        :return: True if there are available slots, False otherwise
        """
        return len(self.monitored_jobs) < self.SLOTS_AVAILABLE
