from typing import Any, Optional

from marie_server.job.common import (
    ActorHandle,
    JobInfo,
    JobInfoStorageClient,
    JobStatus,
)
from marie_server.job.job_distributor import JobDistributor


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
