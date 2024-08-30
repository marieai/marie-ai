import asyncio
from typing import Dict, List, Optional, Union

from docarray import BaseDoc, DocList
from docarray.documents import TextDoc

from marie.logging.logger import MarieLogger
from marie.serve.networking import _NetworkingHistograms, _NetworkingMetrics
from marie.types.request.data import DataRequest
from marie_server.job.common import ActorHandle, JobInfoStorageClient, JobStatus
from marie_server.job.job_distributor import JobDistributor


class JobSupervisor:
    """
    Supervise jobs and keep track of their status on the remote executor.
    """

    DEFAULT_JOB_STOP_WAIT_TIME_S = 3

    def __init__(
        self,
        job_id: str,
        job_info_client: JobInfoStorageClient,
        job_distributor: JobDistributor,
    ):
        self.logger = MarieLogger(self.__class__.__name__)
        self._job_id = job_id
        self._job_info_client = job_info_client
        self._job_distributor = job_distributor
        self.request_info = None

    async def ping(self):
        """Used to check the health of the executor/deployment."""
        request_info = self.request_info
        if request_info is None:
            return True

        request_id = request_info["request_id"]
        address = request_info["address"]
        deployment_name = request_info["deployment"]

        self.logger.debug(
            f"Sending ping to {address} for request {request_id} on deployment {deployment_name}"
        )

        from marie.serve.networking.connection_stub import _ConnectionStubs
        from marie.serve.networking.utils import get_grpc_channel

        channel = get_grpc_channel(address=address, asyncio=True)
        connection_stub = _ConnectionStubs(
            address=address,
            channel=channel,
            deployment_name=deployment_name,
            metrics=_NetworkingMetrics(
                sending_requests_time_metrics=None,
                received_response_bytes=None,
                send_requests_bytes_metrics=None,
            ),
            histograms=_NetworkingHistograms(),
        )

        # print("DryRun - Response: ", response)
        doc = TextDoc(text=f"Text : _jina_dry_run_")
        request = DataRequest()
        request.document_array_cls = DocList[BaseDoc]()
        request.header.exec_endpoint = "_jina_dry_run_"
        request.header.target_executor = deployment_name
        request.parameters = {}
        request.data.docs = DocList([doc])

        try:
            response, _ = await connection_stub.send_requests(
                requests=[request], metadata={}, compression=False
            )
            self.logger.debug(f"DryRun - Response: {response}")
            if response.status.code == response.status.SUCCESS:
                return True
            else:
                raise RuntimeError(
                    f"Endpoint '_jina_dry_run_' failed with status code {response.status.code}"
                )
        except Exception as e:
            self.logger.error(f"Error during ping to {self.request_info} : {e}")
            raise RuntimeError(f"Error during ping to {str(self.request_info)} : {e}")

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

        # this is our gateway address
        driver_agent_http_address = "grpc://127.0.0.1"
        driver_node_id = "CURRENT_NODE_ID"

        await self._job_info_client.put_status(
            self._job_id,
            JobStatus.RUNNING,
            jobinfo_replace_kwargs={
                "driver_agent_http_address": driver_agent_http_address,
                "driver_node_id": driver_node_id,
            },
        )
        # Run the job submission in the background
        task = asyncio.create_task(self._submit_job_in_background(curr_info))
        print("Task: ", task)

    def send_callback(
        self, requests: Union[List[DataRequest] | DataRequest], request_info: Dict
    ):
        """
        Callback when the job is submitted over the network to the executor.

        :param requests: The requests that were sent.
        :param request_info: The request info.
        """
        if isinstance(requests, list):
            request = requests[0]
        else:
            request = [requests]
        self.request_info = request_info

    async def _submit_job_in_background(self, curr_info):
        try:
            response = await self._job_distributor.submit_job(
                curr_info, self.send_callback
            )
            # printing the whole response will trigger a bug in rich.print with stackoverflow
            # format the response
            print("Response type: ", type(response))
            print("Response data: ", response.data)
            print("Response data: ", response.parameters)
            print("Response docs: ", response.data.docs)
            print("Response status: ", response.status)

            job_status = await self._job_info_client.get_status(self._job_id)

            if job_status.is_terminal():
                # If the job is already in a terminal state, then we don't need to update it. This can happen if the
                # job was cancelled while the job was being submitted.
                self.logger.warning(
                    f"Job {self._job_id} is already in terminal state {job_status}."
                )
            else:
                await self._job_info_client.put_status(
                    self._job_id, JobStatus.SUCCEEDED
                )
        except Exception as e:
            await self._job_info_client.put_status(
                self._job_id, JobStatus.FAILED, message=str(e)
            )
            raise e
