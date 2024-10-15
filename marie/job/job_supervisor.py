import asyncio
from typing import Dict, List, Optional, Union

from docarray import BaseDoc, DocList
from docarray.documents import TextDoc

from marie.job.common import ActorHandle, JobInfoStorageClient, JobStatus
from marie.job.event_publisher import EventPublisher
from marie.job.job_distributor import JobDistributor
from marie.logging_core.logger import MarieLogger
from marie.proto import jina_pb2
from marie.serve.networking import _NetworkingHistograms, _NetworkingMetrics
from marie.serve.networking.connection_stub import _ConnectionStubs
from marie.serve.networking.utils import get_grpc_channel
from marie.types_core.request.data import DataRequest


class JobSupervisor:
    """
    Supervise jobs and keep track of their status on the remote executor.

    Executors are responsible for running the job and updating the status of the job, however, the Executor does not update the WorkState.
    The JobSupervisor is responsible for updating the WorkState based on the status of the job.
    """

    DEFAULT_JOB_STOP_WAIT_TIME_S = 3
    DEFAULT_JOB_TIMEOUT_S = 0  # 60 seconds * 60 minutes

    def __init__(
        self,
        job_id: str,
        job_info_client: JobInfoStorageClient,
        job_distributor: JobDistributor,
        event_publisher: EventPublisher,
    ):
        self.logger = MarieLogger(self.__class__.__name__)
        self._job_id = job_id
        self._job_info_client = job_info_client
        self._job_distributor = job_distributor
        self._event_publisher = event_publisher
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

        async with get_grpc_channel(address=address, asyncio=True) as channel:
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

            doc = TextDoc(text=f"ping : _jina_dry_run_")
            request = DataRequest()
            request.document_array_cls = DocList[BaseDoc]()
            request.header.exec_endpoint = "_jina_dry_run_"
            request.header.target_executor = deployment_name
            request.parameters = {
                "job_id": self._job_id,
            }
            request.data.docs = DocList([doc])

            try:
                response, _ = await connection_stub.send_requests(
                    requests=[request], metadata={}, compression=False
                )
                self.logger.debug(f"DryRun - Response: {response}")
                if response.status.code == jina_pb2.StatusProto.SUCCESS:
                    return True
                else:
                    raise RuntimeError(
                        f"Endpoint '_jina_dry_run_' failed with status code {response.status.code}"
                    )
            except Exception as e:
                self.logger.error(f"Error during ping to {self.request_info} : {e}")
                raise RuntimeError(
                    f"Error during ping to {str(self.request_info)} : {e}"
                )

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

        # TODO : This should be moved to the request_handling#_record_started_job
        driver_agent_http_address = "grpc://127.0.0.1"
        driver_node_id = "CURRENT_NODE_ID"

        # check if we a calling floating executor if so then we need to update the job status to RUNNING
        # as floating executors are not part of the main deployment and they don't update the job status.
        # TODO : need to get this from the request_info

        floating_executor = False

        if floating_executor:
            self.logger.info(
                f"Job {self._job_id} is running on a floating executor. "
                f"Updating the job status to RUNNING."
            )
            await self._job_info_client.put_status(
                self._job_id,
                JobStatus.RUNNING,
                jobinfo_replace_kwargs={
                    "driver_agent_http_address": driver_agent_http_address,
                    "driver_node_id": driver_node_id,
                },
            )

        # invoke the job submission in the background
        if self.DEFAULT_JOB_TIMEOUT_S > 0:
            try:
                await asyncio.wait_for(
                    self._submit_job_in_background(curr_info),
                    timeout=self.DEFAULT_JOB_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                self.logger.error(
                    f"Job {self._job_id} timed out after {self.DEFAULT_JOB_TIMEOUT_S} seconds."
                )
                # If the job is still in PENDING state, then mark it as FAILED
                old_info = await self._job_info_client.get_info(self._job_id)
                if old_info is not None:
                    if old_info.status.is_terminal() is False:
                        await self._job_info_client.put_status(
                            self._job_id,
                            JobStatus.FAILED,
                            message="Job submission timed out.",
                        )
        else:
            task = asyncio.create_task(self._submit_job_in_background(curr_info))
        self.logger.debug(f"Job {self._job_id} submitted in the background.")

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
                submission_id=self._job_id,
                job_info=curr_info,
                send_callback=self.send_callback,
            )
            # printing the whole response will trigger a bug in rich.print with stackoverflow
            # format the response
            print("Response type: ", type(response))
            print("Response data: ", response.data)
            print("Response data: ", response.parameters)
            print("Response docs: ", response.data.docs)
            print("Response status: ", response.status)

            # This monitoring strategy allows us to have Floating Executors that can be used to run jobs outside of the main
            # deployment. This is useful for running jobs that are not part of the main deployment, but are still part of the
            # same deployment workflow.
            # Example would be calling a custom API that we don't control.

            # If the job is already in a terminal state, then we don't need to update it. This can happen if the
            # job was cancelled while the job was being submitted.
            # or while the job was marked from the EXECUTOR worker node as "STOPPED", "SUCCEEDED", "FAILED".

            job_status = await self._job_info_client.get_status(self._job_id)
            print(
                "Job status from  _submit_job_in_background: ",
                job_status,
                job_status.is_terminal(),
            )

            if response.status.code == jina_pb2.StatusProto.SUCCESS:
                if job_status.is_terminal():
                    self.logger.warning(
                        f"Job {self._job_id} is already in terminal state {job_status}."
                    )
                    # triggers the event to update the WorkStatus
                    await self._event_publisher.publish(
                        job_status,
                        {
                            "job_id": self._job_id,
                            "status": job_status,
                            "message": f"Job {self._job_id} is already in terminal state {job_status}.",
                            "jobinfo_replace_kwargs": False,
                        },
                    )
                else:
                    await self._job_info_client.put_status(
                        self._job_id, JobStatus.SUCCEEDED
                    )
            else:
                # FIXME : Need to store the exception in the job info
                e: jina_pb2.StatusProto.ExceptionProto = response.status.exception
                if job_status.is_terminal():
                    self.logger.warning(
                        f"Job {self._job_id} is already in terminal state {job_status}."
                    )
                    # triggers the event to update the WorkStatus
                    await self._event_publisher.publish(
                        job_status,
                        {
                            "job_id": self._job_id,
                            "status": job_status,
                            "message": f"Job {self._job_id} is already in terminal state {job_status}.",
                            "jobinfo_replace_kwargs": False,
                        },
                    )
                else:
                    name = str(e.name)
                    # stack = to_json(e.stacks)
                    await self._job_info_client.put_status(
                        self._job_id, JobStatus.FAILED, message=f"{name}"
                    )
        except Exception as e:
            await self._job_info_client.put_status(
                self._job_id, JobStatus.FAILED, message=str(e)
            )
            raise e
