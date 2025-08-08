import asyncio
import time
from collections.abc import Sequence
from typing import Dict, List, Optional, Union

import grpc
from docarray import BaseDoc, DocList
from docarray.documents import TextDoc
from grpc.aio import AioRpcError

from marie.constants import DEPLOYMENT_STATUS_PREFIX
from marie.job.common import ActorHandle, JobInfo, JobInfoStorageClient, JobStatus
from marie.job.event_publisher import EventPublisher
from marie.job.job_callback_executor import job_callback_executor
from marie.job.job_distributor import JobDistributor
from marie.logging_core.logger import MarieLogger
from marie.proto import jina_pb2
from marie.serve.discovery.etcd_client import EtcdClient
from marie.serve.networking import _NetworkingHistograms, _NetworkingMetrics
from marie.serve.networking.connection_stub import _ConnectionStubs
from marie.serve.networking.utils import get_grpc_channel
from marie.serve.runtimes.servers.cluster_state import ClusterState
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
        etcd_client: EtcdClient,
    ):
        self.logger = MarieLogger(self.__class__.__name__)
        self._job_id = job_id
        self._job_info_client = job_info_client
        self._job_distributor = job_distributor
        self._event_publisher = event_publisher
        self._etcd_client = etcd_client
        self.request_info = None

        self._active_tasks = set()
        self._submission_queue = asyncio.Queue()

    async def ping(self) -> bool:
        """
        Perform a health-check ping against the configured executor deployment.

        Returns:
            True if the executor responds successfully;
            False if the RPC was cancelled by the peer;
            otherwise raises RuntimeError.
        """
        if not self.request_info:
            # No target configured, assume healthy
            return True
        # FIXME : this is a hack to skip the health check due to overhead and killing GRPC
        if True:
            self.logger.debug('Skipping health-check ping')
            return True

        request_id = self.request_info.get("request_id")
        address = self.request_info.get("address")
        deployment = self.request_info.get("deployment")
        endpoint = "_jina_dry_run_"

        self.logger.debug(
            "Health-check: sending ping to %s (request_id=%s, deployment=%s)",
            address,
            request_id,
            deployment,
        )

        async with get_grpc_channel(address=address, asyncio=True) as channel:
            stub = _ConnectionStubs(
                address=address,
                channel=channel,
                deployment_name=deployment,
                metrics=_NetworkingMetrics(
                    sending_requests_time_metrics=None,
                    received_response_bytes=None,
                    send_requests_bytes_metrics=None,
                ),
                histograms=_NetworkingHistograms(),
            )

            # dry-run request
            doc = TextDoc(text=f"ping : {deployment}@_jina_dry_run_")
            request = DataRequest()
            request.document_array_cls = DocList[BaseDoc]()
            request.header.exec_endpoint = "_jina_dry_run_"
            request.header.target_executor = deployment
            request.parameters = {
                "job_id": self._job_id,
            }
            request.data.docs = DocList([doc])

            try:
                response, _ = await stub.send_requests(
                    requests=[request], metadata={}, compression=False, timeout=60
                )
                self.logger.debug("Health-check response: %s", response)

                if response.status.code != jina_pb2.StatusProto.SUCCESS:
                    raise RuntimeError(
                        f"Health-check endpoint '{endpoint}' returned status {response.status.code}"
                    )

                return True

            except AioRpcError as rpc_err:
                # Gracefully handle peer-initiated cancellations
                if rpc_err.code() == grpc.StatusCode.CANCELLED:
                    self.logger.warning(
                        "Health-check ping to %s cancelled by peer (request_id=%s)",
                        address,
                        request_id,
                    )
                    return False

                self.logger.error("Health-check ping AioRpcError: %s", rpc_err)
                raise RuntimeError(f"Error during health-check ping: {rpc_err}")

            except Exception as exc:
                msg = (
                    f"Health-check ping to {address} for request_id={request_id} "
                    f"on deployment={deployment} failed: {exc}"
                )
                self.logger.error(msg)
                raise RuntimeError(msg)

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

        # Enqueue job for processing, this will be processed in the background but it guarantees order of execution
        await self._submission_queue.put(curr_info)
        self.logger.debug(f"Job {self._job_id} enqueued for submission.")

        if self.DEFAULT_JOB_TIMEOUT_S > 0:
            try:
                await asyncio.wait_for(
                    self._submit_job_in_background(),
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
            task = asyncio.create_task(self._submit_job_in_background())
            self._active_tasks.add(task)
            task.add_done_callback(lambda t: self._active_tasks.discard(t))
        self.logger.info(f"Job submitted in the background : {self._job_id}")

    def send_callback(self, requests, request_info):
        job_callback_executor.submit(self._send_callback_sync, requests, request_info)

    def _send_callback_sync(
        self, requests: Union[List[DataRequest] | DataRequest], request_info: Dict
    ):
        """
        Callback when the job is submitted over the network to the executor.

        :param requests: The requests that were sent.
        :param request_info: The request info.
        """
        request = (
            requests[0] if isinstance(requests, Sequence) and requests else requests
        )

        if not request:
            self.logger.error("No valid requests provided.")
            return

        self.request_info = request_info

        required_keys = ["request_id", "address", "deployment"]
        if not all(key in request_info for key in required_keys):
            self.logger.error(f"Missing required keys in request_info: {request_info}")
            return

        # FIXME : This is a hack to trigger the event n iJob Scheduler
        request_id = request_info["request_id"]
        address = request_info["address"]
        deployment_name = request_info["deployment"]
        self.logger.info(f"Sent request to {address} on deployment {deployment_name}")

        from grpc_health.v1.health_pb2 import HealthCheckResponse

        status: HealthCheckResponse.ServingStatus = (
            HealthCheckResponse.ServingStatus.SERVICE_UNKNOWN
        )
        status_str = HealthCheckResponse.ServingStatus.Name(status)
        key = f"{DEPLOYMENT_STATUS_PREFIX}/{address}/{deployment_name}"

        # FIXME : This needs to be configured via config
        try:
            _lease_time = 5
            _lease = self._etcd_client.lease(_lease_time)
            res = self._etcd_client.put(key, status_str, lease=_lease)
            self.logger.debug(
                f"Updated Etcd with key {key}, status: {status_str}, lease time: {_lease_time}"
            )
        except Exception as e:
            self.logger.error(f"Failed to update Etcd for key {key}: {e}")
        finally:
            self.logger.debug(
                f"Setting scheduled_event for request_id: {request_info['request_id']}"
            )
            ClusterState.scheduled_event.set()

    async def _submit_job_in_background(self):
        try:
            start_time = time.monotonic()
            job_info: JobInfo = await self._submission_queue.get()

            self.logger.debug(f"Starting background submission for job: {self._job_id}")
            response = await self._job_distributor.send(
                submission_id=self._job_id,
                job_info=job_info,
                send_callback=self.send_callback,
            )

            elapsed = time.monotonic() - start_time
            self.logger.info(
                f"Job processed successfully in {elapsed:.2f}s for job {self._job_id}"
            )

            self.logger.debug(
                f"Response received for job {self._job_id} | "
                f"Status: {response.status.code} | "
                f"Parameters: {response.parameters} | "
                f"Docs: {response.data.docs}"
            )

            # This monitoring strategy allows us to have Floating Executors that can be used to run jobs outside of the main
            # deployment. This is useful for running jobs that are not part of the main deployment, but are still part of the
            # same deployment workflow.
            # Example would be calling a custom API, MPC or tools that we don't control.

            # If the job is already in a terminal state, then we don't need to update it. This can happen if the
            # job was cancelled while the job was being submitted.
            # or while the job was marked from the EXECUTOR worker node as "STOPPED", "SUCCEEDED", "FAILED".

            current_status = await self._job_info_client.get_status(self._job_id)
            if current_status is None:
                self.logger.warning(f"Job {self._job_id} status not found.")
                return

            if response.status.code == jina_pb2.StatusProto.SUCCESS:
                if current_status.is_terminal():
                    self.logger.debug(
                        f"Job {self._job_id} already in terminal state: {current_status}. Event will still be published."
                    )
                    await self._event_publisher.publish(
                        current_status,
                        {
                            "job_id": self._job_id,
                            "status": current_status,
                            "message": f"Job {self._job_id} already completed with status {current_status}.",
                            "jobinfo_replace_kwargs": False,
                        },
                    )
                else:
                    await self._job_info_client.put_status(
                        self._job_id, JobStatus.SUCCEEDED
                    )

            else:
                # Failure path
                exception_proto = response.status.exception
                error_name = str(exception_proto.name)

                if current_status.is_terminal():
                    self.logger.warning(
                        f"Job {self._job_id} failed but is already in terminal state: {current_status}. Event will still be published."
                    )
                    await self._event_publisher.publish(
                        current_status,
                        {
                            "job_id": self._job_id,
                            "status": current_status,
                            "message": f"Job {self._job_id} failed but already marked terminal: {current_status}.",
                            "jobinfo_replace_kwargs": False,
                        },
                    )
                else:
                    await self._job_info_client.put_status(
                        self._job_id, JobStatus.FAILED, message=error_name
                    )

        except Exception as e:
            from marie.excepts import InternalNetworkError

            error_message = str(e)
            error_label = (
                "Communication error with deployment"
                if isinstance(e, InternalNetworkError)
                else "Internal server error - job supervisor"
            )

            self.logger.error(
                f"Exception during job submission for {self._job_id}: {error_message}"
            )

            await self._job_info_client.put_status(
                self._job_id, JobStatus.FAILED, message=error_label
            )

            # Raising here ensures the task failure is visible to debuggers, but you can suppress this in production.
            raise
