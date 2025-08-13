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
    Supervise submitted job and keep track of their status on the remote executor.

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
        confirmation_event: asyncio.Event,
    ):
        self.logger = MarieLogger(self.__class__.__name__)
        self._job_id = job_id
        self._job_info_client = job_info_client
        self._job_distributor = job_distributor
        self._event_publisher = event_publisher
        self._etcd_client = etcd_client
        self.request_info = None
        self.confirmation_event = confirmation_event  # we need to make sure that this is per job confirmation event
        self._active_tasks = set()

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
        if False:
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
        self.logger.debug(f"Job {self._job_id} enqueued for submission.")

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
        # Validate and extract
        req = requests[0] if isinstance(requests, Sequence) and requests else requests
        if not req:
            self.logger.error("No valid requests provided.")
            return
        self.request_info = request_info
        required = ["request_id", "address", "deployment"]
        if not all(k in request_info for k in required):
            self.logger.error(f"Missing required keys in request_info: {request_info}")
            return

        start = time.monotonic()
        request_id = request_info["request_id"]
        address = request_info["address"]
        deployment_name = request_info["deployment"]
        self.logger.info(f"Sent request to {address} on deployment {deployment_name}")

        # Signal immediately to avoid blocking the callback thread
        try:
            ClusterState.scheduled_event.set()
            if self.confirmation_event:
                self.confirmation_event.set()
        except Exception as e:
            self.logger.warning(f"Failed to signal confirmation for {request_id}: {e}")

        t_signal = time.monotonic()

        # Do etcd update best-effort AFTER signaling (still on the callback thread)
        try:
            from grpc_health.v1.health_pb2 import HealthCheckResponse

            status = HealthCheckResponse.ServingStatus.SERVING
            status_str = HealthCheckResponse.ServingStatus.Name(status)
            key = f"{DEPLOYMENT_STATUS_PREFIX}/{address}/{deployment_name}"

            lease_time = 5
            t0 = time.monotonic()
            lease = self._etcd_client.lease(lease_time)
            t1 = time.monotonic()
            self._etcd_client.put(key, status_str, lease=lease)
            t2 = time.monotonic()

            self.logger.info(
                "Etcd update for %s: lease=%.3fs put=%.3fs total=%.3fs",
                key,
                t1 - t0,
                t2 - t1,
                t2 - t0,
            )
        except Exception as e:
            self.logger.error(f"Failed to update Etcd for job {self._job_id}: {e}")

        total = time.monotonic() - start
        self.logger.info(
            "Callback _send_callback_sync executed in %.2fs (signal=%.3fs, post-signal=%.3fs).",
            total,
            t_signal - start,
            total - (t_signal - start),
        )

    async def _submit_job_in_background(self, job_info: JobInfo):
        start_time = time.monotonic()

        try:
            self.logger.debug(
                "Starting background submission for job: %s", self._job_id
            )
            # Submit using the non-blocking wrapper, but DO NOT await the result here.
            send_task = await self._job_distributor.send_nowait(
                submission_id=self._job_id,
                job_info=job_info,
                send_callback=self.send_callback,  # unchanged semantics
                wait_for_ack=0.0,  # no extra wait here
            )

            if self.confirmation_event is not None:
                try:
                    await asyncio.wait_for(self.confirmation_event.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    self.logger.warning(
                        "No ACK within 3s for job %s (continuing)", self._job_id
                    )

            self.logger.info(
                "Job enqueued in %.2fs for job %s",
                time.monotonic() - start_time,
                self._job_id,
            )

            async def _finalize_when_done():
                """
                Finalize the job once the send_task is completed.
                """
                try:
                    start_time = time.monotonic()
                    response = (
                        await send_task
                    )  # this may take long time, so we await it here

                    elapsed = time.monotonic() - start_time
                    self.logger.info(
                        f"Job processed successfully in {elapsed:.2f}s for job {self._job_id}"
                    )

                    current_status = await self._job_info_client.get_status(
                        self._job_id
                    )
                    if current_status is None:
                        self.logger.warning(
                            "Job %s status not found on finalize", self._job_id
                        )
                        return

                    if response.status.code == jina_pb2.StatusProto.SUCCESS:
                        if current_status.is_terminal():
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
                except asyncio.CancelledError:
                    curr = await self._job_info_client.get_status(self._job_id)
                    if curr and not curr.is_terminal():
                        await self._job_info_client.put_status(
                            self._job_id,
                            JobStatus.FAILED,
                            message="Submission cancelled",
                        )
                    raise  # ?? should we re-raise here?
                except Exception as e:
                    curr = await self._job_info_client.get_status(self._job_id)
                    if curr and not curr.is_terminal():
                        await self._job_info_client.put_status(
                            self._job_id,
                            JobStatus.FAILED,
                            message=(
                                "Communication error with deployment"
                                if e.__class__.__name__.endswith("InternalNetworkError")
                                else "Internal server error - job supervisor"
                            ),
                        )
                    self.logger.exception("Finalize failed for job %s", self._job_id)

            asyncio.create_task(_finalize_when_done(), name=f"finalize:{self._job_id}")

        except Exception as e:
            self.logger.error(
                "Exception during job submission for %s: %s", self._job_id, e
            )
            await self._job_info_client.put_status(
                self._job_id, JobStatus.FAILED, message="Submission error"
            )
            raise
