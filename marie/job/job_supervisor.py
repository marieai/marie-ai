import asyncio
import time
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import grpc
from docarray import BaseDoc, DocList
from docarray.documents import TextDoc
from grpc.aio import AioRpcError

from marie.helper import get_or_reuse_loop
from marie.job.common import ActorHandle, JobInfo, JobInfoStorageClient, JobStatus
from marie.job.event_publisher import EventPublisher
from marie.job.job_distributor import JobDistributor
from marie.logging_core.logger import MarieLogger
from marie.proto import jina_pb2
from marie.serve.discovery.etcd_client import EtcdClient
from marie.serve.networking import _NetworkingHistograms, _NetworkingMetrics
from marie.serve.networking.connection_stub import _ConnectionStubs
from marie.serve.networking.utils import get_grpc_channel
from marie.state.state_store import DesiredStore, StatusStore
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
        self._loop = get_or_reuse_loop()
        self._current_job_epoch: Optional[int] = None

        # K8s-style desired/status split
        self._desired_store = DesiredStore(self._etcd_client)
        self._status_store = StatusStore(self._etcd_client)

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
        # Capture the main event loop so callbacks from worker threads
        # self._loop = asyncio.get_running_loop()

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
        self.logger.info(f"Job enqueued for submission : {self._job_id}")

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
            await self._submit_job_in_background(curr_info)
            # task = asyncio.create_task(self._submit_job_in_background(curr_info))
            # self._active_tasks.add(task)
            # task.add_done_callback(lambda t: self._active_tasks.discard(t))

    def _signal_confirmation_threadsafe(self) -> None:
        """
        Signal the asyncio.Event from any thread safely and as early as possible.
        """
        if not self.confirmation_event:
            return
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self.confirmation_event.set)
        else:
            self.logger.warning(
                "No running event loop found, signaling confirmation directly."
            )
            try:
                self.confirmation_event.set()
            except Exception as e:
                self.logger.warning(f"Failed to signal confirmation (no loop): {e}")

    def _await_worker_ack(
        self, node: str, deployment: str, epoch: int, timeout_s: float = 5.0
    ) -> bool:
        """
        Wait up to timeout_s for the worker to write /status with a matching epoch.
        A matching epoch is sufficient to prove the worker has acknowledged the desired state.
        Server remains read-only for /status.
        """
        deadline = time.monotonic() + timeout_s
        self.logger.debug(
            f'_await_worker_ack with deadline: {deadline} : {self._job_id}'
        )
        while time.monotonic() < deadline:
            st = self._status_store.read(node, deployment)
            self.logger.debug(f'Status store read {st}')
            if st and st.epoch == epoch:
                self.logger.debug(
                    f'Worker ack received for epoch {epoch} with status {st.status_name}'
                )
                return True
            time.sleep(0.05)
        self.logger.warning(f'Timed out waiting for ack : {self._job_id}')
        return False

    async def failure_callback(
        self,
        requests: Union[List[DataRequest] | DataRequest],
        ctx: Dict,
        exception: Exception,
    ):
        """
        Callback executed when sending the request fails.
        """
        self.logger.error("Failure callback invoked.")
        node = ctx.get("address", "N/A")
        deployment_name = ctx.get("deployment", "N/A")
        self.logger.error(
            f"Request sending failed for {node}/{deployment_name}, job {self._job_id}. Exception: {exception}"
        )

    async def pre_send_callback(
        self, requests: Union[List[DataRequest], DataRequest], ctx: Dict
    ):
        """
        Only write 'desired' state. Capacity is reserved by PostgreSQLJobScheduler.
        JobSupervisor does not reserve or release capacity anymore.
        """
        self.logger.debug("Pre-send callback invoked (no reservation).")

        try:
            node_addr = ctx["address"]
            deployment = ctx["deployment"]

            node = self._netloc(node_addr)
            params = {"job_id": self._job_id}
            desired = await self._loop.run_in_executor(
                None, self._desired_store.schedule_new_epoch, node, deployment, params
            )
            self._current_job_epoch = desired.epoch if desired else None
            await asyncio.sleep(0.01)

        except Exception as e:
            self.logger.error(f"pre-send callback failed: {e}")
            self._current_job_epoch = None
            raise

    async def after_send_callback(
        self,
        requests: Union[List[DataRequest] | DataRequest],
        ctx: Dict,
        response: "DataRequest",  # WE do not have a response as this is send via GRPC waiting the response
    ):
        """
        Callback executed after a successful response is received. It waits for the worker acknowledgement.
        """
        self.logger.debug("After-send callback invoked.")
        node = ctx["address"]
        deployment_name = ctx["deployment"]
        epoch = self._current_job_epoch

        if epoch is None:
            self.logger.warning("No desired epoch recorded; skipping ack wait")
            return

        try:
            self._signal_confirmation_threadsafe()
            ack = await self._loop.run_in_executor(
                None, self._await_worker_ack, node, deployment_name, epoch
            )
            self.logger.debug(
                "Worker ack for %s/%s epoch=%s: %s", node, deployment_name, epoch, ack
            )
            if not ack:
                self.logger.warning(
                    f"Timed out waiting for worker ack for job {self._job_id}"
                )
        except Exception as e:
            self.logger.error("Ack wait error for %s/%s: %s", node, deployment_name, e)

    async def _submit_job_in_background(self, job_info: JobInfo):
        start_time = time.monotonic()

        try:
            self.logger.debug(
                "Starting background submission for job: %s", self._job_id
            )

            send_callbacks = (
                self.pre_send_callback,
                self.after_send_callback,
                self.failure_callback,
            )

            # Submit using the non-blocking wrapper, but DO NOT await the result here.
            send_task = await self._job_distributor.send_nowait(
                submission_id=self._job_id,
                job_info=job_info,
                send_callback=send_callbacks,
                wait_for_ack=0.0,  # no extra wait here
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
                            # Job is already in terminal state but infrastructure failed
                            # Override to FAILED since the overall operation failed
                            self.logger.error(
                                f"Job {self._job_id} failed after being marked {current_status} "
                                f"(infrastructure error). Overriding to FAILED. "
                                f"error_name = {error_name} \n{exception_proto}"
                            )
                            await self._job_info_client.put_status(
                                self._job_id,
                                JobStatus.FAILED,
                                message=error_name,
                                force=True,
                            )
                        else:
                            # Normal failure - update status to FAILED
                            self.logger.error(
                                f"Job {self._job_id} failed (status was {current_status}). "
                                f"error_name = {error_name} \n{exception_proto}"
                            )
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

    @staticmethod
    def _netloc(addr: str) -> str:
        # Accepts 'grpc://host:port' or 'host:port' and returns 'host:port'
        if "://" in addr:
            try:
                p = urlparse(addr)
                return p.netloc or addr
            except Exception:
                return addr
        return addr
