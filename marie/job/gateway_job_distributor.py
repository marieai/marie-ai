import asyncio
import inspect
import time
from functools import wraps
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from docarray import DocList

from marie.api import AssetKeyDoc, parse_payload_to_docs
from marie.constants import __default_endpoint__
from marie.excepts import ExecutorError, InternalNetworkError
from marie.job.common import JobInfo, JobStatus
from marie.job.job_distributor import DocumentArray, JobDistributor, SendCb
from marie.logging_core.logger import MarieLogger
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.types_core.request.data import DataRequest, Request


class GatewayJobDistributor(JobDistributor):
    def __init__(
        self,
        gateway_streamer: Optional[GatewayStreamer] = None,
        deployment_nodes: Optional[Dict[str, List]] = None,
        logger: Optional[MarieLogger] = None,
        ready_event: Optional[asyncio.Event] = None,
    ):
        """
        GatewayJobDistributor constructor.
        :param gateway_streamer: GatewayStreamer instance to be used to send the jobs to executors.
        :param deployment_nodes: Optional dictionary with the nodes that are going to be used in the graph. If not provided, the graph will be built using the executor_addresses.
        :param logger: Optional logger to be used by the GatewayJobDistributor.
        :param ready_event: Optional asyncio.Event that signals when the gateway is ready to process jobs.
        """
        self.logger = logger or MarieLogger(self.__class__.__name__)
        self.streamer = gateway_streamer
        self.deployment_nodes = deployment_nodes or {}
        self._inflight: Dict[str, asyncio.Task] = {}
        self._ready_event = ready_event
        self._readiness_timeout = 60  # seconds to wait for gateway initialization

    def _validate_preconditions(
        self, submission_id: str, job_info: JobInfo, send_callback: Optional[SendCb]
    ):
        if job_info.status != JobStatus.PENDING:
            raise RuntimeError(
                f"Job {submission_id} not in PENDING state (is {job_info.status})."
            )
        if self.streamer is None:
            raise RuntimeError("Gateway streamer is not initialized")
        if self.deployment_nodes is None:
            raise RuntimeError("Deployment nodes are not initialized")
        if send_callback is None:
            raise ValueError("send_callback must be provided")

    def _wrap_callback(
        self, submission_id: str, send_callback: SendCb
    ) -> Callable[..., Any]:
        @wraps(send_callback)
        async def wrapped(*args, **kwargs):
            start = time.monotonic()
            try:
                result = send_callback(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as e:
                elapsed = time.monotonic() - start
                self.logger.error(
                    f"[{submission_id}] send_callback error after {elapsed:.3f}s: {e}"
                )
                raise
            elapsed = time.monotonic() - start
            self.logger.debug(
                f"[{submission_id}] send_callback succeeded in {elapsed:.3f}s"
            )
            if elapsed > 1:
                self.logger.warning(
                    f"[{submission_id}] send_callback took too long: {elapsed:.3f}s"
                )
            return result

        return wrapped

    async def _build_payload(self, submission_id: str, job_info: JobInfo):
        # parse the incoming metadata into parameters + document
        metadata = job_info.metadata.get("metadata", {})
        parameters, asset_doc = await parse_payload_to_docs(metadata)
        parameters["job_id"] = submission_id
        parameters["payload"] = metadata

        # Pass through DAG tracking parameters for asset materialization
        # These are set by the scheduler when dispatching DAG jobs
        if "dag_id" in job_info.metadata:
            parameters["dag_id"] = job_info.metadata["dag_id"]
        if "node_task_id" in job_info.metadata:
            parameters["node_task_id"] = job_info.metadata["node_task_id"]
        if "partition_key" in job_info.metadata:
            parameters["partition_key"] = job_info.metadata["partition_key"]

        return parameters, asset_doc

    def _resolve_endpoint(self, submission_id: str, entrypoint: str):
        # returns (target_executor, endpoint)
        # split out “executor://path” if present
        if "://" in entrypoint:
            target, ep = entrypoint.split("://", 1)
        else:
            target, ep = None, entrypoint
        if not ep.startswith("/"):
            ep = "/" + ep

        found = False
        for exec_name, nodes in self.deployment_nodes.items():
            if target and exec_name != target:
                continue
            for node in nodes:
                if node["endpoint"] == ep:
                    found = True
                    break
            if not found:
                for node in nodes:
                    if node["endpoint"] == __default_endpoint__:
                        ep = __default_endpoint__
                        found = True
                        break
            if found:
                return exec_name, ep

        avail = [
            f"{e}:{n['endpoint']}"
            for e, nl in self.deployment_nodes.items()
            for n in nl
        ]
        raise RuntimeError(
            f"[{submission_id}] Invalid entrypoint {ep} for executor {target}. "
            f"Available: {avail}"
        )

    async def send(
        self,
        submission_id: str,
        job_info: JobInfo,
        send_callback: Optional[SendCb] = None,
    ) -> DataRequest:
        self._validate_preconditions(submission_id, job_info, send_callback)

        wrapped_cb = self._wrap_callback(submission_id, send_callback)  # type: ignore
        parameters, asset_doc = await self._build_payload(submission_id, job_info)
        target_exec, endpoint = self._resolve_endpoint(
            submission_id, job_info.entrypoint
        )

        # build the DataRequest
        request = DataRequest()
        request.document_array_cls = DocList[AssetKeyDoc]()
        request.header.exec_endpoint = endpoint
        if target_exec:
            request.header.target_executor = target_exec
        request.parameters = parameters
        request.data.docs = DocList[AssetKeyDoc]([asset_doc])

        self.logger.info(f"[{submission_id}] Publishing job via single-send")
        return await self.streamer.process_single_data(
            request=request,
            # send_callback=wrapped_cb
            send_callback=send_callback,
        )

    async def send_stream(
        self,
        submission_id: str,
        job_info: JobInfo,
        send_callback: Optional[SendCb] = None,
    ) -> AsyncIterator[Tuple[Union[DocumentArray, "Request"], ExecutorError]]:
        self._validate_preconditions(submission_id, job_info, send_callback)

        wrapped_cb = self._wrap_callback(submission_id, send_callback)  # type: ignore
        parameters, asset_doc = await self._build_payload(submission_id, job_info)
        target_exec, endpoint = self._resolve_endpoint(
            submission_id, job_info.entrypoint
        )

        self.logger.info(f"[{submission_id}] Publishing job via stream_send")
        docs = DocList[AssetKeyDoc]([asset_doc])

        try:
            async for item, err in self.streamer.stream_docs(
                docs=docs,
                return_results=False,
                exec_endpoint=endpoint,
                target_executor=target_exec,
                parameters=parameters,
                request_id=submission_id,
                # send_callback=wrapped_cb,
                send_callback=send_callback,
            ):
                self.logger.info(
                    f"[{submission_id}] Stream send job published successfully"
                )
                yield item, err

        except InternalNetworkError as err:
            import grpc

            if (
                err.code() == grpc.StatusCode.UNAVAILABLE
                or err.code() == grpc.StatusCode.NOT_FOUND
            ):
                self.logger.error(
                    f"Error while getting responses from deployments SERVICE_UNAVAILABLE: {err.details()}"
                )
            elif err.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                self.logger.error(
                    f"Error while getting responses from deployments DEADLINE_EXCEEDED: {err.details()}"
                )
            else:
                self.logger.error(
                    f"Error while getting responses from deployments: {err.details()}"
                )

            raise err

    async def send_nowait(
        self,
        submission_id: str,
        job_info: JobInfo,
        send_callback: SendCb,
        *,
        wait_for_ack: float = 0.0,  # seconds to optionally wait for early ACK
    ) -> asyncio.Task:  # Task that resolves to the final DataRequest
        """
        Fire-and-forget submit. Runs the existing `send()` (blocking path)
        in the background and returns an asyncio.Task you can await/cancel later.
        If `wait_for_ack` > 0, we wait up to that many seconds for the callback
        to run (serves as an early ACK).
        """

        async def _run():
            # Use the original blocking `send()` (returns final DataRequest)
            return await self.send(
                submission_id=submission_id,
                job_info=job_info,
                send_callback=send_callback,
            )

        task = asyncio.create_task(_run(), name=f"job:{submission_id}")
        self._inflight[submission_id] = task

        def _done(t: asyncio.Task):
            self._inflight.pop(submission_id, None)
            exc = t.exception()
            if exc:
                self.logger.error("Job task crashed for %s: %s", submission_id, exc)

        task.add_done_callback(_done)

        # if ack:
        #     try:
        #         await asyncio.wait_for(ack.wait(), timeout=wait_for_ack)
        #     except asyncio.TimeoutError:
        #         self.logger.warning("[%s] No ACK within %.2fs (continuing)", submission_id, wait_for_ack)

        return task

    async def wait_for_result(
        self, submission_id: str, *, timeout: Optional[float] = None
    ) -> DataRequest:
        task = self._inflight.get(submission_id)
        if not task:
            raise RuntimeError(f"No inflight job for {submission_id}")
        return await asyncio.wait_for(task, timeout=timeout)

    def cancel_job(self, submission_id: str) -> bool:
        t = self._inflight.get(submission_id)
        if not t:
            return False
        t.cancel()
        return True

    async def close(self):
        self.logger.debug("Closing GatewayJobDistributor")
        await self.streamer.close()
