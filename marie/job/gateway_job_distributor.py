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
    ):
        """
        GatewayJobDistributor constructor.
        :param gateway_streamer: GatewayStreamer instance to be used to send the jobs to executors.
        :param deployment_nodes: Optional dictionary with the nodes that are going to be used in the graph. If not provided, the graph will be built using the executor_addresses.
        :param logger: Optional logger to be used by the GatewayJobDistributor.
        """
        self.logger = logger or MarieLogger(self.__class__.__name__)
        self.streamer = gateway_streamer
        self.deployment_nodes = deployment_nodes or {}

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
            request=request, send_callback=wrapped_cb
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
                send_callback=wrapped_cb,
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

    async def close(self):
        self.logger.debug("Closing GatewayJobDistributor")
        await self.streamer.close()
