from typing import Callable, Dict, List, Optional

from docarray import DocList

from marie.api import AssetKeyDoc, parse_payload_to_docs
from marie.constants import __default_endpoint__
from marie.job.common import JobInfo, JobStatus
from marie.job.job_distributor import JobDistributor
from marie.logging_core.logger import MarieLogger
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.types_core.request.data import DataRequest


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

    async def send(
        self,
        submission_id: str,
        job_info: JobInfo,
        send_callback: Callable[[List[DataRequest]], DataRequest] = None,
    ) -> DataRequest:
        self.logger.info(f"Publishing job {job_info} to gateway")
        curr_status = job_info.status
        curr_message = job_info.message

        if curr_status != JobStatus.PENDING:
            raise RuntimeError(
                f"Job {submission_id} is not in PENDING state. "
                f"Current status is {curr_status} with message {curr_message}."
            )

        if self.streamer is None:
            raise RuntimeError("Gateway streamer is not initialized")

        if self.deployment_nodes is None:
            raise RuntimeError("Deployment nodes are not initialized")

        parameters = {"job_id": submission_id}  # "#job_info.job_id,
        if job_info.metadata:
            parameters.update(job_info.metadata)

        metadata = job_info.metadata.get("metadata", {})
        parameters, asset_doc = await parse_payload_to_docs(metadata)
        job_tag = parameters["ref_type"] if "ref_type" in parameters else ""
        parameters["job_id"] = submission_id
        parameters["payload"] = metadata  # THIS IS TEMPORARY HERE

        req_endpoint = (
            job_info.entrypoint
        )  # entrypoint format is executor://endpoint/path or /endpoint/path
        target_executor = None
        found = False

        if "://" in job_info.entrypoint:
            target_executor, req_endpoint = job_info.entrypoint.split("://", 1)
        if not req_endpoint.startswith("/"):
            req_endpoint = f"/{req_endpoint}"

        for executor, nodes in self.deployment_nodes.items():
            print('executor:', executor)
            print('nodes:', nodes)
            if not target_executor or target_executor == executor:
                for node in nodes:
                    if node['endpoint'] == req_endpoint:
                        found = True
                        break
                    if node['endpoint'] == __default_endpoint__:
                        req_endpoint = __default_endpoint__
                        found = True
                        break
                if found:
                    break
        if not found:
            raise RuntimeError(
                f"Invalid entrypoint {req_endpoint} for executor {target_executor}"
            )

        self.logger.info(f"exec_endpoint = {target_executor}@{req_endpoint}")

        request = DataRequest()
        request.document_array_cls = DocList[AssetKeyDoc]()
        request.header.exec_endpoint = req_endpoint
        if target_executor:
            request.header.target_executor = target_executor
        request.parameters = parameters
        request.data.docs = DocList[AssetKeyDoc]([asset_doc])

        return await self.streamer.process_single_data(
            request=request, send_callback=send_callback
        )

    async def close(self):
        """
        Closes the GatewayJobDistributor.

        :return: None
        """
        self.logger.debug(f"Closing GatewayJobDistributor")
        await self.streamer.close()
        self.logger.debug(f"GatewayJobDistributor closed")
