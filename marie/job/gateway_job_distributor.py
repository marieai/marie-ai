from typing import Callable, List, Optional

from docarray import DocList

from marie.api import AssetKeyDoc, parse_payload_to_docs
from marie.job.common import JobInfo, JobStatus
from marie.job.job_distributor import JobDistributor
from marie.logging_core.logger import MarieLogger
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.types_core.request.data import DataRequest


class GatewayJobDistributor(JobDistributor):
    def __init__(
        self,
        gateway_streamer: Optional[GatewayStreamer] = None,
        logger: Optional[MarieLogger] = None,
    ):
        self.streamer = gateway_streamer
        self.logger = logger or MarieLogger(self.__class__.__name__)

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
            self.logger.warning(f"Gateway streamer is not initialized")
            raise RuntimeError("Gateway streamer is not initialized")

        parameters = {"job_id": submission_id}  # "#job_info.job_id,
        if job_info.metadata:
            parameters.update(job_info.metadata)
        print(f"entrypoint = {job_info.entrypoint}")
        # metadata for payload is nested in metadata
        metadata = job_info.metadata.get("metadata", {})
        print(f"metadata: {metadata}")
        parameters, asset_doc = await parse_payload_to_docs(metadata)
        job_tag = parameters["ref_type"] if "ref_type" in parameters else ""
        parameters["job_id"] = submission_id
        parameters["payload"] = metadata  # THIS IS TEMPORARY HERE

        # entrypoint format is executor://endpoint/path or /endpoint/path
        exec_endpoint = job_info.entrypoint
        target_executor = None

        if "://" in job_info.entrypoint:
            target_executor, exec_endpoint = job_info.entrypoint.split("://", 1)

        print(f"exec_endpoint = {exec_endpoint}")
        print(f"target_executor = {target_executor}")

        request = DataRequest()
        request.document_array_cls = DocList[AssetKeyDoc]()
        request.header.exec_endpoint = exec_endpoint
        if target_executor:
            request.header.target_executor = target_executor
        request.parameters = parameters
        request.data.docs = DocList[AssetKeyDoc]([asset_doc])

        # doc = TextDoc(text=f"sample text : {job_info.entrypoint}")

        # metadata = job_info.metadata
        # print(f"metadata: {metadata}")
        # request = DataRequest()
        # request.document_array_cls = DocList[BaseDoc]()
        # request.header.exec_endpoint = "/extract"
        # # request.header.target_executor = "executor0"  # job_info.entrypoint
        # request.parameters = parameters
        # request.data.docs = DocList([doc])

        response = await self.streamer.process_single_data(
            request=request, send_callback=send_callback
        )

        return response

    async def close(self):
        """
        Closes the GatewayJobDistributor.

        :return: None
        """
        self.logger.debug(f"Closing GatewayJobDistributor")
        await self.streamer.close()
        self.logger.debug(f"GatewayJobDistributor closed")
