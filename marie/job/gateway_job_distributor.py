from typing import Callable, List, Optional

from docarray import BaseDoc, DocList
from docarray.documents import TextDoc

from marie.job.common import JobInfo, JobStatus
from marie.job.job_distributor import JobDistributor
from marie.logging.logger import MarieLogger
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.types.request.data import DataRequest


class GatewayJobDistributor(JobDistributor):
    def __init__(
        self,
        gateway_streamer: Optional[GatewayStreamer] = None,
        logger: Optional[MarieLogger] = None,
    ):
        self.streamer = gateway_streamer
        self.logger = logger or MarieLogger(self.__class__.__name__)

    async def submit_job(
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

        # attempt to get gateway streamer if not initialized
        if self.streamer is None:
            self.logger.warning(f"Gateway streamer is not initialized")
            raise RuntimeError("Gateway streamer is not initialized")

        parameters = {"job_id": submission_id}  # "#job_info.job_id,
        if job_info.metadata:
            parameters.update(job_info.metadata)

        doc = TextDoc(text=f"sample text : {job_info.entrypoint}")

        request = DataRequest()
        request.document_array_cls = DocList[BaseDoc]()
        request.header.exec_endpoint = "/extract"
        request.header.target_executor = "executor0"  # job_info.entrypoint
        request.parameters = parameters
        request.data.docs = DocList([doc])

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
