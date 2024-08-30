from typing import Callable, List, Optional

from django.views.debug import CallableSettingWrapper
from docarray import BaseDoc, DocList
from docarray.documents import TextDoc

from marie import DocumentArray
from marie.logging.logger import MarieLogger
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.types.request.data import DataRequest
from marie_server.job.common import JobInfo, JobStatus
from marie_server.job.job_distributor import JobDistributor


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
        job_info: JobInfo,
        send_callback: Callable[[List[DataRequest]], DataRequest] = None,
    ) -> DataRequest:
        self.logger.info(f"Publishing job {job_info} to gateway")
        curr_status = job_info.status
        curr_message = job_info.message

        if curr_status != JobStatus.PENDING:
            raise RuntimeError(
                f"Job {job_info._job_id} is not in PENDING state. "
                f"Current status is {curr_status} with message {curr_message}."
            )

        # attempt to get gateway streamer if not initialized
        if self.streamer is None:
            self.logger.warning(f"Gateway streamer is not initialized")
            raise RuntimeError("Gateway streamer is not initialized")

        doc = TextDoc(text=f"sample text : {job_info.entrypoint}")
        request = DataRequest()
        request.document_array_cls = DocList[BaseDoc]()
        request.header.exec_endpoint = "/extract"
        request.header.target_executor = "executor0"  # job_info.entrypoint
        request.parameters = {}  # job_info.metadata
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
