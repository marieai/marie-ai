from typing import Optional, Any

from marie.clients.request import asyncio
from marie.logging.logger import MarieLogger
from marie.serve.runtimes.gateway.streamer import GatewayStreamer
from marie.types.request.data import DataRequest
from marie_server.job.common import JobInfo, JobStatus
from marie_server.job.job_distributor import JobDistributor
from m import Document
from marie import DocumentArray


class GatewayJobDistributor(JobDistributor):
    def __init__(
        self,
        gateway_streamer: Optional[GatewayStreamer] = None,
        logger: Optional[MarieLogger] = None,
    ):
        self.gateway_streamer = gateway_streamer
        self._logger = logger or MarieLogger(self.__class__.__name__)

    async def submit_job(self, job_info: JobInfo) -> DataRequest:
        self._logger.info(f"Publishing job {job_info} to gateway")
        curr_status = job_info.status
        curr_message = job_info.message

        if curr_status != JobStatus.PENDING:
            raise RuntimeError(
                f"Job {job_info._job_id} is not in PENDING state. "
                f"Current status is {curr_status} with message {curr_message}."
            )

        # attempt to get gateway streamer if not initialized
        if self.gateway_streamer is None:
            self._logger.warning(f"Gateway streamer is not initialized")
            self.gateway_streamer = GatewayStreamer.get_streamer()

        if self.gateway_streamer is None:
            raise Exception("Gateway streamer is not initialized")

        # convert job_info to DataRequest
        request = DataRequest()
        # request.header.exec_endpoint = on
        request.header.target_executor = job_info.entrypoint
        request.parameters = job_info.metadata

        request.data.docs = DocumentArray([Document(text="sample text")])
        response = await self.gateway_streamer.process_single_data(request=request)

        return response
