from typing import Optional

from marie import Document, DocumentArray
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
        self._logger = logger or MarieLogger(self.__class__.__name__)

    async def submit_job(self, job_info: JobInfo, doc: Document) -> DataRequest:
        self._logger.info(f"Publishing job {job_info} to gateway")
        curr_status = job_info.status
        curr_message = job_info.message

        if curr_status != JobStatus.PENDING:
            raise RuntimeError(
                f"Job {job_info._job_id} is not in PENDING state. "
                f"Current status is {curr_status} with message {curr_message}."
            )

        # attempt to get gateDDDway streamer if not initialized
        if self.streamer is None:
            self._logger.warning(f"Gateway streamer is not initialized")
            raise RuntimeError("Gateway streamer is not initialized")

        async for docs in self.streamer.stream_docs(
            doc=doc,
            # exec_endpoint="/extract",  # _jina_dry_run_
            exec_endpoint="_jina_dry_run_",  # _jina_dry_run_
            # target_executor="executor0",
            return_results=False,
        ):
            self._logger.info(f"Received {len(docs)} docs from gateway")
            print(docs)
            result = docs[0].text

        return result

        if False:
            # convert job_info to DataRequest
            request = DataRequest()
            # request.header.exec_endpoint = on
            request.header.target_executor = job_info.entrypoint
            request.parameters = job_info.metadata

            request.data.docs = DocumentArray([Document(text="sample text")])
            response = await self.streamer.process_single_data(request=request)

            return response
