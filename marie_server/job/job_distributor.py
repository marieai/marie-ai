import abc

from marie import Document
from marie.types.request.data import DataRequest
from marie_server.job.common import JobInfo


class JobDistributor(abc.ABC):
    """
    Job Distributor is responsible for publishing jobs to the underlying executor which can be a gateway/flow/deployment.
    """

    @abc.abstractmethod
    async def submit_job(self, job_info: JobInfo, doc: Document) -> DataRequest:
        """
        Publish a job.

        :param job_info: The job info to publish.
        :param doc: The document to process.
        :return:
        """
        ...
