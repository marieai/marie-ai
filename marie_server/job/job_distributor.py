import abc

from marie.types.request.data import DataRequest
from marie_server.job.common import JobInfo


class JobDistributor(abc.ABC):
    """
    Job Distributor is responsible for publishing jobs to the underlying executor which can be a gateway/flow.
    """

    @abc.abstractmethod
    async def submit_job(self, job_info: JobInfo) -> DataRequest:
        """
        Publish a job.

        :param job_info: The job info to publish.
        :return:
        """
        ...
