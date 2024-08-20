import abc
from typing import Callable, Dict, List, Optional

from marie.types.request import Request
from marie.types.request.data import DataRequest
from marie_server.job.common import JobInfo


class JobDistributor(abc.ABC):
    """
    Job Distributor is responsible for publishing jobs to the underlying executor which can be a gateway/flow/deployment.
    """

    @abc.abstractmethod
    async def submit_job(
        self,
        job_info: JobInfo,
        send_callback: Optional[Callable[[List[Request], Dict[str, str]], None]] = None,
    ) -> DataRequest:
        """
        Publish a job.

        :param job_info: The job info to publish.
        :param send_callback:  The callback after the job is submitted over the network.
        :return:
        """
        ...
