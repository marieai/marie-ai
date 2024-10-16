import abc
from typing import Callable, Dict, List, Optional

from marie.job.common import JobInfo
from marie.types_core.request import Request
from marie.types_core.request.data import DataRequest


class JobDistributor(abc.ABC):
    """
    Job Distributor is responsible for publishing jobs to the underlying executor which can be a gateway/flow/deployment.
    """

    @abc.abstractmethod
    async def send(
        self,
        submission_id: str,
        job_info: JobInfo,
        send_callback: Optional[Callable[[List[Request], Dict[str, str]], None]] = None,
    ) -> DataRequest:
        """
        Publish a job to the underlying executor.

        :param submission_id: The submission id of the job.
        :param job_info: The job info to publish.
        :param send_callback:  The callback after the job is submitted over the network.
        :return:
        """
        ...
