import abc
import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional, Tuple, Union

from marie.excepts import ExecutorError
from marie.job.common import JobInfo
from marie.types_core.request.data import DataRequest, Request

# SendCb = Callable[[List[DataRequest]], Union[DataRequest, Awaitable[DataRequest]]]
SendCb = Callable[..., Any]  # now fully generic


class DocumentArray:
    pass


class JobDistributor(abc.ABC):
    """
    Job Distributor is responsible for publishing jobs to the underlying executor which can be a gateway/flow/deployment.
    """

    @abc.abstractmethod
    async def send(
        self,
        submission_id: str,
        job_info: JobInfo,
        send_callback: Optional[SendCb] = None,
    ) -> DataRequest:
        """
        Publish a job to the underlying executor. This is a synchronous method that will block until the job is completed
        or an error occurs.

        :param submission_id: The submission id of the job.
        :param job_info: The job info to publish.
        :param send_callback:  The callback after the job is submitted over the network.
        :return:
        """
        ...

    @abc.abstractmethod
    async def send_stream(
        self,
        submission_id: str,
        job_info: JobInfo,
        send_callback: Optional[SendCb] = None,
    ) -> AsyncIterator[Tuple[Union[DocumentArray, "Request"], "ExecutorError"]]:
        """
        Publish a job to the underlying executor.

        :param submission_id: The submission id of the job.
        :param job_info: The job info to publish.
        :param send_callback:  The callback after the job is submitted over the network.
        :return:
        """
        ...

    @abc.abstractmethod
    async def send_nowait(
        self,
        submission_id: str,
        job_info: JobInfo,
        send_callback: SendCb,
        *,
        wait_for_ack: float = 0.0,  # seconds to optionally wait for early ACK
    ) -> asyncio.Task:  # Task that resolves to the final DataRequest
        """
        Publish a job to the underlying executor without waiting for the result.
        :param submission_id:
        :param job_info:
        :param send_callback:
        :param wait_for_ack:
        :return:
        """
        ...
