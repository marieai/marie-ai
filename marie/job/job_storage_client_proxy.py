from typing import Any, Dict, Optional

from marie.job.common import JobInfoStorageClient, JobStatus
from marie.job.event_publisher import EventPublisher
from marie.storage.kv.storage_client import StorageArea


class JobInfoStorageClientProxy(JobInfoStorageClient):
    """
    This class represents a proxy client for accessing a Job Info Storage service. The Job Info Storage
    service is responsible for storing and retrieving job information.

    ## Usage

    1. Create an instance of the JobInfoStorageClientProxy class, providing the necessary dependencies:
    ```
    job_info_storage_client = JobInfoStorageClient()
    event_publisher = EventPublisher()
    proxy_client = JobInfoStorageClientProxy(job_info_storage_client, event_publisher)
    ```

    Note: The implementation of the JobInfoStorageClientProxy class assumes the existence of the
    JobInfoStorageClient and EventPublisher classes.
    """

    def __init__(self, event_publisher: EventPublisher, storage: StorageArea):
        super().__init__(storage)
        self._event_publisher = event_publisher

    async def put_status(
        self,
        job_id: str,
        status: JobStatus,
        message: Optional[str] = None,
        jobinfo_replace_kwargs: Optional[Dict[str, Any]] = None,
    ):
        print("put_status called : ", job_id, status)
        await super().put_status(job_id, status, message, jobinfo_replace_kwargs)
        await self._event_publisher.publish(
            status,
            {
                "job_id": job_id,
                "status": status,
                "message": message,
                "jobinfo_replace_kwargs": jobinfo_replace_kwargs,
            },
        )
