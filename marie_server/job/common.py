import asyncio
import json
import time
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from marie.constants import KV_NAMESPACE_JOB
from marie_server.storage.storage_client import StorageArea

JOB_ID_METADATA_KEY = "job_submission_id"
JOB_NAME_METADATA_KEY = "job_name"

INTERNAL_NAMESPACE_PREFIX = "marie_internal"
JOB_STATUS_KEY = f"{INTERNAL_NAMESPACE_PREFIX}/job_status"

ActorHandle = Any


class JobStatus(str, Enum):
    """An enumeration for describing the status of a job."""

    #: The job has not started yet, likely waiting for the runtime_env to be set up.
    PENDING = "PENDING"
    #: The job is currently running.
    RUNNING = "RUNNING"
    #: The job was intentionally stopped by the user.
    STOPPED = "STOPPED"
    #: The job finished successfully.
    SUCCEEDED = "SUCCEEDED"
    #: The job failed.
    FAILED = "FAILED"

    def __str__(self) -> str:
        return f"{self.value}"

    def is_terminal(self) -> bool:
        """Return whether or not this status is terminal.

        A terminal status is one that cannot transition to any other status.
        The terminal statuses are "STOPPED", "SUCCEEDED" and "FAILED"

        Returns:
            True if this status is terminal, otherwise False.
        """
        return self.value in {"STOPPED", "SUCCEEDED", "FAILED"}


@dataclass
class JobInfo:
    """A class for recording information associated with a job and its execution.

    Please keep this in sync with the JobsAPIInfo proto in src/ray/protobuf/gcs.proto.
    """

    #: The status of the job.
    status: JobStatus
    #: The entrypoint command for this job.
    entrypoint: str
    #: A message describing the status in more detail.
    message: Optional[str] = None
    # TODO(architkulkarni): Populate this field with e.g. Runtime env setup failure,
    # Internal error, user script error
    error_type: Optional[str] = None
    #: The time when the job was started.  A Unix timestamp in ms.
    start_time: Optional[int] = None
    #: The time when the job moved into a terminal state.  A Unix timestamp in ms.
    end_time: Optional[int] = None
    #: Arbitrary user-provided metadata for the job.
    metadata: Optional[Dict[str, str]] = None
    #: The runtime environment for the job.
    runtime_env: Optional[Dict[str, Any]] = None
    #: The quantity of CPU cores to reserve for the entrypoint command.
    entrypoint_num_cpus: Optional[Union[int, float]] = None
    #: The number of GPUs to reserve for the entrypoint command.
    entrypoint_num_gpus: Optional[Union[int, float]] = None
    #: The quantity of various custom resources to reserve for the entrypoint command.
    entrypoint_resources: Optional[Dict[str, float]] = None
    #: Driver agent http address
    driver_agent_http_address: Optional[str] = None
    #: The node id that driver running on. It will be None only when the job status
    # is PENDING, and this field will not be deleted or modified even if the driver dies
    driver_node_id: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = JobStatus(self.status)
        if self.message is None:
            if self.status == JobStatus.PENDING:
                self.message = "Job has not started yet."
                if any(
                    [
                        self.entrypoint_num_cpus is not None
                        and self.entrypoint_num_cpus > 0,
                        self.entrypoint_num_gpus is not None
                        and self.entrypoint_num_gpus > 0,
                        self.entrypoint_resources not in [None, {}],
                    ]
                ):
                    self.message += (
                        " It may be waiting for resources "
                        "(CPUs, GPUs, custom resources) to become available."
                    )
                if self.runtime_env not in [None, {}]:
                    self.message += (
                        " It may be waiting for the runtime environment to be set up."
                    )
            elif self.status == JobStatus.RUNNING:
                self.message = "Job is currently running."
            elif self.status == JobStatus.STOPPED:
                self.message = "Job was intentionally stopped."
            elif self.status == JobStatus.SUCCEEDED:
                self.message = "Job finished successfully."
            elif self.status == JobStatus.FAILED:
                self.message = "Job failed."

    def to_json(self) -> Dict[str, Any]:
        """Convert this object to a JSON-serializable dictionary.

        Note that the runtime_env field is converted to a JSON-serialized string
        and the field is renamed to runtime_env_json.

        Returns:
            A JSON-serializable dictionary representing the JobInfo object.
        """

        json_dict = asdict(self)

        # Convert enum values to strings.
        json_dict["status"] = str(json_dict["status"])

        # Convert runtime_env to a JSON-serialized string.
        if "runtime_env" in json_dict:
            if json_dict["runtime_env"] is not None:
                json_dict["runtime_env_json"] = json.dumps(json_dict["runtime_env"])
            del json_dict["runtime_env"]

        # Assert that the dictionary is JSON-serializable.
        json.dumps(json_dict)

        return json_dict

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> None:
        """Initialize this object from a JSON dictionary.

        Note that the runtime_env_json field is converted to a dictionary and
        the field is renamed to runtime_env.

        Args:
            json_dict: A JSON dictionary to use to initialize the JobInfo object.
        """
        # Convert enum values to enum objects.
        json_dict["status"] = JobStatus(json_dict["status"])

        # Convert runtime_env from a JSON-serialized string to a dictionary.
        if "runtime_env_json" in json_dict:
            if json_dict["runtime_env_json"] is not None:
                json_dict["runtime_env"] = json.loads(json_dict["runtime_env_json"])
            del json_dict["runtime_env_json"]

        return cls(**json_dict)


class JobInfoStorageClient:
    """
    Interface to put and get job data from the Internal KV store.
    """

    # Please keep this format in sync with JobDataKey()
    # in src/ray/gcs/gcs_server/gcs_job_manager.h.
    JOB_DATA_KEY_PREFIX = f"{INTERNAL_NAMESPACE_PREFIX}job_info_"
    JOB_DATA_KEY = f"{JOB_DATA_KEY_PREFIX}{{job_id}}"

    def __init__(self, storage: StorageArea):
        self.storage = storage

    async def put_info(
        self, job_id: str, job_info: JobInfo, overwrite: bool = True
    ) -> bool:
        """Put job info to the internal kv store.

        Args:
            job_id: The job id.
            job_info: The job info.
            overwrite: Whether to overwrite the existing job info.

        Returns:
            True if a new key is added.
        """
        added_num = await self.storage.internal_kv_put(
            self.JOB_DATA_KEY.format(job_id=job_id).encode(),
            json.dumps(job_info.to_json()).encode(),
            overwrite,
            namespace=KV_NAMESPACE_JOB,
        )
        return added_num == 1

    async def get_info(self, job_id: str, timeout: int = 30) -> Optional[JobInfo]:
        serialized_info = await self.storage.internal_kv_get(
            self.JOB_DATA_KEY.format(job_id=job_id).encode(),
            namespace=KV_NAMESPACE_JOB,
            timeout=timeout,
        )
        if serialized_info is None:
            return None
        elif isinstance(serialized_info, bytes):
            return JobInfo.from_json(json.loads(serialized_info))
        elif isinstance(serialized_info, dict):
            return JobInfo.from_json(serialized_info)

    async def delete_info(self, job_id: str, timeout: int = 30):
        await self.storage.internal_kv_del(
            self.JOB_DATA_KEY.format(job_id=job_id).encode(),
            False,
            namespace=KV_NAMESPACE_JOB,
            timeout=timeout,
        )

    async def put_status(
        self,
        job_id: str,
        status: JobStatus,
        message: Optional[str] = None,
        jobinfo_replace_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Puts or updates job status.  Sets end_time if status is terminal."""

        old_info = await self.get_info(job_id)

        if jobinfo_replace_kwargs is None:
            jobinfo_replace_kwargs = dict()
        jobinfo_replace_kwargs.update(status=status, message=message)
        if old_info is not None:
            if status != old_info.status and old_info.status.is_terminal():
                assert False, "Attempted to change job status from a terminal state."
            new_info = replace(old_info, **jobinfo_replace_kwargs)
        else:
            new_info = JobInfo(
                entrypoint="Entrypoint not found.", **jobinfo_replace_kwargs
            )

        if status.is_terminal():
            new_info.end_time = int(time.time() * 1000)

        await self.put_info(job_id, new_info)

    async def get_status(self, job_id: str) -> Optional[JobStatus]:
        job_info = await self.get_info(job_id)
        if job_info is None:
            return None
        else:
            return job_info.status

    async def get_all_jobs(self, timeout: int = 30) -> Dict[str, JobInfo]:
        raw_job_ids_with_prefixes = await self.storage.internal_kv_keys(
            self.JOB_DATA_KEY_PREFIX.encode(),
            namespace=KV_NAMESPACE_JOB,
            timeout=timeout,
        )
        job_ids_with_prefixes = [
            job_id.decode() if isinstance(job_id, bytes) else job_id
            for job_id in raw_job_ids_with_prefixes
        ]
        job_ids = []
        for job_id_with_prefix in job_ids_with_prefixes:
            assert job_id_with_prefix.startswith(
                self.JOB_DATA_KEY_PREFIX
            ), "Unexpected format for internal_kv key for Job submission"
            job_ids.append(job_id_with_prefix[len(self.JOB_DATA_KEY_PREFIX) :])

        async def get_job_info(job_id: str):
            job_info = await self.get_info(job_id, timeout)
            return job_id, job_info

        return {
            job_id: job_info
            for job_id, job_info in await asyncio.gather(
                *[get_job_info(job_id) for job_id in job_ids]
            )
        }
