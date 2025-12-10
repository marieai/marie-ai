"""
S3StateBackend - Amazon S3 state backend for distributed storage.

This backend stores state as JSON objects in S3, suitable for
serverless or distributed environments where shared filesystem
or database access is not available.

Requires the 's3' optional dependency:
    pip install marie-kernel[s3]
"""

import json
from typing import Any, Iterable, Optional

from marie_kernel.ref import TaskInstanceRef

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore


class S3StateBackend:
    """
    Amazon S3 backend for task state storage.

    Stores each state entry as a JSON object in S3. Object keys follow
    the pattern: {prefix}/{tenant_id}/{dag_name}/{dag_id}/{task_id}/{try_number}/{key}.json

    This backend is eventually consistent (S3 standard behavior) and
    suitable for workflows where strong consistency is not required.

    Example:
        ```python
        import boto3

        s3_client = boto3.client('s3')
        backend = S3StateBackend(
            s3_client=s3_client, bucket="my-state-bucket", prefix="marie-state"
        )

        ti = TaskInstanceRef(
            tenant_id="acme",
            dag_name="document_pipeline",
            dag_id="run_2024_001",
            task_id="extract_text",
            try_number=1,
        )
        backend.push(ti, "result", {"text": "Hello World"})
        ```

    Object Structure:
        Each object contains:
        ```json
        {
            "value": <the stored value>,
            "metadata": {<optional metadata>},
            "task_instance": {
                "tenant_id": "...",
                "dag_name": "...",
                "dag_id": "...",
                "task_id": "...",
                "try_number": 1
            }
        }
        ```
    """

    def __init__(
        self,
        s3_client: Any,
        bucket: str,
        prefix: str = "marie-state",
    ) -> None:
        """
        Initialize with an S3 client and bucket configuration.

        Args:
            s3_client: boto3 S3 client instance
            bucket: S3 bucket name for state storage
            prefix: Key prefix for all state objects (default: "marie-state")
        """
        if boto3 is None:
            raise ImportError(
                "boto3 is required for S3StateBackend. "
                "Install with: pip install marie-kernel[s3]"
            )
        self._client = s3_client
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

    def _build_key(
        self,
        ti: TaskInstanceRef,
        key: str,
        *,
        task_id_override: Optional[str] = None,
    ) -> str:
        """Build S3 object key for a state entry."""
        task_id = task_id_override or ti.task_id
        return (
            f"{self._prefix}/{ti.tenant_id}/{ti.dag_name}/"
            f"{ti.dag_id}/{task_id}/{ti.try_number}/{key}.json"
        )

    def _build_task_prefix(
        self,
        ti: TaskInstanceRef,
        *,
        include_try_number: bool = True,
    ) -> str:
        """Build S3 prefix for listing task state."""
        base = (
            f"{self._prefix}/{ti.tenant_id}/{ti.dag_name}/" f"{ti.dag_id}/{ti.task_id}/"
        )
        if include_try_number:
            return f"{base}{ti.try_number}/"
        return base

    def push(
        self,
        ti: TaskInstanceRef,
        key: str,
        value: Any,
        *,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a value in S3.

        Creates or overwrites an object at the computed key path.

        Args:
            ti: Task instance reference
            key: State key
            value: JSON-serializable value to store
            metadata: Optional metadata dict

        Raises:
            TypeError: If value is not JSON-serializable
            ClientError: If S3 operation fails
        """
        s3_key = self._build_key(ti, key)

        payload = {
            "value": value,
            "metadata": metadata or {},
            "task_instance": {
                "tenant_id": ti.tenant_id,
                "dag_name": ti.dag_name,
                "dag_id": ti.dag_id,
                "task_id": ti.task_id,
                "try_number": ti.try_number,
            },
        }

        self._client.put_object(
            Bucket=self._bucket,
            Key=s3_key,
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
        )

    def pull(
        self,
        ti: TaskInstanceRef,
        key: str,
        *,
        from_tasks: Optional[Iterable[str]] = None,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a value from S3.

        If from_tasks is provided, searches those task IDs in order
        and returns the first match.

        Args:
            ti: Task instance reference (provides dag context)
            key: State key to retrieve
            from_tasks: Optional list of task IDs to search
            default: Value to return if key not found

        Returns:
            The stored value, or default if not found
        """
        task_ids = list(from_tasks) if from_tasks else [ti.task_id]

        for task_id in task_ids:
            s3_key = self._build_key(ti, key, task_id_override=task_id)
            try:
                response = self._client.get_object(
                    Bucket=self._bucket,
                    Key=s3_key,
                )
                body = response["Body"].read().decode("utf-8")
                payload = json.loads(body)
                return payload.get("value", default)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "404"):
                    continue
                raise

        return default

    def clear_for_task(self, ti: TaskInstanceRef) -> None:
        """
        Clear all state for a task instance.

        Deletes ALL objects for the given (tenant, dag_name, dag_id, task)
        regardless of try_number. This is called BEFORE retry to
        ensure clean slate.

        Args:
            ti: Task instance reference identifying the task to clear
        """
        # List all objects for this task (all try_numbers)
        prefix = self._build_task_prefix(ti, include_try_number=False)

        paginator = self._client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._bucket, Prefix=prefix)

        objects_to_delete = []
        for page in pages:
            for obj in page.get("Contents", []):
                objects_to_delete.append({"Key": obj["Key"]})

        # Delete in batches (S3 allows up to 1000 per request)
        while objects_to_delete:
            batch = objects_to_delete[:1000]
            objects_to_delete = objects_to_delete[1000:]

            if batch:
                self._client.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": batch},
                )

    def get_all_for_task(self, ti: TaskInstanceRef) -> dict[str, Any]:
        """
        Retrieve all state for a task instance (debugging helper).

        Args:
            ti: Task instance reference

        Returns:
            Dictionary of {key: value} for all state belonging to this task
        """
        prefix = self._build_task_prefix(ti, include_try_number=True)
        result = {}

        paginator = self._client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._bucket, Prefix=prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                # Extract the state key from the object key
                # Format: prefix/tenant/dag_name/dag_id/task/try/KEY.json
                key_name = s3_key.rsplit("/", 1)[-1]
                if key_name.endswith(".json"):
                    key_name = key_name[:-5]

                try:
                    response = self._client.get_object(
                        Bucket=self._bucket,
                        Key=s3_key,
                    )
                    body = response["Body"].read().decode("utf-8")
                    payload = json.loads(body)
                    result[key_name] = payload.get("value")
                except ClientError:
                    continue

        return result

    def exists(self, ti: TaskInstanceRef, key: str) -> bool:
        """
        Check if a key exists for a task instance.

        Args:
            ti: Task instance reference
            key: State key to check

        Returns:
            True if the key exists, False otherwise
        """
        s3_key = self._build_key(ti, key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=s3_key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False
            raise
