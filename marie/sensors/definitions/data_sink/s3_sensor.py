"""
S3 Data Sink sensor implementation.

Monitors AWS S3 or S3-compatible storage (MinIO, DigitalOcean Spaces, Backblaze B2)
for new files and triggers jobs to process them.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.data_sink.base import (
    DataSinkProvider,
    DataSinkSensor,
    FileObject,
)
from marie.sensors.exceptions import SensorConfigError, SensorEvaluationError
from marie.sensors.registry import register_sensor
from marie.sensors.types import SensorType


@register_sensor(SensorType.DATA_SINK)
class S3DataSinkSensor(DataSinkSensor):
    """
    S3 Data Sink sensor for monitoring S3 buckets for new files.

    Supports AWS S3 and S3-compatible storage services via custom endpoint URLs.

    Configuration:
        provider: "s3"
        bucket: str - S3 bucket name
        prefix: str - Key prefix to filter (e.g., "incoming/")
        region: str - AWS region (default: us-east-1)
        endpoint_url: str - Custom endpoint for S3-compatible storage
        file_patterns: list[str] - Glob patterns to match files
        max_files_per_tick: int - Maximum files per evaluation (default: 100)
        batch_mode: bool - Emit single RunRequest with all files

    Credentials:
        Uses boto3's credential chain (environment, IAM role, profile, etc.)
        For custom endpoints, ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
        are set appropriately.

    Cursor:
        ISO timestamp of the last processed file's LastModified time.
        On each tick, only objects with LastModified > cursor are returned.

    Run Config (individual mode):
        {
            "provider": "s3",
            "bucket": "my-bucket",
            "key": "incoming/file.pdf",
            "size": 245678,
            "last_modified": "2026-02-11T12:30:00+00:00",
            "etag": "abc123def456",
            "uri": "s3://my-bucket/incoming/file.pdf"
        }

    Run Config (batch mode):
        {
            "provider": "s3",
            "bucket": "my-bucket",
            "files": [...],
            "file_count": 5
        }
    """

    def __init__(self, sensor_data: Dict[str, Any]):
        super().__init__(sensor_data)
        self.region: str = self.get_config_value("region", "us-east-1")
        self.endpoint_url: Optional[str] = self.get_config_value("endpoint_url")
        self._client = None

    def _get_client(self):
        """Get or create the S3 client."""
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise SensorEvaluationError(
                    "boto3 is required for S3 data sink sensor. "
                    "Install with: pip install boto3",
                    sensor_id=self.sensor_id,
                )

            client_kwargs = {"region_name": self.region}
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url

            self._client = boto3.client("s3", **client_kwargs)

        return self._client

    async def list_objects(
        self, after_timestamp: Optional[datetime] = None
    ) -> List[FileObject]:
        """
        List objects in the S3 bucket.

        Uses pagination to handle large buckets. Filters by prefix and
        LastModified timestamp.
        """
        import asyncio

        # Run boto3 sync calls in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._list_objects_sync, after_timestamp
        )

    def _list_objects_sync(
        self, after_timestamp: Optional[datetime] = None
    ) -> List[FileObject]:
        """Synchronous implementation of list_objects."""
        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")

        list_kwargs = {"Bucket": self.bucket}
        if self.prefix:
            list_kwargs["Prefix"] = self.prefix

        objects: List[FileObject] = []

        for page in paginator.paginate(**list_kwargs):
            for obj in page.get("Contents", []):
                last_modified = obj["LastModified"]

                # Ensure timezone-aware
                if last_modified.tzinfo is None:
                    last_modified = last_modified.replace(tzinfo=timezone.utc)

                # Filter by timestamp
                if after_timestamp:
                    # Ensure after_timestamp is also timezone-aware for comparison
                    if after_timestamp.tzinfo is None:
                        after_timestamp = after_timestamp.replace(tzinfo=timezone.utc)
                    if last_modified <= after_timestamp:
                        continue

                # Skip directories (keys ending with /)
                if obj["Key"].endswith("/"):
                    continue

                objects.append(
                    FileObject(
                        key=obj["Key"],
                        size=obj["Size"],
                        last_modified=last_modified,
                        etag=obj.get("ETag", "").strip('"'),
                    )
                )

        # Sort by last_modified ascending
        objects.sort(key=lambda x: x.last_modified)

        return objects

    def get_uri(self, key: str) -> str:
        """Get the S3 URI for a key."""
        return f"s3://{self.bucket}/{key}"

    def validate_config(self) -> None:
        """Validate S3 sensor configuration."""
        super().validate_config()

        if self.provider != DataSinkProvider.S3:
            raise SensorConfigError(
                f"S3DataSinkSensor requires provider='s3', got '{self.provider.value}'",
                field="provider",
            )

        # Validate endpoint_url if provided
        if self.endpoint_url:
            if not self.endpoint_url.startswith(("http://", "https://")):
                raise SensorConfigError(
                    "endpoint_url must start with http:// or https://",
                    field="endpoint_url",
                )
