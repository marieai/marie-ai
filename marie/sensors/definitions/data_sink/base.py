"""
Base classes for Data Sink sensors.

Data Sink sensors monitor cloud storage services (S3, GCS, Google Drive, etc.)
for new files and trigger jobs to process them.
"""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.base import BaseSensor
from marie.sensors.types import RunRequest, SensorResult, SensorType


class DataSinkProvider(str, Enum):
    """Supported cloud storage providers."""

    S3 = "s3"  # AWS S3 and S3-compatible (MinIO, DigitalOcean Spaces)
    GCS = "gcs"  # Google Cloud Storage
    GOOGLE_DRIVE = "google_drive"  # Google Drive
    DROPBOX = "dropbox"  # Dropbox
    AZURE_BLOB = "azure_blob"  # Azure Blob Storage


@dataclass
class FileObject:
    """Represents a file object from a cloud storage provider."""

    key: str  # Full path/key of the file
    size: int  # Size in bytes
    last_modified: datetime  # Last modification timestamp
    etag: Optional[str] = None  # Entity tag for change detection
    content_type: Optional[str] = None  # MIME type if available

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for run_config."""
        return {
            "key": self.key,
            "size": self.size,
            "last_modified": self.last_modified.isoformat(),
            "etag": self.etag,
            "content_type": self.content_type,
        }


class DataSinkSensor(BaseSensor):
    """
    Abstract base class for Data Sink sensors.

    Data Sink sensors monitor cloud storage for new files and emit
    RunRequests when new files are detected. State is tracked via
    a cursor (typically the last processed timestamp).

    Configuration:
        provider: str - Storage provider (s3, gcs, etc.)
        bucket: str - Bucket/container name
        prefix: str - Optional key prefix to filter objects
        file_patterns: list[str] - Glob patterns to match (e.g., ["*.pdf", "*.png"])
        max_files_per_tick: int - Maximum files to process per evaluation
        batch_mode: bool - If True, emit single RunRequest with all files

    Cursor:
        ISO timestamp of the last processed file's last_modified time
    """

    sensor_type = SensorType.DATA_SINK

    def __init__(self, sensor_data: Dict[str, Any]):
        super().__init__(sensor_data)
        self.provider = DataSinkProvider(self.get_required_config("provider"))
        self.bucket: str = self.get_required_config("bucket")
        self.prefix: str = self.get_config_value("prefix", "")
        self.file_patterns: List[str] = self.get_config_value("file_patterns", [])
        self.max_files_per_tick: int = self.get_config_value("max_files_per_tick", 100)
        self.batch_mode: bool = self.get_config_value("batch_mode", False)

    @abstractmethod
    async def list_objects(
        self, after_timestamp: Optional[datetime] = None
    ) -> List[FileObject]:
        """
        List objects in the storage location.

        Args:
            after_timestamp: Only return objects modified after this timestamp

        Returns:
            List of FileObject instances sorted by last_modified
        """
        raise NotImplementedError

    @abstractmethod
    def get_uri(self, key: str) -> str:
        """
        Get the full URI for a file key.

        Args:
            key: The file key/path

        Returns:
            Full URI (e.g., s3://bucket/key)
        """
        raise NotImplementedError

    def matches_patterns(self, key: str) -> bool:
        """Check if a key matches the configured file patterns."""
        if not self.file_patterns:
            return True

        import fnmatch

        filename = key.rsplit("/", 1)[-1]
        return any(fnmatch.fnmatch(filename, pattern) for pattern in self.file_patterns)

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        """
        Evaluate the sensor by listing new files in the storage location.

        Returns RunRequest(s) for any new files detected since the last cursor.
        """
        # Parse cursor as ISO timestamp
        cursor_timestamp: Optional[datetime] = None
        if context.cursor:
            try:
                cursor_timestamp = datetime.fromisoformat(context.cursor)
            except ValueError:
                context.log_warning(f"Invalid cursor format: {context.cursor}")

        # List objects after the cursor timestamp
        try:
            objects = await self.list_objects(after_timestamp=cursor_timestamp)
        except Exception as e:
            context.log_error(f"Failed to list objects: {e}")
            from marie.sensors.exceptions import SensorEvaluationError

            raise SensorEvaluationError(
                f"Failed to list objects in {self.bucket}: {e}",
                sensor_id=self.sensor_id,
                cause=e,
            )

        # Filter by patterns
        objects = [obj for obj in objects if self.matches_patterns(obj.key)]

        # Limit to max_files_per_tick
        if len(objects) > self.max_files_per_tick:
            objects = objects[: self.max_files_per_tick]
            context.log_info(
                f"Limited to {self.max_files_per_tick} files " f"(more files available)"
            )

        if not objects:
            return SensorResult.skip(
                "No new files detected",
                cursor=context.cursor,
            )

        # Update cursor to latest file's timestamp
        latest_timestamp = max(obj.last_modified for obj in objects)
        new_cursor = latest_timestamp.isoformat()

        context.log_info(
            f"Detected {len(objects)} new file(s) in {self.bucket}/{self.prefix}"
        )

        if self.batch_mode:
            # Single RunRequest with all files
            run_key = self.build_run_key(
                "data_sink",
                self.sensor_id,
                new_cursor,
            )
            return SensorResult.fire(
                run_key=run_key,
                job_name=self.target_job_name,
                dag_id=self.target_dag_id,
                run_config={
                    "provider": self.provider.value,
                    "bucket": self.bucket,
                    "files": [
                        {**obj.to_dict(), "uri": self.get_uri(obj.key)}
                        for obj in objects
                    ],
                    "file_count": len(objects),
                },
                tags={
                    "trigger": "data_sink",
                    "provider": self.provider.value,
                    "sensor_id": self.sensor_id,
                },
                cursor=new_cursor,
            )
        else:
            # Individual RunRequest per file
            run_requests = []
            for obj in objects:
                run_key = self.build_run_key(
                    "data_sink",
                    self.sensor_id,
                    obj.key,
                    obj.last_modified.isoformat(),
                )
                run_requests.append(
                    RunRequest(
                        run_key=run_key,
                        job_name=self.target_job_name,
                        dag_id=self.target_dag_id,
                        run_config={
                            "provider": self.provider.value,
                            "bucket": self.bucket,
                            "key": obj.key,
                            "size": obj.size,
                            "last_modified": obj.last_modified.isoformat(),
                            "etag": obj.etag,
                            "uri": self.get_uri(obj.key),
                        },
                        tags={
                            "trigger": "data_sink",
                            "provider": self.provider.value,
                            "sensor_id": self.sensor_id,
                        },
                    )
                )

            return SensorResult.fire_multiple(run_requests, cursor=new_cursor)

    def validate_config(self) -> None:
        """Validate data sink sensor configuration."""
        from marie.sensors.exceptions import SensorConfigError

        if not self.bucket:
            raise SensorConfigError(
                "Data sink sensor requires 'bucket' configuration",
                field="bucket",
            )

        try:
            DataSinkProvider(self.get_required_config("provider"))
        except ValueError as e:
            raise SensorConfigError(
                f"Invalid provider: {e}",
                field="provider",
            )
