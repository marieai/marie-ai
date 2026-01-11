"""
S3 Storage - Payload storage for ALL LLM tracking data.

ALL payloads (prompts, responses, raw LLM data) are stored in S3.
PostgreSQL stores only metadata for analytics.

Delegates to StorageManager for actual S3 operations.
"""

import gzip
import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from marie.llm_tracking.config import get_settings
from marie.storage import StorageManager

logger = logging.getLogger(__name__)


class S3StorageError(Exception):
    """Exception raised when S3 storage operations fail."""

    pass


class S3Storage:
    """
    S3/MinIO storage for ALL LLM tracking payloads.

    ALL payloads are stored in S3.
    Delegates to StorageManager for actual S3 operations while providing:
    - Automatic gzip compression
    - Time-based key partitioning
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        compress: bool = True,
    ):
        """
        Initialize S3 storage.

        Args:
            bucket: S3 bucket name (or from config)
            compress: Whether to gzip compress payloads
        """
        settings = get_settings()
        self._bucket = bucket or settings.S3_BUCKET
        self._compress = compress
        self._started = False

    def start(self) -> None:
        """Verify S3 bucket is accessible via StorageManager."""
        if self._started:
            return

        if not self._bucket:
            raise ValueError(
                "S3 bucket not configured. "
                "Set MARIE_LLM_TRACKING_S3_BUCKET environment variable."
            )

        try:
            # Verify StorageManager can access the bucket
            StorageManager.ensure_connection(f"s3://{self._bucket}")
            self._started = True
            logger.info(f"S3 storage verified: bucket={self._bucket}")
        except Exception as e:
            logger.error(f"Failed to verify S3 bucket: {e}")
            raise

    def stop(self) -> None:
        """No-op - StorageManager manages S3 client lifecycle."""
        self._started = False
        logger.debug("S3 storage stopped (managed by StorageManager)")

    def _generate_key(
        self,
        trace_id: str,
        event_id: str,
        event_type: str,
    ) -> str:
        """
        Generate S3 object key with time-based partitioning.

        Format: llm-events/{year}/{month}/{day}/{hour}/{trace_id}/{event_id}.json.gz

        Args:
            trace_id: Trace ID
            event_id: Event ID
            event_type: Type of event

        Returns:
            S3 object key
        """
        now = datetime.utcnow()
        extension = ".json.gz" if self._compress else ".json"

        return (
            f"llm-events/"
            f"{now.year:04d}/{now.month:02d}/{now.day:02d}/{now.hour:02d}/"
            f"{trace_id}/{event_type}_{event_id}{extension}"
        )

    def save_payload(
        self,
        payload: Dict[str, Any],
        trace_id: str,
        event_id: str,
        event_type: str,
    ) -> str:
        """
        Save a payload to S3 via StorageManager.

        Args:
            payload: Payload data to store
            trace_id: Associated trace ID
            event_id: Event ID
            event_type: Type of event

        Returns:
            S3 key of saved object
        """
        if not self._started:
            raise RuntimeError("S3 storage not started")

        key = self._generate_key(trace_id, event_id, event_type)
        s3_path = f"s3://{self._bucket}/{key}"

        try:
            # Serialize to JSON
            data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")

            # Compress if enabled
            if self._compress:
                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                    gz.write(data)
                data = buffer.getvalue()

            # Write via StorageManager (supports BytesIO after S3StorageHandler enhancement)
            StorageManager.write(io.BytesIO(data), s3_path, overwrite=True)

            logger.debug(f"Saved payload to S3: {key} ({len(data)} bytes)")
            return key

        except Exception as e:
            logger.error(f"Failed to save payload to S3: {e}")
            raise

    def get_payload(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a payload from S3 via StorageManager.

        Args:
            key: S3 object key

        Returns:
            Payload data or None if not found (legitimate 404)

        Raises:
            S3StorageError: If S3 operation fails (non-404 errors)
        """
        if not self._started:
            raise RuntimeError("S3 storage not started")

        s3_path = f"s3://{self._bucket}/{key}"

        try:
            # Read via StorageManager
            data = StorageManager.read(s3_path)

            # Decompress if needed
            if self._compress or key.endswith(".gz"):
                buffer = io.BytesIO(data)
                with gzip.GzipFile(fileobj=buffer, mode="rb") as gz:
                    data = gz.read()

            return json.loads(data.decode("utf-8"))

        except FileNotFoundError:
            # Legitimate "not found" - return None
            logger.warning(f"Payload not found in S3: {key}")
            return None
        except Exception as e:
            # Other errors should be raised for caller to handle
            logger.exception(f"Failed to get payload from S3: {key}")
            raise S3StorageError(f"Failed to get payload from S3: {key}") from e

    def delete_payload(self, key: str) -> bool:
        """
        Delete a payload from S3.

        Note: StorageManager doesn't have delete method.
        Use S3 lifecycle policies for cleanup instead.

        Args:
            key: S3 object key

        Returns:
            False (not implemented via StorageManager)
        """
        logger.warning(f"Delete not implemented via StorageManager: {key}")
        return False

    def list_keys(
        self,
        prefix: str = "llm-events/",
        max_keys: int = 1000,
    ) -> List[str]:
        """
        List objects in the bucket with a prefix.

        Args:
            prefix: Key prefix to filter
            max_keys: Maximum number of keys to return

        Returns:
            List of object keys
        """
        if not self._started:
            raise RuntimeError("S3 storage not started")

        s3_path = f"s3://{self._bucket}/{prefix}"
        try:
            keys = StorageManager.list(s3_path)
            return keys[:max_keys]
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            raise
