"""
S3 Storage - Large payload storage for LLM tracking.

Stores large payloads (input/output data) in S3/MinIO when they exceed
the inline storage threshold for PostgreSQL.
"""

import gzip
import io
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from marie.llm_tracking.config import get_settings

logger = logging.getLogger(__name__)


class S3Storage:
    """
    S3/MinIO storage for large LLM tracking payloads.

    Features:
    - Automatic compression (gzip)
    - Key generation with timestamp-based partitioning
    - Configurable endpoint (for MinIO compatibility)
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        compress: bool = True,
    ):
        """
        Initialize S3 storage.

        Args:
            bucket: S3 bucket name (or from config)
            endpoint_url: S3 endpoint URL (for MinIO)
            region: AWS region
            access_key: AWS access key (or from env)
            secret_key: AWS secret key (or from env)
            compress: Whether to gzip compress payloads
        """
        settings = get_settings()
        self._bucket = bucket or settings.S3_BUCKET
        self._endpoint_url = endpoint_url or settings.S3_ENDPOINT
        self._region = region or settings.S3_REGION
        self._access_key = access_key or settings.S3_ACCESS_KEY
        self._secret_key = secret_key or settings.S3_SECRET_KEY
        self._compress = compress

        self._client: Optional[Any] = None
        self._started = False

    def start(self) -> None:
        """Initialize S3 client."""
        if self._started:
            return

        if not self._bucket:
            raise ValueError(
                "S3 bucket not configured. "
                "Set MARIE_LLM_TRACKING_S3_BUCKET environment variable."
            )

        try:
            import boto3
            from botocore.config import Config

            config = Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3, "mode": "standard"},
            )

            client_kwargs = {
                "service_name": "s3",
                "region_name": self._region,
                "config": config,
            }

            if self._endpoint_url:
                client_kwargs["endpoint_url"] = self._endpoint_url

            if self._access_key and self._secret_key:
                client_kwargs["aws_access_key_id"] = self._access_key
                client_kwargs["aws_secret_access_key"] = self._secret_key

            self._client = boto3.client(**client_kwargs)

            # Verify bucket exists
            self._client.head_bucket(Bucket=self._bucket)

            self._started = True
            logger.info(
                f"S3 storage started: bucket={self._bucket}, "
                f"endpoint={self._endpoint_url or 'AWS'}"
            )
        except ImportError:
            raise ImportError("boto3 is required for S3 storage")
        except Exception as e:
            logger.error(f"Failed to start S3 storage: {e}")
            raise

    def stop(self) -> None:
        """Close S3 client."""
        self._client = None
        self._started = False
        logger.debug("S3 storage stopped")

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
        Save a payload to S3.

        Args:
            payload: Payload data to store
            trace_id: Associated trace ID
            event_id: Event ID
            event_type: Type of event

        Returns:
            S3 key of saved object
        """
        if not self._started or self._client is None:
            raise RuntimeError("S3 storage not started")

        key = self._generate_key(trace_id, event_id, event_type)

        try:
            # Serialize to JSON
            data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")

            # Compress if enabled
            content_type = "application/json"
            content_encoding = None
            if self._compress:
                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                    gz.write(data)
                data = buffer.getvalue()
                content_encoding = "gzip"

            # Upload to S3
            extra_args = {
                "ContentType": content_type,
            }
            if content_encoding:
                extra_args["ContentEncoding"] = content_encoding

            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                **extra_args,
            )

            logger.debug(f"Saved payload to S3: {key} ({len(data)} bytes)")
            return key

        except Exception as e:
            logger.error(f"Failed to save payload to S3: {e}")
            raise

    def get_payload(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a payload from S3.

        Args:
            key: S3 object key

        Returns:
            Payload data or None if not found
        """
        if not self._started or self._client is None:
            raise RuntimeError("S3 storage not started")

        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            data = response["Body"].read()

            # Decompress if needed
            content_encoding = response.get("ContentEncoding", "")
            if content_encoding == "gzip" or key.endswith(".gz"):
                buffer = io.BytesIO(data)
                with gzip.GzipFile(fileobj=buffer, mode="rb") as gz:
                    data = gz.read()

            return json.loads(data.decode("utf-8"))

        except self._client.exceptions.NoSuchKey:
            logger.warning(f"S3 object not found: {key}")
            return None
        except Exception as e:
            logger.error(f"Failed to get payload from S3: {e}")
            raise

    def delete_payload(self, key: str) -> bool:
        """
        Delete a payload from S3.

        Args:
            key: S3 object key

        Returns:
            True if deleted, False if not found
        """
        if not self._started or self._client is None:
            raise RuntimeError("S3 storage not started")

        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            logger.debug(f"Deleted payload from S3: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete payload from S3: {e}")
            return False

    def should_use_s3(self, payload: Dict[str, Any]) -> bool:
        """
        Determine if a payload should be stored in S3 vs inline in Postgres.

        Args:
            payload: Payload to check

        Returns:
            True if payload is large enough for S3
        """
        settings = get_settings()
        try:
            size = len(
                json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            )
            return size > settings.PAYLOAD_SIZE_THRESHOLD_BYTES
        except Exception:
            return False

    def list_keys(
        self,
        prefix: str = "llm-events/",
        max_keys: int = 1000,
    ) -> list:
        """
        List objects in the bucket with a prefix.

        Args:
            prefix: Key prefix to filter
            max_keys: Maximum number of keys to return

        Returns:
            List of object keys
        """
        if not self._started or self._client is None:
            raise RuntimeError("S3 storage not started")

        try:
            response = self._client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
            )

            contents = response.get("Contents", [])
            return [obj["Key"] for obj in contents]
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            raise

    def cleanup_old_payloads(self, days: int = 30) -> int:
        """
        Delete payloads older than specified days.

        Note: This is a best-effort cleanup. For large buckets,
        consider using S3 lifecycle rules instead.

        Args:
            days: Delete payloads older than this many days

        Returns:
            Number of deleted objects
        """
        if not self._started or self._client is None:
            raise RuntimeError("S3 storage not started")

        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=days)
        deleted_count = 0

        try:
            paginator = self._client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket, Prefix="llm-events/")

            for page in pages:
                for obj in page.get("Contents", []):
                    if obj["LastModified"].replace(tzinfo=None) < cutoff:
                        self._client.delete_object(
                            Bucket=self._bucket,
                            Key=obj["Key"],
                        )
                        deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} old S3 payloads")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup S3 payloads: {e}")
            raise
