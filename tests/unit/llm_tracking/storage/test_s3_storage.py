"""
Unit tests for S3 storage with mocked StorageManager.

Tests payload saving, retrieval, compression, and error handling.
"""

import gzip
import io
import json
from unittest.mock import MagicMock, patch

import pytest

from marie.llm_tracking.config import configure_from_yaml, reset_settings
from marie.llm_tracking.storage.s3 import S3Storage, S3StorageError


@pytest.fixture(autouse=True)
def reset_settings_between_tests():
    """Reset settings before and after each test."""
    reset_settings()
    configure_from_yaml({
        "enabled": True,
        "s3": {
            "bucket": "test-bucket",
        },
    })
    yield
    reset_settings()


@pytest.fixture
def mock_storage_manager():
    """Create mock StorageManager."""
    with patch("marie.llm_tracking.storage.s3.StorageManager") as mock:
        yield mock


class TestSavePayloadCompresses:
    """Test payload is gzip compressed."""

    def test_save_payload_compresses(self, mock_storage_manager):
        """Test payload is gzip compressed before saving."""
        storage = S3Storage(bucket="test-bucket", compress=True)
        storage._started = True

        payload = {"test": "data", "nested": {"key": "value"}}

        storage.save_payload(
            payload=payload,
            trace_id="trace-123",
            event_id="event-456",
            event_type="generation",
        )

        mock_storage_manager.write.assert_called_once()
        call_args = mock_storage_manager.write.call_args[0]

        # First argument should be BytesIO with compressed data
        data_buffer = call_args[0]
        assert isinstance(data_buffer, io.BytesIO)

        # Verify it's gzip compressed by decompressing
        data_buffer.seek(0)
        with gzip.GzipFile(fileobj=data_buffer, mode="rb") as gz:
            decompressed = gz.read()

        decompressed_payload = json.loads(decompressed.decode("utf-8"))
        assert decompressed_payload == payload

    def test_save_payload_without_compression(self, mock_storage_manager):
        """Test payload is not compressed when compress=False."""
        storage = S3Storage(bucket="test-bucket", compress=False)
        storage._started = True

        payload = {"test": "data"}

        storage.save_payload(
            payload=payload,
            trace_id="trace-123",
            event_id="event-456",
            event_type="generation",
        )

        mock_storage_manager.write.assert_called_once()
        call_args = mock_storage_manager.write.call_args[0]

        # First argument should be BytesIO with uncompressed JSON
        data_buffer = call_args[0]
        data_buffer.seek(0)
        content = data_buffer.read().decode("utf-8")
        result = json.loads(content)

        assert result == payload


class TestSavePayloadKeyFormat:
    """Test S3 key follows expected pattern."""

    def test_save_payload_key_format(self, mock_storage_manager):
        """Test S3 key includes date-based partitioning."""
        storage = S3Storage(bucket="test-bucket")
        storage._started = True

        result = storage.save_payload(
            payload={"test": "data"},
            trace_id="trace-123",
            event_id="event-456",
            event_type="generation",
        )

        # Verify key format: llm-events/{year}/{month}/{day}/{hour}/{trace_id}/{event_type}_{event_id}.json.gz
        assert result.startswith("llm-events/")
        assert "trace-123" in result
        assert "generation_event-456" in result
        assert result.endswith(".json.gz")

    def test_save_payload_key_format_uncompressed(self, mock_storage_manager):
        """Test S3 key ends with .json when not compressed."""
        storage = S3Storage(bucket="test-bucket", compress=False)
        storage._started = True

        result = storage.save_payload(
            payload={"test": "data"},
            trace_id="trace-123",
            event_id="event-456",
            event_type="generation",
        )

        assert result.endswith(".json")
        assert not result.endswith(".json.gz")


class TestGetPayloadDecompresses:
    """Test payload is decompressed on read."""

    def test_get_payload_decompresses(self, mock_storage_manager):
        """Test payload is decompressed when reading .gz file."""
        storage = S3Storage(bucket="test-bucket")
        storage._started = True

        # Create compressed payload
        original_payload = {"test": "data", "nested": {"key": "value"}}
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
            gz.write(json.dumps(original_payload).encode("utf-8"))

        mock_storage_manager.read.return_value = buffer.getvalue()

        result = storage.get_payload("some-key.json.gz")

        assert result == original_payload

    def test_get_payload_uncompressed(self, mock_storage_manager):
        """Test uncompressed payload is read correctly."""
        storage = S3Storage(bucket="test-bucket", compress=False)
        storage._started = True

        original_payload = {"test": "data"}
        mock_storage_manager.read.return_value = json.dumps(original_payload).encode(
            "utf-8"
        )

        result = storage.get_payload("some-key.json")

        assert result == original_payload


class TestGetPayloadNotFoundReturnsNone:
    """Test FileNotFoundError returns None."""

    def test_get_payload_not_found_returns_none(self, mock_storage_manager):
        """Test FileNotFoundError returns None instead of raising."""
        storage = S3Storage(bucket="test-bucket")
        storage._started = True

        mock_storage_manager.read.side_effect = FileNotFoundError("Key not found")

        result = storage.get_payload("nonexistent-key.json.gz")

        assert result is None


class TestGetPayloadErrorRaisesS3Error:
    """Test other errors raise S3StorageError."""

    def test_get_payload_error_raises_s3_error(self, mock_storage_manager):
        """Test non-404 errors raise S3StorageError."""
        storage = S3Storage(bucket="test-bucket")
        storage._started = True

        mock_storage_manager.read.side_effect = Exception("Connection refused")

        with pytest.raises(S3StorageError) as exc_info:
            storage.get_payload("some-key.json.gz")

        assert "Failed to get payload from S3" in str(exc_info.value)
        assert "some-key.json.gz" in str(exc_info.value)

    def test_s3_storage_error_preserves_original(self, mock_storage_manager):
        """Test S3StorageError preserves original exception."""
        storage = S3Storage(bucket="test-bucket")
        storage._started = True

        original_error = Exception("Network error")
        mock_storage_manager.read.side_effect = original_error

        with pytest.raises(S3StorageError) as exc_info:
            storage.get_payload("key.json.gz")

        # Check the cause chain
        assert exc_info.value.__cause__ is original_error


class TestStorageNotStarted:
    """Test operations fail when not started."""

    def test_save_payload_not_started_raises(self, mock_storage_manager):
        """Test save_payload raises when not started."""
        storage = S3Storage(bucket="test-bucket")
        # Don't call start()

        with pytest.raises(RuntimeError) as exc_info:
            storage.save_payload(
                payload={"test": "data"},
                trace_id="trace-123",
                event_id="event-456",
                event_type="generation",
            )

        assert "not started" in str(exc_info.value)

    def test_get_payload_not_started_raises(self, mock_storage_manager):
        """Test get_payload raises when not started."""
        storage = S3Storage(bucket="test-bucket")
        # Don't call start()

        with pytest.raises(RuntimeError) as exc_info:
            storage.get_payload("some-key.json.gz")

        assert "not started" in str(exc_info.value)


class TestStartStop:
    """Test S3 storage lifecycle."""

    def test_start_verifies_bucket(self, mock_storage_manager):
        """Test start() verifies bucket is accessible."""
        storage = S3Storage(bucket="test-bucket")

        storage.start()

        mock_storage_manager.ensure_connection.assert_called_once_with(
            "s3://test-bucket"
        )
        assert storage._started is True

    def test_start_raises_on_failure(self, mock_storage_manager):
        """Test start() raises if bucket not accessible."""
        storage = S3Storage(bucket="test-bucket")

        mock_storage_manager.ensure_connection.side_effect = Exception(
            "Bucket not found"
        )

        with pytest.raises(Exception) as exc_info:
            storage.start()

        assert "Bucket not found" in str(exc_info.value)

    def test_stop_clears_started_flag(self, mock_storage_manager):
        """Test stop() clears the started flag."""
        storage = S3Storage(bucket="test-bucket")
        storage.start()

        storage.stop()

        assert storage._started is False

    def test_start_idempotent(self, mock_storage_manager):
        """Test multiple start() calls are safe."""
        storage = S3Storage(bucket="test-bucket")

        storage.start()
        storage.start()
        storage.start()

        # ensure_connection should only be called once
        assert mock_storage_manager.ensure_connection.call_count == 1


class TestListKeys:
    """Test list_keys functionality."""

    def test_list_keys(self, mock_storage_manager):
        """Test list_keys returns object keys."""
        storage = S3Storage(bucket="test-bucket")
        storage._started = True

        mock_storage_manager.list.return_value = [
            "llm-events/2024/01/15/trace-1/gen_event-1.json.gz",
            "llm-events/2024/01/15/trace-1/gen_event-2.json.gz",
            "llm-events/2024/01/15/trace-2/gen_event-3.json.gz",
        ]

        result = storage.list_keys(prefix="llm-events/2024/01/15/")

        assert len(result) == 3
        mock_storage_manager.list.assert_called_once()

    def test_list_keys_with_max_keys(self, mock_storage_manager):
        """Test list_keys respects max_keys limit."""
        storage = S3Storage(bucket="test-bucket")
        storage._started = True

        mock_storage_manager.list.return_value = [f"key-{i}" for i in range(100)]

        result = storage.list_keys(max_keys=10)

        assert len(result) == 10


class TestBucketRequired:
    """Test bucket configuration is required."""

    def test_start_without_bucket_raises(self, mock_storage_manager):
        """Test start() raises if bucket not configured."""
        # Configure without bucket
        reset_settings()
        configure_from_yaml({
            "enabled": True,
            # No S3 bucket
        })

        storage = S3Storage(bucket=None)

        with pytest.raises(ValueError) as exc_info:
            storage.start()

        assert "bucket not configured" in str(exc_info.value).lower()


class TestUnicodePayload:
    """Test handling of unicode in payloads."""

    def test_save_unicode_payload(self, mock_storage_manager):
        """Test unicode characters are preserved in payloads."""
        storage = S3Storage(bucket="test-bucket")
        storage._started = True

        payload = {
            "text": "Hello, 世界! 🌍",
            "japanese": "こんにちは",
            "emoji": "😀🎉",
        }

        storage.save_payload(
            payload=payload,
            trace_id="trace-123",
            event_id="event-456",
            event_type="generation",
        )

        mock_storage_manager.write.assert_called_once()
        call_args = mock_storage_manager.write.call_args[0]

        # Decompress and verify unicode is preserved
        data_buffer = call_args[0]
        data_buffer.seek(0)
        with gzip.GzipFile(fileobj=data_buffer, mode="rb") as gz:
            decompressed = gz.read()

        result = json.loads(decompressed.decode("utf-8"))
        assert result["text"] == "Hello, 世界! 🌍"
        assert result["japanese"] == "こんにちは"
        assert result["emoji"] == "😀🎉"
