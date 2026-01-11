"""
Unit tests for PostgreSQL storage with mocked database.

Tests event saving, retrieval, and status updates with mocked connections.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from marie.llm_tracking.config import configure_from_yaml, reset_settings
from marie.llm_tracking.types import EventType, RawEvent


@pytest.fixture(autouse=True)
def reset_settings_between_tests():
    """Reset settings before and after each test."""
    reset_settings()
    configure_from_yaml({
        "enabled": True,
        "postgres": {
            "url": "postgresql://user:pass@localhost:5432/testdb",
        },
    })
    yield
    reset_settings()


@pytest.fixture
def mock_connection():
    """Create a mock database connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


class TestSaveEventFields:
    """Test save_event inserts all fields."""

    def test_save_event_fields(self, mock_connection):
        """Test save_event inserts all fields correctly."""
        conn, cursor = mock_connection

        # Import here to avoid config issues
        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb"
        )
        storage._started = True

        # Mock the _get_connection and _close_connection methods
        storage._get_connection = MagicMock(return_value=conn)
        storage._close_connection = MagicMock()

        # Create test event
        event = RawEvent(
            id="event-123",
            trace_id="trace-456",
            event_type=EventType.GENERATION_CREATE,
            s3_key="llm-events/2024/01/15/trace-456/gen_event-123.json.gz",
            model_name="gpt-4",
            model_provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            duration_ms=1500,
            time_to_first_token_ms=200,
            cost_usd=Decimal("0.005"),
            user_id="user-789",
            session_id="session-abc",
            tags=["tag1", "tag2"],
            status="pending",
        )

        # Mock cursor.fetchone to return the ID
        cursor.fetchone.return_value = ("event-123",)

        result = storage.save_event(event)

        assert result == "event-123"
        cursor.execute.assert_called_once()

        # Verify SQL contains expected columns
        sql_call = cursor.execute.call_args[0][0]
        assert "INSERT INTO" in sql_call
        assert "llm_raw_events" in sql_call
        assert "trace_id" in sql_call
        assert "s3_key" in sql_call
        assert "model_name" in sql_call
        assert "prompt_tokens" in sql_call
        assert "cost_usd" in sql_call

    def test_save_event_with_none_optional_fields(self, mock_connection):
        """Test save_event handles None optional fields."""
        conn, cursor = mock_connection

        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb"
        )
        storage._started = True
        storage._get_connection = MagicMock(return_value=conn)
        storage._close_connection = MagicMock()

        event = RawEvent(
            id="event-123",
            trace_id="trace-456",
            event_type=EventType.TRACE_CREATE,
            s3_key="llm-events/trace.json.gz",
            # Optional fields left as None
        )

        cursor.fetchone.return_value = ("event-123",)

        result = storage.save_event(event)

        assert result == "event-123"


class TestSaveFailedEvent:
    """Test save_failed_event inserts to DLQ table."""

    def test_save_failed_event(self, mock_connection):
        """Test save_failed_event inserts to DLQ correctly."""
        conn, cursor = mock_connection

        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb"
        )
        storage._started = True
        storage._get_connection = MagicMock(return_value=conn)
        storage._close_connection = MagicMock()

        cursor.fetchone.return_value = ("dlq-123",)

        result = storage.save_failed_event(
            event_id="event-123",
            trace_id="trace-456",
            event_type="generation-create",
            error_message="Connection timeout",
            payload={"test": "data"},
            error_type="TimeoutError",
            stack_trace="Traceback...",
        )

        assert result == "dlq-123"
        cursor.execute.assert_called_once()

        # Verify SQL targets the DLQ table
        sql_call = cursor.execute.call_args[0][0]
        assert "INSERT INTO" in sql_call
        assert "llm_failed_events" in sql_call
        assert "error_message" in sql_call
        assert "payload_json" in sql_call


class TestGetEventNotFound:
    """Test get_event returns None for missing ID."""

    def test_get_event_not_found_returns_none(self, mock_connection):
        """Test get_event returns None when event not found."""
        conn, cursor = mock_connection

        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb"
        )
        storage._started = True
        storage._get_connection = MagicMock(return_value=conn)
        storage._close_connection = MagicMock()

        # No row found
        cursor.fetchone.return_value = None

        result = storage.get_event("nonexistent-id")

        assert result is None

    def test_get_event_returns_raw_event(self, mock_connection):
        """Test get_event returns RawEvent when found."""
        conn, cursor = mock_connection

        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb"
        )
        storage._started = True
        storage._get_connection = MagicMock(return_value=conn)
        storage._close_connection = MagicMock()

        # Mock row data
        cursor.fetchone.return_value = (
            "event-123",  # id
            "trace-456",  # trace_id
            "generation-create",  # event_type
            "s3-key",  # s3_key
            "gpt-4",  # model_name
            "openai",  # model_provider
            100,  # prompt_tokens
            50,  # completion_tokens
            150,  # total_tokens
            1500,  # duration_ms
            200,  # time_to_first_token_ms
            Decimal("0.005"),  # cost_usd
            "user-123",  # user_id
            "session-456",  # session_id
            ["tag1"],  # tags
            "pending",  # status
            None,  # error_message
            datetime.now(),  # created_at
            None,  # processed_at
        )

        result = storage.get_event("event-123")

        assert result is not None
        assert result.id == "event-123"
        assert result.trace_id == "trace-456"


class TestMarkProcessed:
    """Test mark_processed updates status."""

    def test_mark_processed_updates_status(self, mock_connection):
        """Test mark_processed sets status and timestamp."""
        conn, cursor = mock_connection

        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb"
        )
        storage._started = True
        storage._get_connection = MagicMock(return_value=conn)
        storage._close_connection = MagicMock()

        storage.mark_processed("event-123")

        cursor.execute.assert_called_once()
        sql_call = cursor.execute.call_args[0][0]

        assert "UPDATE" in sql_call
        assert "llm_raw_events" in sql_call
        assert "status" in sql_call
        assert "processed_at" in sql_call


class TestMarkFailed:
    """Test mark_failed updates status with error."""

    def test_mark_failed_updates_status_with_error(self, mock_connection):
        """Test mark_failed sets status, error message, and timestamp."""
        conn, cursor = mock_connection

        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb"
        )
        storage._started = True
        storage._get_connection = MagicMock(return_value=conn)
        storage._close_connection = MagicMock()

        storage.mark_failed("event-123", "Processing error")

        cursor.execute.assert_called_once()
        sql_call = cursor.execute.call_args[0][0]
        params = cursor.execute.call_args[0][1]

        assert "UPDATE" in sql_call
        assert "status" in sql_call
        assert "error_message" in sql_call
        assert "failed" in params or "Processing error" in params


class TestConnectionTimeout:
    """Test connect_timeout is passed to pool."""

    def test_connection_timeout_configured(self):
        """Test connect_timeout is included in pool config."""
        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb",
            connect_timeout=15,
        )

        config = storage._parse_url()

        assert config["connect_timeout"] == 15

    def test_default_connection_timeout(self):
        """Test default connection timeout is 10 seconds."""
        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb"
        )

        config = storage._parse_url()

        assert config["connect_timeout"] == 10


class TestUrlParsing:
    """Test PostgreSQL URL parsing."""

    def test_parse_url_extracts_components(self):
        """Test URL parsing extracts all components."""
        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://myuser:mypass@db.example.com:5433/mydb"
        )

        config = storage._parse_url()

        assert config["hostname"] == "db.example.com"
        assert config["port"] == 5433
        assert config["username"] == "myuser"
        assert config["password"] == "mypass"
        assert config["database"] == "mydb"

    def test_parse_url_with_defaults(self):
        """Test URL parsing uses defaults for missing components."""
        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://localhost/"
        )

        config = storage._parse_url()

        assert config["hostname"] == "localhost"
        assert config["port"] == 5432
        assert config["username"] == "postgres"


class TestGetFailedEvents:
    """Test get_failed_events retrieves from DLQ."""

    def test_get_failed_events_returns_list(self, mock_connection):
        """Test get_failed_events returns list of failed events."""
        conn, cursor = mock_connection

        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb"
        )
        storage._started = True
        storage._get_connection = MagicMock(return_value=conn)
        storage._close_connection = MagicMock()

        # Mock fetchall to return some rows
        cursor.fetchall.return_value = [
            (
                "dlq-1",
                "event-1",
                "trace-1",
                "trace-create",
                "Error 1",
                "RuntimeError",
                "Traceback...",
                '{"test": "data1"}',
                0,
                3,
                "pending",
                datetime.now(),
                None,
            ),
            (
                "dlq-2",
                "event-2",
                "trace-2",
                "generation-create",
                "Error 2",
                "ValueError",
                "Traceback...",
                '{"test": "data2"}',
                1,
                3,
                "pending",
                datetime.now(),
                None,
            ),
        ]

        # Mock column names
        cursor.description = [
            ("id",), ("event_id",), ("trace_id",), ("event_type",),
            ("error_message",), ("error_type",), ("stack_trace",),
            ("payload_json",), ("retry_count",), ("max_retries",),
            ("status",), ("created_at",), ("resolved_at",),
        ]

        result = storage.get_failed_events(status="pending", limit=10)

        assert len(result) == 2
        assert result[0]["event_id"] == "event-1"
        assert result[1]["event_id"] == "event-2"


class TestMarkFailedEventResolved:
    """Test mark_failed_event_resolved updates DLQ status."""

    def test_mark_failed_event_resolved(self, mock_connection):
        """Test marking a failed event as resolved."""
        conn, cursor = mock_connection

        from marie.llm_tracking.storage.postgres import PostgresStorage

        storage = PostgresStorage(
            postgres_url="postgresql://user:pass@localhost:5432/testdb"
        )
        storage._started = True
        storage._get_connection = MagicMock(return_value=conn)
        storage._close_connection = MagicMock()

        storage.mark_failed_event_resolved("dlq-123", "Manually reprocessed")

        cursor.execute.assert_called_once()
        sql_call = cursor.execute.call_args[0][0]

        assert "UPDATE" in sql_call
        assert "llm_failed_events" in sql_call
        assert "status" in sql_call
        assert "resolved_at" in sql_call
