"""
PostgreSQL Storage - Raw event storage for LLM tracking.

Stores events in PostgreSQL before they are processed and written to ClickHouse.
This provides durability and crash recovery.
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urlparse

from marie.llm_tracking.config import get_settings
from marie.llm_tracking.types import EventType, RawEvent

logger = logging.getLogger(__name__)

# SQL for creating the raw events table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_raw_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    s3_key VARCHAR(500),
    payload JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    CONSTRAINT valid_status CHECK (status IN ('pending', 'processed', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_llm_raw_events_status
    ON llm_raw_events(status);
CREATE INDEX IF NOT EXISTS idx_llm_raw_events_trace_id
    ON llm_raw_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_llm_raw_events_created_at
    ON llm_raw_events(created_at);
"""


class PostgresStorage:
    """
    PostgreSQL storage for raw LLM tracking events.

    Features:
    - Connection pooling via psycopg2
    - Automatic table creation
    - Retry logic for transient failures
    """

    def __init__(
        self,
        postgres_url: Optional[str] = None,
        pool_size: int = 5,
        auto_create_table: bool = True,
    ):
        """
        Initialize PostgreSQL storage.

        Args:
            postgres_url: PostgreSQL connection URL (or from config)
            pool_size: Connection pool size
            auto_create_table: Whether to create table on start
        """
        settings = get_settings()
        self._postgres_url = postgres_url or settings.POSTGRES_URL

        if not self._postgres_url:
            raise ValueError(
                "PostgreSQL URL not configured. "
                "Set MARIE_LLM_TRACKING_POSTGRES_URL environment variable."
            )

        self._pool_size = pool_size
        self._auto_create_table = auto_create_table
        self._pool: Optional[Any] = None
        self._started = False

    def _parse_url(self) -> Dict[str, Any]:
        """Parse PostgreSQL URL into connection parameters."""
        parsed = urlparse(self._postgres_url)

        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "user": parsed.username or "postgres",
            "password": parsed.password or "",
            "database": parsed.path.lstrip("/") or "marie",
        }

    def start(self) -> None:
        """Initialize connection pool and create table if needed."""
        if self._started:
            return

        try:
            import psycopg2
            import psycopg2.pool

            config = self._parse_url()
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=self._pool_size,
                host=config["host"],
                port=config["port"],
                user=config["user"],
                password=config["password"],
                database=config["database"],
                options="-c timezone=UTC",
                application_name="marie_llm_tracking",
            )

            if self._auto_create_table:
                self._create_table()

            self._started = True
            logger.info(
                f"PostgreSQL storage started: {config['host']}:{config['port']}"
            )
        except ImportError:
            raise ImportError("psycopg2 is required for PostgreSQL storage")
        except Exception as e:
            logger.error(f"Failed to start PostgreSQL storage: {e}")
            raise

    def stop(self) -> None:
        """Close connection pool."""
        if self._pool is not None:
            try:
                self._pool.closeall()
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL pool: {e}")
            finally:
                self._pool = None
                self._started = False
        logger.debug("PostgreSQL storage stopped")

    @contextmanager
    def _get_connection(self) -> Generator[Any, None, None]:
        """Get a connection from the pool."""
        if not self._started or self._pool is None:
            raise RuntimeError("PostgreSQL storage not started")

        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _create_table(self) -> None:
        """Create the raw events table if it doesn't exist."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
        logger.info("Created llm_raw_events table (if not exists)")

    def save_event(self, event: RawEvent) -> str:
        """
        Save a raw event to PostgreSQL.

        Args:
            event: RawEvent to save

        Returns:
            Event ID
        """
        import psycopg2.extras

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_raw_events
                        (id, trace_id, event_type, s3_key, payload, status)
                    VALUES
                        (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        event.id,
                        event.trace_id,
                        event.event_type.value,
                        event.s3_key,
                        json.dumps(event.payload) if event.payload else None,
                        event.status,
                    ),
                )
                result = cur.fetchone()
                return str(result[0]) if result else event.id

    def save_events(self, events: List[RawEvent]) -> List[str]:
        """
        Save multiple raw events in a batch.

        Args:
            events: List of RawEvents to save

        Returns:
            List of event IDs
        """
        if not events:
            return []

        import psycopg2.extras

        values = [
            (
                e.id,
                e.trace_id,
                e.event_type.value,
                e.s3_key,
                json.dumps(e.payload) if e.payload else None,
                e.status,
            )
            for e in events
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    """
                    INSERT INTO llm_raw_events
                        (id, trace_id, event_type, s3_key, payload, status)
                    VALUES
                        (%s, %s, %s, %s, %s, %s)
                    """,
                    values,
                )

        return [e.id for e in events]

    def get_event(self, event_id: str) -> Optional[RawEvent]:
        """
        Get a raw event by ID.

        Args:
            event_id: Event ID

        Returns:
            RawEvent or None if not found
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, trace_id, event_type, s3_key, payload,
                           status, error_message, created_at, processed_at
                    FROM llm_raw_events
                    WHERE id = %s
                    """,
                    (event_id,),
                )
                row = cur.fetchone()

                if row is None:
                    return None

                return RawEvent(
                    id=str(row[0]),
                    trace_id=str(row[1]),
                    event_type=EventType(row[2]),
                    s3_key=row[3],
                    payload=row[4],
                    status=row[5],
                    error_message=row[6],
                    created_at=row[7],
                    processed_at=row[8],
                )

    def get_pending_events(
        self,
        limit: int = 100,
        older_than_seconds: int = 0,
    ) -> List[RawEvent]:
        """
        Get pending events for processing.

        Args:
            limit: Maximum number of events to return
            older_than_seconds: Only return events older than this

        Returns:
            List of pending RawEvents
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if older_than_seconds > 0:
                    cur.execute(
                        """
                        SELECT id, trace_id, event_type, s3_key, payload,
                               status, error_message, created_at, processed_at
                        FROM llm_raw_events
                        WHERE status = 'pending'
                          AND created_at < NOW() - INTERVAL '%s seconds'
                        ORDER BY created_at ASC
                        LIMIT %s
                        """,
                        (older_than_seconds, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, trace_id, event_type, s3_key, payload,
                               status, error_message, created_at, processed_at
                        FROM llm_raw_events
                        WHERE status = 'pending'
                        ORDER BY created_at ASC
                        LIMIT %s
                        """,
                        (limit,),
                    )

                rows = cur.fetchall()
                return [
                    RawEvent(
                        id=str(row[0]),
                        trace_id=str(row[1]),
                        event_type=EventType(row[2]),
                        s3_key=row[3],
                        payload=row[4],
                        status=row[5],
                        error_message=row[6],
                        created_at=row[7],
                        processed_at=row[8],
                    )
                    for row in rows
                ]

    def mark_processed(self, event_id: str) -> None:
        """
        Mark an event as processed.

        Args:
            event_id: Event ID
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE llm_raw_events
                    SET status = 'processed', processed_at = NOW()
                    WHERE id = %s
                    """,
                    (event_id,),
                )

    def mark_failed(self, event_id: str, error_message: str) -> None:
        """
        Mark an event as failed.

        Args:
            event_id: Event ID
            error_message: Error description
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE llm_raw_events
                    SET status = 'failed', error_message = %s
                    WHERE id = %s
                    """,
                    (error_message, event_id),
                )

    def cleanup_old_events(self, days: int = 30) -> int:
        """
        Delete processed events older than specified days.

        Args:
            days: Delete events older than this many days

        Returns:
            Number of deleted events
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM llm_raw_events
                    WHERE status = 'processed'
                      AND processed_at < NOW() - INTERVAL '%s days'
                    """,
                    (days,),
                )
                return cur.rowcount
