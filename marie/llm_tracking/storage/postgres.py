"""
PostgreSQL Storage - METADATA ONLY for LLM tracking.

Stores essential tracking metadata in PostgreSQL. All payload data
(prompts, responses) is stored in S3.
"""

import logging
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from marie.llm_tracking.config import get_settings
from marie.llm_tracking.types import EventType, RawEvent
from marie.logging_core.logger import MarieLogger
from marie.storage.database.postgres import PostgresqlMixin

logger = logging.getLogger(__name__)


class PostgresStorage(PostgresqlMixin):
    """
    PostgreSQL storage for LLM tracking EVENT METADATA ONLY.

    All payload data (prompts, responses) is stored in S3.
    PostgreSQL stores only tracking metadata for analytics:
    - Event IDs and trace relationships
    - S3 keys (where payload is stored)
    - Token counts, costs, latency metrics
    - Model info, user/session IDs
    - Processing status
    """

    def __init__(
        self,
        postgres_url: Optional[str] = None,
        schema: str = "marie_scheduler",
        pool_size: int = 5,
        auto_create_table: bool = True,
        connect_timeout: int = 10,
    ):
        """
        Initialize PostgreSQL storage.

        Args:
            postgres_url: PostgreSQL connection URL (or from config)
            schema: PostgreSQL schema name (default: marie_scheduler)
            pool_size: Connection pool size
            auto_create_table: Whether to verify table exists on start
            connect_timeout: Connection timeout in seconds (default: 10s)
        """
        settings = get_settings()
        self._postgres_url = postgres_url or settings.POSTGRES_URL

        if not self._postgres_url:
            raise ValueError(
                "PostgreSQL URL not configured. "
                "Set postgres.url in llm_tracking section of YAML config."
            )

        self._schema = schema
        self._pool_size = pool_size
        self._auto_create_table = auto_create_table
        self._connect_timeout = connect_timeout
        self._started = False
        # Required by PostgresqlMixin
        self.logger = MarieLogger("llm_postgres_storage")

    def _parse_url(self) -> Dict[str, Any]:
        """Parse PostgreSQL URL into connection parameters for mixin."""
        parsed = urlparse(self._postgres_url)

        return {
            "hostname": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "username": parsed.username or "postgres",
            "password": parsed.password or "",
            "database": parsed.path.lstrip("/") or "marie",
            "schema": self._schema,
            "max_connections": self._pool_size,
            "min_connections": 1,
            "default_table": "llm_raw_events",
            "application_name": "marie_llm_tracking",
            # Connection timeout prevents indefinite hangs
            "connect_timeout": self._connect_timeout,
        }

    @property
    def _table_name(self) -> str:
        """Return fully qualified table name (schema.table)."""
        return f"{self._schema}.llm_raw_events"

    @property
    def _failed_table_name(self) -> str:
        """Return fully qualified failed events table name (schema.table)."""
        return f"{self._schema}.llm_failed_events"

    def start(self) -> None:
        """Initialize connection pool using PostgresqlMixin."""
        if self._started:
            return

        try:
            config = self._parse_url()
            self._setup_storage(
                config,
                create_table_callback=(
                    self._create_events_table if self._auto_create_table else None
                ),
            )
            self._started = True
            logger.info(
                f"PostgreSQL storage started: {config['hostname']}:{config['port']}"
            )
        except Exception as e:
            logger.error(f"Failed to start PostgreSQL storage: {e}")
            raise

    def stop(self) -> None:
        """Close connection pool."""
        if hasattr(self, 'postgreSQL_pool') and self.postgreSQL_pool is not None:
            try:
                self.postgreSQL_pool.closeall()
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL pool: {e}")
            finally:
                self.postgreSQL_pool = None
                self._started = False
        logger.debug("PostgreSQL storage stopped")

    def _create_events_table(self, table_name: str) -> None:
        """
        Verify llm_raw_events table exists in the configured schema.

        Schema is defined in: config/psql/schema/048_llm_tracking.sql
        Table should be created via schema migrations, not Python code.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = %s AND table_name = 'llm_raw_events'
                    );
                    """,
                    (self._schema,),
                )
                exists = cur.fetchone()[0]
                if not exists:
                    # Rollback to end transaction before raising
                    conn.rollback()
                    raise RuntimeError(
                        f"Table '{self._schema}.llm_raw_events' does not exist. "
                        "Run schema migration: config/psql/schema/048_llm_tracking.sql"
                    )
                logger.debug(f"Verified {self._table_name} table exists")
            # Commit to end transaction before returning connection to pool
            conn.commit()
        finally:
            self._close_connection(conn)

    def save_event(self, event: RawEvent) -> str:
        """
        Save event metadata to PostgreSQL (payload already in S3).

        Uses UPSERT to handle updates to existing events (e.g., trace updates).

        Args:
            event: RawEvent with metadata to save

        Returns:
            Event ID
        """
        import json

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._table_name} (
                        id, trace_id, event_type, s3_key,
                        model_name, model_provider,
                        prompt_tokens, completion_tokens, total_tokens,
                        duration_ms, time_to_first_token_ms,
                        cost_usd,
                        user_id, session_id, tags,
                        status
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s,
                        %s, %s, %s,
                        %s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        event_type = EXCLUDED.event_type,
                        s3_key = EXCLUDED.s3_key,
                        model_name = COALESCE(EXCLUDED.model_name, {self._table_name}.model_name),
                        model_provider = COALESCE(EXCLUDED.model_provider, {self._table_name}.model_provider),
                        prompt_tokens = COALESCE(EXCLUDED.prompt_tokens, {self._table_name}.prompt_tokens),
                        completion_tokens = COALESCE(EXCLUDED.completion_tokens, {self._table_name}.completion_tokens),
                        total_tokens = COALESCE(EXCLUDED.total_tokens, {self._table_name}.total_tokens),
                        duration_ms = COALESCE(EXCLUDED.duration_ms, {self._table_name}.duration_ms),
                        time_to_first_token_ms = COALESCE(EXCLUDED.time_to_first_token_ms, {self._table_name}.time_to_first_token_ms),
                        cost_usd = COALESCE(EXCLUDED.cost_usd, {self._table_name}.cost_usd),
                        user_id = COALESCE(EXCLUDED.user_id, {self._table_name}.user_id),
                        session_id = COALESCE(EXCLUDED.session_id, {self._table_name}.session_id),
                        tags = COALESCE(EXCLUDED.tags, {self._table_name}.tags),
                        status = COALESCE(EXCLUDED.status, {self._table_name}.status)
                    RETURNING id
                    """,
                    (
                        event.id,
                        event.trace_id,
                        event.event_type.value,
                        event.s3_key,
                        event.model_name,
                        event.model_provider,
                        event.prompt_tokens,
                        event.completion_tokens,
                        event.total_tokens,
                        event.duration_ms,
                        event.time_to_first_token_ms,
                        float(event.cost_usd) if event.cost_usd else None,
                        event.user_id,
                        event.session_id,
                        json.dumps(event.tags) if event.tags else None,
                        event.status,
                    ),
                )
                conn.commit()
                result = cur.fetchone()
                return str(result[0]) if result else event.id
        finally:
            self._close_connection(conn)

    def save_events(self, events: List[RawEvent]) -> List[str]:
        """
        Save multiple event metadata records in a batch.

        Uses UPSERT to handle updates to existing events.

        Args:
            events: List of RawEvents to save

        Returns:
            List of event IDs
        """
        if not events:
            return []

        import json

        import psycopg2.extras

        values = [
            (
                e.id,
                e.trace_id,
                e.event_type.value,
                e.s3_key,
                e.model_name,
                e.model_provider,
                e.prompt_tokens,
                e.completion_tokens,
                e.total_tokens,
                e.duration_ms,
                e.time_to_first_token_ms,
                float(e.cost_usd) if e.cost_usd else None,
                e.user_id,
                e.session_id,
                json.dumps(e.tags) if e.tags else None,
                e.status,
            )
            for e in events
        ]

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    f"""
                    INSERT INTO {self._table_name} (
                        id, trace_id, event_type, s3_key,
                        model_name, model_provider,
                        prompt_tokens, completion_tokens, total_tokens,
                        duration_ms, time_to_first_token_ms,
                        cost_usd,
                        user_id, session_id, tags,
                        status
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s,
                        %s, %s, %s,
                        %s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        event_type = EXCLUDED.event_type,
                        s3_key = EXCLUDED.s3_key,
                        model_name = COALESCE(EXCLUDED.model_name, {self._table_name}.model_name),
                        model_provider = COALESCE(EXCLUDED.model_provider, {self._table_name}.model_provider),
                        prompt_tokens = COALESCE(EXCLUDED.prompt_tokens, {self._table_name}.prompt_tokens),
                        completion_tokens = COALESCE(EXCLUDED.completion_tokens, {self._table_name}.completion_tokens),
                        total_tokens = COALESCE(EXCLUDED.total_tokens, {self._table_name}.total_tokens),
                        duration_ms = COALESCE(EXCLUDED.duration_ms, {self._table_name}.duration_ms),
                        time_to_first_token_ms = COALESCE(EXCLUDED.time_to_first_token_ms, {self._table_name}.time_to_first_token_ms),
                        cost_usd = COALESCE(EXCLUDED.cost_usd, {self._table_name}.cost_usd),
                        user_id = COALESCE(EXCLUDED.user_id, {self._table_name}.user_id),
                        session_id = COALESCE(EXCLUDED.session_id, {self._table_name}.session_id),
                        tags = COALESCE(EXCLUDED.tags, {self._table_name}.tags),
                        status = COALESCE(EXCLUDED.status, {self._table_name}.status)
                    """,
                    values,
                )
            conn.commit()
        finally:
            self._close_connection(conn)

        return [e.id for e in events]

    def get_event(self, event_id: str) -> Optional[RawEvent]:
        """
        Get event metadata by ID.

        Args:
            event_id: Event ID

        Returns:
            RawEvent or None if not found
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, trace_id, event_type, s3_key,
                           model_name, model_provider,
                           prompt_tokens, completion_tokens, total_tokens,
                           duration_ms, time_to_first_token_ms,
                           cost_usd,
                           user_id, session_id, tags,
                           status, error_message, created_at, processed_at
                    FROM {self._table_name}
                    WHERE id = %s
                    """,
                    (event_id,),
                )
                row = cur.fetchone()

                if row is None:
                    conn.rollback()
                    return None

                result = RawEvent(
                    id=str(row[0]),
                    trace_id=str(row[1]),
                    event_type=EventType(row[2]),
                    s3_key=row[3],
                    model_name=row[4],
                    model_provider=row[5],
                    prompt_tokens=row[6],
                    completion_tokens=row[7],
                    total_tokens=row[8],
                    duration_ms=row[9],
                    time_to_first_token_ms=row[10],
                    cost_usd=Decimal(str(row[11])) if row[11] else None,
                    user_id=row[12],
                    session_id=row[13],
                    tags=row[14],
                    status=row[15],
                    error_message=row[16],
                    created_at=row[17],
                    processed_at=row[18],
                )
            # Rollback to end transaction (read-only, nothing to commit)
            conn.rollback()
            return result
        finally:
            self._close_connection(conn)

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
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if older_than_seconds > 0:
                    cur.execute(
                        f"""
                        SELECT id, trace_id, event_type, s3_key,
                               model_name, model_provider,
                               prompt_tokens, completion_tokens, total_tokens,
                               duration_ms, time_to_first_token_ms,
                               cost_usd,
                               user_id, session_id, tags,
                               status, error_message, created_at, processed_at
                        FROM {self._table_name}
                        WHERE status = 'pending'
                          AND created_at < NOW() - INTERVAL '1 second' * %s
                        ORDER BY created_at ASC
                        LIMIT %s
                        """,
                        (older_than_seconds, limit),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT id, trace_id, event_type, s3_key,
                               model_name, model_provider,
                               prompt_tokens, completion_tokens, total_tokens,
                               duration_ms, time_to_first_token_ms,
                               cost_usd,
                               user_id, session_id, tags,
                               status, error_message, created_at, processed_at
                        FROM {self._table_name}
                        WHERE status = 'pending'
                        ORDER BY created_at ASC
                        LIMIT %s
                        """,
                        (limit,),
                    )

                rows = cur.fetchall()
                result = [
                    RawEvent(
                        id=str(row[0]),
                        trace_id=str(row[1]),
                        event_type=EventType(row[2]),
                        s3_key=row[3],
                        model_name=row[4],
                        model_provider=row[5],
                        prompt_tokens=row[6],
                        completion_tokens=row[7],
                        total_tokens=row[8],
                        duration_ms=row[9],
                        time_to_first_token_ms=row[10],
                        cost_usd=Decimal(str(row[11])) if row[11] else None,
                        user_id=row[12],
                        session_id=row[13],
                        tags=row[14],
                        status=row[15],
                        error_message=row[16],
                        created_at=row[17],
                        processed_at=row[18],
                    )
                    for row in rows
                ]
            # Rollback to end transaction (read-only, nothing to commit)
            conn.rollback()
            return result
        finally:
            self._close_connection(conn)

    def mark_processed(self, event_id: str) -> None:
        """
        Mark an event as processed.

        Args:
            event_id: Event ID
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self._table_name}
                    SET status = 'processed', processed_at = NOW()
                    WHERE id = %s
                    """,
                    (event_id,),
                )
            conn.commit()
        finally:
            self._close_connection(conn)

    def mark_failed(self, event_id: str, error_message: str) -> None:
        """
        Mark an event as failed.

        Args:
            event_id: Event ID
            error_message: Error description
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self._table_name}
                    SET status = 'failed', error_message = %s
                    WHERE id = %s
                    """,
                    (error_message, event_id),
                )
            conn.commit()
        finally:
            self._close_connection(conn)

    def mark_pending(self, event_id: str) -> None:
        """
        Reset event status to pending for reprocessing.

        Used by requeue script to retry failed events.

        Args:
            event_id: Event ID to reset
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self._table_name}
                    SET status = 'pending', error_message = NULL, processed_at = NULL
                    WHERE id = %s
                    """,
                    (event_id,),
                )
            conn.commit()
        finally:
            self._close_connection(conn)

    def cleanup_old_events(self, days: int = 30) -> int:
        """
        Delete processed events older than specified days.

        Args:
            days: Delete events older than this many days

        Returns:
            Number of deleted events
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self._table_name}
                    WHERE status = 'processed'
                      AND processed_at < NOW() - INTERVAL '1 day' * %s
                    """,
                    (days,),
                )
            conn.commit()
            return cur.rowcount
        finally:
            self._close_connection(conn)

    def update_metadata(
        self,
        event_id: str,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        duration_ms: Optional[int] = None,
        time_to_first_token_ms: Optional[int] = None,
        cost_usd: Optional[Decimal] = None,
    ) -> None:
        """
        Update metadata fields for an event.

        Used when metadata is calculated after initial event creation
        (e.g., token counting, cost calculation).

        Args:
            event_id: Event ID to update
            model_name: Model name
            model_provider: Model provider
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total token count
            duration_ms: Duration in milliseconds
            time_to_first_token_ms: Time to first token in milliseconds
            cost_usd: Cost in USD
        """
        updates = []
        params = []

        if model_name is not None:
            updates.append("model_name = %s")
            params.append(model_name)
        if model_provider is not None:
            updates.append("model_provider = %s")
            params.append(model_provider)
        if prompt_tokens is not None:
            updates.append("prompt_tokens = %s")
            params.append(prompt_tokens)
        if completion_tokens is not None:
            updates.append("completion_tokens = %s")
            params.append(completion_tokens)
        if total_tokens is not None:
            updates.append("total_tokens = %s")
            params.append(total_tokens)
        if duration_ms is not None:
            updates.append("duration_ms = %s")
            params.append(duration_ms)
        if time_to_first_token_ms is not None:
            updates.append("time_to_first_token_ms = %s")
            params.append(time_to_first_token_ms)
        if cost_usd is not None:
            updates.append("cost_usd = %s")
            params.append(float(cost_usd))

        if not updates:
            return

        params.append(event_id)

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self._table_name}
                    SET {', '.join(updates)}
                    WHERE id = %s
                    """,
                    tuple(params),
                )
            conn.commit()
        finally:
            self._close_connection(conn)

    def save_failed_event(
        self,
        event_id: Optional[str],
        trace_id: Optional[str],
        event_type: str,
        error_message: str,
        payload: Dict[str, Any],
        error_type: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> str:
        """
        Save a failed event to the Dead Letter Queue for later retry/investigation.

        This method is called when event processing fails (S3 error, export error, etc.)
        to preserve the event data for debugging and potential retry.

        Args:
            event_id: Original event ID (may be None if failed before ID assignment)
            trace_id: Trace ID for correlation
            event_type: Type of event (trace-create, generation-create, etc.)
            error_message: Description of what went wrong
            payload: Full event payload to preserve
            error_type: Exception class name (e.g., "S3StorageError")
            stack_trace: Full stack trace for debugging

        Returns:
            ID of the failed event record
        """
        import json
        import uuid

        conn = self._get_connection()
        try:
            failed_id = str(uuid.uuid4())
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._failed_table_name} (
                        id, event_id, trace_id, event_type,
                        error_message, error_type, stack_trace,
                        payload_json, status
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, 'pending'
                    )
                    RETURNING id
                    """,
                    (
                        failed_id,
                        event_id,
                        trace_id,
                        event_type,
                        error_message,
                        error_type,
                        stack_trace,
                        json.dumps(payload, default=str),
                    ),
                )
                result = cur.fetchone()
            conn.commit()
            logger.warning(
                f"Saved failed event to DLQ: id={failed_id}, "
                f"event_id={event_id}, type={event_type}, error={error_message}"
            )
            return result[0] if result else failed_id
        except Exception as e:
            logger.error(f"Failed to save to DLQ (data may be lost): {e}")
            conn.rollback()
            raise
        finally:
            self._close_connection(conn)

    def get_failed_events(
        self,
        status: str = "pending",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get failed events from the DLQ for retry or investigation.

        Args:
            status: Filter by status (pending, retrying, resolved, abandoned)
            limit: Maximum number of events to return

        Returns:
            List of failed event records
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, event_id, trace_id, event_type,
                           error_message, error_type, stack_trace,
                           payload_json, retry_count, max_retries,
                           status, created_at, last_retry_at
                    FROM {self._failed_table_name}
                    WHERE status = %s
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (status, limit),
                )
                rows = cur.fetchall()
            conn.rollback()  # Read-only, end transaction

            return [
                {
                    "id": row[0],
                    "event_id": row[1],
                    "trace_id": row[2],
                    "event_type": row[3],
                    "error_message": row[4],
                    "error_type": row[5],
                    "stack_trace": row[6],
                    "payload": row[7],
                    "retry_count": row[8],
                    "max_retries": row[9],
                    "status": row[10],
                    "created_at": row[11],
                    "last_retry_at": row[12],
                }
                for row in rows
            ]
        finally:
            self._close_connection(conn)

    def mark_failed_event_resolved(
        self,
        failed_event_id: str,
        resolution_notes: Optional[str] = None,
    ) -> None:
        """
        Mark a failed event as resolved.

        Args:
            failed_event_id: ID of the failed event record
            resolution_notes: Optional notes about how it was resolved
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self._failed_table_name}
                    SET status = 'resolved',
                        resolved_at = NOW(),
                        resolution_notes = %s
                    WHERE id = %s
                    """,
                    (resolution_notes, failed_event_id),
                )
            conn.commit()
        finally:
            self._close_connection(conn)
