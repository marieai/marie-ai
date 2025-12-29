"""
ClickHouse Writer - Batched writer for high-throughput inserts.

Ported from Langfuse's TypeScript ClickhouseWriter to Python.
Provides batched inserts with size-based and time-based flushing.
"""

import atexit
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID

from marie.llm_tracking.clickhouse.client import ClickHouseClientManager
from marie.llm_tracking.config import get_settings
from marie.llm_tracking.types import Observation, Score, Trace

logger = logging.getLogger(__name__)


class TableName(str, Enum):
    """ClickHouse table names for LLM tracking."""

    TRACES = "traces"
    OBSERVATIONS = "observations"
    SCORES = "scores"


T = TypeVar("T", Trace, Observation, Score)


@dataclass
class QueueItem(Generic[T]):
    """Item in the write queue."""

    created_at: float  # timestamp in seconds
    attempts: int
    data: T


@dataclass
class ClickHouseWriterConfig:
    """Configuration for ClickHouse writer."""

    batch_size: int = 1000
    flush_interval_seconds: float = 5.0
    max_attempts: int = 3
    max_field_size: int = 1024 * 1024  # 1MB per field


class ClickHouseWriter:
    """
    Batched ClickHouse writer for LLM tracking data.

    Features:
    - Batched inserts (size-based + time-based flushing)
    - Thread-safe queue management
    - Automatic retries with backoff
    - Oversized record truncation
    - Graceful shutdown with final flush
    """

    _instance: Optional["ClickHouseWriter"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ClickHouseWriter":
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize the writer."""
        if getattr(self, "_initialized", False):
            return

        settings = get_settings()
        self._config = ClickHouseWriterConfig(
            batch_size=settings.CLICKHOUSE_BATCH_SIZE,
            flush_interval_seconds=settings.CLICKHOUSE_FLUSH_INTERVAL_S,
            max_attempts=settings.CLICKHOUSE_MAX_ATTEMPTS,
        )

        # Queues for each table type
        self._queues: Dict[TableName, List[QueueItem]] = {
            TableName.TRACES: [],
            TableName.OBSERVATIONS: [],
            TableName.SCORES: [],
        }
        self._queue_locks: Dict[TableName, threading.Lock] = {
            TableName.TRACES: threading.Lock(),
            TableName.OBSERVATIONS: threading.Lock(),
            TableName.SCORES: threading.Lock(),
        }

        self._client: Optional[ClickHouseClientManager] = None
        self._flush_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._started = False
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "ClickHouseWriter":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None

    def start(self) -> None:
        """Start the writer and background flush thread."""
        if self._started:
            return

        try:
            self._client = ClickHouseClientManager.get_instance()
            self._client.start()

            # Start background flush thread
            self._shutdown_event.clear()
            self._flush_thread = threading.Thread(
                target=self._flush_loop,
                name="clickhouse-writer-flush",
                daemon=True,
            )
            self._flush_thread.start()

            self._started = True
            logger.info(
                f"ClickHouse writer started: batch_size={self._config.batch_size}, "
                f"flush_interval={self._config.flush_interval_seconds}s"
            )

            # Register shutdown hook
            atexit.register(self.shutdown)

        except Exception as e:
            logger.error(f"Failed to start ClickHouse writer: {e}")
            raise

    def shutdown(self) -> None:
        """Shutdown the writer, flushing all pending data."""
        if not self._started:
            return

        logger.info("Shutting down ClickHouse writer...")

        # Signal flush thread to stop
        self._shutdown_event.set()

        # Wait for flush thread to finish
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=10.0)

        # Final flush of all queues
        self._flush_all(full_queue=True)

        self._started = False
        logger.info("ClickHouse writer shutdown complete")

    def _flush_loop(self) -> None:
        """Background thread that periodically flushes queues."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for flush interval or shutdown
                self._shutdown_event.wait(timeout=self._config.flush_interval_seconds)
                if not self._shutdown_event.is_set():
                    self._flush_all()
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")

    def _flush_all(self, full_queue: bool = False) -> None:
        """Flush all queues."""
        for table_name in TableName:
            try:
                self._flush(table_name, full_queue=full_queue)
            except Exception as e:
                logger.error(f"Error flushing {table_name.value}: {e}")

    def _flush(self, table_name: TableName, full_queue: bool = False) -> None:
        """Flush a specific queue to ClickHouse."""
        queue = self._queues[table_name]
        lock = self._queue_locks[table_name]

        with lock:
            if not queue:
                return

            # Take items from queue
            batch_size = len(queue) if full_queue else self._config.batch_size
            items = queue[:batch_size]
            del queue[:batch_size]

        if not items:
            return

        # Log wait times
        now = time.time()
        for item in items:
            wait_time = now - item.created_at
            if wait_time > 10.0:
                logger.warning(
                    f"High queue wait time for {table_name.value}: {wait_time:.2f}s"
                )

        # Prepare records for insert
        records = [self._prepare_record(table_name, item.data) for item in items]

        # Try to write with retries
        for attempt in range(self._config.max_attempts):
            try:
                self._write_to_clickhouse(table_name, records)
                logger.debug(f"Flushed {len(records)} records to {table_name.value}")
                return
            except Exception as e:
                if attempt < self._config.max_attempts - 1:
                    logger.warning(
                        f"ClickHouse write failed for {table_name.value} "
                        f"(attempt {attempt + 1}/{self._config.max_attempts}): {e}"
                    )
                    # Check if we should truncate
                    if self._is_size_error(e):
                        records = [
                            self._truncate_record(table_name, r) for r in records
                        ]
                    time.sleep(0.1 * (attempt + 1))  # Simple backoff
                else:
                    logger.error(
                        f"ClickHouse write failed after {self._config.max_attempts} attempts: {e}"
                    )
                    # Re-add failed items to queue with incremented attempts
                    with lock:
                        for item in items:
                            if item.attempts < self._config.max_attempts:
                                item.attempts += 1
                                queue.append(item)
                            else:
                                logger.error(
                                    f"Dropping record after max attempts: {item.data}"
                                )

    def _prepare_record(
        self, table_name: TableName, data: Union[Trace, Observation, Score]
    ) -> Dict[str, Any]:
        """Prepare a record for ClickHouse insert."""
        record = data.to_dict()

        # Convert string UUIDs to UUID objects for ClickHouse
        for key in ["id", "trace_id", "observation_id", "parent_observation_id"]:
            if key in record and record[key] is not None:
                if isinstance(record[key], str):
                    try:
                        record[key] = UUID(record[key])
                    except ValueError:
                        pass  # Keep as string if not a valid UUID

        # Parse datetime strings back to datetime objects
        for key in [
            "timestamp",
            "created_at",
            "updated_at",
            "start_time",
            "end_time",
            "completion_start_time",
            "processed_at",
        ]:
            if key in record and record[key] is not None:
                if isinstance(record[key], str):
                    try:
                        record[key] = datetime.fromisoformat(record[key])
                    except ValueError:
                        pass

        return record

    def _write_to_clickhouse(
        self, table_name: TableName, records: List[Dict[str, Any]]
    ) -> None:
        """Write records to ClickHouse."""
        if not self._client:
            raise RuntimeError("ClickHouse client not initialized")

        if not records:
            return

        # Get column names from first record
        columns = list(records[0].keys())

        self._client.insert(
            table=table_name.value,
            data=records,
            columns=columns,
        )

    def _is_size_error(self, error: Exception) -> bool:
        """Check if error is due to record size."""
        error_msg = str(error).lower()
        return (
            "size of json object" in error_msg
            or "extremely large" in error_msg
            or "invalid string length" in error_msg
        )

    def _truncate_record(
        self, table_name: TableName, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Truncate oversized fields in a record."""
        max_size = self._config.max_field_size
        truncation_msg = "[TRUNCATED: Field exceeded size limit]"

        def truncate_field(value: Optional[str]) -> Optional[str]:
            if not value:
                return value
            if len(value) > max_size:
                return value[: max_size // 2] + truncation_msg
            return value

        # Truncate large string fields
        for field_name in ["input", "output", "metadata"]:
            if field_name in record and isinstance(record[field_name], str):
                original_len = len(record[field_name])
                record[field_name] = truncate_field(record[field_name])
                if original_len > max_size:
                    logger.warning(
                        f"Truncated {field_name} field in {table_name.value} "
                        f"record {record.get('id')}: {original_len} -> {len(record[field_name])}"
                    )

        return record

    # ========== Public API ==========

    def add_trace(self, trace: Trace) -> None:
        """Add a trace to the write queue."""
        self._add_to_queue(TableName.TRACES, trace)

    def add_observation(self, observation: Observation) -> None:
        """Add an observation to the write queue."""
        self._add_to_queue(TableName.OBSERVATIONS, observation)

    def add_score(self, score: Score) -> None:
        """Add a score to the write queue."""
        self._add_to_queue(TableName.SCORES, score)

    def _add_to_queue(
        self, table_name: TableName, data: Union[Trace, Observation, Score]
    ) -> None:
        """Add an item to the appropriate queue."""
        if not self._started:
            self.start()

        item = QueueItem(
            created_at=time.time(),
            attempts=1,
            data=data,
        )

        with self._queue_locks[table_name]:
            self._queues[table_name].append(item)
            queue_len = len(self._queues[table_name])

        # Trigger flush if batch size reached
        if queue_len >= self._config.batch_size:
            logger.debug(f"Queue full, flushing {table_name.value}...")
            try:
                self._flush(table_name)
            except Exception as e:
                logger.error(f"Error during batch flush: {e}")

    def flush(self) -> None:
        """Manually flush all queues."""
        self._flush_all(full_queue=True)

    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes for monitoring."""
        return {
            table_name.value: len(self._queues[table_name]) for table_name in TableName
        }


# Convenience function
def get_clickhouse_writer() -> ClickHouseWriter:
    """Get the singleton ClickHouse writer."""
    return ClickHouseWriter.get_instance()
