"""
LLM Tracking Worker - RabbitMQ consumer for event processing.

This worker:
1. Consumes events from RabbitMQ
2. Fetches full payloads from Postgres/S3
3. Normalizes events (token counting, cost calculation)
4. Writes to ClickHouse in batches

Usage:
    # As standalone process
    python -m marie.llm_tracking.worker

    # Programmatically
    from marie.llm_tracking.worker import LLMTrackingWorker
    worker = LLMTrackingWorker()
    worker.start()
"""

import atexit
import json
import logging
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

from marie.llm_tracking.clickhouse.writer import ClickHouseWriter, get_clickhouse_writer
from marie.llm_tracking.config import get_settings
from marie.llm_tracking.normalizer import EventNormalizer, get_normalizer
from marie.llm_tracking.storage.postgres import PostgresStorage
from marie.llm_tracking.storage.s3 import S3Storage
from marie.llm_tracking.types import EventType, QueueMessage, RawEvent

logger = logging.getLogger(__name__)


class LLMTrackingWorker:
    """
    RabbitMQ consumer worker for LLM event processing.

    Listens to the configured queue and processes events:
    - Fetches raw events from Postgres
    - Retrieves large payloads from S3
    - Normalizes events (token counting, cost calculation)
    - Writes to ClickHouse in batches
    """

    _instance: Optional["LLMTrackingWorker"] = None

    def __init__(self):
        """Initialize the worker."""
        self._settings = get_settings()
        self._connection = None
        self._channel = None
        self._postgres: Optional[PostgresStorage] = None
        self._s3: Optional[S3Storage] = None
        self._clickhouse: Optional[ClickHouseWriter] = None
        self._normalizer: Optional[EventNormalizer] = None
        self._started = False
        self._shutdown_event = threading.Event()
        self._consumer_tag: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "LLMTrackingWorker":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self) -> None:
        """Start the worker."""
        if self._started:
            return

        logger.info("Starting LLM tracking worker...")

        try:
            # Initialize components
            self._init_postgres()
            self._init_s3()
            self._init_clickhouse()
            self._init_normalizer()
            self._init_rabbitmq()

            self._started = True
            logger.info(
                f"LLM tracking worker started: queue={self._settings.RABBITMQ_QUEUE}"
            )

            # Register shutdown hook
            atexit.register(self.stop)

        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            raise

    def _init_postgres(self) -> None:
        """Initialize Postgres storage."""
        if not self._settings.POSTGRES_URL:
            logger.warning(
                "Postgres URL not configured, worker will fail to fetch events"
            )
            return

        try:
            self._postgres = PostgresStorage()
            self._postgres.start()
            logger.debug("Postgres storage initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Postgres: {e}")
            raise

    def _init_s3(self) -> None:
        """Initialize S3 storage for large payloads."""
        if not self._settings.S3_BUCKET:
            logger.debug("S3 not configured, large payloads will fail")
            return

        try:
            self._s3 = S3Storage()
            self._s3.start()
            logger.debug("S3 storage initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize S3: {e}")

    def _init_clickhouse(self) -> None:
        """Initialize ClickHouse writer."""
        try:
            self._clickhouse = get_clickhouse_writer()
            self._clickhouse.start()
            logger.debug("ClickHouse writer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ClickHouse: {e}")
            raise

    def _init_normalizer(self) -> None:
        """Initialize event normalizer."""
        self._normalizer = get_normalizer()
        logger.debug("Event normalizer initialized")

    def _init_rabbitmq(self) -> None:
        """Initialize RabbitMQ connection and channel."""
        try:
            import pika

            # Parse connection URL
            params = pika.URLParameters(self._settings.RABBITMQ_URL)
            self._connection = pika.BlockingConnection(params)
            self._channel = self._connection.channel()

            # Declare exchange and queue
            self._channel.exchange_declare(
                exchange=self._settings.RABBITMQ_EXCHANGE,
                exchange_type="topic",
                durable=True,
            )

            self._channel.queue_declare(
                queue=self._settings.RABBITMQ_QUEUE,
                durable=True,
            )

            self._channel.queue_bind(
                exchange=self._settings.RABBITMQ_EXCHANGE,
                queue=self._settings.RABBITMQ_QUEUE,
                routing_key=self._settings.RABBITMQ_ROUTING_KEY,
            )

            # Set QoS
            self._channel.basic_qos(prefetch_count=10)

            logger.debug("RabbitMQ connection initialized")

        except ImportError:
            raise ImportError(
                "pika is required for RabbitMQ support. Install with: pip install pika"
            )
        except Exception as e:
            logger.error(f"Failed to initialize RabbitMQ: {e}")
            raise

    def stop(self) -> None:
        """Stop the worker gracefully."""
        if not self._started:
            return

        logger.info("Stopping LLM tracking worker...")

        self._shutdown_event.set()

        # Cancel consumer
        if self._channel and self._consumer_tag:
            try:
                self._channel.basic_cancel(self._consumer_tag)
            except Exception as e:
                logger.warning(f"Error canceling consumer: {e}")

        # Flush ClickHouse
        if self._clickhouse:
            try:
                self._clickhouse.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down ClickHouse: {e}")

        # Close Postgres
        if self._postgres:
            try:
                self._postgres.stop()
            except Exception as e:
                logger.warning(f"Error closing Postgres: {e}")

        # Close S3
        if self._s3:
            try:
                self._s3.stop()
            except Exception as e:
                logger.warning(f"Error closing S3: {e}")

        # Close RabbitMQ
        if self._connection and self._connection.is_open:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing RabbitMQ: {e}")

        self._started = False
        logger.info("LLM tracking worker stopped")

    def run(self) -> None:
        """Run the worker (blocking)."""
        if not self._started:
            self.start()

        logger.info("Starting message consumption...")

        try:
            # Start consuming
            self._consumer_tag = self._channel.basic_consume(
                queue=self._settings.RABBITMQ_QUEUE,
                on_message_callback=self._on_message,
                auto_ack=False,
            )

            # Block until shutdown
            while not self._shutdown_event.is_set():
                self._connection.process_data_events(time_limit=1)

        except Exception as e:
            logger.error(f"Error in worker run loop: {e}")
            raise
        finally:
            self.stop()

    def _on_message(
        self,
        channel,
        method,
        properties,
        body: bytes,
    ) -> None:
        """Handle incoming message."""
        try:
            # Parse message
            message_data = json.loads(body.decode("utf-8"))
            message = QueueMessage.from_dict(message_data)

            logger.debug(
                f"Processing event: {message.event_id} ({message.event_type.value})"
            )

            # Process the event
            self._process_event(message)

            # Ack the message
            channel.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Nack and requeue
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def _process_event(self, message: QueueMessage) -> None:
        """
        Process a single event.

        1. Fetch raw event from Postgres
        2. Fetch payload from S3 if needed
        3. Normalize the event
        4. Write to ClickHouse
        """
        if not self._postgres:
            raise RuntimeError("Postgres not initialized")
        if not self._clickhouse:
            raise RuntimeError("ClickHouse not initialized")
        if not self._normalizer:
            raise RuntimeError("Normalizer not initialized")

        # 1. Fetch raw event from Postgres
        raw_event = self._postgres.get_event(message.event_id)
        if raw_event is None:
            logger.warning(f"Event not found: {message.event_id}")
            return

        # 2. Fetch payload from S3 if needed
        payload = raw_event.payload
        if raw_event.s3_key and self._s3:
            try:
                payload = self._s3.get_payload(raw_event.s3_key)
            except Exception as e:
                logger.error(f"Failed to fetch S3 payload: {e}")
                return

        # 3. Normalize and write based on event type
        if message.event_type in [EventType.TRACE_CREATE, EventType.TRACE_UPDATE]:
            normalized = self._normalizer.normalize_trace(raw_event, payload)
            self._clickhouse.add_trace(normalized.trace)

        elif message.event_type in [
            EventType.SPAN_CREATE,
            EventType.SPAN_UPDATE,
            EventType.GENERATION_CREATE,
            EventType.GENERATION_UPDATE,
            EventType.EVENT_CREATE,
        ]:
            normalized = self._normalizer.normalize_observation(raw_event, payload)
            self._clickhouse.add_observation(normalized.observation)

        elif message.event_type == EventType.SCORE_CREATE:
            normalized = self._normalizer.normalize_score(raw_event, payload)
            self._clickhouse.add_score(normalized.score)

        else:
            logger.warning(f"Unknown event type: {message.event_type}")

        # 4. Mark event as processed in Postgres
        try:
            self._postgres.mark_processed(message.event_id)
        except Exception as e:
            logger.warning(f"Failed to mark event as processed: {e}")

    def process_single_event(self, message: QueueMessage) -> None:
        """
        Process a single event (for testing/direct calls).

        Args:
            message: The queue message to process
        """
        if not self._started:
            self.start()
        self._process_event(message)


def run_worker() -> None:
    """Run the worker as a standalone process."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = get_settings()
    if not settings.ENABLED:
        logger.error("LLM tracking is disabled. Set MARIE_LLM_TRACKING_ENABLED=true")
        sys.exit(1)

    worker = LLMTrackingWorker()

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")
        worker.stop()
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        worker.stop()
        sys.exit(1)


if __name__ == "__main__":
    run_worker()
