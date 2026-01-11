"""
LLM Tracking Worker - RabbitMQ consumer for event processing.

This worker:
1. Consumes events from RabbitMQ
2. Fetches event metadata from PostgreSQL
3. Fetches full payloads from S3 (all payloads stored in S3)
4. Normalizes events (token counting, cost calculation)
5. Writes to ClickHouse for analytics

Usage:
    # As standalone process (requires --config)
    python -m marie.llm_tracking.worker --config /path/to/config.yml

    # Programmatically (after configure_from_yaml has been called)
    from marie.llm_tracking.worker import LLMTrackingWorker
    worker = LLMTrackingWorker()
    worker.start()
"""

import argparse
import atexit
import json
import logging
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

import pika
import yaml

from marie.llm_tracking.clickhouse.writer import ClickHouseWriter, get_clickhouse_writer
from marie.llm_tracking.config import configure_from_yaml, get_settings
from marie.llm_tracking.normalizer import EventNormalizer, get_normalizer
from marie.llm_tracking.storage.postgres import PostgresStorage
from marie.llm_tracking.storage.s3 import S3Storage
from marie.llm_tracking.types import EventType, QueueMessage, RawEvent
from marie.messaging.rabbitmq import RabbitMQConsumer

logger = logging.getLogger(__name__)

# Maximum retry attempts before marking event as permanently failed
MAX_RETRY_ATTEMPTS = 3


class LLMTrackingWorker:
    """
    RabbitMQ consumer worker for LLM event processing.

    Listens to the configured queue and processes events:
    - Fetches event metadata from PostgreSQL
    - Fetches ALL payloads from S3 (no inline storage)
    - Normalizes events (token counting, cost calculation)
    - Writes to ClickHouse for analytics
    """

    _instance: Optional["LLMTrackingWorker"] = None

    def __init__(self):
        """Initialize the worker."""
        self._settings = get_settings()
        self._consumer: Optional[RabbitMQConsumer] = None
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
            self._init_postgres()
            self._init_s3()
            self._init_clickhouse()
            self._init_normalizer()
            self._init_rabbitmq()

            self._started = True
            logger.info(
                f"LLM tracking worker started: queue={self._settings.RABBITMQ_QUEUE}"
            )

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
        """Initialize S3 storage (required - all payloads stored in S3)."""
        if not self._settings.S3_BUCKET:
            raise ValueError(
                "S3 bucket not configured. All payloads are stored in S3. "
                "Set MARIE_LLM_TRACKING_S3_BUCKET environment variable."
            )

        try:
            self._s3 = S3Storage()
            self._s3.start()
            logger.debug("S3 storage initialized (via StorageManager)")
        except Exception as e:
            logger.error(f"Failed to initialize S3: {e}")
            raise

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
        """Initialize RabbitMQ connection using RabbitMQConsumer."""
        try:
            # Create consumer from URL
            self._consumer = RabbitMQConsumer.from_url(self._settings.RABBITMQ_URL)

            # Connect with retry logic
            self._consumer.connect()

            # Setup topology (exchange, queue, binding, QoS)
            self._consumer.setup_topology(
                exchange=self._settings.RABBITMQ_EXCHANGE,
                queue=self._settings.RABBITMQ_QUEUE,
                routing_key=self._settings.RABBITMQ_ROUTING_KEY,
                prefetch_count=10,
            )

            logger.debug("RabbitMQ connection initialized via RabbitMQConsumer")

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

        # Close RabbitMQ consumer (handles cancel and connection close)
        if self._consumer:
            try:
                self._consumer.close()
            except Exception as e:
                logger.warning(f"Error closing RabbitMQ consumer: {e}")

        self._started = False
        logger.info("LLM tracking worker stopped")

    def run(self) -> None:
        """Run the worker (blocking)."""
        if not self._started:
            self.start()

        if not self._consumer:
            raise RuntimeError("RabbitMQ consumer not initialized")

        logger.info("Starting message consumption...")

        try:
            # Start consuming using the consumer's channel
            channel = self._consumer.channel
            self._consumer_tag = channel.basic_consume(
                queue=self._settings.RABBITMQ_QUEUE,
                on_message_callback=self._on_message,
                auto_ack=False,
            )

            # Block until shutdown (use connection from consumer)
            while not self._shutdown_event.is_set():
                self._consumer.connection.process_data_events(time_limit=1)

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
        """Handle incoming message with retry tracking."""
        event_id = None
        try:
            # Parse message
            message_data = json.loads(body.decode("utf-8"))
            message = QueueMessage.from_dict(message_data)
            event_id = message.event_id

            # Get retry count from message headers
            headers = properties.headers or {}
            retry_count = headers.get("x-retry-count", 0)

            logger.debug(
                f"Processing event: {event_id} ({message.event_type.value}) "
                f"[attempt {retry_count + 1}/{MAX_RETRY_ATTEMPTS}]"
            )

            # Process the event
            self._process_event(message)

            # Ack the message on success
            channel.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

            # Get current retry count and increment
            headers = properties.headers or {}
            retry_count = headers.get("x-retry-count", 0) + 1

            if retry_count >= MAX_RETRY_ATTEMPTS:
                # Max retries exceeded - mark as permanently failed
                logger.error(
                    f"Event {event_id} failed after {MAX_RETRY_ATTEMPTS} retries, "
                    "marking as permanently failed"
                )
                if self._postgres and event_id:
                    self._postgres.mark_failed(
                        event_id, f"Max retries exceeded ({MAX_RETRY_ATTEMPTS}): {e}"
                    )
                # ACK to remove from queue (don't requeue)
                channel.basic_ack(delivery_tag=method.delivery_tag)
            else:
                # Apply exponential backoff before retrying
                # Backoff: 1s, 2s, 4s (capped at 4s)
                backoff_seconds = min(2 ** (retry_count - 1), 4)
                logger.info(
                    f"Backing off {backoff_seconds}s before retry {retry_count} "
                    f"for event {event_id}"
                )
                time.sleep(backoff_seconds)

                # NACK without requeue, then republish with retry header
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                # Republish with incremented retry count
                channel.basic_publish(
                    exchange=self._settings.RABBITMQ_EXCHANGE,
                    routing_key=self._settings.RABBITMQ_ROUTING_KEY,
                    body=body,
                    properties=pika.BasicProperties(
                        headers={"x-retry-count": retry_count},
                        delivery_mode=pika.DeliveryMode.Persistent,
                    ),
                )
                logger.debug(f"Requeued event {event_id} (attempt {retry_count + 1})")

    def _process_event(self, message: QueueMessage) -> None:
        """
        Process a single event.

        1. Fetch event metadata from PostgreSQL
        2. Fetch payload from S3 (all payloads in S3)
        3. Normalize the event
        4. Write to ClickHouse
        """
        if not self._postgres:
            raise RuntimeError("Postgres not initialized")
        if not self._s3:
            raise RuntimeError("S3 not initialized")
        if not self._clickhouse:
            raise RuntimeError("ClickHouse not initialized")
        if not self._normalizer:
            raise RuntimeError("Normalizer not initialized")

        # 1. Fetch event metadata from PostgreSQL
        raw_event = self._postgres.get_event(message.event_id)
        if raw_event is None:
            logger.warning(f"Event not found: {message.event_id}")
            return

        # 2. Fetch payload from S3 (all payloads stored in S3)
        if not raw_event.s3_key:
            logger.error(
                f"Event {message.event_id} missing s3_key - cannot fetch payload"
            )
            self._postgres.mark_failed(message.event_id, "Missing s3_key")
            return

        try:
            payload = self._s3.get_payload(raw_event.s3_key)
            if payload is None:
                logger.error(f"Failed to fetch payload from S3: {raw_event.s3_key}")
                self._postgres.mark_failed(
                    message.event_id, f"S3 payload not found: {raw_event.s3_key}"
                )
                return
        except Exception as e:
            logger.error(f"Failed to fetch S3 payload: {e}")
            self._postgres.mark_failed(message.event_id, f"S3 fetch error: {e}")
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

        # 4. Mark event as processed in PostgreSQL
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
    """Run the worker as a standalone process.

    Requires a YAML config file to be specified via --config argument.
    The config file must contain an 'llm_tracking' section with the
    required settings (enabled, exporter, rabbitmq, postgres, etc.).
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="LLM Tracking Worker - RabbitMQ consumer for event processing"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # Load YAML configuration
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML configuration: {e}")
        sys.exit(1)

    # Configure LLM tracking settings from YAML
    llm_tracking_config = config.get("llm_tracking", {})
    storage_config = config.get("storage")

    if not llm_tracking_config:
        logger.error("No 'llm_tracking' section found in configuration file")
        sys.exit(1)

    configure_from_yaml(llm_tracking_config, storage_config)

    # Now get_settings() will work
    settings = get_settings()
    if not settings.ENABLED:
        logger.error(
            "LLM tracking is disabled in configuration. "
            "Set llm_tracking.enabled: true in your config file."
        )
        sys.exit(1)

    logger.info(f"Loaded configuration from: {args.config}")
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
