"""
RabbitMQ Exporter - Production exporter for async event processing.

Publishes events to RabbitMQ for consumption by the ingestion worker.
Uses the existing marie-ai RabbitMQ infrastructure.
"""

import json
import logging
import urllib.parse
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from marie.llm_tracking.config import get_settings
from marie.llm_tracking.exporters.base import AbstractExporter
from marie.llm_tracking.types import (
    EventType,
    Observation,
    QueueMessage,
    Score,
    Trace,
)

logger = logging.getLogger(__name__)


class RabbitMQExporter(AbstractExporter):
    """
    Exporter that publishes events to RabbitMQ.

    Uses the existing marie-ai BlockingPikaClient for synchronous operations.
    Events are published as lightweight queue messages that reference
    the full event data stored in Postgres/S3.
    """

    def __init__(
        self,
        rabbitmq_url: Optional[str] = None,
        exchange: Optional[str] = None,
        routing_key: Optional[str] = None,
        queue: Optional[str] = None,
    ):
        """
        Initialize the RabbitMQ exporter.

        Args:
            rabbitmq_url: RabbitMQ connection URL (or from config)
            exchange: Exchange name (or from config)
            routing_key: Routing key (or from config)
            queue: Queue name (or from config)
        """
        settings = get_settings()
        self._rabbitmq_url = rabbitmq_url or settings.RABBITMQ_URL
        self._exchange = exchange or settings.RABBITMQ_EXCHANGE
        self._routing_key = routing_key or settings.RABBITMQ_ROUTING_KEY
        self._queue = queue or settings.RABBITMQ_QUEUE

        self._client: Optional[Any] = None
        self._started = False

    def _parse_url(self) -> Dict[str, Any]:
        """Parse RabbitMQ URL into connection config."""
        parsed = urlparse(self._rabbitmq_url)
        # Extract vhost from URL path (URL-decode it)
        # Empty path or "/" means default vhost "/"
        # "/marie" means vhost "marie"
        vhost_path = parsed.path.lstrip("/")
        vhost = urllib.parse.unquote(vhost_path) if vhost_path else "/"

        return {
            "provider": "rabbitmq",
            "hostname": parsed.hostname or "localhost",
            "port": parsed.port or 5672,
            "username": parsed.username or "guest",
            "password": parsed.password or "guest",
            "tls": parsed.scheme == "amqps",
            "vhost": vhost,
        }

    def start(self) -> None:
        """Initialize RabbitMQ connection and setup topology."""
        if self._started:
            return

        try:
            from marie.messaging.rabbitmq.client import BlockingPikaClient

            conf = self._parse_url()
            self._client = BlockingPikaClient(conf)

            # Declare exchange
            from pika.exchange_type import ExchangeType

            self._client.exchange_declare(
                exchange=self._exchange,
                exchange_type=ExchangeType.topic,
                durable=True,
            )

            # Declare queue and bind to exchange so messages are queued
            # even when the worker isn't consuming yet
            self._client.declare_queue(queue_name=self._queue, durable=True)
            self._client.queue_bind(
                queue=self._queue,
                exchange=self._exchange,
                routing_key=self._routing_key,
            )

            self._started = True
            logger.info(
                f"RabbitMQ exporter started: exchange={self._exchange}, "
                f"queue={self._queue}, routing_key={self._routing_key}"
            )
        except ImportError as e:
            logger.error(f"Failed to import RabbitMQ client: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def stop(self) -> None:
        """Close RabbitMQ connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing RabbitMQ connection: {e}")
            finally:
                self._client = None
                self._started = False
        logger.debug("RabbitMQ exporter stopped")

    def _reconnect_client(self) -> bool:
        """
        Attempt to reconnect the RabbitMQ client.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        try:
            logger.info("Attempting to reconnect RabbitMQ exporter client...")

            # Close existing client
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass

            # Recreate client
            from marie.messaging.rabbitmq.client import BlockingPikaClient

            conf = self._parse_url()
            self._client = BlockingPikaClient(conf)

            # Re-declare topology
            from pika.exchange_type import ExchangeType

            self._client.exchange_declare(
                exchange=self._exchange,
                exchange_type=ExchangeType.topic,
                durable=True,
            )
            self._client.declare_queue(queue_name=self._queue, durable=True)
            self._client.queue_bind(
                queue=self._queue,
                exchange=self._exchange,
                routing_key=self._routing_key,
            )

            logger.info("RabbitMQ exporter client reconnected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reconnect RabbitMQ exporter client: {e}")
            return False

    def _publish(
        self,
        event_type: EventType,
        event_id: str,
        trace_id: str,
        project_id: str,
        max_retries: int = 2,
    ) -> None:
        """
        Publish a queue message to RabbitMQ.

        Includes retry logic with client reconnection on failure.

        Args:
            event_type: Type of the event
            event_id: ID of the raw event in Postgres
            trace_id: Associated trace ID
            project_id: Project ID
            max_retries: Number of retries at exporter level (client also retries internally)

        Raises:
            RuntimeError: If exporter is not started
            Exception: If publish fails after all retries (re-raised for caller to handle)
        """
        if not self._started or self._client is None:
            # Raise instead of silently skipping - caller should handle with DLQ
            raise RuntimeError(
                f"RabbitMQ exporter not started, cannot publish event {event_id}"
            )

        message = QueueMessage(
            event_id=event_id,
            event_type=event_type,
            trace_id=trace_id,
            project_id=project_id,
        )

        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                # BlockingPikaClient.publish_message() has its own retry logic
                # This outer retry handles cases where the client itself needs rebuilding
                self._client.publish_message(
                    exchange=self._exchange,
                    routing_key=self._routing_key,
                    message=message.to_dict(),
                )
                logger.debug(f"Published {event_type.value} event: {event_id}")
                return  # Success

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Publish failed for event {event_id} (attempt {attempt + 1}/{max_retries}), "
                        f"attempting client reconnection: {e}"
                    )
                    if self._reconnect_client():
                        continue  # Retry with new client
                    else:
                        logger.error("Client reconnection failed, giving up")
                        break

        # All retries exhausted
        logger.error(f"Failed to publish event {event_id} after {max_retries} attempts")
        if last_error:
            raise last_error

    def export_trace(self, trace: Trace) -> None:
        """
        Export a trace to RabbitMQ.

        Note: The full trace data should already be stored in Postgres/S3.
        This only publishes a lightweight notification.
        """
        self._publish(
            event_type=EventType.TRACE_CREATE,
            event_id=trace.id,
            trace_id=trace.id,
            project_id=trace.project_id,
        )

    def export_traces(self, traces: List[Trace]) -> None:
        """Export multiple traces."""
        for trace in traces:
            self.export_trace(trace)

    def export_observation(self, observation: Observation) -> None:
        """
        Export an observation to RabbitMQ.

        Note: The full observation data should already be stored in Postgres/S3.
        """
        event_type = EventType.GENERATION_CREATE
        if observation.type.value == "SPAN":
            event_type = EventType.SPAN_CREATE
        elif observation.type.value == "EVENT":
            event_type = EventType.EVENT_CREATE

        self._publish(
            event_type=event_type,
            event_id=observation.id,
            trace_id=observation.trace_id,
            project_id=observation.project_id,
        )

    def export_observations(self, observations: List[Observation]) -> None:
        """Export multiple observations."""
        for observation in observations:
            self.export_observation(observation)

    def export_score(self, score: Score) -> None:
        """Export a score to RabbitMQ."""
        self._publish(
            event_type=EventType.SCORE_CREATE,
            event_id=score.id,
            trace_id=score.trace_id,
            project_id=score.project_id,
        )

    def export_scores(self, scores: List[Score]) -> None:
        """Export multiple scores."""
        for score in scores:
            self.export_score(score)


class AsyncRabbitMQExporter(AbstractExporter):
    """
    Async version of RabbitMQExporter.

    Uses the AsyncPikaClient for non-blocking operations.
    """

    def __init__(
        self,
        rabbitmq_url: Optional[str] = None,
        exchange: Optional[str] = None,
        routing_key: Optional[str] = None,
        queue: Optional[str] = None,
    ):
        """
        Initialize the async RabbitMQ exporter.

        Args:
            rabbitmq_url: RabbitMQ connection URL (or from config)
            exchange: Exchange name (or from config)
            routing_key: Routing key (or from config)
            queue: Queue name (or from config)
        """
        settings = get_settings()
        self._rabbitmq_url = rabbitmq_url or settings.RABBITMQ_URL
        self._exchange = exchange or settings.RABBITMQ_EXCHANGE
        self._routing_key = routing_key or settings.RABBITMQ_ROUTING_KEY
        self._queue = queue or settings.RABBITMQ_QUEUE

        self._client: Optional[Any] = None
        self._started = False

    def _parse_url(self) -> Dict[str, Any]:
        """Parse RabbitMQ URL into connection config."""
        parsed = urlparse(self._rabbitmq_url)
        # Extract vhost from URL path (URL-decode it)
        # Empty path or "/" means default vhost "/"
        # "/marie" means vhost "marie"
        vhost_path = parsed.path.lstrip("/")
        vhost = urllib.parse.unquote(vhost_path) if vhost_path else "/"

        return {
            "provider": "rabbitmq",
            "hostname": parsed.hostname or "localhost",
            "port": parsed.port or 5672,
            "username": parsed.username or "guest",
            "password": parsed.password or "guest",
            "tls": parsed.scheme == "amqps",
            "vhost": vhost,
        }

    async def start_async(self) -> None:
        """Initialize async RabbitMQ connection."""
        if self._started:
            return

        try:
            from pika.exchange_type import ExchangeType

            from marie.messaging.rabbitmq.client import AsyncPikaClient

            conf = self._parse_url()
            self._client = await AsyncPikaClient.get_instance(conf)

            # Ensure exchange and queue exist
            await self._client.ensure_topology(
                exchange=self._exchange,
                queue=self._queue,
                exchange_type=ExchangeType.topic,
                durable=True,
                routing_key=self._routing_key,
            )

            self._started = True
            logger.info(
                f"Async RabbitMQ exporter started: exchange={self._exchange}, "
                f"queue={self._queue}, routing_key={self._routing_key}"
            )
        except ImportError as e:
            logger.error(f"Failed to import async RabbitMQ client: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def start(self) -> None:
        """Sync wrapper - use start_async() in async contexts."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            # Already in async context - this shouldn't be called
            logger.warning("start() called in async context, use start_async()")
        except RuntimeError:
            # No running loop - create one
            asyncio.run(self.start_async())

    async def stop_async(self) -> None:
        """Close async RabbitMQ connection."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning(f"Error closing async RabbitMQ connection: {e}")
            finally:
                self._client = None
                self._started = False
        logger.debug("Async RabbitMQ exporter stopped")

    def stop(self) -> None:
        """Sync wrapper - use stop_async() in async contexts."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            logger.warning("stop() called in async context, use stop_async()")
        except RuntimeError:
            asyncio.run(self.stop_async())

    async def _publish_async(
        self, event_type: EventType, event_id: str, trace_id: str, project_id: str
    ) -> None:
        """Publish a queue message asynchronously."""
        if not self._started or self._client is None:
            logger.warning("Async RabbitMQ exporter not started, skipping publish")
            return

        message = QueueMessage(
            event_id=event_id,
            event_type=event_type,
            trace_id=trace_id,
            project_id=project_id,
        )

        try:
            await self._client.publish_message(
                exchange=self._exchange,
                routing_key=self._routing_key,
                message=message.to_dict(),
            )
            logger.debug(f"Published {event_type.value} event: {event_id}")
        except Exception as e:
            logger.error(f"Failed to publish event {event_id}: {e}")

    def _run_async_publish(
        self,
        event_type: EventType,
        event_id: str,
        trace_id: str,
        project_id: str,
    ) -> None:
        """
        Run async publish handling both sync and async contexts.

        Uses create_task in async context, asyncio.run in sync context.
        """
        import asyncio

        coro = self._publish_async(
            event_type=event_type,
            event_id=event_id,
            trace_id=trace_id,
            project_id=project_id,
        )

        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            # We're in async context - schedule as task
            loop.create_task(coro)
        except RuntimeError:
            # No running loop - we're in sync context
            # Use asyncio.run() to run in a new loop
            asyncio.run(coro)

    def export_trace(self, trace: Trace) -> None:
        """Sync wrapper for trace export."""
        self._run_async_publish(
            event_type=EventType.TRACE_CREATE,
            event_id=trace.id,
            trace_id=trace.id,
            project_id=trace.project_id,
        )

    def export_observation(self, observation: Observation) -> None:
        """Sync wrapper for observation export."""
        event_type = EventType.GENERATION_CREATE
        if observation.type.value == "SPAN":
            event_type = EventType.SPAN_CREATE
        elif observation.type.value == "EVENT":
            event_type = EventType.EVENT_CREATE

        self._run_async_publish(
            event_type=event_type,
            event_id=observation.id,
            trace_id=observation.trace_id,
            project_id=observation.project_id,
        )

    def export_score(self, score: Score) -> None:
        """Sync wrapper for score export."""
        self._run_async_publish(
            event_type=EventType.SCORE_CREATE,
            event_id=score.id,
            trace_id=score.trace_id,
            project_id=score.project_id,
        )
