"""
RabbitMQ Consumer with connection resilience and graceful shutdown.

Inspired by:
- https://github.com/donexfience/streamRx-Auth-service/blob/main/src/infrastructure/rabbitmq/rabbitmqConsumer.py
- https://github.com/samctur/ai-tools/blob/main/app/utils/rabbitMQConsumer.py
"""

import ssl
import time
import urllib.parse
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import pika
from pika.exchange_type import ExchangeType

from marie.logging_core.predefined import default_logger as logger


class RabbitMQConsumer:
    """
    RabbitMQ consumer with connection resilience and topology management.

    Usage:
        consumer = RabbitMQConsumer.from_url("amqp://user:pass@localhost:5672/")
        consumer.connect()
        consumer.setup_topology(
            exchange="my-exchange",
            queue="my-queue",
            routing_key="my.routing.key"
        )
        consumer.start_consuming(callback=my_handler)
        # ... on shutdown:
        consumer.close()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        retry_delay: float = 2.0,
        max_retries: int = 0,  # 0 = unlimited
    ):
        """
        Initialize RabbitMQ consumer.

        :param config: Connection config dict with keys:
            - provider: 'rabbitmq' or 'amazon-rabbitmq'
            - hostname: RabbitMQ host
            - port: RabbitMQ port (default 5672)
            - username: Auth username
            - password: Auth password
            - tls: Enable TLS (default False)
        :param retry_delay: Seconds to wait between connection retries
        :param max_retries: Max connection attempts (0 = unlimited)
        """
        self._config = config
        self._retry_delay = retry_delay
        self._max_retries = max_retries

        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.channel.Channel] = None
        self._consumer_tag: Optional[str] = None
        self._is_running = False

    @classmethod
    def from_url(
        cls,
        url: str,
        provider: str = "rabbitmq",
        **kwargs,
    ) -> "RabbitMQConsumer":
        """
        Create consumer from AMQP URL string.

        :param url: AMQP URL (e.g., amqp://user:pass@localhost:5672/vhost)
        :param provider: Provider type ('rabbitmq' or 'amazon-rabbitmq')
        :param kwargs: Additional arguments passed to constructor
        :return: RabbitMQConsumer instance
        """
        parsed = urlparse(url)
        # Extract vhost from URL path (URL-decode it)
        # Empty path or "/" means default vhost "/"
        # "/marie" means vhost "marie"
        # "/%2F" means vhost "/" (URL-encoded)
        vhost_path = parsed.path.lstrip("/")
        vhost = urllib.parse.unquote(vhost_path) if vhost_path else "/"
        config = {
            "provider": provider,
            "hostname": parsed.hostname or "localhost",
            "port": parsed.port or 5672,
            "username": parsed.username or "guest",
            "password": parsed.password or "guest",
            "tls": parsed.scheme == "amqps",
            "vhost": vhost,
        }
        return cls(config, **kwargs)

    @property
    def channel(self) -> pika.channel.Channel:
        """Get the channel (for direct access if needed)."""
        if self._channel is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")
        return self._channel

    @property
    def connection(self) -> pika.BlockingConnection:
        """Get the connection (for advanced usage like process_data_events)."""
        if self._connection is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")
        return self._connection

    @property
    def is_connected(self) -> bool:
        """Check if consumer is connected."""
        return (
            self._connection is not None
            and self._connection.is_open
            and self._channel is not None
            and self._channel.is_open
        )

    @property
    def is_running(self) -> bool:
        """Check if consumer is actively consuming."""
        return self._is_running

    def connect(self) -> None:
        """
        Establish connection to RabbitMQ with retry logic.

        Raises:
            pika.exceptions.AMQPConnectionError: If max_retries exceeded
        """
        attempt = 0
        while True:
            try:
                parameters = self._build_parameters()
                self._connection = pika.BlockingConnection(parameters)
                self._channel = self._connection.channel()
                logger.info("RabbitMQ consumer connected successfully")
                return
            except pika.exceptions.AMQPConnectionError as e:
                attempt += 1
                if self._max_retries > 0 and attempt >= self._max_retries:
                    logger.error(
                        f"Max connection retries ({self._max_retries}) exceeded"
                    )
                    raise
                logger.warning(
                    f"RabbitMQ connection failed (attempt {attempt}), "
                    f"retrying in {self._retry_delay}s: {e}"
                )
                time.sleep(self._retry_delay)

    def _build_parameters(self) -> pika.URLParameters:
        """Build pika connection parameters from config."""
        hostname = self._config.get("hostname", "localhost")
        port = int(self._config.get("port", 5672))
        username = self._config.get("username", "guest")
        password = self._config.get("password", "guest")
        tls_enabled = self._config.get("tls", False)
        vhost = self._config.get("vhost", "/")

        # URL-encode vhost for inclusion in URL
        # Default "/" becomes "/" (trailing slash)
        # Non-default "marie" becomes "/marie"
        if vhost == "/":
            vhost_part = "/"
        else:
            vhost_part = "/" + urllib.parse.quote(vhost, safe="")

        if tls_enabled:
            url = f"amqps://{username}:{password}@{hostname}:{port}{vhost_part}?connection_attempts=3&heartbeat=3600"
        else:
            url = f"amqp://{username}:{password}@{hostname}:{port}{vhost_part}?connection_attempts=3&heartbeat=3600"

        parameters = pika.URLParameters(url)

        # Handle Amazon RabbitMQ SSL
        if self._config.get("provider") == "amazon-rabbitmq":
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            ssl_context.set_ciphers("ECDHE+AESGCM:!ECDSA")
            parameters.ssl_options = pika.SSLOptions(context=ssl_context)

        return parameters

    def setup_topology(
        self,
        exchange: str,
        queue: str,
        routing_key: str,
        exchange_type: ExchangeType = ExchangeType.topic,
        durable: bool = True,
        prefetch_count: int = 10,
    ) -> None:
        """
        Declare exchange, queue, binding and set QoS.

        :param exchange: Exchange name
        :param queue: Queue name
        :param routing_key: Routing key pattern
        :param exchange_type: Exchange type (default: topic)
        :param durable: Make exchange/queue durable (default: True)
        :param prefetch_count: QoS prefetch count (default: 10)
        """
        if self._channel is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        # Declare exchange
        self._channel.exchange_declare(
            exchange=exchange,
            exchange_type=exchange_type,
            durable=durable,
        )

        # Declare queue
        self._channel.queue_declare(queue=queue, durable=durable)

        # Bind queue to exchange
        self._channel.queue_bind(
            queue=queue,
            exchange=exchange,
            routing_key=routing_key,
        )

        # Set QoS for fair dispatch
        self._channel.basic_qos(prefetch_count=prefetch_count)

        logger.debug(
            f"Topology setup complete: exchange={exchange}, queue={queue}, "
            f"routing_key={routing_key}"
        )

    def start_consuming(
        self,
        queue: str,
        callback: Callable[
            [
                pika.channel.Channel,
                pika.spec.Basic.Deliver,
                pika.spec.BasicProperties,
                bytes,
            ],
            None,
        ],
        auto_ack: bool = False,
    ) -> None:
        """
        Start consuming messages (blocking).

        :param queue: Queue name to consume from
        :param callback: Function called for each message
            Signature: (channel, method, properties, body) -> None
        :param auto_ack: If True, messages are auto-acknowledged
        """
        if self._channel is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        self._is_running = True
        self._consumer_tag = self._channel.basic_consume(
            queue=queue,
            on_message_callback=callback,
            auto_ack=auto_ack,
        )

        logger.info(f"Starting to consume from queue: {queue}")
        try:
            self._channel.start_consuming()
        except KeyboardInterrupt:
            self.stop_consuming()

    def stop_consuming(self) -> None:
        """Stop consuming messages gracefully."""
        self._is_running = False

        if self._consumer_tag and self._channel:
            try:
                self._channel.basic_cancel(self._consumer_tag)
            except Exception as e:
                logger.warning(f"Error canceling consumer: {e}")

        if self._channel:
            try:
                self._channel.stop_consuming()
            except Exception as e:
                logger.warning(f"Error stopping consumption: {e}")

        logger.info("Consumer stopped")

    def close(self) -> None:
        """Close connection and cleanup resources."""
        self.stop_consuming()

        if self._connection and self._connection.is_open:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")

        self._connection = None
        self._channel = None
        self._consumer_tag = None

        logger.info("RabbitMQ consumer closed")
