import asyncio
import random
import ssl
from typing import Any, Dict, Optional, Set, Tuple

import pika
from pika.adapters.asyncio_connection import AsyncioConnection
from pika.exchange_type import ExchangeType

from marie.logging_core.predefined import default_logger as logger
from marie.utils.json import to_json


class BlockingPikaClient:
    def __init__(self, conf: Dict[str, str]):
        # "amazon-rabbitmq"  "rabbitmq"

        provider = conf.get("provider", "rabbitmq")
        hostname = conf.get("hostname", "localhost")
        port = int(conf.get("port", 5672))
        username = conf.get("username", "guest")
        password = conf.get("password", "guest")
        tls_enabled = conf.get("tls", False)

        # b-ff3d0999-a25b-4a5a-9775-e7ba76f8fa3d.mq.us-east-1.amazonaws.com
        if tls_enabled:
            url = f"amqps://{username}:{password}@{hostname}:{port}?connection_attempts=3&heartbeat=3600"
        else:
            url = f"amqp://{username}:{password}@{hostname}:{port}?connection_attempts=3&heartbeat=3600"

        parameters = pika.URLParameters(url)
        if provider == "amazon-rabbitmq":
            # SSL Context for TLS configuration of Amazon MQ for RabbitMQ
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            ssl_context.set_ciphers("ECDHE+AESGCM:!ECDSA")
            parameters.ssl_options = pika.SSLOptions(context=ssl_context)

        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        # Turn on delivery confirmations
        self.channel.confirm_delivery()
        self.logger = logger

    def close(self):
        self.channel.close()
        self.connection.close()

    def declare_queue(self, queue_name: str, durable: Optional[bool] = True) -> Any:
        return self.channel.queue_declare(queue=queue_name, durable=durable)

    def exchange_declare(
        self,
        exchange,
        exchange_type=ExchangeType.direct,
        passive=False,
        durable=False,
    ) -> Any:
        """
        Declare an exchange on the RabbitMQ server
        :param exchange:
        :param exchange_type:
        :param passive:
        :param durable:
        :return:
        """
        return self.channel.exchange_declare(
            exchange=exchange,
            exchange_type=exchange_type,
            passive=passive,
            durable=durable,
        )

    def publish_message(self, exchange: str, routing_key: str, message) -> None:
        """
        Publish a message to an exchange with a routing key
        :param exchange:
        :param routing_key:
        :param message:
        """
        # channel = self.connection.channel()
        channel = self.channel
        hdrs = {"key": "val"}

        # this should be a configuration
        # we do an expiration of 24 hours https://www.rabbitmq.com/ttl.html
        hour = 60 * 60 * 1000
        mils = hour * 24

        properties = pika.BasicProperties(
            app_id="marie-service",
            content_type="application/json",
            headers=hdrs,
            delivery_mode=pika.DeliveryMode.Transient,
            expiration=str(mils),
        )
        # body = json.dumps(message, ensure_ascii=False)
        body = to_json(message)
        body = body.encode("utf-8")

        # Send a message
        try:
            channel.basic_publish(
                exchange, routing_key, body, properties, mandatory=True
            )
            self.logger.debug(
                f"Sent message. Exchange: {exchange}, Routing Key: {routing_key}"
            )
        except pika.exceptions.UnroutableError as e:
            self.logger.error(e)


class AsyncPikaClient:
    """
    Async RabbitMQ client using pika.AsyncioConnection.
    Reuses a single connection and channel and caches topology declarations.
    """

    _instance: Optional["AsyncPikaClient"] = None
    _instance_lock = asyncio.Lock()

    def __init__(self, conf: Dict[str, str]):
        self._conf = conf
        self._logger = logger
        self._connection: Optional[AsyncioConnection] = None
        self._channel: Optional[pika.channel.Channel] = None
        self._connected_event = asyncio.Event()
        self._closing = False
        self._connect_lock = asyncio.Lock()
        self._reconnect_lock = asyncio.Lock()
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0

        # Backoff tuning (seconds)
        self._initial_backoff = float(conf.get("reconnect_initial_backoff", 0.5))
        self._max_backoff = float(conf.get("reconnect_max_backoff", 30.0))
        # None or 0 means unlimited attempts
        self._max_reconnect_attempts = int(conf.get("reconnect_max_attempts", 0))

        # Topology cache
        self._topology_cache: Set[Tuple[str, str]] = set()  # (exchange, queue)

    @classmethod
    async def get_instance(cls, conf: Dict[str, str]) -> "AsyncPikaClient":
        async with cls._instance_lock:
            if cls._instance is None:
                cls._instance = AsyncPikaClient(conf)
                await cls._instance.connect()
        return cls._instance

    def _build_parameters(self) -> pika.URLParameters:
        provider = self._conf.get("provider", "rabbitmq")
        hostname = self._conf.get("hostname", "localhost")
        port = int(self._conf.get("port", 5672))
        username = self._conf.get("username", "guest")
        password = self._conf.get("password", "guest")
        tls_enabled = self._conf.get("tls", False)

        if tls_enabled:
            url = f"amqps://{username}:{password}@{hostname}:{port}?connection_attempts=3&heartbeat=3600"
        else:
            url = f"amqp://{username}:{password}@{hostname}:{port}?connection_attempts=3&heartbeat=3600"

        parameters = pika.URLParameters(url)
        if provider == "amazon-rabbitmq":
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            ssl_context.set_ciphers("ECDHE+AESGCM:!ECDSA")
            parameters.ssl_options = pika.SSLOptions(context=ssl_context)
        return parameters

    def _is_connected(self) -> bool:
        return bool(
            self._connection
            and self._channel
            and getattr(self._connection, "is_open", False)
            and getattr(self._channel, "is_open", False)
        )

    def _compute_backoff(self, attempts: int) -> float:
        # attempts starts at 1; capped exponential with jitter (0..50%)
        base = min(
            self._initial_backoff * (2 ** max(0, attempts - 1)), self._max_backoff
        )
        jitter = random.uniform(0, base * 0.5)
        return base + jitter

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._is_connected():
                self._connected_event.set()
                return

            loop = asyncio.get_running_loop()
            params = self._build_parameters()
            connected_fut: asyncio.Future = loop.create_future()

            def on_open(conn: AsyncioConnection) -> None:
                self._logger.info("RabbitMQ Async connection opened")
                if not connected_fut.done():
                    connected_fut.set_result(conn)

            def on_open_error(_conn: AsyncioConnection, exc: Exception) -> None:
                if not connected_fut.done():
                    connected_fut.set_exception(exc)

            def on_closed(_conn: AsyncioConnection, code, text):
                self._logger.warning(
                    f"RabbitMQ connection closed: code={code}, text={text}"
                )
                self._connected_event.clear()
                # Trigger automatic reconnect unless we're intentionally closing
                if not self._closing:
                    self._schedule_reconnect()

            self._connection = AsyncioConnection(
                parameters=params,
                on_open_callback=on_open,
                on_open_error_callback=on_open_error,
                on_close_callback=on_closed,
            )

            await connected_fut

            # Open channel
            channel_fut: asyncio.Future = loop.create_future()

            def on_channel_open(ch: pika.channel.Channel) -> None:
                self._logger.info("RabbitMQ channel opened")
                self._channel = ch
                if not channel_fut.done():
                    channel_fut.set_result(ch)

            self._connection.channel(on_open_callback=on_channel_open)
            await channel_fut

            # Return callback for unroutable messages when mandatory=True
            def on_return(channel, method, properties, body):
                self._logger.warning(
                    f"Message returned: exchange={method.exchange}, routing_key={method.routing_key}, "
                    f"reply_code={method.reply_code}, reply_text={method.reply_text}"
                )

            assert self._channel is not None
            self._channel.add_on_return_callback(on_return)

            self._reconnect_attempts = 0  # reset attempts on successful connect
            self._connected_event.set()

    def _schedule_reconnect(self) -> None:
        # Ensure only one reconnect task is active
        if self._reconnect_task and not self._reconnect_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; let the next ensure_connection drive it
            return
        self._reconnect_task = loop.create_task(
            self._reconnect_loop(), name="rabbitmq-reconnect"
        )

    async def _reconnect_loop(self) -> None:
        async with self._reconnect_lock:
            if self._is_connected() or self._closing:
                return
            self._logger.info("Starting RabbitMQ reconnect loop")
            while not self._closing and not self._is_connected():
                # Check max attempts if configured (>0)
                if (
                    self._max_reconnect_attempts > 0
                    and self._reconnect_attempts >= self._max_reconnect_attempts
                ):
                    self._logger.error(
                        "Max RabbitMQ reconnect attempts reached; giving up until next manual trigger"
                    )
                    return

                self._reconnect_attempts += 1
                delay = self._compute_backoff(self._reconnect_attempts)
                self._logger.warning(
                    f"Reconnecting to RabbitMQ (attempt {self._reconnect_attempts}) in {delay:.2f}s"
                )
                try:
                    await asyncio.sleep(delay)
                    # Attempt a fresh connect
                    await self.connect()
                except Exception as e:
                    self._logger.error(
                        f"Reconnect attempt {self._reconnect_attempts} failed: {e}",
                        exc_info=1,
                    )
                    continue

            if self._is_connected():
                self._logger.info("RabbitMQ reconnect successful")
            else:
                self._logger.warning(
                    "RabbitMQ reconnect loop exited without connection"
                )

    async def ensure_connection(self) -> None:
        if self._is_connected():
            return
        self._schedule_reconnect()
        try:
            await asyncio.wait_for(
                self._connected_event.wait(), timeout=self._max_backoff + 5.0
            )
        except asyncio.TimeoutError:
            # Give one more direct connect attempt as a fallback
            await self.connect()

    async def ensure_topology(
        self,
        exchange: str,
        queue: str,
        exchange_type: ExchangeType = ExchangeType.topic,
        durable: bool = True,
        routing_key: str = "#",
    ) -> None:
        await self.ensure_connection()
        assert self._channel is not None
        key = (exchange, queue)
        if key in self._topology_cache:
            return

        loop = asyncio.get_running_loop()

        # exchange_declare
        ex_fut: asyncio.Future = loop.create_future()

        def on_exchange_declare_ok(_frame):
            if not ex_fut.done():
                ex_fut.set_result(True)

        self._channel.exchange_declare(
            exchange=exchange,
            exchange_type=exchange_type,
            durable=durable,
            passive=False,
            callback=on_exchange_declare_ok,
        )
        await asyncio.wait_for(ex_fut, timeout=10.0)

        # queue_declare
        q_fut: asyncio.Future = loop.create_future()

        def on_queue_declare_ok(_method_frame):
            if not q_fut.done():
                q_fut.set_result(True)

        self._channel.queue_declare(
            queue=queue, durable=durable, callback=on_queue_declare_ok
        )
        await asyncio.wait_for(q_fut, timeout=10.0)

        # queue_bind
        bind_fut: asyncio.Future = loop.create_future()

        def on_bind_ok(_frame):
            if not bind_fut.done():
                bind_fut.set_result(True)

        self._channel.queue_bind(
            queue=queue, exchange=exchange, routing_key=routing_key, callback=on_bind_ok
        )
        await asyncio.wait_for(bind_fut, timeout=10.0)

        self._topology_cache.add(key)

    async def publish_message(
        self, exchange: str, routing_key: str, message: Any
    ) -> None:
        await self.ensure_connection()
        assert self._channel is not None

        hdrs = {"key": "val"}
        hour = 60 * 60 * 1000
        mils = hour * 24

        properties = pika.BasicProperties(
            app_id="marie-service",
            content_type="application/json",
            headers=hdrs,
            delivery_mode=pika.DeliveryMode.Transient,
            expiration=str(mils),
        )
        body = to_json(message).encode("utf-8")

        self._channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key or "*",
            body=body,
            properties=properties,
            mandatory=True,
        )

    async def close(self) -> None:
        self._closing = True
        # Cancel reconnect task if running
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self._channel and self._channel.is_open:
            ch_closed = asyncio.get_running_loop().create_future()

            def on_ch_closed(_frame):
                if not ch_closed.done():
                    ch_closed.set_result(True)

            # Using add_on_close_callback prevents param signature pitfalls
            self._channel.add_on_close_callback(
                lambda _ch, _code, _text: on_ch_closed(None)
            )
            try:
                self._channel.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(ch_closed, timeout=5)
            except asyncio.TimeoutError:
                pass

        if self._connection and self._connection.is_open:
            conn_closed = asyncio.get_running_loop().create_future()

            def on_conn_closed(_conn, _code, _text):
                if not conn_closed.done():
                    conn_closed.set_result(True)

            try:
                self._connection.add_on_close_callback(on_conn_closed)
                self._connection.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(conn_closed, timeout=5)
            except asyncio.TimeoutError:
                pass

        self._connected_event.clear()
