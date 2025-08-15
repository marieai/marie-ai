# python
import asyncio
from typing import Any, List

from pika.exchange_type import ExchangeType

from marie.excepts import BadConfigSource
from marie.logging_core.logger import MarieLogger
from marie.messaging.events import EventMessage
from marie.messaging.rabbitmq import AsyncPikaClient  # switched to async client
from marie.messaging.toast_handler import ToastHandler


class RabbitMQToastHandler(ToastHandler):
    """
    RabbitMQ Toast Handler that publishes events to RabbitMQ using a single async client.
    """

    def __init__(self, config: Any, **kwargs: Any):
        self.config = config
        self.logger = MarieLogger(context=self.__class__.__name__)
        self._client: AsyncPikaClient | None = None
        self._client_ready = asyncio.Lock()  # serialize first-time init
        self.logger.info("RabbitMQ Toast Handler started.")
        # Warm-up connection in background (doesn't block constructor)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._warmup())
        except RuntimeError:
            # No running loop yet; warm-up will happen on first use
            pass

    async def _warmup(self) -> None:
        ok = await self.verify_connection()
        if ok:
            self.logger.info("RabbitMQ connection verified.")
        else:
            self.logger.warning(
                "RabbitMQ connection not available yet; will retry on demand."
            )

    async def verify_connection(self) -> bool:
        """
        Ensure the AsyncPikaClient is connected and channel is open.
        Returns True if connected; otherwise logs and returns False.
        """
        try:
            client = await self._get_client()
            await client.ensure_connection()
            # Inspect connection/channel state
            conn = getattr(client, "_connection", None)
            ch = getattr(client, "_channel", None)
            ok = bool(
                conn
                and ch
                and getattr(conn, "is_open", False)
                and getattr(ch, "is_open", False)
            )
            if not ok:
                self.logger.error(
                    "RabbitMQ connection verification failed: channel/connection not open."
                )
            return ok
        except Exception as e:
            self.logger.error(
                f"RabbitMQ connection verification error: {e}", exc_info=1
            )
            return False

    async def _get_client(self) -> AsyncPikaClient:
        if self._client is not None:
            return self._client
        async with self._client_ready:
            # double-checked after acquiring the lock
            if self._client is None:
                self._client = await AsyncPikaClient.get_instance(self.config)
        return self._client

    def get_supported_events(self) -> List[str]:
        return ["*"]

    async def __notify_task(
        self,
        notification: EventMessage,
        silence_exceptions: bool = False,
        **kwargs: Any,
    ) -> None:
        try:
            self.logger.info(f"Sending notification to RabbitMQ : {notification.event}")
            if notification.api_key is None:
                raise ValueError(
                    f"'api_key' not present in notification : {notification}"
                )

            # Removed duplicate connectivity check here; notify() does the fail-fast gate.

            api_key = notification.api_key
            exchange = f"{api_key}.events"
            queue = f"{api_key}.all-events"
            routing_key = notification.event if notification.event else "*"

            self.logger.info(
                f"exchange: {exchange}, queue: {queue}, routing_key: {routing_key}"
            )

            client = await self._get_client()

            await client.ensure_topology(
                exchange=exchange,
                queue=queue,
                exchange_type=ExchangeType.topic,
                durable=True,
                routing_key="#",
            )

            await client.publish_message(
                exchange=exchange, routing_key=routing_key, message=notification
            )

        except Exception as e:
            if silence_exceptions:
                self.logger.warning(
                    "Toast enabled but config not setup correctly", exc_info=1
                )
            else:
                raise BadConfigSource(
                    "Toast enabled but config not setup correctly"
                ) from e

    async def notify(self, notification: EventMessage, **kwargs: Any) -> bool:
        if not self.config or not self.config.get("enabled"):
            return False

        # Single fail-fast connectivity check
        if not await self.verify_connection():
            self.logger.warning("Skipping notification: RabbitMQ not connected.")
            return False

        await self.__notify_task(notification, True, **kwargs)
        return True
