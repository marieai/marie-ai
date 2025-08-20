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

        q_cfg = (self.config or {}).get("queue", {})
        self._queue_maxsize: int = int(q_cfg.get("maxsize", 2048))
        self._drop_if_full: bool = bool(q_cfg.get("drop_if_full", False))
        self._enqueue_timeout_s: float = float(
            q_cfg.get("enqueue_timeout_s", 0.0)
        )  # 0 => no wait
        self._queue: asyncio.Queue[EventMessage] = asyncio.Queue(
            maxsize=self._queue_maxsize
        )
        self._shutdown_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

        # NEW: retry/backoff knobs for worker
        r_cfg = (self.config or {}).get("retry", {})
        self._backoff_base_s: float = float(r_cfg.get("backoff_base_s", 0.5))
        self._backoff_max_s: float = float(r_cfg.get("backoff_max_s", 10.0))

        self.logger.info("RabbitMQ Toast Handler started.")
        # Warm-up connection in background
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._warmup())
            self._worker_task = loop.create_task(self._worker())
        except RuntimeError:
            # No running loop yet; worker will start on first use via notify()
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
        """
        Single publish action (no retries here). The worker ensures ordering and retries.
        """
        try:
            if notification.api_key is None:
                raise ValueError(
                    f"'api_key' not present in notification : {notification}"
                )

            api_key = notification.api_key
            exchange = f"{api_key}.events"
            queue = f"{api_key}.all-events"
            routing_key = notification.event if notification.event else "*"

            client = await self._get_client()
            # ensure topology once per stream; harmless if idempotent
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
                    f"RabbitMQ publish failed (silenced): {e}", exc_info=1
                )
            else:
                raise BadConfigSource(
                    "Toast enabled but config not setup correctly"
                ) from e

    async def _worker(self) -> None:
        """
        Single consumer that preserves ordering by:
        - Getting one item at a time
        - Retrying the same item on failure with backoff
        - Only then moving to the next item
        """
        self.logger.info("RabbitMQToastHandler worker started.")
        backoff = self._backoff_base_s
        while not self._shutdown_event.is_set():
            try:
                notification = await self._queue.get()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker queue error: {e}", exc_info=1)
                await asyncio.sleep(0.5)
                continue

            # retry loop for the current item
            while not self._shutdown_event.is_set():
                try:
                    # Ensure connection before trying to publish
                    ok = await self.verify_connection()
                    if not ok:
                        raise RuntimeError("RabbitMQ not connected")

                    # publish (no reordering: we retry this item until success or shutdown)
                    await self.__notify_task(notification, silence_exceptions=False)
                    # success => reset backoff and go to next message
                    backoff = self._backoff_base_s
                    break
                except Exception as e:
                    self.logger.warning(
                        f"Publish failed; will retry after {backoff:.2f}s. Error: {e}"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(
                        self._backoff_max_s, backoff * 2 or self._backoff_base_s
                    )

            self._queue.task_done()

        self.logger.info("RabbitMQToastHandler worker stopped.")

    async def notify(self, notification: EventMessage, **kwargs: Any) -> bool:
        """
        Enqueue the notification for ordered delivery.
        - Preserves ordering via single worker.
        - Applies backpressure according to queue settings.
        """
        if not self.config or not self.config.get("enabled"):
            return False

        # Start worker if we were constructed before loop was running
        if self._worker_task is None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._warmup())
                self._worker_task = loop.create_task(self._worker())
            except RuntimeError:
                self.logger.warning("notify() called without a running loop.")
                raise

        try:
            # apply backpressure policy
            if self._drop_if_full and self._queue.full():
                self.logger.warning(
                    "Toast queue full; dropping message due to drop_if_full=True."
                )
                return False

            if self._enqueue_timeout_s and self._enqueue_timeout_s > 0:
                await asyncio.wait_for(
                    self._queue.put(notification), timeout=self._enqueue_timeout_s
                )
            else:
                await self._queue.put(notification)

            return True
        except asyncio.TimeoutError:
            self.logger.warning("Toast enqueue timed out; message dropped.")
            return False
        except Exception as e:
            self.logger.error(f"Failed to enqueue toast notification: {e}", exc_info=1)
            return False

    async def close(self, drain: bool = True, timeout: float = 5.0) -> None:
        """
        Graceful shutdown:
        - Optionally drain the queue
        - Stop the worker
        """
        if drain:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for queue to drain.")

        self._shutdown_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
