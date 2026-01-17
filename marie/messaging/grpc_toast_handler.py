"""
gRPC Toast Handler - bridges Toast registry to GrpcEventBroker.

Follows the same patterns as SseToastHandler:
- Async queue for decoupling
- Background worker for ordering
- Retry with exponential backoff
"""

import asyncio
import logging
from typing import Any, List, Optional

from marie.messaging.events import EventMessage
from marie.messaging.grpc_event_broker import GrpcEventBroker
from marie.messaging.toast_handler import ToastHandler

logger = logging.getLogger(__name__)


class GrpcToastHandler(ToastHandler):
    """
    Toast handler that publishes events to GrpcEventBroker.

    Follows same patterns as SseToastHandler:
    - Async queue for decoupling
    - Background worker for ordering
    - Retry with exponential backoff
    """

    def __init__(
        self,
        config: Any,
        *,
        broker: GrpcEventBroker,
        **kwargs: Any,
    ):
        super().__init__()
        self.config = config or {}
        self.broker = broker

        # Queue configuration
        q_cfg = self.config.get("queue", {})
        self._queue_maxsize: int = int(q_cfg.get("maxsize", 4096))
        self._drop_if_full: bool = bool(q_cfg.get("drop_if_full", False))
        self._enqueue_timeout_s: float = float(q_cfg.get("enqueue_timeout_s", 0.0))

        # Retry configuration
        r_cfg = self.config.get("retry", {})
        self._backoff_base_s: float = float(r_cfg.get("backoff_base_s", 0.1))
        self._backoff_max_s: float = float(r_cfg.get("backoff_max_s", 2.0))
        self._max_attempts: int = int(r_cfg.get("max_attempts", 0))  # 0 = infinite

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info("GrpcToastHandler initialized")
        # Try to start worker if event loop is already running
        try:
            loop = asyncio.get_running_loop()
            self._worker_task = loop.create_task(self._worker())
            logger.info("GrpcToastHandler worker started in __init__")
        except RuntimeError:
            # No running loop yet; worker will start on first notify()
            pass

    def get_supported_events(self) -> List[str]:
        """Returns list of supported event patterns."""
        return ["*"]

    @property
    def priority(self) -> int:
        """Handler priority. Same as SSE for parallel operation during migration."""
        return 1

    async def notify(self, notification: EventMessage, **kwargs: Any) -> bool:
        """Enqueue event for async processing."""
        self._check_kwargs(kwargs)

        # Start worker if we were constructed before loop was running
        if self._worker_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._worker_task = loop.create_task(self._worker())
                logger.info("GrpcToastHandler worker started lazily in notify()")
            except RuntimeError:
                logger.warning("notify() called without a running loop")
                raise

        try:
            if self._enqueue_timeout_s > 0:
                await asyncio.wait_for(
                    self._queue.put(notification),
                    timeout=self._enqueue_timeout_s,
                )
            else:
                self._queue.put_nowait(notification)
            return True
        except asyncio.QueueFull:
            if self._drop_if_full:
                logger.warning(f"gRPC queue full, dropping event: {notification.id}")
                return False
            raise
        except asyncio.TimeoutError:
            logger.warning(f"gRPC queue timeout, dropping event: {notification.id}")
            return False

    async def start(self) -> None:
        """Start background worker."""
        self._shutdown_event.clear()
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("GrpcToastHandler worker started")

    async def _worker(self) -> None:
        """Process queue with retry logic."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for item with timeout to check shutdown
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                await self._publish_with_retry(item)
                self._queue.task_done()

            except Exception as e:
                logger.exception(f"GrpcToastHandler worker error: {e}")

    async def _publish_with_retry(self, msg: EventMessage) -> None:
        """Publish with exponential backoff retry."""
        backoff = self._backoff_base_s
        attempt = 0

        while True:
            attempt += 1
            try:
                await self._publish_once(msg)
                return
            except Exception as e:
                if self._max_attempts > 0 and attempt >= self._max_attempts:
                    logger.error(f"Max retries exceeded for event {msg.id}: {e}")
                    return

                logger.warning(
                    f"Publish failed (attempt {attempt}), "
                    f"retrying in {backoff}s: {e}"
                )
                await asyncio.sleep(backoff)
                backoff = min(self._backoff_max_s, backoff * 2)

    async def _publish_once(self, msg: EventMessage) -> None:
        """Single publish attempt."""
        print(f"Publishing event {msg.id} to gRPC broker")
        await self.broker.publish_event_message(msg)

    async def close(self, drain: bool = True, timeout: float = 5.0) -> None:
        """Graceful shutdown."""
        self._shutdown_event.set()

        if drain and not self._queue.empty():
            logger.info(f"Draining {self._queue.qsize()} events...")
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Drain timeout, {self._queue.qsize()} events dropped")

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("GrpcToastHandler closed")
