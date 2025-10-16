from __future__ import annotations

import asyncio
from typing import Any, List

from marie.excepts import BadConfigSource
from marie.logging_core.logger import MarieLogger
from marie.messaging.events import EventMessage
from marie.messaging.toast_handler import ToastHandler

from .sse_broker import SseBroker


class SseToastHandler(ToastHandler):
    """
    SSE Toast Handler:
      - Enqueues EventMessage
      - Single worker preserves order and retries with backoff
      - Publishes to in-process SseBroker under topic = {api_key}
      - UI subscribes via /sse/{api_key}
    """

    def __init__(self, config: Any, *, broker: SseBroker, **kwargs: Any):
        self.logger = MarieLogger(self.__class__.__name__)
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.broker = broker

        q_cfg = (self.config or {}).get("queue", {})
        self._queue_maxsize: int = int(q_cfg.get("maxsize", 4096))
        self._drop_if_full: bool = bool(q_cfg.get("drop_if_full", False))
        self._enqueue_timeout_s: float = float(q_cfg.get("enqueue_timeout_s", 0.0))

        r_cfg = (self.config or {}).get("retry", {})
        self._backoff_base_s: float = float(r_cfg.get("backoff_base_s", 0.1))
        self._backoff_max_s: float = float(r_cfg.get("backoff_max_s", 2.0))
        self._max_attempts: int = int(r_cfg.get("max_attempts", 0))  # 0 => infinite

        self._queue: asyncio.Queue[EventMessage] = asyncio.Queue(
            maxsize=self._queue_maxsize
        )
        self._shutdown_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        self.logger.info("SseToastHandler initialized.")

    def get_supported_events(self) -> List[str]:
        return ["*"]

    async def _publish_once(self, msg: EventMessage) -> None:
        # topic is per-tenant/api_key; event type drives "event:" name
        if msg.api_key is None or msg.api_key == "":
            raise BadConfigSource("'api_key' missing on EventMessage")
        event_name = msg.event or "event"
        api_key = msg.api_key
        payload = msg if isinstance(msg, (dict, list)) else msg.__dict__
        source = msg.source or "unknown"

        await self.broker.publish(
            source=source,
            topic=api_key,
            event=event_name,
            payload=payload,
        )

    async def _worker(self) -> None:
        self.logger.info("SseToastHandler worker started.")
        backoff = self._backoff_base_s
        while not self._shutdown_event.is_set():
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"SseToastHandler worker queue error: {e}")
                await asyncio.sleep(0.1)
                continue

            attempts = 0
            while not self._shutdown_event.is_set():
                try:
                    await self._publish_once(item)
                    backoff = self._backoff_base_s
                    break
                except Exception as e:
                    attempts += 1
                    self.logger.warning(
                        f"SSE publish failed (attempt {attempts}); retry in {backoff:.2f}s: {e}"
                    )
                    if self._max_attempts and attempts >= self._max_attempts:
                        self.logger.error(
                            f"Dropping SSE event after {attempts} attempts: {getattr(item, 'event', 'unknown')}"
                        )
                        break
                    await asyncio.sleep(backoff)
                    backoff = min(
                        self._backoff_max_s, backoff * 2 or self._backoff_base_s
                    )

            self._queue.task_done()
        self.logger.info("SseToastHandler worker stopped.")

    async def notify(self, notification: EventMessage, **kwargs: Any) -> bool:
        if not self.enabled:
            return False

        if self._worker_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._worker_task = loop.create_task(self._worker())
            except RuntimeError:
                self.logger.warning("notify() called without a running loop.")
                raise

        try:
            if self._drop_if_full and self._queue.full():
                self.logger.warning("SSE toast queue full; dropping message.")
                return False

            if self._enqueue_timeout_s and self._enqueue_timeout_s > 0:
                await asyncio.wait_for(
                    self._queue.put(notification), timeout=self._enqueue_timeout_s
                )
            else:
                await self._queue.put(notification)
            return True
        except asyncio.TimeoutError:
            self.logger.warning("SSE toast enqueue timed out; message dropped.")
            return False
        except Exception as e:
            self.logger.error(f"SSE toast enqueue failed: {e}", exc_info=True)
            return False

    async def close(self, drain: bool = True, timeout: float = 5.0) -> None:
        if drain:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning("SSE Toast: timeout while draining queue.")
        self._shutdown_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
