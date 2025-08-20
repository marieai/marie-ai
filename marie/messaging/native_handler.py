import asyncio
import json
import logging
from typing import Any, List

from marie.messaging.events import EventMessage
from marie.messaging.toast_registry import ToastHandler


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.

    @param dict fmt_dict: Key: logging format attribute pairs. Defaults to {"message": "message"}.
    @param str time_format: time.strftime() format string. Default: "%Y-%m-%dT%H:%M:%S"
    @param str msec_format: Microsecond formatting. Appended at the end. Default: "%s.%03dZ"
    """

    def __init__(
        self,
        fmt_dict: dict = None,
        time_format: str = "%Y-%m-%dT%H:%M:%S",
        msec_format: str = "%s.%03dZ",
    ):
        super().__init__()
        self.fmt_dict = fmt_dict if fmt_dict is not None else {"message": "message"}
        self.default_time_format = time_format
        self.default_msec_format = msec_format
        self.datefmt = None

    def usesTime(self) -> bool:
        """
        Overwritten to look for the attribute in the format dict values instead of the fmt string.
        """
        return "asctime" in self.fmt_dict.values()

    def formatMessage(self, record) -> dict:
        """
        Overwritten to return a dictionary of the relevant LogRecord attributes instead of a string.
        KeyError is raised if an unknown attribute is provided in the fmt_dict.
        """
        return {
            fmt_key: record.__dict__[fmt_val]
            for fmt_key, fmt_val in self.fmt_dict.items()
        }

    def format(self, record) -> str:
        """
        Mostly the same as the parent's class method, the difference being that a dict is manipulated and dumped as JSON
        instead of a string.
        """
        record.message = record.getMessage()

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        message_dict = self.formatMessage(record)

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            message_dict["exc_info"] = record.exc_text

        if record.stack_info:
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(message_dict, default=str)


class NativeToastHandler(ToastHandler):
    """
    Native Toast Handler that writes events using JSON format
    """

    def __init__(self, filename: str, **kwargs: Any):
        json_handler = logging.FileHandler(filename, delay=True)
        json_formatter = JsonFormatter(
            {
                # "level": "levelname",
                # "loggerName": "name",
                # "processName": "processName",
                # "processID": "process",
                # "threadName": "threadName",
                # "threadID": "thread",e
                "ts": "asctime",
                "event": "message",
            }
        )

        json_handler.setFormatter(json_formatter)

        self.event_logger = logging.getLogger(__name__)
        self.event_logger.setLevel(logging.INFO)
        self.event_logger.addHandler(json_handler)

        # ordered, backpressure-aware async pipeline ---
        q_cfg = kwargs.get("queue", {})
        self._queue_maxsize: int = int(q_cfg.get("maxsize", 4096))
        self._drop_if_full: bool = bool(q_cfg.get("drop_if_full", False))
        self._enqueue_timeout_s: float = float(
            q_cfg.get("enqueue_timeout_s", 0.0)
        )  # 0 => wait forever
        self._queue: asyncio.Queue[EventMessage] = asyncio.Queue(
            maxsize=self._queue_maxsize
        )

        r_cfg = kwargs.get("retry", {})
        self._backoff_base_s: float = float(r_cfg.get("backoff_base_s", 0.05))
        self._backoff_max_s: float = float(r_cfg.get("backoff_max_s", 1.0))

        self._shutdown_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

        # Attempt to start worker immediately if loop is running
        try:
            loop = asyncio.get_running_loop()
            self._worker_task = loop.create_task(self._worker())
        except RuntimeError:
            # Loop will be created later; worker starts on first notify()
            pass

    def get_supported_events(self) -> List[str]:
        """
        Native handler handles all requests, this handles should have the lowest priority
        :return:
        """
        return ["*"]

    def _serialize_event(self, notification: EventMessage) -> str:
        # Convert to a concise, stable string for the 'message' field.
        # If EventMessage has dict-like interface, prefer that:
        try:
            # pydantic/dataclass-like
            payload = getattr(notification, "dict", None) or getattr(
                notification, "model_dump", None
            )
            if callable(payload):
                payload = payload()
            elif hasattr(notification, "__dict__"):
                payload = notification.__dict__
            else:
                payload = {"value": str(notification)}
        except Exception:
            payload = {"value": str(notification)}
        # We pass a string to logger; JsonFormatter will wrap into {"ts":..., "event": "..."}
        return json.dumps(payload, default=str, ensure_ascii=False)

    async def _write(self, notification: EventMessage) -> None:
        # Use the standard logger synchronously (it’s thread-safe)
        # Call within worker to preserve ordering
        msg = self._serialize_event(notification)
        self.event_logger.info(msg)

    async def _worker(self) -> None:
        """
        Single consumer that preserves ordering:
        - Dequeues one item at a time
        - Retries the same item with exponential backoff on failure
        """
        backoff = self._backoff_base_s
        while not self._shutdown_event.is_set():
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Unexpected queue error; small pause then continue
                self.event_logger.warning(f"NativeToastHandler worker queue error: {e}")
                await asyncio.sleep(0.1)
                continue

            # Retry loop for this specific item to preserve order
            while not self._shutdown_event.is_set():
                try:
                    await self._write(item)
                    backoff = self._backoff_base_s
                    break
                except Exception as e:
                    self.event_logger.warning(
                        f"NativeToastHandler write failed, retrying in {backoff:.2f}s: {e}"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(
                        self._backoff_max_s, backoff * 2 or self._backoff_base_s
                    )

            self._queue.task_done()

    async def notify(self, notification: EventMessage, **kwargs: Any) -> bool:
        """
        Enqueue events to preserve ordering and avoid per-call I/O bottlenecks.
        Applies backpressure via bounded queue.
        """
        # Start worker if not started yet (e.g., created before loop running)
        if self._worker_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._worker_task = loop.create_task(self._worker())
            except RuntimeError:
                raise

        try:
            if self._drop_if_full and self._queue.full():
                self.event_logger.warning(
                    "NativeToastHandler queue full; dropping message."
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
            self.event_logger.warning(
                "NativeToastHandler enqueue timed out; dropping message."
            )
            return False
        except Exception as e:
            self.event_logger.error(
                f"NativeToastHandler enqueue failed: {e}", exc_info=True
            )
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
                self.event_logger.warning(
                    "NativeToastHandler: timeout while draining queue."
                )

        self._shutdown_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    @property
    def priority(self) -> int:
        return 0
