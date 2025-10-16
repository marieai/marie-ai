import asyncio
import time
from typing import Any, Iterable, MutableMapping, Optional, OrderedDict, Tuple, Union

from marie.excepts import BadConfigSource
from marie.logging_core.predefined import default_logger as logger
from marie.messaging.events import (
    ASSET_EVENTS,
    FAILURE_EVENTS,
    MARKER_EVENTS,
    SUCCESS_EVENTS,
    EventMessage,
    MarieEvent,
    MarieEventType,
)
from marie.messaging.toast_handler import ToastHandler


class Toast:
    """
    A class for users to access notifications from specific providers

        Handlers:
            native          : Notification are stored int JSON format
            amazon-rabbitmq : Amazon MQ
            sns             : Amazon SNS
            rabbitmq        : Amazon MQ,
            database : Database
    """

    _NOTIFICATION_HANDLERS: MutableMapping[str, ToastHandler] = OrderedDict()  # type: ignore
    __NATIVE_NOTIFICATION_HANDLER = None

    # Queue-backed dispatch to preserve ordering and avoid blocking producers
    _QUEUE_MAXSIZE = 2048
    _queue: Optional[asyncio.Queue[Tuple[str, EventMessage, dict]]] = None
    _worker_task: Optional[asyncio.Task] = None
    _shutdown_event: Optional[asyncio.Event] = None
    # sentinel for graceful shutdown
    _SENTINEL: Tuple[str, EventMessage, dict] | None = None

    # soft limits so a burst of events doesn't explode the runtime
    _MAX_CONCURRENT = 16
    _HANDLER_TIMEOUT_SECS = 5

    # --- High-water mark settings / metrics ---
    _WARN_QSIZE_THRESHOLD: Optional[int] = (
        None  # if set, absolute threshold (e.g., 256)
    )
    _WARN_QSIZE_RATIO: Optional[float] = (
        0.75  # if set, fraction of maxsize (e.g., 0.75 => 75%)
    )
    _WARN_INTERVAL_S: float = 5.0  # min seconds between warnings
    _last_warn_ts: float = 0.0
    _ENQ_COUNT: int = 0
    _DROP_COUNT: int = 0

    @staticmethod
    def configure(
        *,
        warn_qsize_threshold: Optional[int] = None,
        warn_qsize_ratio: Optional[float] = None,
        warn_interval_s: Optional[float] = None,
    ) -> None:
        """runtime tuning for high-water warnings."""
        if warn_qsize_threshold is not None:
            Toast._WARN_QSIZE_THRESHOLD = max(1, int(warn_qsize_threshold))
        if warn_qsize_ratio is not None:
            # clamp between 0.0 and 1.0
            Toast._WARN_QSIZE_RATIO = max(0.0, min(1.0, float(warn_qsize_ratio)))
        if warn_interval_s is not None:
            Toast._WARN_INTERVAL_S = max(0.1, float(warn_interval_s))

    @staticmethod
    def _high_water_threshold() -> Optional[int]:
        """Compute effective threshold in queue items."""
        if Toast._queue is None:
            return None
        if Toast._WARN_QSIZE_THRESHOLD:
            return Toast._WARN_QSIZE_THRESHOLD
        if Toast._WARN_QSIZE_RATIO and Toast._queue.maxsize:
            return int(Toast._queue.maxsize * Toast._WARN_QSIZE_RATIO)
        return None

    @staticmethod
    def _maybe_warn_high_water(qsz: int) -> None:
        thr = Toast._high_water_threshold()
        if thr is None:
            return
        if qsz < thr:
            return
        now = time.monotonic()
        if now - Toast._last_warn_ts >= Toast._WARN_INTERVAL_S:
            maxsize = getattr(Toast._queue, "maxsize", 0) if Toast._queue else 0
            logger.warning(
                "Toast queue high-water mark: size=%d threshold=%d max=%d enq=%d drop=%d",
                qsz,
                thr,
                maxsize,
                Toast._ENQ_COUNT,
                Toast._DROP_COUNT,
            )
            Toast._last_warn_ts = now

    @staticmethod
    def stats() -> dict:
        """
        Return current runtime metrics for Toast.
        Keys:
          - running: bool
          - queue_size: int
          - queue_maxsize: int
          - enqueued: int
          - dropped: int
          - high_water_threshold: int | None
          - last_high_water_warn_ts: float
        """
        q = Toast._queue
        wt = Toast._worker_task
        running = bool(wt) and not wt.done()
        return {
            "running": running,
            "queue_size": q.qsize() if q else 0,
            "queue_maxsize": q.maxsize if q else 0,
            "enqueued": Toast._ENQ_COUNT,
            "dropped": Toast._DROP_COUNT,
            "high_water_threshold": Toast._high_water_threshold(),
            "last_high_water_warn_ts": Toast._last_warn_ts,
        }

    @staticmethod
    def _ensure_worker():
        if Toast._queue is None:
            Toast._queue = asyncio.Queue(maxsize=Toast._QUEUE_MAXSIZE)
            Toast._shutdown_event = asyncio.Event()
        if Toast._worker_task is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                raise RuntimeError("Toast requires a running asyncio loop")
            Toast._worker_task = loop.create_task(Toast._worker())

    @staticmethod
    async def _worker():
        logger.info("Toast worker started.")
        try:
            while Toast._shutdown_event and not Toast._shutdown_event.is_set():
                try:
                    item = await Toast._queue.get()
                    if not isinstance(item, tuple) or len(item) != 3:
                        logger.warning("Toast worker received malformed item: %r", item)
                        continue
                    event_name, msg, kw = item
                    if event_name is None or msg is None:
                        # drop harmless wake-ups / bad inputs
                        continue
                    if not isinstance(kw, dict):
                        kw = {}

                    await Toast._dispatch_to_handlers(event_name, msg, **kw)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Toast dispatch failed: {e}")
                finally:
                    try:
                        Toast._queue.task_done()
                    except Exception:
                        pass
        finally:
            logger.info("Toast worker stopped.")

    @staticmethod
    def _enqueue(event_name: str, msg: EventMessage, **kwargs: Any) -> None:
        if Toast._queue is None or Toast._worker_task is None:
            Toast._ensure_worker()
        try:
            kw = dict(kwargs) if kwargs else {}
            Toast._queue.put_nowait((event_name, msg, kw))
            Toast._ENQ_COUNT += 1
            # high-water check after enqueue
            qsz = Toast._queue.qsize()
            Toast._maybe_warn_high_water(qsz)
        except asyncio.QueueFull:
            # drop-oldest then try once more
            try:
                _ = Toast._queue.get_nowait()
                Toast._queue.task_done()
            except asyncio.QueueEmpty:
                pass
            try:
                kw = dict(kwargs) if kwargs else {}
                Toast._queue.put_nowait((event_name, msg, kw))
                Toast._ENQ_COUNT += 1
                qsz = Toast._queue.qsize()
                Toast._maybe_warn_high_water(qsz)
            except asyncio.QueueFull:
                Toast._DROP_COUNT += 1
                logger.error(
                    "Toast queue full (cap=%d). Dropping event=%s jobid=%s drops=%d",
                    Toast._queue.maxsize,
                    event_name,
                    getattr(msg, "jobid", None),
                    Toast._DROP_COUNT,
                )

    @staticmethod
    async def close(drain: bool = True, timeout: float = 5.0):
        """
        Closes the Toast notification system gracefully. This method manages the draining of the
        underlying task queue and handles the cancellation of the worker task.
        """
        if Toast._queue is None:
            return
        if drain:
            try:
                await asyncio.wait_for(Toast._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Toast: timeout while draining queue.")
        if Toast._shutdown_event and not Toast._shutdown_event.is_set():
            Toast._shutdown_event.set()
        try:
            Toast._queue.put_nowait((None, None, {}))
        except asyncio.QueueFull:
            pass
        if Toast._worker_task:
            try:
                await Toast._worker_task
            except asyncio.CancelledError:
                pass
            Toast._worker_task = None

    @staticmethod
    def __get_event_handlers(event: str) -> Iterable[ToastHandler]:
        _iterables = (
            [Toast.__NATIVE_NOTIFICATION_HANDLER]
            if Toast.__NATIVE_NOTIFICATION_HANDLER
            else []
        )
        for p in Toast._NOTIFICATION_HANDLERS.keys():
            # exact “*” or prefix match like "RUN_" to catch RUN_START, RUN_SUCCESS, etc.
            if p == "*" or event.startswith(p):
                _iterables.extend(Toast._NOTIFICATION_HANDLERS[p])  # type: ignore
        # sort by priority (lower runs first)
        _iterables.sort(key=lambda x: x.priority, reverse=False)
        return _iterables

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def _status_for_event_type(ev_type: MarieEventType) -> str:
        if ev_type in FAILURE_EVENTS:
            return "FAILURE"
        if ev_type in SUCCESS_EVENTS:
            return "SUCCESS"
        return "INFO"

    @staticmethod
    def _namespace_for_event_type(ev_type: MarieEventType) -> str:
        # Coarse namespaces for routing/UX
        if ev_type == MarieEventType.ENGINE_EVENT:
            return "engine"
        if ev_type.name.startswith("RESOURCE_"):
            return "resource"
        if ev_type.name.startswith("RUN_"):
            return "run"
        if ev_type.name.startswith("STEP_"):
            return "step"
        if ev_type in ASSET_EVENTS:
            return "asset"
        return "marie"

    @staticmethod
    def marie_event_to_message(
        ev: MarieEvent,
        *,
        api_key: str = "system",
        node: Optional[str] = None,
        jobid: Optional[str] = None,
        timestamp: Optional[int] = None,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> EventMessage:
        import time as _time

        ns = Toast._namespace_for_event_type(ev.event_type)
        dotted = ev.event_type.as_event_name()  # e.g., "run.success", "engine.event"
        event_name = f"{ns}.{dotted}" if not dotted.startswith(f"{ns}.") else dotted

        print('event_name : ', event_name)

        payload: dict[str, Any] = {"message": ev.message}

        es = ev.event_specific_data
        if es is not None:
            # Always include normalized metadata
            payload["metadata"] = dict(es.metadata or {})
            # Include markers for marker events
            if ev.event_type in MARKER_EVENTS:
                if es.marker_start:
                    payload["marker_start"] = es.marker_start
                if es.marker_end:
                    payload["marker_end"] = es.marker_end
            # Include error if any
            if es.error:
                try:
                    payload["error"] = es.error.to_dict()  # type: ignore[attr-defined]
                except Exception:
                    payload["error"] = str(es.error)

        if extra_payload:
            payload.update(extra_payload)

        ts = int(timestamp if timestamp is not None else _time.time())
        job_tag = (
            (es.marker_start if es and es.marker_start else None)
            or (node if node else None)
            or ns
        )

        return EventMessage(
            api_key=api_key,
            jobid=jobid or (node or ns),
            event=event_name,
            jobtag=job_tag,
            status=Toast._status_for_event_type(ev.event_type),
            timestamp=ts,
            payload=payload,
        )

    @staticmethod
    async def _dispatch_to_handlers(
        event_name: str, notification: EventMessage, **kwargs: Any
    ):
        """
        Dispatches the given event and notification to all associated handlers asynchronously.
        This method gathers all event handlers for the specified event and dispatches the given
        notification to each handler. The method ensures that the maximum number of concurrent
        handlers are respected. If a handler times out or raises an exception, it is handled
        and logged accordingly.

        :param event_name: A string indicating the name of the event being dispatched.
        :param notification: The EventMessage containing details about the event that needs
            to be processed by the handlers.
        :param kwargs: Additional keyword arguments to be passed to handlers when invoking them.
        :return: A list of results returned by the event handlers, including `False` for timed-out
            or failed handlers. Exceptions are wrapped in the `return_exceptions` mechanism.
        """
        handlers = list(Toast.__get_event_handlers(event_name))
        if not handlers:
            return []

        sem = asyncio.Semaphore(Toast._MAX_CONCURRENT)

        async def _run_handler(h: ToastHandler):
            try:
                async with sem:
                    return await asyncio.wait_for(
                        h.notify(notification, **kwargs),
                        timeout=Toast._HANDLER_TIMEOUT_SECS,
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "Toast handler %s timed out for event=%s jobid=%s",
                    h.__class__.__name__,
                    event_name,
                    getattr(notification, "jobid", None),
                )
                return False
            except Exception:
                logger.exception(
                    "Toast handler %s crashed for event=%s jobid=%s",
                    h.__class__.__name__,
                    event_name,
                    getattr(notification, "jobid", None),
                )
                return False

        return await asyncio.gather(
            *[asyncio.create_task(_run_handler(h)) for h in handlers],
            return_exceptions=False,
        )

    @staticmethod
    async def notify(
        event: Union[str, MarieEvent],
        notification: Optional[EventMessage] = None,
        **kwargs: Any,
    ) -> bool:
        """Enqueue a toast event; returns True if enqueued."""
        if isinstance(event, MarieEvent):
            msg = Toast.marie_event_to_message(
                event,
                api_key=kwargs.pop("api_key", "system"),
                node=kwargs.pop("node", None),
                jobid=kwargs.pop("jobid", None),
                timestamp=kwargs.pop("timestamp", None),
                extra_payload=kwargs.pop("extra_payload", None),
            )
            Toast._enqueue(msg.event, msg, **kwargs)
            return True

        event_name: str = event
        if notification is None:
            raise ValueError(
                "Toast.notify(event_name, notification) requires EventMessage"
            )
        if notification.api_key is None:
            raise ValueError(f"'api_key' not present in notification : {notification}")
        Toast._enqueue(event_name, notification, **kwargs)
        return True

    @staticmethod
    def register(handler: ToastHandler, native: Optional[bool] = False) -> None:
        """
        Register specific handlers for notifications
        :param handler: The handler to register
        :param native: If the handler is native or not
        """
        assert isinstance(handler, ToastHandler), handler

        if native:
            if Toast.__NATIVE_NOTIFICATION_HANDLER is None:
                Toast.__NATIVE_NOTIFICATION_HANDLER = handler
            else:
                raise BadConfigSource(
                    f"Native handler already registered as : {Toast.__NATIVE_NOTIFICATION_HANDLER}"
                )
        else:
            # add items to a dictionary of lists
            for prefix in handler.get_supported_events():
                # assert prefix not in Toast._NOTIFICATION_HANDLERS
                if prefix not in Toast._NOTIFICATION_HANDLERS:
                    Toast._NOTIFICATION_HANDLERS[prefix] = []
                Toast._NOTIFICATION_HANDLERS[prefix].append(handler)
