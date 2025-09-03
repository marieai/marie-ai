import asyncio
from typing import Any, Iterable, MutableMapping, Optional, OrderedDict

from marie.excepts import BadConfigSource
from marie.logging_core.predefined import default_logger as logger
from marie.messaging.events import EventMessage
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

    # soft limits so a burst of events doesn't explode the runtime
    _MAX_CONCURRENT = 16
    _HANDLER_TIMEOUT_SECS = 5

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
    async def notify(event: str, notification: EventMessage, **kwargs: Any):
        """
        Push notification event to the client
        :param event:
        :param notification:
        :param kwargs:
        """
        if notification.api_key is None:
            raise ValueError(f"'api_key' not present in notification : {notification}")

        # # Create tasks for each handler.
        # tasks = [
        #     asyncio.create_task(handler.notify(notification, **kwargs))
        #     for handler in Toast.__get_event_handlers(event)
        # ]

        handlers = list(Toast.__get_event_handlers(event))
        if not handlers:
            return []

        sem = asyncio.Semaphore(Toast._MAX_CONCURRENT)

        async def _run_handler(h: ToastHandler):
            try:
                async with sem:
                    # enforce per-handler timeout so one bad sink doesn’t block the fan-out
                    return await asyncio.wait_for(
                        h.notify(notification, **kwargs),
                        timeout=Toast._HANDLER_TIMEOUT_SECS,
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "Toast handler %s timed out for event=%s jobid=%s",
                    h.__class__.__name__,
                    event,
                    notification.jobid,
                )
                return False
            except Exception:  # noqa
                logger.exception(
                    "Toast handler %s crashed for event=%s jobid=%s",
                    h.__class__.__name__,
                    event,
                    notification.jobid,
                )
                return False

        tasks = [asyncio.create_task(_run_handler(h)) for h in handlers]

    @staticmethod
    def notify_sync(event: str, notification: EventMessage, **kwargs: Any):
        """
        Push notification event to the client
        :param event:
        :param notification:
        :param kwargs:
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(Toast.notify(event, notification, **kwargs))

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
