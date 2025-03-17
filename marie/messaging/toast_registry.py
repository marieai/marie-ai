import asyncio
from typing import Any, Iterable, MutableMapping, Optional, OrderedDict

from marie.excepts import BadConfigSource
from marie.helper import add_sync_version, get_or_reuse_loop
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

    @staticmethod
    def __get_event_handlers(event: str) -> Iterable[ToastHandler]:
        _iterables = [Toast.__NATIVE_NOTIFICATION_HANDLER]

        for p in Toast._NOTIFICATION_HANDLERS.keys():
            if p == "*" or event.startswith(p):
                _iterables.extend(Toast._NOTIFICATION_HANDLERS[p])  # type: ignore

        # sort by priority
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

        # tasks = [
        #     asyncio.ensure_future(handler.notify(notification, **kwargs))
        #     for handler in Toast.__get_event_handlers(event)
        # ]
        # # await asyncio.gather(*tasks)

        # Create tasks for each handler.
        tasks = [
            asyncio.create_task(handler.notify(notification, **kwargs))
            for handler in Toast.__get_event_handlers(event)
        ]

    @staticmethod
    def notify_sync(event: str, notification: EventMessage, **kwargs: Any):
        """
        Push notification event to the client
        :param event:
        :param notification:
        :param kwargs:
        """
        get_or_reuse_loop().run_until_complete(
            Toast.notify(event, notification, **kwargs)
        )

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
