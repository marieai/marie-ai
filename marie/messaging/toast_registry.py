import asyncio
from typing import Any
from typing import (
    MutableMapping,
    OrderedDict,
    Iterable,
    Optional,
)

from marie.excepts import BadConfigSource
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
    async def notify(event: str, notification: Any, **kwargs: Any):
        # lets fire and forget
        tasks = [
            asyncio.ensure_future(handler.notify(notification, **kwargs))
            for handler in Toast.__get_event_handlers(event)
        ]

        # await asyncio.gather(*tasks)

    @staticmethod
    def register(handler: ToastHandler, native: Optional[bool] = False) -> None:

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
