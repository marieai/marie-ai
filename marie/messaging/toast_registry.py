import logging
from typing import Any, Dict, MutableMapping, OrderedDict, Union, Iterable, List
from typing import Dict, Any

import asyncio
from marie.messaging.publisher import MessagePublisher
from marie.logging.predefined import default_logger

logger = default_logger


class ToastHandler:
    """
    Push notification service
    """

    _strict_kwargs_check = True

    def _check_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Checks if the given arguments are empty. Throws a ValueError if strict
        kwargs checking is enabled and args are non-empty. If strict kwargs
        checking is disabled, only a warning is logged.

        Args:
            kwargs (Dict[str, Any])
        """
        if self._strict_kwargs_check:
            if len(kwargs) > 0:
                raise ValueError("Unused arguments: {}".format(kwargs))
        else:
            logger = logging.getLogger(__name__)
            for k, v in kwargs.items():
                logger.warning("{}={} argument ignored".format(k, v))

    def get_supported_events(self) -> List[str]:
        """
        Returns the list of events this handler supports
        Valid events include '*' as well as 'name.*'
        @return:
        """
        raise NotImplementedError()

    async def notify(self, notification: Any, **kwargs: Any) -> None:
        """
        Send push notification

        @param notification: A Notification object that contains information about the event that occurred.
        @param kwargs:
        @return:
        """

        raise NotImplementedError()


class NativeToastHandler(ToastHandler):
    """
    Native toast registry
    """

    def __init__(self, **kwargs: Any):
        pass

    def get_supported_events(self) -> List[str]:
        return ["*"]

    def notify(self, notification: Any, **kwargs: Any) -> None:
        logger.info(f"Pushing notification")
        self._check_kwargs(kwargs)
        logger.info(notification)


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

    _NOTIFICATION_HANDLERS: MutableMapping[str, ToastHandler] = OrderedDict()
    __NATIVE_NOTIFICATION_HANDLER = NativeToastHandler()

    @staticmethod
    def __get_event_handlers(event: str) -> Iterable[ToastHandler]:
        _iterables = []
        for p in Toast._NOTIFICATION_HANDLERS.keys():
            if event.startswith(p):
                _iterables.append(Toast._NOTIFICATION_HANDLERS[p])

        return [Toast.__NATIVE_NOTIFICATION_HANDLER]

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def notify(event: str, notification: Any, **kwargs: Any) -> None:
        for handler in Toast.__get_event_handlers(event):
            handler.notify(notification, **kwargs)

            if False:
                task = asyncio.ensure_future(MessagePublisher.publish(None))
                sync = False
                if sync:
                    results = await asyncio.gather(task)
                    return results[0]

    @staticmethod
    def register_handler(handler: ToastHandler) -> None:

        assert isinstance(handler, ToastHandler), handler
        for prefix in handler.get_supported_events():
            assert prefix not in Toast._NOTIFICATION_HANDLERS
            Toast._NOTIFICATION_HANDLERS[prefix] = handler

        Toast._NOTIFICATION_HANDLERS = OrderedDict(
            sorted(
                Toast._NOTIFICATION_HANDLERS.items(),
                key=lambda t: t[0],
                reverse=True,
            )
        )


Toast.register_handler(NativeToastHandler())
