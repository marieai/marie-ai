from typing import Dict, Any, List

import logging


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

    async def notify(self, notification: Any, **kwargs: Any) -> bool:
        """
        Send push notification

        @param notification: A Notification object that contains information about the event that occurred.
        @param kwargs:
        @return: returns notification status, true if sucessfull false otherwise
        """

        raise NotImplementedError()

    @property
    def priority(self) -> int:
        """The lower the priority, the earlier the handler is called. Default is 1. to make it the last handler to be called"""
        return 1
