from typing import Any, List

from marie.excepts import BadConfigSource
from marie.logging.logger import MarieLogger
from marie.messaging.events import EventMessage
from marie.messaging.toast_handler import ToastHandler


class GrpcToastHandler(ToastHandler):
    """
    This is a toast handler that will send a message to a grpc client.
    """

    def __init__(self, config: Any, **kwargs: Any):
        self.config = config
        self.logger = MarieLogger(context=self.__class__.__name__)

    def get_supported_events(self) -> List[str]:
        return ["*"]

    async def __notify_task(
        self,
        notification: EventMessage,
        silence_exceptions: bool = False,
        **kwargs: Any,
    ) -> None:
        try:
            if notification.api_key is None:
                raise ValueError(
                    f"'api_key' not present in notification : {notification}"
                )

            msg_config = self.config
            api_key = notification.api_key

            exchange = f"{api_key}.events"
            queue = f"{api_key}.all-events"
            event_key = notification.event if notification.event else "*"

        except Exception as e:
            if silence_exceptions:
                self.logger.warning(
                    "Toast enabled but config not setup correctly", exc_info=1
                )
            else:
                raise BadConfigSource(
                    "Toast enabled but config not setup correctly"
                ) from e
        finally:
            pass

    async def notify(self, notification: EventMessage, **kwargs: Any) -> bool:
        if not self.config or not self.config["enabled"]:
            return False
        await self.__notify_task(notification, True, **kwargs)
        # task = asyncio.ensure_future(self.__notify_task(notification, True, **kwargs))

        return True
