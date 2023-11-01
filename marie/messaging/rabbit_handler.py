from typing import Any, List

from pika.exchange_type import ExchangeType

from marie.excepts import BadConfigSource
from marie.logging.logger import MarieLogger
from marie.messaging.events import EventMessage
from marie.messaging.rabbitmq import BlockingPikaClient
from marie.messaging.toast_handler import ToastHandler


class RabbitMQToastHandler(ToastHandler):
    """
    Amazon Toast Handler that publishes events to Amazon MQ or RabbitMQ.
    TODO: Need to add support for SNS and make this asynchronous.
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
        client = None
        try:
            self.logger.info(f"Sending notification to RabbitMQ : {notification.event}")
            if notification.api_key is None:
                raise ValueError(
                    f"'api_key' not present in notification : {notification}"
                )

            msg_config = self.config
            api_key = notification.api_key

            exchange = f"{api_key}.events"
            queue = f"{api_key}.all-events"

            routing_key = notification.event if notification.event else "*"
            self.logger.debug(
                f"exchange: {exchange}, queue: {queue}, routing_key: {routing_key}"
            )

            client = BlockingPikaClient(conf=msg_config)
            # Declare the destination exchange with the topic exchange type to allow routing
            client.exchange_declare(
                exchange, durable=True, exchange_type=ExchangeType.topic
            )

            # ensure the queue exists and is bound to the exchange
            # The mandatory flag tells RabbitMq that the message must be routable
            # to a queue. If it is not, the message will be returned to the publisher
            # this is why we need to declare a catch-all (default) queue and bind it to the exchange

            client.declare_queue(queue, durable=True)
            # Bind the queue to the destination exchange
            client.channel.queue_bind(queue, exchange=exchange, routing_key="#")

            client.publish_message(
                exchange=exchange, routing_key=routing_key, message=notification
            )
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
            if client is not None:
                client.close()

    async def notify(self, notification: EventMessage, **kwargs: Any) -> bool:
        if not self.config or not self.config["enabled"]:
            return False

        await self.__notify_task(notification, True, **kwargs)
        # task = asyncio.ensure_future(self.__notify_task(notification, True, **kwargs))

        return True
