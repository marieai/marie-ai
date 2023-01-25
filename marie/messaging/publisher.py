from typing import Any
from marie.logging.predefined import default_logger
from marie.messaging.rabbitmq import BlockingPikaClient


class MessagePublisher:
    @staticmethod
    async def publish_to_rabbitmq(message: Any) -> None:
        import logging

        logging.basicConfig(level=logging.DEBUG)

        msg_config = {
            "provider": "amazon-rabbitmq",
            "hostname": "",
            "port": 5671,
            "username": "marie-ai",
            "password": "",
            "virtualhost": "/marie-ai",
            "tls": True,
        }

        completion_queue = "marie.completed"
        exchange = "marie"

        basic_message_sender = BlockingPikaClient(conf=msg_config)
        basic_message_sender.exchange_declare(exchange, durable=True)
        basic_message_sender.declare_queue(completion_queue, durable=True)

        basic_message_sender.publish_message(
            exchange="", routing_key=completion_queue, message=message
        )

        basic_message_sender.close()

    @staticmethod
    async def publish(message: Any) -> None:
        default_logger.info(f"Publishing msg : {message}")
