import ssl
from typing import Dict, Any

import json
from typing import Optional

import pika
from pika.exchange_type import ExchangeType


class BlockingPikaClient:
    def __init__(self, conf: Dict[str, str]):
        # "amazon-rabbitmq"  "rabbitmq"
        provider = conf.get("provider", "rabbitmq")
        hostname = conf.get("hostname", "localhost")
        port = int(conf.get("port", 5672))
        username = conf.get("username", "guest")
        password = conf.get("password", "guest")
        tls_enabled = conf.get("tls", False)

        # b-ff3d0999-a25b-4a5a-9775-e7ba76f8fa3d.mq.us-east-1.amazonaws.com
        url = f"amqps://{username}:{password}@{hostname}:{port}?connection_attempts=3&heartbeat=3600"

        parameters = pika.URLParameters(url)
        if provider == "amazon-rabbitmq":
            # SSL Context for TLS configuration of Amazon MQ for RabbitMQ
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            ssl_context.set_ciphers("ECDHE+AESGCM:!ECDSA")
            parameters.ssl_options = pika.SSLOptions(context=ssl_context)

        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        # Turn on delivery confirmations
        self.channel.confirm_delivery()

    def close(self):
        self.channel.close()
        self.connection.close()

    def declare_queue(self, queue_name: str, durable: Optional[bool] = True) -> Any:
        print(f"Trying to declare queue({queue_name})...")
        return self.channel.queue_declare(queue=queue_name, durable=durable)

    def exchange_declare(
        self,
        exchange,
        exchange_type=ExchangeType.direct,
        passive=False,
        durable=False,
    ) -> Any:

        print(f"Trying to declare exchange({exchange})...")
        return self.channel.exchange_declare(
            exchange=exchange,
            exchange_type=exchange_type,
            passive=passive,
            durable=durable,
        )

    def publish_message(self, exchange, routing_key, message):
        # channel = self.connection.channel()
        channel = self.channel
        hdrs = {"key": "val"}

        properties = pika.BasicProperties(
            app_id="marie-service",
            content_type="application/json",
            headers=hdrs,
            delivery_mode=pika.DeliveryMode.Transient,
        )
        body = json.dumps(message, ensure_ascii=False)

        # Send a message
        try:
            # channel.basic_publish(
            #     exchange=exchange,
            #     routing_key="test",
            #     body="Hello World!",
            #     properties=pika.BasicProperties(
            #         content_type="text/plain", delivery_mode=pika.DeliveryMode.Transient
            #     ),
            #     mandatory=True,
            # )

            channel.basic_publish(
                exchange, routing_key, body, properties, mandatory=True
            )
            print(
                f"Sent message. Exchange: {exchange}, Routing Key: {routing_key}, Body: {body}"
            )
        except pika.exceptions.UnroutableError as e:
            print(e)
            print("Message was returned")
