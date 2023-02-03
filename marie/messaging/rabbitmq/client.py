import ssl
from typing import Dict

import json
from typing import Optional

import pika
from pika.exchange_type import ExchangeType


class BlockingPikaClient:
    def __init__(self, conf: Dict[str, str]):

        print("conf")
        print(conf)

        # "amazon-rabbitmq"  "rabbitmq"
        provider = conf.get("provider", "rabbitmq")
        hostname = conf.get("hostname", "localhost")
        port = int(conf.get("port", 5672))
        username = conf.get("username", "guest")
        password = conf.get("password", "guest")
        tls_enabled = conf.get("tls", False)

        # b-ff3d0999-a25b-4a5a-9775-e7ba76f8fa3d.mq.us-east-1.amazonaws.com
        url = f"amqps://{username}:{password}@{hostname}:{port}"

        print("url")
        print(url)
        parameters = pika.URLParameters(url)

        if provider == "amazon-rabbitmq":
            # SSL Context for TLS configuration of Amazon MQ for RabbitMQ
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            ssl_context.set_ciphers("ECDHE+AESGCM:!ECDSA")
            parameters.ssl_options = pika.SSLOptions(context=ssl_context)

        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

    def close(self):
        self.channel.close()
        self.connection.close()

    def declare_queue(self, queue_name: str, durable: Optional[bool] = True):
        print(f"Trying to declare queue({queue_name})...")
        self.channel.queue_declare(queue=queue_name, durable=durable)

    def exchange_declare(
        self,
        exchange,
        exchange_type=ExchangeType.direct,
        passive=False,
        durable=False,
    ):

        print(f"Trying to declare exchange({exchange})...")
        self.channel.exchange_declare(
            exchange=exchange,
            exchange_type=exchange_type,
            passive=passive,
            durable=durable,
        )

    def publish_message(self, exchange, routing_key, message):
        channel = self.connection.channel()
        hdrs = {"key": "val"}
        properties = pika.BasicProperties(
            app_id="marie-service", content_type="application/json", headers=hdrs
        )
        body = json.dumps(message, ensure_ascii=False)
        channel.basic_publish(exchange, routing_key, body, properties)

        print(
            f"Sent message. Exchange: {exchange}, Routing Key: {routing_key}, Body: {body}"
        )
