import sys
import threading
from datetime import datetime
from functools import partial
from threading import Thread
import pika
import json
import requests
from pika.exchange_type import ExchangeType


def online(api_url) -> bool:
    """
    Check if the API server is online
    :param api_url:
    :return:
    """
    r = requests.head(api_url)
    # The 308 (Permanent Redirect)
    return r.status_code == 200 or r.status_code == 308


def setup_queue(
    api_key: str,
    queue: str = "events",
    routing_key: str = "#",
    stop_event: threading.Event = None,
    exit_on_event: [str | list] = None,
    callback: callable = None,
):
    """
    Setup a queue to receive events from the API server

    :param api_key: The API key
    :param queue: The queue name
    :param routing_key: The routing key to use for the queue binding
    :param stop_event: The event to stop the consumer thread
    :param exit_on_event: The event list to exit the consumer thread
    :param callback: The callback function to call when a message is received
    """

    print(f"Setting up queue for : {api_key}")

    def on_message_callback(
        exit_on_event: [str | list], channel, method_frame, header_frame, body
    ):
        # get current time formatted as string
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        payload = body.decode("utf-8")
        payload = json.loads(payload)
        event = payload["event"]
        jobid = payload["jobid"]
        print(f"Received event : {now},  {event} : {jobid}")
        if callback is not None:
            callback(payload)

        # exit the consumer thread when the extract is completed
        if exit_on_event is not None and event in exit_on_event:
            sys.exit(0)

    def consume(
        api_key: str,
        queue_: str,
        routing_key: str,
        stop_event: threading.Event,
        exit_on_event: str,
        on_message_received: callable,
    ):
        print(f"Consuming queue : {queue_} with routing key : {routing_key}")
        exchange = f"{api_key}.events"
        queue = f"{api_key}.{queue_}"

        while not stop_event.is_set():
            try:
                credentials = pika.PlainCredentials("guest", "guest")
                parameters = pika.ConnectionParameters(
                    "localhost", 5672, "/", credentials
                )

                connection = pika.BlockingConnection(parameters)
                channel = connection.channel()
                channel.basic_qos(prefetch_count=1)

                # Declare the destination exchange with the topic exchange type to allow routing
                channel.exchange_declare(
                    exchange, durable=True, exchange_type=ExchangeType.topic
                )
                channel.queue_declare(queue=queue, durable=True)
                # Bind the queue to the destination exchange
                channel.queue_bind(queue, exchange=exchange, routing_key=routing_key)

                # noinspection PyTypeChecker
                channel.basic_consume(
                    queue, partial(on_message_received, exit_on_event)
                )
                channel.start_consuming()
            # Don't recover if connection was closed by broker
            except pika.exceptions.ConnectionClosedByBroker:
                break
            # Don't recover on channel errors
            except pika.exceptions.AMQPChannelError:
                break
            # Recover on all other connection errors
            except pika.exceptions.AMQPConnectionError:
                continue
        print("Consumer thread has exited")

    consumer_thread = Thread(
        target=consume,
        args=(
            api_key,
            queue,
            routing_key,
            stop_event,
            exit_on_event,
            on_message_callback,
        ),
    )
    consumer_thread.start()
