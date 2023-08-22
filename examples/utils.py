import sys
import threading
from datetime import datetime
from functools import partial
from threading import Thread
import pika
import json


def setup_queue(
    api_key: str,
    queue: str = "events",
    stop_event: threading.Event = None,
    exit_on_event: [str | list] = None,
    callback: callable = None,
):
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
        print(payload)
        if callback is not None:
            callback(payload)

        # exit the consumer thread when the extract is completed
        if exit_on_event is not None and event in exit_on_event:
            sys.exit(0)

    def consume(
        api_key: str,
        queue: str,
        stop_event: threading.Event,
        exit_on_event: str,
        on_message_received: callable,
    ):
        print(f"Consuming queue : {queue}")
        while not stop_event.is_set():
            try:
                connection = pika.BlockingConnection()
                channel = connection.channel()
                # noinspection PyTypeChecker
                channel.basic_consume(
                    f"{api_key}.{queue}", partial(on_message_received, exit_on_event)
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
        args=(api_key, queue, stop_event, exit_on_event, on_message_callback),
    )
    consumer_thread.start()
