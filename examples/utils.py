import sys
from threading import Thread
import pika
import json


def setup_queue(
    api_key: str,
    queue: str = "events",
    exit_on_event: str = None,
    callback: callable = None,
):
    print(f"Setting up queue for : {api_key}")

    def on_message_callback(channel, method_frame, header_frame, body):
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        payload = body.decode("utf-8")
        payload = json.loads(payload)
        event = payload["event"]
        jobid = payload["jobid"]
        print(f"Received event : {event} : {jobid}")
        print(payload)
        if callback is not None:
            callback(payload)

        # exit the consumer thread when the extract is completed
        if exit_on_event is not None and event == exit_on_event:
            sys.exit(0)

    def consume(api_key, queue, on_message_received):
        print(f"Consuming queue : {queue}")

        while True:
            queue = f"{api_key}.{queue}"
            try:
                connection = pika.BlockingConnection()
                channel = connection.channel()
                channel.basic_consume(queue, on_message_received)
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

    consumer_thread = Thread(target=consume, args=(api_key, queue, on_message_callback))
    consumer_thread.start()
