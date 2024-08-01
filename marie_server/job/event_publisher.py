import inspect
from typing import Callable, Dict, List, TypeVar, Union

T = TypeVar("T")


class EventPublisher:
    """
    Publisher

    This class represents a publisher that allows subscribers to subscribe to specific event types and receive messages when those events are published.

    Example Usage:
        publisher = Publisher()

        def subscriber1(message):
            print("Subscriber 1:", message)

        def subscriber2(message):
            print("Subscriber 2:", message)

        publisher.subscribe("event_type1", subscriber1)
        publisher.subscribe("event_type2", subscriber2)

        publisher.publish("event_type1", "Message 1")  # Output: Subscriber 1: Message 1
        publisher.publish("event_type2", "Message 2")  # Output: Subscriber 2: Message 2

        publisher.unsubscribe("event_type1", subscriber1)
        publisher.publish("event_type1", "Message 3")  # No output

    Notes:
        - The publish method is asynchronous, allowing for non-blocking message publishing in asynchronous contexts.
        - Subscribers can be any callable function that accepts a single string argument. This allows for flexible handling of messages by subscribers.
        - Subscribers can subscribe to multiple event types by calling the subscribe method multiple times with different event types.

    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[T], None]]] = {}

    def subscribe(
        self, event_type: Union[str, List[str]], subscriber: Callable[[T], None]
    ):
        """
        Subscribes a subscriber function to a specific event type. The subscriber should be a callable function that accepts a single string argument representing the message.

        :param event_type: A string representing the type of event to subscribe to.
        :param subscriber: A callable function that takes a string parameter and does not return anything.
        :return: None
        """

        if isinstance(event_type, str):
            event_type = [event_type]
        for et in event_type:
            self._subscribers.setdefault(et, []).append(subscriber)

    def unsubscribe(self, event_type: str, subscriber: Callable[[T], None]):
        """

        Unsubscribe method removes a subscriber from the event_type's subscriber list.

        :param event_type: The type of event to unsubscribe from.
        :type event_type: str
        :param subscriber: The subscriber function to be removed.
        :type subscriber: Callable[[str], None]
        :return: None
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(subscriber)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    async def publish(self, event_type: str, message: T):
        """
        Publishes a message associated with a specific event type to all subscribers of that event type.
        :param event_type: The type of event to be published.
        :param message: The message associated with the event.
        :return: None
        """
        if event_type in self._subscribers:
            for subscriber in self._subscribers[event_type]:
                if inspect.iscoroutinefunction(subscriber):
                    await subscriber(message)
                else:
                    subscriber(message)
