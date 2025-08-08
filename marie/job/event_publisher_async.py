import asyncio
import inspect
from typing import Callable, Dict, List, TypeVar, Union

T = TypeVar("T")


class EventPublisher:
    """
    EventPublisher with async event queue (global ordering)

    - Events are published into a FIFO queue.
    - A single dispatcher consumes the queue in strict order.
    - Each event is delivered to all subscribers before the next event is processed.
    - Guarantees global ordering across *all* events and subscribers.

    Example Usage:
        async def async_subscriber(event_type, message):
            await asyncio.sleep(1)
            print("Async subscriber:", event_type, message)

        def sync_subscriber(event_type, message):
            print("Sync subscriber:", event_type, message)

        async def main():
            publisher = EventPublisher()
            publisher.subscribe("event_type1", async_subscriber)
            publisher.subscribe("event_type1", sync_subscriber)

            publisher.start()

            await publisher.publish("event_type1", "Message 1")
            await publisher.publish("event_type1", "Message 2")

            await asyncio.sleep(3)  # allow events to process
            await publisher.stop()

        asyncio.run(main())

    Notes:
        - Global FIFO ordering is guaranteed (no parallel dispatch).
        - Subscribers may be sync or async functions.
        - Sync subscribers run in a background thread pool but are awaited in order.
        - A slow subscriber delays all subsequent events.
        - Start the dispatcher via `start()` and stop it via `stop()`.

    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[str, T], None]]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._dispatcher_task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self.start()

    def subscribe(
        self, event_type: Union[str, List[str]], subscriber: Callable[[str, T], None]
    ):
        """
        Subscribe a callable to one or more event types.

        :param event_type: A string or list of strings representing event types.
        :param subscriber: A callable (sync or async) that accepts (event_type, message).
        """
        if isinstance(event_type, str):
            event_type = [event_type]
        for et in event_type:
            self._subscribers.setdefault(et, []).append(subscriber)

    def unsubscribe(self, event_type: str, subscriber: Callable[[str, T], None]):
        """
        Unsubscribe a subscriber from a specific event type.
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(subscriber)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    async def publish(self, event_type: str, message: T):
        """
        Enqueue a message for dispatching (non-blocking).

        :param event_type: The type of event being published.
        :param message: The message payload to deliver.
        """
        await self._queue.put((event_type, message))

    async def _dispatcher(self):
        """
        Dispatcher consumes events in strict FIFO order
        and processes subscribers synchronously to guarantee ordering.
        """
        while not self._stopped.is_set():
            try:
                event_type, message = await self._queue.get()
                if event_type in self._subscribers:
                    for subscriber in list(self._subscribers[event_type]):
                        try:
                            if inspect.iscoroutinefunction(subscriber):
                                await subscriber(event_type, message)
                            else:
                                # Sync function: run in thread but wait for completion
                                await asyncio.get_running_loop().run_in_executor(
                                    None, subscriber, event_type, message
                                )
                        except Exception as e:
                            print(f"Subscriber error: {e}")
            except asyncio.CancelledError:
                break

    def start(self):
        """
        Start the dispatcher loop (must be called inside an event loop).
        """
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(self._dispatcher())

    async def stop(self):
        """
        Stop the dispatcher gracefully.
        """
        self._stopped.set()
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
            self._dispatcher_task = None
