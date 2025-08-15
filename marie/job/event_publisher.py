# Python
import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, TypeVar, Union

from marie.logging_core.predefined import default_logger as logger

T = TypeVar("T")


class EventPublisher:
    """
    EventPublisher with async event queue (global ordering)

    - Events are published into a FIFO queue (bounded to provide backpressure).
    - A single dispatcher consumes the queue in strict order.
    - Each event is delivered to all subscribers before the next event is processed.
    - Guarantees global ordering across all events and subscribers (by default).

    Notes:
        - Sync subscribers run on a dedicated, bounded thread pool (not the default loop executor).
        - A slow subscriber can still delay, but timeouts prevent indefinite stalls.
        - Start the dispatcher via constructor (creates task) and stop via stop().
    """

    def __init__(
        self,
        *,
        max_queue_size: int = 1024,
        subscriber_timeout_s: float = 5.0,
        max_thread_workers: int = 4,
        warn_qsize_threshold: int = 256,
        publish_blocking: bool = False,
    ):
        """
        :param max_queue_size: Bounded size for event queue.
        :param subscriber_timeout_s: Per-subscriber timeout when delivering an event.
        :param max_thread_workers: Bounded pool size for sync subscribers.
        :param warn_qsize_threshold: Emit a warning when queue size reaches this value.
        :param publish_blocking: If True, publish() will await a queue slot; otherwise it drops when full.
        """
        self._subscribers: Dict[str, List[Callable[[str, T], None]]] = {}
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._dispatcher_task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

        self._subscriber_timeout_s = max(0.0, float(subscriber_timeout_s))
        self._warn_qsize_threshold = max(0, int(warn_qsize_threshold))
        self._publish_blocking = bool(publish_blocking)

        # Dedicated bounded pool for sync subscribers
        self._executor = ThreadPoolExecutor(
            max_workers=max_thread_workers, thread_name_prefix="EventPub"
        )

        self.start()

    def subscribe(
        self, event_type: Union[str, List[str]], subscriber: Callable[[str, T], None]
    ):
        if isinstance(event_type, str):
            event_type = [event_type]
        for et in event_type:
            self._subscribers.setdefault(et, []).append(subscriber)

    def unsubscribe(self, event_type: str, subscriber: Callable[[str, T], None]):
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(subscriber)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    async def publish(
        self, event_type: str, message: T, timeout_s: Optional[float] = None
    ):
        """
        Enqueue a message for dispatching.

        :param event_type: Event type.
        :param message: Payload.
        :param timeout_s: Optional max time to wait for a queue slot when publish_blocking=True.
        """
        # Backpressure handling
        if self._publish_blocking:
            if timeout_s is None:
                await self._queue.put((event_type, message))
            else:
                try:
                    await asyncio.wait_for(
                        self._queue.put((event_type, message)), timeout=timeout_s
                    )
                except asyncio.TimeoutError:
                    # Drop or raise on publish timeout
                    logger.error(
                        f"EventPublisher: publish timeout for event '{event_type}'"
                    )
                    return
        else:
            try:
                self._queue.put_nowait((event_type, message))
            except asyncio.QueueFull:
                # Drop newest when full to prevent unbounded latency growth
                return

        # Soft visibility into pressure
        try:
            qsz = self._queue.qsize()
            if self._warn_qsize_threshold and qsz >= self._warn_qsize_threshold:
                logger.warning(f"EventPublisher queue high-water mark: size={qsz}")
        except Exception:
            pass

    async def _dispatcher(self):
        """
        Dispatcher consumes events in strict FIFO order.
        It processes all subscribers for an event concurrently, but applies per-subscriber timeouts.
        """
        loop = asyncio.get_running_loop()

        while not self._stopped.is_set():
            try:
                event_type, message = await self._queue.get()
                if event_type in self._subscribers:
                    # Build tasks for all subscribers
                    subscriber_tasks = []
                    for subscriber in list(self._subscribers[event_type]):
                        try:
                            if inspect.iscoroutinefunction(subscriber):
                                coro = subscriber(event_type, message)
                                # Wrap with timeout if configured
                                if self._subscriber_timeout_s > 0:
                                    task = asyncio.create_task(
                                        asyncio.wait_for(
                                            coro, timeout=self._subscriber_timeout_s
                                        )
                                    )
                                else:
                                    task = asyncio.create_task(coro)
                            else:
                                fut = loop.run_in_executor(
                                    self._executor, subscriber, event_type, message
                                )
                                if self._subscriber_timeout_s > 0:
                                    task = asyncio.create_task(
                                        asyncio.wait_for(
                                            fut, timeout=self._subscriber_timeout_s
                                        )
                                    )
                                else:
                                    task = asyncio.create_task(fut)
                            subscriber_tasks.append(task)
                        except Exception as e:
                            logger.error(f"Subscriber creation error: {e}")

                    if subscriber_tasks:
                        results = await asyncio.gather(
                            *subscriber_tasks, return_exceptions=True
                        )
                        # Log timeouts/errors but continue to preserve forward progress
                        for r in results:
                            if isinstance(r, asyncio.TimeoutError):
                                logger.warning("EventPublisher: subscriber timed out")
                            elif isinstance(r, Exception):
                                logger.error(f"EventPublisher: subscriber error: {r}")

                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EventPublisher dispatcher error: {e}")

    def start(self):
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(self._dispatcher())

    async def stop(self):
        """
        Stop the dispatcher gracefully (best effort).
        Note: Pending events in the queue are not drained on stop.
        """
        self._stopped.set()
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
            self._dispatcher_task = None

        self._executor.shutdown(wait=False)
