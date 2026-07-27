import asyncio
import inspect
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, TypeVar, Union

from marie.logging_core.predefined import default_logger as logger
from marie.utils.scheduler_trace import scheduler_trace

T = TypeVar("T")


class EventPublisher:
    """
    EventPublisher with bounded, keyed async queues.

    Events for one job are handled in order by the same worker. Unrelated jobs
    can be delivered concurrently by different workers.

    Notes:
        - Sync subscribers run on a dedicated, bounded thread pool (not the default loop executor).
        - Queue capacity is divided across workers to provide bounded backpressure.
        - Start the workers via constructor (creates tasks) and stop via stop().
    """

    def __init__(
        self,
        *,
        max_queue_size: int = 1024,
        subscriber_timeout_s: float = 5.0,
        max_thread_workers: int = 4,
        warn_qsize_threshold: int = 256,
        publish_blocking: bool = False,
        worker_count: int = 8,
    ):
        """
        :param max_queue_size: Bounded size for event queue.
        :param subscriber_timeout_s: Per-subscriber timeout when delivering an event.
        :param max_thread_workers: Bounded pool size for sync subscribers.
        :param warn_qsize_threshold: Emit a warning when queue size reaches this value.
        :param publish_blocking: If True, publish() will await a queue slot; otherwise it drops when full.
        :param worker_count: Number of keyed publisher workers.
        """
        if worker_count <= 0:
            raise ValueError("worker_count must be greater than zero")
        if max_queue_size < worker_count:
            raise ValueError("max_queue_size must be at least worker_count")

        self._subscribers: Dict[str, List[Callable[[str, T], None]]] = {}
        base_size, extra_slots = divmod(max_queue_size, worker_count)
        self._queues = tuple(
            asyncio.Queue[tuple[str, T, float]](
                maxsize=base_size + (1 if worker_id < extra_slots else 0)
            )
            for worker_id in range(worker_count)
        )
        self.worker_count = worker_count
        self.queue_capacity = max_queue_size
        self._worker_tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._accepting = True
        self._active_publishes = 0
        self._publishes_done = asyncio.Event()
        self._publishes_done.set()
        self._stop_lock = asyncio.Lock()
        self._dequeue_times: deque[float] = deque()

        self._subscriber_timeout_s = max(0.0, float(subscriber_timeout_s))
        self._warn_qsize_threshold = max(0, int(warn_qsize_threshold))
        self._publish_blocking = bool(publish_blocking)

        # Dedicated bounded pool for sync subscribers
        self._executor = ThreadPoolExecutor(
            max_workers=max_thread_workers, thread_name_prefix="EventPub"
        )

        self.start()

    @property
    def queue_size(self) -> int:
        return sum(queue.qsize() for queue in self._queues)

    @property
    def queue_sizes(self) -> tuple[int, ...]:
        return tuple(queue.qsize() for queue in self._queues)

    @property
    def publish_blocking(self) -> bool:
        return self._publish_blocking

    @property
    def subscriber_timeout_s(self) -> float:
        return self._subscriber_timeout_s

    async def join(self) -> None:
        await asyncio.gather(*(queue.join() for queue in self._queues))

    def subscribe(
        self, event_type: Union[str, List[str]], subscriber: Callable[[str, T], None]
    ) -> None:
        if isinstance(event_type, str):
            event_type = [event_type]
        for et in event_type:
            self._subscribers.setdefault(et, []).append(subscriber)

    def unsubscribe(
        self, event_type: str, subscriber: Callable[[str, T], None]
    ) -> None:
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(subscriber)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    async def publish(
        self, event_type: str, message: T, timeout_s: Optional[float] = None
    ) -> None:
        """
        Enqueue a message for dispatching.

        :param event_type: Event type.
        :param message: Payload.
        :param timeout_s: Optional max time to wait for a queue slot when publish_blocking=True.
        """
        if not self._accepting:
            raise RuntimeError("EventPublisher is stopped")

        self._active_publishes += 1
        self._publishes_done.clear()
        try:
            publish_started = time.perf_counter()
            enqueued_at = time.perf_counter()
            job_id = message.get("job_id") if isinstance(message, dict) else None
            trace_fields = {
                "job_id": job_id,
                "status": str(event_type),
                "subscriber_count": len(self._subscribers.get(event_type, [])),
            }
            trace_job_event = isinstance(job_id, str) and bool(job_id)
            worker_id = self._worker_for(job_id, event_type)
            queue = self._queues[worker_id]
            queued_message = dict(message) if isinstance(message, dict) else message

            if self._publish_blocking:
                if timeout_s is None:
                    await queue.put((event_type, queued_message, enqueued_at))
                else:
                    try:
                        await asyncio.wait_for(
                            queue.put((event_type, queued_message, enqueued_at)),
                            timeout=timeout_s,
                        )
                    except asyncio.TimeoutError:
                        if trace_job_event:
                            scheduler_trace(
                                "job_status_event_dropped",
                                **trace_fields,
                                reason="publish_timeout",
                                worker_id=worker_id,
                                worker_queue_size=queue.qsize(),
                                queue_size=self.queue_size,
                                queue_capacity=self.queue_capacity,
                                elapsed_ms=(time.perf_counter() - publish_started)
                                * 1000.0,
                            )
                        logger.error(
                            f"EventPublisher: publish timeout for event '{event_type}'"
                        )
                        return
            else:
                try:
                    queue.put_nowait((event_type, queued_message, enqueued_at))
                except asyncio.QueueFull:
                    if trace_job_event:
                        scheduler_trace(
                            "job_status_event_dropped",
                            **trace_fields,
                            reason="queue_full",
                            worker_id=worker_id,
                            worker_queue_size=queue.qsize(),
                            queue_size=self.queue_size,
                            queue_capacity=self.queue_capacity,
                            elapsed_ms=(time.perf_counter() - publish_started) * 1000.0,
                        )
                    return

            if trace_job_event:
                scheduler_trace(
                    "job_status_event_enqueued",
                    **trace_fields,
                    worker_id=worker_id,
                    worker_queue_size=queue.qsize(),
                    queue_size=self.queue_size,
                    queue_capacity=self.queue_capacity,
                    elapsed_ms=(time.perf_counter() - publish_started) * 1000.0,
                )

            queue_size = self.queue_size
            if self._warn_qsize_threshold and queue_size >= self._warn_qsize_threshold:
                logger.warning(
                    f"EventPublisher queue high-water mark: size={queue_size}"
                )
        finally:
            self._active_publishes -= 1
            if self._active_publishes == 0:
                self._publishes_done.set()

    async def _worker(self, worker_id: int) -> None:
        loop = asyncio.get_running_loop()
        queue = self._queues[worker_id]

        while not self._stopped.is_set():
            queued_event: tuple[str, T, float] | None = None
            try:
                queued_event = await queue.get()
                event_type, message, enqueued_at = queued_event
                dispatch_started = time.perf_counter()
                subscriber_count = len(self._subscribers.get(event_type, []))
                job_id = message.get("job_id") if isinstance(message, dict) else None
                trace_fields = {
                    "job_id": job_id,
                    "status": str(event_type),
                    "subscriber_count": subscriber_count,
                }
                trace_job_event = isinstance(job_id, str) and bool(job_id)
                if trace_job_event:
                    scheduler_trace(
                        "job_status_event_dequeued",
                        **trace_fields,
                        worker_id=worker_id,
                        worker_queue_size=queue.qsize(),
                        queue_size=self.queue_size,
                        queue_capacity=self.queue_capacity,
                        queue_wait_ms=(dispatch_started - enqueued_at) * 1000.0,
                        dequeue_rate_per_second=self._record_dequeue(dispatch_started),
                    )
                timeout_count = 0
                error_count = 0
                delivery_started = time.perf_counter()
                if event_type in self._subscribers:
                    subscriber_tasks = []
                    for subscriber in list(self._subscribers[event_type]):
                        try:
                            if inspect.iscoroutinefunction(subscriber):
                                delivery = subscriber(event_type, message)
                            else:
                                delivery = loop.run_in_executor(
                                    self._executor, subscriber, event_type, message
                                )
                            if self._subscriber_timeout_s > 0:
                                delivery = asyncio.wait_for(
                                    delivery, timeout=self._subscriber_timeout_s
                                )
                            subscriber_tasks.append(asyncio.ensure_future(delivery))
                        except Exception as e:
                            logger.error(f"Subscriber creation error: {e}")

                    if subscriber_tasks:
                        results = await asyncio.gather(
                            *subscriber_tasks, return_exceptions=True
                        )
                        # Log timeouts/errors but continue to preserve forward progress
                        for r in results:
                            if isinstance(r, asyncio.TimeoutError):
                                timeout_count += 1
                                logger.warning(
                                    "EventPublisher: subscriber timed out for "
                                    f"event '{event_type}', job_id={job_id}"
                                )
                            elif isinstance(r, Exception):
                                error_count += 1
                                logger.error(
                                    "EventPublisher: subscriber error for "
                                    f"event '{event_type}', job_id={job_id}: {r}"
                                )

                subscriber_delivery_ms = (
                    time.perf_counter() - delivery_started
                ) * 1000.0
                if trace_job_event:
                    scheduler_trace(
                        "job_status_event_dispatch_completed",
                        **trace_fields,
                        worker_id=worker_id,
                        worker_queue_size=queue.qsize(),
                        queue_size=self.queue_size,
                        queue_capacity=self.queue_capacity,
                        timeout_count=timeout_count,
                        error_count=error_count,
                        subscriber_delivery_ms=subscriber_delivery_ms,
                        elapsed_ms=(time.perf_counter() - dispatch_started) * 1000.0,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"EventPublisher dispatcher error: {e}")
            finally:
                if queued_event is not None:
                    queue.task_done()

    def _worker_for(self, job_id: object, event_type: str) -> int:
        routing_key = job_id if isinstance(job_id, str) and job_id else str(event_type)
        return zlib.crc32(routing_key.encode("utf-8")) % self.worker_count

    def _record_dequeue(self, now: float) -> float:
        self._dequeue_times.append(now)
        cutoff = now - 1.0
        while self._dequeue_times and self._dequeue_times[0] < cutoff:
            self._dequeue_times.popleft()
        return float(len(self._dequeue_times))

    def start(self) -> None:
        if self._worker_tasks:
            return
        if self._stopped.is_set():
            raise RuntimeError("EventPublisher cannot restart after stop")
        self._worker_tasks = [
            asyncio.create_task(
                self._worker(worker_id), name=f"event-publisher-{worker_id}"
            )
            for worker_id in range(self.worker_count)
        ]

    async def stop(self) -> None:
        """Stop accepting events, drain queued work, and stop publisher workers."""
        async with self._stop_lock:
            if self._stopped.is_set():
                return

            self._accepting = False
            await self._publishes_done.wait()
            await self.join()
            self._stopped.set()
            for task in self._worker_tasks:
                task.cancel()
            if self._worker_tasks:
                await asyncio.gather(*self._worker_tasks, return_exceptions=True)
                self._worker_tasks.clear()

            self._executor.shutdown(wait=False)
