"""
Event Queue - In-memory batching queue for LLM tracking events.

Provides thread-safe batching with size and time-based flushing.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar

from marie.llm_tracking.config import get_settings
from marie.llm_tracking.types import Observation, QueueMessage, RawEvent, Score, Trace

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class BatchItem(Generic[T]):
    """Container for a queued item with timestamp."""

    item: T
    timestamp: float


class EventQueue(Generic[T]):
    """
    Thread-safe in-memory queue with batching support.

    Features:
    - Size-based flushing (when batch_size items accumulated)
    - Time-based flushing (when flush_interval_seconds elapsed)
    - Thread-safe operations
    - Graceful shutdown with final flush
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        flush_interval_seconds: Optional[float] = None,
        on_flush: Optional[Callable[[List[T]], None]] = None,
    ):
        """
        Initialize the event queue.

        Args:
            batch_size: Number of items to batch before flushing
            flush_interval_seconds: Maximum time between flushes
            on_flush: Callback function when batch is flushed
        """
        settings = get_settings()
        self._batch_size = batch_size or settings.BATCH_SIZE
        self._flush_interval = flush_interval_seconds or settings.FLUSH_INTERVAL_SECONDS
        self._on_flush = on_flush

        self._queue: List[BatchItem[T]] = []
        self._lock = threading.Lock()
        self._last_flush_time = time.time()

        self._shutdown = False
        self._flush_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background flush thread."""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return

        self._shutdown = False
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="llm-tracking-flush",
        )
        self._flush_thread.start()
        logger.debug("Event queue flush thread started")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the flush thread and flush remaining items.

        Args:
            timeout: Maximum time to wait for thread to stop
        """
        self._shutdown = True

        if self._flush_thread is not None and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=timeout)

        # Final flush
        self.flush()
        logger.debug("Event queue stopped")

    def add(self, item: T) -> None:
        """
        Add an item to the queue.

        Triggers immediate flush if batch size is reached.

        Args:
            item: Item to add to the queue
        """
        with self._lock:
            self._queue.append(BatchItem(item=item, timestamp=time.time()))

            if len(self._queue) >= self._batch_size:
                self._flush_internal()

    def add_batch(self, items: List[T]) -> None:
        """
        Add multiple items to the queue.

        Args:
            items: Items to add
        """
        with self._lock:
            now = time.time()
            for item in items:
                self._queue.append(BatchItem(item=item, timestamp=now))

            while len(self._queue) >= self._batch_size:
                self._flush_internal()

    def flush(self) -> List[T]:
        """
        Manually flush the queue.

        Returns:
            List of flushed items
        """
        with self._lock:
            return self._flush_internal()

    def _flush_internal(self) -> List[T]:
        """Internal flush (must be called with lock held)."""
        if not self._queue:
            return []

        # Take up to batch_size items
        batch_items = self._queue[: self._batch_size]
        self._queue = self._queue[self._batch_size :]
        self._last_flush_time = time.time()

        items = [bi.item for bi in batch_items]

        if items and self._on_flush:
            try:
                self._on_flush(items)
            except Exception as e:
                logger.error(f"Error in flush callback: {e}")
                # Put items back at the front of the queue
                self._queue = batch_items + self._queue

        return items

    def _flush_loop(self) -> None:
        """Background thread that periodically checks for time-based flush."""
        while not self._shutdown:
            time.sleep(1.0)  # Check every second

            with self._lock:
                if self._queue:
                    elapsed = time.time() - self._last_flush_time
                    if elapsed >= self._flush_interval:
                        self._flush_internal()

    @property
    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0


class AsyncEventQueue(Generic[T]):
    """
    Async version of EventQueue for use in async contexts.

    Provides the same batching functionality but with async callbacks.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        flush_interval_seconds: Optional[float] = None,
        on_flush: Optional[Callable[[List[T]], None]] = None,
    ):
        """
        Initialize the async event queue.

        Args:
            batch_size: Number of items to batch before flushing
            flush_interval_seconds: Maximum time between flushes
            on_flush: Async callback function when batch is flushed
        """
        settings = get_settings()
        self._batch_size = batch_size or settings.BATCH_SIZE
        self._flush_interval = flush_interval_seconds or settings.FLUSH_INTERVAL_SECONDS
        self._on_flush = on_flush

        self._queue: List[BatchItem[T]] = []
        self._lock = asyncio.Lock()
        self._last_flush_time = time.time()

        self._shutdown = False
        self._flush_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background flush task."""
        if self._flush_task is not None and not self._flush_task.done():
            return

        self._shutdown = False
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.debug("Async event queue flush task started")

    async def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the flush task and flush remaining items.

        Args:
            timeout: Maximum time to wait for task to stop
        """
        self._shutdown = True

        if self._flush_task is not None and not self._flush_task.done():
            try:
                await asyncio.wait_for(self._flush_task, timeout=timeout)
            except asyncio.TimeoutError:
                self._flush_task.cancel()

        # Final flush
        await self.flush()
        logger.debug("Async event queue stopped")

    async def add(self, item: T) -> None:
        """
        Add an item to the queue.

        Args:
            item: Item to add
        """
        async with self._lock:
            self._queue.append(BatchItem(item=item, timestamp=time.time()))

            if len(self._queue) >= self._batch_size:
                await self._flush_internal()

    async def add_batch(self, items: List[T]) -> None:
        """
        Add multiple items to the queue.

        Args:
            items: Items to add
        """
        async with self._lock:
            now = time.time()
            for item in items:
                self._queue.append(BatchItem(item=item, timestamp=now))

            while len(self._queue) >= self._batch_size:
                await self._flush_internal()

    async def flush(self) -> List[T]:
        """
        Manually flush the queue.

        Returns:
            List of flushed items
        """
        async with self._lock:
            return await self._flush_internal()

    async def _flush_internal(self) -> List[T]:
        """Internal flush (must be called with lock held)."""
        if not self._queue:
            return []

        # Take up to batch_size items
        batch_items = self._queue[: self._batch_size]
        self._queue = self._queue[self._batch_size :]
        self._last_flush_time = time.time()

        items = [bi.item for bi in batch_items]

        if items and self._on_flush:
            try:
                result = self._on_flush(items)
                # Handle both sync and async callbacks
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in async flush callback: {e}")
                # Put items back at the front of the queue
                self._queue = batch_items + self._queue

        return items

    async def _flush_loop(self) -> None:
        """Background task that periodically checks for time-based flush."""
        while not self._shutdown:
            await asyncio.sleep(1.0)  # Check every second

            async with self._lock:
                if self._queue:
                    elapsed = time.time() - self._last_flush_time
                    if elapsed >= self._flush_interval:
                        await self._flush_internal()

    @property
    async def size(self) -> int:
        """Get current queue size."""
        async with self._lock:
            return len(self._queue)

    @property
    async def is_empty(self) -> bool:
        """Check if queue is empty."""
        async with self._lock:
            return len(self._queue) == 0
