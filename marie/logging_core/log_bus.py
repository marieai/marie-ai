import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from logging.handlers import QueueHandler
from typing import Callable, Iterable, List, Optional


def _stderr(msg: str) -> None:
    try:
        sys.stderr.write(msg)
    except Exception:
        pass


class NonBlockingQueueHandler(QueueHandler):
    """
    Producer-side handler that never blocks.

    Behavior
    --------
    - Uses a bounded queue; if full, the record is dropped.
    - During shutdown (when `stop_event` is set), records are dropped immediately.
    - Optionally invokes `on_drop(record)` and increments `drop_counter[0]`.

    Thread-safety
    -------------
    - `enqueue()` is called from arbitrary producer threads.
    - Uses lock-free queue operations and avoids logging recursively on failure.
    """

    _WARN_INTERVAL = 5.0  # seconds between stderr warnings

    def __init__(
        self,
        q: queue.Queue,
        on_drop: Optional[Callable[[logging.LogRecord], None]] = None,
        drop_counter: Optional[List[int]] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        super().__init__(q)
        self._on_drop = on_drop
        self._drop_counter = drop_counter
        self._stop_event = stop_event
        self._last_warn: float = 0.0
        self._drops_since_warn: int = 0

    def enqueue(self, record: logging.LogRecord) -> None:
        # During shutdown: drop immediately (avoid re-enqueue loops)
        if self._stop_event is not None and self._stop_event.is_set():
            if self._drop_counter is not None:
                self._drop_counter[0] += 1
            return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            if self._drop_counter is not None:
                self._drop_counter[0] += 1
            self._drops_since_warn += 1
            now = time.monotonic()
            if now - self._last_warn >= self._WARN_INTERVAL:
                _stderr(
                    f"[logbus] enqueue failed: queue.Full"
                    f" (dropped {self._drops_since_warn} since last warning)\n"
                )
                self._last_warn = now
                self._drops_since_warn = 0
            if self._on_drop:
                try:
                    self._on_drop(record)
                except Exception:
                    pass


class BatchingQueueListener:
    """
    Single consumer that drains a bounded queue in batches to amortize I/O.

    Behavior
    --------
    - Dequeues records, batches up to `batch_size` or `flush_interval`, then emits to sinks.
    - Keeps a *shared* in-memory buffer for items already dequeued; on `stop()`/`flush()` we
      drain both the queue and the shared buffer to avoid tail-loss.
    - Worker thread is daemonized to avoid blocking interpreter exit if sinks misbehave.

    Public API
    ----------
    - start() : start the worker thread (idempotent)
    - flush() : drain queue + in-memory buffer and flush sinks (non-stopping)
    - stop()  : idempotent; best-effort join + final drain + flush sinks

    Thread-safety
    -------------
    - Producer threads: safe (they only use the Queue).
    - Consumer thread: single-threaded; uses `_buf_lock` to guard the shared buffer.
    - `flush()` and `stop()` run in caller thread; they are safe to call concurrently
      with the worker due to the buffer lock and bounded queue semantics.
    """

    _SENTINEL = object()

    def __init__(
        self,
        q: queue.Queue,
        handlers: Iterable[logging.Handler],
        batch_size: int = 512,
        flush_interval: float = 0.05,
    ):
        self.q = q
        self.handlers: List[logging.Handler] = list(handlers)
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._stop_evt = threading.Event()
        self._thr: Optional[threading.Thread] = None

        # Shared buffer (records already dequeued from the queue)
        # self._buf: List[logging.LogRecord] = []
        self._buf: deque[logging.LogRecord] = deque()
        self._buf_lock = threading.Lock()

    # -------------------- lifecycle --------------------

    def start(self) -> None:
        """Start the worker thread (idempotent)."""
        if self._thr:
            return
        self._thr = threading.Thread(target=self._run, name="log-listener", daemon=True)
        self._thr.start()

    def flush(self) -> None:
        """
        Drain queue + shared buffer and flush sinks (non-stopping).

        Use this in long-running services (e.g., before rotating files or at checkpoints)
        or in tests to ensure all pending records are emitted.
        """
        sinks = list(self.handlers)

        # Drain queue
        q_recs: List[logging.LogRecord] = []
        while True:
            try:
                rec = self.q.get_nowait()
            except queue.Empty:
                break
            if rec is self._SENTINEL:
                continue
            q_recs.append(rec)

        # Drain shared buffer
        buf_recs = self._pop_buffer()

        # Emit & flush
        all_recs = buf_recs + q_recs
        if all_recs:
            self._emit_to_handlers(all_recs, sinks)
        self._flush_sinks(sinks)

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """
        Idempotent shutdown: signal stop, best-effort join, final drain, emit & flush.

        If called multiple times, later calls are cheap no-ops (the owning bus ensures this).
        """
        self._stop_evt.set()
        try:
            self.q.put_nowait(self._SENTINEL)
        except Exception:
            pass

        worker = self._thr
        try:
            if worker and worker.is_alive():
                worker.join(timeout=timeout)
        except Exception:
            pass

        if worker and worker.is_alive():
            _stderr("[logbus] listener stop timed out; leaving worker alive\n")
            return

        self._thr = None

        self.flush()

    # -------------------- worker --------------------

    def _run(self) -> None:
        last_flush = time.monotonic()
        while not self._stop_evt.is_set():
            try:
                try:
                    rec = self.q.get(timeout=self.flush_interval)
                    if rec is not self._SENTINEL and rec is not None:
                        self._push_batch_to_buffer(rec)
                except queue.Empty:
                    pass

                now = time.monotonic()
                recs = self._pop_ready_batch(now, last_flush)
                if recs:
                    self._flush_safe(recs)
                    last_flush = now

            except Exception as e:
                _stderr(f"[logbus] listener error: {e}\n")

        # Final flush when stop is set
        recs = self._pop_buffer()
        recs.extend(self._drain_queue())
        if recs:
            try:
                self._flush_safe(recs)
            except Exception:
                pass
        self._flush_sinks()

    # -------------------- helpers --------------------

    def _pop_buffer(self) -> List[logging.LogRecord]:
        with self._buf_lock:
            if not self._buf:
                return []
            batch, self._buf = list(self._buf), deque()
            return batch

    def _drain_queue(self, limit: Optional[int] = None) -> List[logging.LogRecord]:
        recs: List[logging.LogRecord] = []
        while limit is None or len(recs) < limit:
            try:
                rec = self.q.get_nowait()
            except queue.Empty:
                break
            if rec is self._SENTINEL or rec is None:
                continue
            recs.append(rec)
        return recs

    def _push_batch_to_buffer(self, first: logging.LogRecord) -> None:
        recs = [first]
        if self.batch_size > 1:
            recs.extend(self._drain_queue(limit=self.batch_size - 1))
        with self._buf_lock:
            self._buf.extend(recs)

    def _pop_ready_batch(
        self, now: float, last_flush: float
    ) -> List[logging.LogRecord]:
        with self._buf_lock:
            if not self._buf:
                return []
            if (
                len(self._buf) < self.batch_size
                and (now - last_flush) < self.flush_interval
            ):
                return []
            batch_len = min(len(self._buf), self.batch_size)
            return [self._buf.popleft() for _ in range(batch_len)]

    def _emit_to_handlers(
        self, recs: List[logging.LogRecord], sinks: Iterable[logging.Handler]
    ) -> None:
        for h in sinks:
            level = getattr(h, "level", logging.NOTSET)
            handler_recs = [r for r in recs if r.levelno >= level]
            if not handler_recs:
                continue
            try:
                handle_many = getattr(h, "handle_many", None)
                if callable(handle_many):
                    handle_many(handler_recs)
                else:
                    for r in handler_recs:
                        h.handle(r)
            except Exception as e:
                _stderr(f"[logbus] sink {h.__class__.__name__} failed: {e}\n")

    def _flush_sinks(self, sinks: Optional[Iterable[logging.Handler]] = None) -> None:
        targets = list(sinks if sinks is not None else self.handlers)
        for h in targets:
            try:
                flush = getattr(h, "flush", None)
                if callable(flush):
                    flush()
            except Exception:
                pass

    def _flush_safe(self, recs: List[logging.LogRecord]) -> None:
        self._emit_to_handlers(recs, self.handlers)


class _GlobalLogBus:
    """
    Singleton bus that connects producers to the batching listener.

    Behavior
    --------
    - `attach_logger(logger)` installs a NonBlockingQueueHandler onto the logger.
    - `set_sinks(handlers)` sets the downstream sinks (e.g., file/console handlers).
    - `flush()` drains queue + buffer and flushes sinks without stopping.
    - `stop()` is idempotent and performs a final drain + flush.

    Thread-safety
    -------------
    - Public methods acquire an internal RLock to guard one-time start/stop and sink updates.
    """

    def __init__(
        self,
        maxsize: int = 10000,
        batch_size: int = 512,
        flush_interval: float = 0.03,
    ):
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue(maxsize)
        self._listener = BatchingQueueListener(
            self._queue,
            [],  # sinks set later via set_sinks()
            batch_size=batch_size,
            flush_interval=flush_interval,
        )
        self._listener_started = False
        self._drops = [0]
        self._stopped = False

    def _ensure_started(self) -> None:
        if not self._listener_started:
            self._listener.start()
            self._listener_started = True

    def attach_logger(self, logger: logging.Logger) -> None:
        with self._lock:
            self._ensure_started()
            for h in logger.handlers:
                if isinstance(h, NonBlockingQueueHandler) and getattr(
                    h, "_marie_global_bus", False
                ):
                    return
            qh = NonBlockingQueueHandler(
                self._queue,
                drop_counter=self._drops,
                stop_event=self._listener._stop_evt,
            )
            qh._marie_global_bus = True
            logger.addHandler(qh)

    def set_sinks(self, sinks: Iterable[logging.Handler]) -> None:
        with self._lock:
            self._listener.handlers = list(sinks)

    def flush(self) -> None:
        """Drain queue + buffer and flush sinks without stopping (thread-safe)."""
        with self._lock:
            if not self._stopped:
                try:
                    self._listener.flush()
                except Exception:
                    pass

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """Idempotent final drain + flush + best-effort join (thread-safe)."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            try:
                self._listener.stop(timeout=timeout)
            except Exception:
                pass

    @property
    def dropped_count(self) -> int:
        return self._drops[0]


GLOBAL_LOG_BUS = _GlobalLogBus(
    maxsize=int(os.getenv("MARIE_LOG_QUEUE_MAXSIZE", "50000")),
    batch_size=int(os.getenv("MARIE_LOG_BATCH_SIZE", "512")),
    flush_interval=float(os.getenv("MARIE_LOG_FLUSH_INTERVAL", "0.05")),
)
