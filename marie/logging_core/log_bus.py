import atexit
import logging
import queue
import sys
import threading
import time
from logging.handlers import QueueHandler
from typing import Callable, Iterable, List, Optional


class NonBlockingQueueHandler(QueueHandler):
    """Never blocks producers; drops when bounded queue is full. Optional hook on drop."""

    def __init__(
        self,
        q: queue.Queue,
        on_drop: Optional[Callable[[logging.LogRecord], None]] = None,
        drop_counter: Optional[List[int]] = None,
    ):
        super().__init__(q)
        self._on_drop = on_drop
        self._drop_counter = drop_counter

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            sys.stderr.write("[logbus] error during enqueue queue.Full\n")
            if self._drop_counter is not None:
                self._drop_counter[0] += 1
            if self._on_drop:
                try:
                    self._on_drop(record)
                except Exception:
                    pass  # never let metrics/logging here recurse


class BatchingQueueListener:
    """Drain in batches to amortize formatting/I/O."""

    def __init__(
        self,
        q: queue.Queue,
        handlers: Iterable[logging.Handler],
        batch_size: int = 512,
        flush_interval: float = 0.05,
    ):
        self.q = q
        self.handlers = list(handlers)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def start(self):
        if self._thr:
            return
        self._thr = threading.Thread(target=self._run, name="log-listener", daemon=True)
        self._thr.start()

    def stop(self):
        # Ask thread to stop, then try to flush any remaining records safely
        self._stop.set()
        if self._thr and self._thr.is_alive():
            # Best-effort wakeup
            try:
                self.q.put_nowait(None)  # type: ignore[arg-type]
            except Exception:
                pass
            self._thr.join(timeout=1.0)
        self._thr = None

        # Final drain (in caller thread), protected
        try:
            recs: List[logging.LogRecord] = []
            while True:
                try:
                    rec = self.q.get_nowait()
                    if rec is None:
                        continue
                    recs.append(rec)
                except queue.Empty:
                    break
            if recs:
                self._flush_safe(recs)
        except Exception:
            # Never raise from stop(); just report to stderr
            try:
                sys.stderr.write("[logbus] error during final flush in stop()\n")
            except Exception:
                pass

    def _run(self):
        recs: List[logging.LogRecord] = []
        last_flush = time.monotonic()
        while not self._stop.is_set():
            try:
                try:
                    rec = self.q.get(timeout=self.flush_interval)
                    if rec is not None:
                        recs.append(rec)
                        # drain up to batch_size
                        for _ in range(self.batch_size - 1):
                            nxt = self.q.get_nowait()
                            if nxt is None:
                                continue
                            recs.append(nxt)
                except queue.Empty:
                    pass

                # flush on batch full or timeout
                now = time.monotonic()
                if recs and (
                    len(recs) >= self.batch_size
                    or (now - last_flush) >= self.flush_interval
                ):
                    self._flush_safe(recs)
                    recs.clear()
                    last_flush = now

            except Exception as e:
                # Swallow errors so the listener thread survives; report once per incident
                try:
                    sys.stderr.write(f"[logbus] listener error: {e}\n")
                except Exception:
                    pass
                # continue loop

    def _flush_safe(self, recs: List[logging.LogRecord]) -> None:
        """
        Emit each record to each handler; never let a single sink crash the listener
        """ ""
        for r in recs:
            for h in self.handlers:
                if r.levelno >= h.level:
                    try:
                        h.handle(r)
                    except Exception as e:
                        try:
                            sys.stderr.write(
                                f"[logbus] sink {h.__class__.__name__} failed: {e}\n"
                            )
                        except Exception:
                            pass


class _GlobalLogBus:
    def __init__(self, maxsize: int = 10000):
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue(maxsize)
        self._listener = BatchingQueueListener(
            self._queue, [], batch_size=512, flush_interval=0.03
        )
        self._listener_started = False
        self._drops = [0]

    def _ensure_started(self):
        if not self._listener_started:
            self._listener.start()
            atexit.register(self.stop)
            self._listener_started = True

    def attach_logger(self, logger: logging.Logger) -> None:
        with self._lock:
            self._ensure_started()
            for h in logger.handlers:
                if isinstance(h, NonBlockingQueueHandler) and getattr(
                    h, "_marie_global_bus", False
                ):
                    return
            qh = NonBlockingQueueHandler(self._queue, drop_counter=self._drops)
            qh._marie_global_bus = True
            logger.addHandler(qh)

    def set_sinks(self, sinks: Iterable[logging.Handler]) -> None:
        with self._lock:
            self._listener.handlers = list(sinks)
            # No error if empty; per-handler try/except is in _flush_safe

    def stop(self):
        with self._lock:
            self._listener.stop()

    @property
    def dropped_count(self) -> int:
        return self._drops[0]


GLOBAL_LOG_BUS = _GlobalLogBus(maxsize=10000)
