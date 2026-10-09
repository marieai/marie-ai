import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from marie.logging_core.logger import MarieLogger


class JobCallbackExecutor:
    def __init__(
        self,
        max_queue_size: Optional[int] = None,
        callback_timeout: float = 5.0,
        max_workers: int = 8,
        warn_qsize_threshold: int = 256,
    ) -> None:
        self.logger = MarieLogger(self.__class__.__name__)
        # Bounded queue for backpressure (default to 1024 if None/0)
        qsize = max_queue_size if (max_queue_size and max_queue_size > 0) else 1024
        self._queue = queue.Queue(maxsize=qsize)

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="CallbackPool"
        )

        self._shutdown_event = threading.Event()
        self._submission_lock = threading.Lock()
        self._active_submitters = 0
        self._callback_timeout = callback_timeout
        self._warn_qsize_threshold = warn_qsize_threshold

        self._thread = threading.Thread(
            target=self._run, name="JobCallbackExecutorThread", daemon=True
        )
        self._thread.start()

    def submit(
        self, fn: Callable, *args, block: bool = False, timeout: Optional[float] = None
    ) -> None:
        with self._submission_lock:
            if self._shutdown_event.is_set():
                self.logger.warning(
                    "Callback executor has been shut down. Ignoring task."
                )
                return
            self._active_submitters += 1
        try:
            if block:
                self._queue.put((fn, args), timeout=timeout)
            else:
                self._queue.put_nowait((fn, args))
            qsz = self._queue.qsize()
            if qsz >= self._warn_qsize_threshold:
                self.logger.warning("Callback queue high-water mark: size=%d", qsz)
        except queue.Full:
            # Drop current task, log; alternative: drop oldest
            self.logger.error(
                "Callback executor queue is full (cap=%d). Task dropped.",
                self._queue.maxsize,
            )
        finally:
            with self._submission_lock:
                self._active_submitters -= 1

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Drain accepted callbacks; timeout bounds the dispatcher join only."""
        with self._submission_lock:
            self._shutdown_event.set()

        if wait:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.logger.warning(
                    "Callback dispatcher did not terminate within timeout."
                )
                return
            self._executor.shutdown(wait=True)

    def _run(self) -> None:
        self.logger.info("JobCallbackExecutor dispatcher started.")
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                with self._submission_lock:
                    # An admitted submitter may still be about to enqueue its callback.
                    if (
                        self._shutdown_event.is_set()
                        and self._active_submitters == 0
                        and self._queue.empty()
                    ):
                        break
                continue

            fn, args = item
            start_time = time.monotonic()
            try:
                future = self._executor.submit(fn, *args)

                # Log completion asynchronously; do not block dispatcher
                def _on_done(fut):
                    elapsed = time.monotonic() - start_time
                    exc = fut.exception()
                    if exc is None:
                        self.logger.info(
                            "Callback %s executed in %.2fs.", fn.__name__, elapsed
                        )
                    else:
                        self.logger.error(
                            "Exception in callback %s after %.2fs: %s",
                            fn.__name__,
                            elapsed,
                            exc,
                        )

                future.add_done_callback(_on_done)

            except Exception as e:
                self.logger.error("Failed to submit callback %s: %s", fn.__name__, e)
            finally:
                self._queue.task_done()

        self._executor.shutdown(wait=False)
        self.logger.info("JobCallbackExecutor dispatcher terminated.")


job_callback_executor = JobCallbackExecutor(
    max_queue_size=1024, callback_timeout=5.0, max_workers=4
)
