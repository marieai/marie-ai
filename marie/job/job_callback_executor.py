import queue
import threading
import time
from typing import Callable, Optional

from marie.logging_core.logger import MarieLogger

SENTINEL = object()


class JobCallbackExecutor:
    def __init__(
        self, max_queue_size: Optional[int] = None, callback_timeout: float = 5.0
    ):
        self.logger = MarieLogger(self.__class__.__name__)
        self._queue = queue.Queue(maxsize=max_queue_size or 0)
        self._thread = threading.Thread(
            target=self._run, name="JobCallbackExecutorThread", daemon=True
        )
        self._shutdown_event = threading.Event()
        self._callback_timeout = callback_timeout
        self._thread.start()

    def submit(self, fn: Callable, *args):
        if self._shutdown_event.is_set():
            self.logger.warning("Callback executor has been shut down. Ignoring task.")
            return
        try:
            self._queue.put_nowait((fn, args))
        except queue.Full:
            self.logger.error("Callback executor queue is full. Task dropped.")

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        self._shutdown_event.set()
        self._queue.put(SENTINEL)
        if wait:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.logger.warning("Callback thread did not terminate within timeout.")

    def _run(self):
        self.logger.info("JobCallbackExecutor thread started.")
        while True:
            item = self._queue.get()
            if item is SENTINEL:
                self.logger.info("JobCallbackExecutor received shutdown signal.")
                self._queue.task_done()
                break

            fn, args = item
            start_time = time.monotonic()
            try:
                t = threading.Thread(target=fn, args=args, name="CallbackTask")
                t.start()
                t.join(timeout=self._callback_timeout)

                elapsed = time.monotonic() - start_time
                if t.is_alive():
                    self.logger.warning(
                        f"Callback {fn.__name__} exceeded timeout of {self._callback_timeout:.2f}s "
                        f"(ran for {elapsed:.2f}s). Skipping wait."
                    )
                else:
                    self.logger.info(
                        f"Callback {fn.__name__} executed in {elapsed:.2f}s."
                    )
            except Exception as e:
                self.logger.error(f"Exception while executing callback: {e}")
            finally:
                self._queue.task_done()

        self.logger.info("JobCallbackExecutor thread terminated.")


job_callback_executor = JobCallbackExecutor(max_queue_size=0)
