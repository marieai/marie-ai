import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from docarray import DocList

from marie.api.docs import StorageDoc
from marie.excepts import BadConfigSource
from marie.executor.mixin import StorageMixin
from marie.logging_core.logger import MarieLogger
from marie.messaging.events import EventMessage
from marie.messaging.toast_handler import ToastHandler


class PsqlToastHandler(ToastHandler, StorageMixin):
    """
    PSQL Toast Handler that writes events using JSONB format to a postgres database
    utilizing the marie Document Storage API

    """

    def __init__(self, config: Any, **kwargs: Any):
        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info("Initializing PSQL Toast Handler")

        if not config:
            self.storage_enabled = False
            self.logger.warning("Storage config not set - storage disabled")
            return
        self.storage_enabled = config.get("enabled", False)
        self.setup_storage(self.storage_enabled, config)

        # our db driver is sync so we we will execute in another thread until we update
        self._db_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="db-executor-toast"
        )

        # --- ordered, backpressure-aware async pipeline ---
        q_cfg = kwargs.get("queue") or (config or {}).get("queue") or {}
        self._queue_maxsize: int = int(q_cfg.get("maxsize", 4096))
        self._drop_if_full: bool = bool(q_cfg.get("drop_if_full", False))
        self._enqueue_timeout_s: float = float(
            q_cfg.get("enqueue_timeout_s", 0.0)
        )  # 0 => wait forever
        self._queue: asyncio.Queue[EventMessage] = asyncio.Queue(
            maxsize=self._queue_maxsize
        )

        r_cfg = kwargs.get("retry") or (config or {}).get("retry") or {}
        self._backoff_base_s: float = float(r_cfg.get("backoff_base_s", 0.1))
        self._backoff_max_s: float = float(r_cfg.get("backoff_max_s", 2.0))
        self._max_attempts: int = int(r_cfg.get("max_attempts", 0))  # 0 => infinite

        self._shutdown_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

    def get_supported_events(self) -> List[str]:
        return ["*"]

    async def __notify_task(
        self,
        notification: EventMessage,
        silence_exceptions: bool = False,
        **kwargs: Any,
    ) -> None:
        try:
            if not self.storage_enabled:
                return
            await self.persist(
                ref_id=notification.jobid,
                ref_type=notification.event if notification.event else "NA",
                results=notification,
            )
        except Exception as e:
            if silence_exceptions:
                self.logger.warning(
                    "Toast enabled but config not setup correctly", exc_info=1
                )
            else:
                raise BadConfigSource(
                    "Toast enabled but config not setup correctly"
                ) from e

    async def _worker(self) -> None:
        """
        Single consumer that preserves FIFO ordering:
        - Dequeues one item at a time
        - Retries the same item with exponential backoff (and optional max attempts)
        - Only after success (or dropping after max attempts) moves to next
        """
        self.logger.info("PsqlToastHandler worker started.")
        backoff = self._backoff_base_s
        while not self._shutdown_event.is_set():
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"PsqlToastHandler worker queue error: {e}")
                await asyncio.sleep(0.1)
                continue

            attempts = 0
            while not self._shutdown_event.is_set():
                try:
                    await self.__notify_task(item, silence_exceptions=False)
                    backoff = self._backoff_base_s
                    break
                except Exception as e:
                    attempts += 1
                    self.logger.warning(
                        f"PsqlToastHandler write failed (attempt {attempts}); retry in {backoff:.2f}s: {e}"
                    )
                    if self._max_attempts and attempts >= self._max_attempts:
                        # Optional: log to a dead-letter sink or metrics
                        self.logger.error(
                            f"Dropping event after {attempts} attempts: {getattr(item, 'event', 'unknown')}"
                        )
                        break
                    await asyncio.sleep(backoff)
                    backoff = min(
                        self._backoff_max_s, backoff * 2 or self._backoff_base_s
                    )

            self._queue.task_done()

        self.logger.info("PsqlToastHandler worker stopped.")

    async def notify(self, notification: EventMessage, **kwargs: Any) -> bool:
        if not self.storage_enabled:
            return False

        # Ensure worker is running
        if self._worker_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._worker_task = loop.create_task(self._worker())
            except RuntimeError:
                self.logger.warning("notify() called without a running loop.")
                raise

        try:
            if self._drop_if_full and self._queue.full():
                self.logger.warning("PsqlToastHandler queue full; dropping message.")
                return False

            if self._enqueue_timeout_s and self._enqueue_timeout_s > 0:
                await asyncio.wait_for(
                    self._queue.put(notification), timeout=self._enqueue_timeout_s
                )
            else:
                await self._queue.put(notification)
            return True
        except asyncio.TimeoutError:
            self.logger.warning("PsqlToastHandler enqueue timed out; message dropped.")
            return False
        except Exception as e:
            self.logger.error(f"PsqlToastHandler enqueue failed: {e}", exc_info=True)
            return False

    async def persist(self, ref_id: str, ref_type: str, results: Any) -> None:
        """
        Persist results to storage backend
        :param ref_id:
        :param ref_type:
        :param results:
        :return:
        """
        if self.storage_enabled:
            docs = DocList[StorageDoc](
                [
                    StorageDoc(
                        content=results,
                        tags={
                            "action": "job",
                            "ttl": 48 * 60,
                        },
                    )
                ]
            )

            store_func = functools.partial(
                self.store,
                ref_id=ref_id,
                ref_type=ref_type,
                store_mode="content",
                docs=docs,
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._db_executor, store_func)

    async def close(self, drain: bool = True, timeout: float = 5.0) -> None:
        """
        Graceful shutdown:
        - Optionally drain the queue
        - Stop the worker
        """
        if drain:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning("PsqlToastHandler: timeout while draining queue.")

        self._shutdown_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
