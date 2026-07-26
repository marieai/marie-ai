import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from marie.state.state_store import DesiredDoc, DesiredStore


class DesiredStateExecutor:
    def __init__(
        self,
        store: DesiredStore,
        *,
        max_workers: int = 16,
        max_pending: int = 128,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")
        if max_pending <= 0:
            raise ValueError("max_pending must be greater than zero")

        self._store = store
        self._slots = asyncio.Semaphore(max_pending)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="desired-state",
        )
        self._closed = False

    async def schedule_new_epoch(
        self,
        node: str,
        deployment: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> DesiredDoc:
        if self._closed:
            raise RuntimeError("Desired-state executor is closed")

        await self._slots.acquire()
        if self._closed:
            self._slots.release()
            raise RuntimeError("Desired-state executor is closed")

        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(
                self._executor,
                self._store.schedule_new_epoch,
                node,
                deployment,
                params,
            )
        except Exception:
            self._slots.release()
            raise

        future.add_done_callback(lambda _: self._slots.release())
        return await asyncio.shield(future)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
