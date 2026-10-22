from __future__ import annotations

import asyncio
from types import TracebackType
from weakref import WeakValueDictionary


class _JobLockContext:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    def locked(self) -> bool:
        return self._lock.locked()

    async def __aenter__(self) -> _JobLockContext:
        await self._lock.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._lock.release()


class AsyncJobLock:
    """Own per-job locks through ``async with locks[job_id]`` contexts."""

    def __init__(self) -> None:
        self._locks: WeakValueDictionary[str, _JobLockContext] = WeakValueDictionary()

    def __getitem__(self, job_id: str) -> _JobLockContext:
        lock = self._locks.get(job_id)
        if lock is None:
            lock = _JobLockContext()
            self._locks[job_id] = lock
        return lock
