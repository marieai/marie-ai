import asyncio
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class AsyncJobLock:
    """
    Provides a dictionary of asyncio.Lock objects for per-job synchronization.
    This implementation uses an OrderedDict to act as an LRU cache, preventing
    the collection of locks from growing indefinitely.
    """

    def __init__(self, max_size: int = 4096):
        self._locks = OrderedDict()
        self._max_size = max_size

    def __getitem__(self, job_id: str) -> asyncio.Lock:
        """
        Retrieves or creates a lock for the given job_id.
        Moves the accessed lock to the end of the OrderedDict to mark it as
        recently used.
        """
        if job_id in self._locks:
            self._locks.move_to_end(job_id)
        else:
            if len(self._locks) >= self._max_size:
                # Evict the least recently used (oldest) item.
                old_job_id, old_lock = self._locks.popitem(last=False)
                if old_lock.locked():
                    logger.warning(
                        f"Evicted a lock for job '{old_job_id}' that was still locked. "
                        f"This may indicate a lock is being held for too long."
                    )
            self._locks[job_id] = asyncio.Lock()
        return self._locks[job_id]

    def release(self, job_id: str):
        """
        Explicitly removes a lock from the cache.
        This should be called when a job is known to be in a terminal state.
        """
        if job_id in self._locks:
            lock = self._locks[job_id]
            if not lock.locked():
                del self._locks[job_id]
            else:
                logger.warning(
                    f"Attempted to release a locked lock for job '{job_id}'. "
                    f"The lock will not be removed yet."
                )
