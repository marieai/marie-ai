import asyncio

import pytest

from marie.scheduler.job_lock import AsyncJobLock


def test_reuses_referenced_lock() -> None:
    locks = AsyncJobLock()
    referenced = locks["job-a"]

    for index in range(5_000):
        locks[f"job-{index}"]

    assert locks["job-a"] is referenced


@pytest.mark.asyncio
async def test_reuses_held_lock_beyond_previous_capacity() -> None:
    locks = AsyncJobLock()
    held = locks["job-a"]
    await held.acquire()

    references = [locks[f"job-{index}"] for index in range(4_097)]

    assert len(references) == 4_097
    assert locks["job-a"] is held

    held.release()
