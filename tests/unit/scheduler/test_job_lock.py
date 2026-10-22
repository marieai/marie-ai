import asyncio
import gc

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
    async with locks["job-a"] as held:
        references = [locks[f"job-{index}"] for index in range(4_097)]

        assert len(references) == 4_097
        assert locks["job-a"] is held


async def test_temporary_lock_handle_requires_context_manager() -> None:
    locks = AsyncJobLock()

    with pytest.raises(AttributeError):
        await locks['job-a'].acquire()
    with pytest.raises(AttributeError):
        locks['job-a'].release()


async def test_temporary_context_managers_serialize_same_job() -> None:
    locks = AsyncJobLock()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def first() -> None:
        async with locks['job-a']:
            order.append('first')
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        async with locks['job-a']:
            order.append('second')

    owner = asyncio.create_task(first())
    await first_entered.wait()
    waiter = asyncio.create_task(second())
    try:
        await asyncio.sleep(0)
        gc.collect()
        assert locks['job-a'].locked()
        assert not waiter.done()
        assert order == ['first']
    finally:
        release_first.set()
        await asyncio.gather(owner, waiter)

    assert order == ['first', 'second']
    gc.collect()
    assert len(locks._locks) == 0


async def test_cancelled_waiter_does_not_release_owner_lock() -> None:
    locks = AsyncJobLock()

    async def wait_for_lock() -> None:
        async with locks['job-a']:
            raise AssertionError('Waiter entered before owner released the lock')

    async with locks['job-a'] as held:
        waiter = asyncio.create_task(wait_for_lock())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert held.locked()
        assert locks['job-a'] is held

    assert not held.locked()


async def test_context_exception_releases_lock_and_allows_collection() -> None:
    locks = AsyncJobLock()

    with pytest.raises(RuntimeError, match='critical section failed'):
        async with locks['job-a']:
            raise RuntimeError('critical section failed')
    gc.collect()

    assert len(locks._locks) == 0
    async with locks['job-a']:
        assert locks['job-a'].locked()
