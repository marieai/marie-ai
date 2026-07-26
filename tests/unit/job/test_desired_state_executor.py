import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from marie.job.desired_state_executor import DesiredStateExecutor
from marie.state.state_store import DesiredDoc, DesiredStore


class ExistingDesiredEtcd:
    def __init__(self) -> None:
        self.get_count = 0
        self.update_count = 0
        self.value = json.dumps(
            {
                "phase": "SCHEDULED",
                "epoch": 4,
                "params": {"existing": True},
                "updated_at": "2026-07-26T00:00:00+00:00",
            }
        ).encode()

    def get(self, key, *, metadata, serializable):
        self.get_count += 1
        return self.value, SimpleNamespace(mod_revision=9)

    def update_if_unchanged(self, key, value, mod_revision):
        self.update_count += 1
        self.value = value.encode()
        return True


class TrackingStore:
    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def schedule_new_epoch(self, node, deployment, params):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return DesiredDoc("SCHEDULED", 1, params or {}, "now")
        finally:
            with self._lock:
                self.active -= 1


class BlockingStore:
    def __init__(self) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def schedule_new_epoch(self, node, deployment, params):
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=2)
        return DesiredDoc("SCHEDULED", 1, params or {}, "now")


def test_schedule_new_epoch_reuses_initial_linearizable_read() -> None:
    etcd = ExistingDesiredEtcd()
    store = DesiredStore(etcd)

    result = store.schedule_new_epoch("node-1", "executor-1", {"job_id": "job-1"})

    assert result.epoch == 5
    assert result.params == {"existing": True, "job_id": "job-1"}
    assert etcd.get_count == 1
    assert etcd.update_count == 1


@pytest.mark.asyncio
async def test_desired_state_executor_handles_dispatch_sized_fanout() -> None:
    store = TrackingStore()
    executor = DesiredStateExecutor(store, max_workers=16, max_pending=128)
    try:
        results = await asyncio.gather(
            *(
                executor.schedule_new_epoch(
                    f"node-{index}", "executor", {"job_id": f"job-{index}"}
                )
                for index in range(70)
            )
        )
    finally:
        executor.shutdown()

    assert len(results) == 70
    assert store.calls == 70
    assert store.max_active == 16


@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_capacity_until_write_finishes() -> None:
    store = BlockingStore()
    executor = DesiredStateExecutor(store, max_workers=1, max_pending=1)
    try:
        first = asyncio.create_task(
            executor.schedule_new_epoch("node-1", "executor", {"job_id": "job-1"})
        )
        assert await asyncio.to_thread(store.started.wait, 1)

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(
            executor.schedule_new_epoch("node-2", "executor", {"job_id": "job-2"})
        )
        await asyncio.sleep(0.05)
        assert store.calls == 1

        store.release.set()
        await asyncio.wait_for(second, timeout=1)
        assert store.calls == 2
    finally:
        store.release.set()
        executor.shutdown()


@pytest.mark.asyncio
async def test_shutdown_rejects_new_writes() -> None:
    executor = DesiredStateExecutor(TrackingStore(), max_workers=1, max_pending=1)
    executor.shutdown()

    with pytest.raises(RuntimeError, match="executor is closed"):
        await executor.schedule_new_epoch("node-1", "executor", {})
