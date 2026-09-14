import asyncio
import threading
from types import SimpleNamespace

import pytest
from marie.engine.llm_queue import registry
from marie.engine.llm_queue.request_dispatcher import RequestDispatcher
from marie.engine.llm_queue.store import RequestStore, StoreLimits


class MetadataStore:
    ready_metadata = RequestStore.ready_metadata

    def __init__(self, pool, *, block=None, malformed=True):
        self.keys = SimpleNamespace(fabric_id='fabric', ready=lambda pool: pool)
        self.limits = StoreLimits()
        self.pool = pool
        self.calls = []
        self.block = block
        self.entered, self.release = threading.Event(), threading.Event()
        self.malformed = malformed

    def _page(self, limit):
        assert 1 <= limit <= 100

    def record(self, kind, identity):
        self.calls.append((kind, identity))
        if kind == self.block:
            self.entered.set()
            assert self.release.wait(2)

    def _read(self, method, key, start, end):
        self.record('ids', key)
        return [str(n) for n in range(end + 1)]

    def metadata(self, attempt):
        self.record('metadata', attempt)
        if self.malformed:
            raise ValueError('malformed numeric metadata')
        return SimpleNamespace(
            attempt_id=attempt,
            pool_id=self.pool,
            endpoint_id='endpoint',
            config_revision=1,
            state='ready',
            payload_bytes=10,
            cost=1,
            expires_at_ms=1000,
            execution_seq=0,
            next_eligible=0,
            last_error=None,
            model='model',
            admitted_at_ms=100,
        )

    def usage(self):
        self.record('usage', 'fabric')
        return {'reserved_items': 0}

    def endpoint_status(self, identity):
        self.record('endpoint', identity)
        return {'circuit': 'closed'}

    def ready_depth(self, pool):
        self.record('depth', pool)
        return 10


class Runtime:
    sample_pending_requests = RequestDispatcher.sample_pending_requests

    def __init__(self, store, *, real_health=False):
        self.store = store
        self.lanes = [
            SimpleNamespace(
                pool_id=store.pool,
                endpoint_id='endpoint',
                enabled=True,
                execution_limit=1,
            )
        ]
        self.endpoints = {'endpoint': SimpleNamespace(enabled=True)}
        self.metadata_unavailable = 0
        self.real_health = real_health

    def health(self, **kwargs):
        if self.real_health:
            return RequestDispatcher.health(self, **kwargs)
        return {
            'pool_id': self.store.pool,
            'request_queue_depth': 10,
            'contract_version': 'v3',
        }

    def inflight_requests_snapshot(self):
        return []


@pytest.fixture(autouse=True)
def registry_state(monkeypatch):
    monkeypatch.setattr(registry, '_DISPATCHERS', {})


def test_malformed_candidates_consume_global_allowance_before_next_dispatcher():
    stores = [MetadataStore('p1'), MetadataStore('p2')]
    for number, store in enumerate(stores):
        registry.register_dispatcher(str(number), Runtime(store))
    result = registry.dispatch_runtime_live_state(
        limit_per_pool=3, fabric_group_id='fabric'
    )
    assert result['live_requests'] == []
    assert sum(kind == 'metadata' for store in stores for kind, _ in store.calls) == 3
    assert stores[1].calls == []


def test_valid_rows_survive_malformed_candidates_with_shared_allowance():
    first, second = MetadataStore('p1'), MetadataStore('p2', malformed=False)
    first._read = lambda *args: ['corrupt']
    registry.register_dispatcher('first', Runtime(first))
    registry.register_dispatcher('second', Runtime(second))
    result = registry.dispatch_runtime_live_state(
        limit_per_pool=3, fabric_group_id='fabric'
    )
    assert [row['request_id'] for row in result['live_requests']] == ['0', '1']
    assert (
        sum(kind == 'metadata' for store in (first, second) for kind, _ in store.calls)
        == 3
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'real_health,block',
    [
        (False, 'ids'),
        (False, 'metadata'),
        (True, 'ids'),
        (True, 'metadata'),
        (True, 'depth'),
    ],
)
@pytest.mark.parametrize('timeout', [False, True])
async def test_cancel_or_timeout_stops_before_next_store_unit(
    real_health, block, timeout
):
    store = MetadataStore('p1', block=block)
    runtime = Runtime(store, real_health=real_health)
    runtime.lanes.append(
        SimpleNamespace(
            pool_id='p2', endpoint_id='endpoint', enabled=True, execution_limit=1
        )
    )
    registry.register_dispatcher('runtime', runtime)
    task = asyncio.create_task(
        registry.read_runtime_snapshot(
            fabric_group_id='fabric',
            limit=5,
            timeout_seconds=0.05 if timeout else 1,
        )
    )
    try:
        assert await asyncio.to_thread(store.entered.wait, 1)
        before = list(store.calls)
        if timeout:
            with pytest.raises(registry.SnapshotUnavailable):
                await task
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        store.release.set()
    await asyncio.get_running_loop().run_in_executor(
        registry._READ_WORKER, lambda: None
    )
    assert store.calls == before


def test_store_command_allowance_stops_before_another_metadata_read():
    store = MetadataStore('p1')
    budget = registry.SnapshotReadBudget(5, registry.time.monotonic() + 1, units_left=2)
    with pytest.raises(registry.SnapshotUnavailable):
        Runtime(store).sample_pending_requests(5, read_budget=budget)
    assert store.calls == [('ids', 'p1'), ('metadata', '0')]
    assert budget.candidates_left == 4


def test_deadline_is_checked_between_metadata_reads_without_cancellation(monkeypatch):
    clock = [0]
    monkeypatch.setattr(registry.time, 'monotonic', lambda: clock[0])
    store = MetadataStore('p1')
    metadata = store.metadata

    def expire_during_read(attempt):
        clock[0] = 2
        return metadata(attempt)

    store.metadata = expire_during_read
    budget = registry.SnapshotReadBudget(5, deadline=1)
    with pytest.raises(registry.SnapshotUnavailable):
        Runtime(store).sample_pending_requests(5, read_budget=budget)
    assert store.calls == [('ids', 'p1'), ('metadata', '0')]


@pytest.mark.asyncio
@pytest.mark.parametrize('block_at', [1, 2])
async def test_v2_health_stops_before_next_depth_read(block_at):
    from marie.engine.llm_queue.config import LlmQueueConfig
    from marie.engine.llm_queue.dispatcher import DrrQueuedBatchDispatcher
    from marie.engine.llm_queue.scheduler import DrrLaneConfig

    entered, release = threading.Event(), threading.Event()
    reads = []

    def depth(pool):
        reads.append(pool)
        if len(reads) == block_at:
            entered.set()
            assert release.wait(2)
        return 1

    runtime = DrrQueuedBatchDispatcher(
        queue_client=SimpleNamespace(request_queue_depth=depth),
        execution_adapter=object(),
        logger=object(),
        config=LlmQueueConfig(enabled=True, fabric_group_id='fabric'),
        lanes=[DrrLaneConfig('p1'), DrrLaneConfig('p2')],
        total_concurrent_dispatch=1,
    )
    registry.register_dispatcher('runtime', runtime)
    task = asyncio.create_task(registry.read_runtime_snapshot(fabric_group_id='fabric'))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        before = list(reads)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    await asyncio.get_running_loop().run_in_executor(
        registry._READ_WORKER, lambda: None
    )
    assert reads == before


def test_health_reads_distinct_endpoints_once_and_uses_complete_shared_budget():
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import DispatchLane

    store = MetadataStore('p1', malformed=False)
    store.route_status = lambda pool: {'reserved_items': 0, 'reserved_bytes': 0}
    store.usage = lambda: store.record('usage', 'fabric') or {'reserved_items': 0}
    store.endpoint_status = lambda identity: store.record('endpoint', identity) or {
        'circuit': 'closed'
    }
    runtime = RequestDispatcher(
        store=store,
        endpoints=[RegisteredEndpoint('endpoint', 'https://example.test')],
        lanes=[DispatchLane('p1', 'endpoint'), DispatchLane('p2', 'endpoint')],
    )
    try:
        budget = registry.SnapshotReadBudget(
            5, registry.time.monotonic() + 1, units_left=10
        )
        health = runtime.health(read_budget=budget)
        assert store.calls.count(('usage', 'fabric')) == 1
        assert store.calls.count(('endpoint', 'endpoint')) == 1
        assert budget.units_left == 0
        assert health['endpoints']['endpoint']['circuit'] == 'closed'
        assert health['lanes'][0]['waiting_reason'] is None
        with pytest.raises(registry.SnapshotUnavailable):
            runtime.health(
                read_budget=registry.SnapshotReadBudget(
                    5, registry.time.monotonic() + 1, units_left=9
                )
            )
    finally:
        for worker in (
            runtime._store_workers,
            runtime._lease_worker,
            runtime._refresh_worker,
        ):
            worker.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('block', ['usage', 'endpoint'])
@pytest.mark.parametrize('timeout', [False, True])
async def test_shared_health_cancellation_stops_before_next_unit(block, timeout):
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import DispatchLane

    store = MetadataStore('p1', block=block, malformed=False)
    store.usage = lambda: store.record('usage', 'fabric') or {'reserved_items': 0}
    store.endpoint_status = lambda identity: store.record('endpoint', identity) or {
        'circuit': 'closed'
    }
    store.route_status = lambda pool: {'reserved_items': 0, 'reserved_bytes': 0}
    runtime = RequestDispatcher(
        store=store,
        endpoints=[RegisteredEndpoint('endpoint', 'https://example.test')],
        lanes=[DispatchLane('p1', 'endpoint')],
    )
    registry.register_dispatcher('runtime', runtime)
    task = asyncio.create_task(
        registry.read_runtime_snapshot(
            fabric_group_id='fabric', timeout_seconds=0.1 if timeout else 2
        )
    )
    try:
        assert await asyncio.to_thread(store.entered.wait, 1)
        before = list(store.calls)
        if timeout:
            with pytest.raises(registry.SnapshotUnavailable):
                await task
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        store.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.get_running_loop().run_in_executor(
            registry._READ_WORKER, lambda: None
        )
        for worker in (
            runtime._store_workers,
            runtime._lease_worker,
            runtime._refresh_worker,
        ):
            worker.shutdown()
    assert store.calls == before


@pytest.mark.parametrize('failed', ['usage', 'endpoint_status'])
def test_shared_health_read_failure_is_unavailable_without_partial_projection(failed):
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import DispatchLane

    store = MetadataStore('p1', malformed=False)

    def unavailable(*args):
        raise RuntimeError('SENTINEL private store error')

    setattr(store, failed, unavailable)
    runtime = RequestDispatcher(
        store=store,
        endpoints=[RegisteredEndpoint('endpoint', 'https://example.test')],
        lanes=[DispatchLane('p1', 'endpoint')],
    )
    registry.register_dispatcher('runtime', runtime)
    try:
        with pytest.raises(registry.SnapshotUnavailable, match='^runtime_read_failed$'):
            registry.dispatch_runtime_live_state(fabric_group_id='fabric')
        assert not any(kind == 'ids' for kind, _ in store.calls)
    finally:
        for worker in (
            runtime._store_workers,
            runtime._lease_worker,
            runtime._refresh_worker,
        ):
            worker.shutdown()
