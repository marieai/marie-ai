import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from marie.engine.llm_queue import registry


class Runtime:
    def __init__(self, fabric='a', delay=0):
        self.config = SimpleNamespace(fabric_group_id=fabric)
        self.calls = 0
        self.delay = delay

    def health(self, *, read_budget=None):
        self.calls += 1
        time.sleep(self.delay)
        return {
            'fabric_group_id': self.config.fabric_group_id,
            'pool_id': 'default',
            'running': True,
            'contract_version': 'v3',
            'request_queue_depth': 10,
            'endpoint_url': 'https://secret',
            'last_error': 'PHI raw secret',
            'lanes': [{'pool_id': str(n), 'request_queue_depth': 1} for n in range(10)],
        }

    def sample_pending_requests(self, limit, *, read_budget=None):
        return [
            {
                'request_id': str(n),
                'pool_id': str(n),
                'submitted_at': 1,
                'prompt': 'PHI data:image secret',
            }
            for n in range(limit)
        ]

    def inflight_requests_snapshot(self):
        return []


@pytest.fixture(autouse=True)
def registry_state(monkeypatch):
    monkeypatch.setattr(registry, '_DISPATCHERS', {})


def test_snapshot_reads_health_once_filters_before_read_and_has_global_budget():
    first, second, other = Runtime(), Runtime(), Runtime('b')
    for name, runtime in [('first', first), ('second', second), ('other', other)]:
        registry.register_dispatcher(name, runtime)
    result = registry.dispatch_runtime_live_state(limit_per_pool=5, fabric_group_id='a')
    assert first.calls == second.calls == 1
    assert other.calls == 0
    assert len(result['live_requests']) <= 5
    assert result['runtime_summary']['pending_request_count'] == 10
    assert 'secret' not in json.dumps(result) and 'PHI' not in json.dumps(result)


@pytest.mark.asyncio
async def test_snapshot_timeout_keeps_event_loop_responsive_and_bounds_workers():
    registry.register_dispatcher('slow', Runtime(delay=0.2))
    started = time.monotonic()
    with pytest.raises(registry.SnapshotUnavailable):
        await registry.read_runtime_snapshot(fabric_group_id='a', timeout_seconds=0.02)
    assert time.monotonic() - started < 0.1
    with pytest.raises(registry.SnapshotUnavailable):
        await registry.read_runtime_snapshot(fabric_group_id='a', timeout_seconds=0.02)
    await asyncio.sleep(0.22)


def test_v2_sampling_never_reads_payload():
    from marie.engine.llm_queue.dispatcher import QueuedBatchDispatcher

    runtime = object.__new__(QueuedBatchDispatcher)
    runtime.queue_client = SimpleNamespace(
        sample_requests=lambda *a: pytest.fail('payload read')
    )
    runtime.config = SimpleNamespace(pool_id='default')
    assert runtime.sample_pending_requests(5) == []


def test_drr_status_does_not_peek_payload():
    from marie.engine.llm_queue.scheduler import DrrLaneConfig, DrrLaneScheduler

    queue = SimpleNamespace(
        request_queue_depth=lambda _: 2,
        peek_request=lambda _: pytest.fail('payload read'),
    )
    scheduler = DrrLaneScheduler(
        queue_client=queue,
        lanes=[DrrLaneConfig(pool_id='default')],
        total_concurrent_dispatch=1,
    )
    assert scheduler.lane_snapshots()[0].head_cost_units is None


def test_total_control_and_sample_rows_obey_one_budget(monkeypatch):
    monkeypatch.setattr(registry, 'MAX_SNAPSHOT_ROWS', 12)
    registry.register_dispatcher('runtime', Runtime())
    with pytest.raises(registry.SnapshotUnavailable):
        registry.dispatch_runtime_live_state(limit_per_pool=5, fabric_group_id='a')


@pytest.mark.asyncio
async def test_cancelled_snapshot_stops_before_next_dispatcher():
    first, second = Runtime(delay=0.1), Runtime()
    registry.register_dispatcher('first', first)
    registry.register_dispatcher('second', second)
    task = asyncio.create_task(registry.read_runtime_snapshot(fabric_group_id='a'))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.12)
    assert second.calls == 0


def test_dispatch_error_trace_does_not_record_raw_exception():
    from marie.engine.llm_queue.dispatcher import _set_dispatch_span_error_attributes

    values = []
    span = SimpleNamespace(
        set_attribute=lambda *args: values.append(args),
        record_exception=lambda *args: values.append(args),
        set_status=lambda *args: values.append(args),
    )
    request = SimpleNamespace(submitted_at=time.time())
    _set_dispatch_span_error_attributes(
        span,
        exc=RuntimeError('PHI data:image credential'),
        started_monotonic=time.monotonic(),
        request=request,
    )
    assert 'PHI' not in str(values) and 'data:image' not in str(values)


def test_legacy_depth_failure_is_unavailable_not_successful_zero():
    runtime = Runtime()
    runtime.health = lambda **kwargs: {
        'request_queue_depth': None,
        'request_queue_depth_error': 'queue_unavailable',
    }
    registry.register_dispatcher('failed', runtime)
    with pytest.raises(registry.SnapshotUnavailable):
        registry.dispatch_runtime_live_state(fabric_group_id='a')


def test_corrupted_numeric_metadata_does_not_become_text_content():
    clean = registry._clean({'reserved_items': 'PHI secret', 'failures': '2'})
    assert clean['reserved_items'] is None
    assert clean['failures'] == 2
