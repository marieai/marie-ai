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


class RoutingRuntime(Runtime):
    def health(self, *, read_budget=None):
        self.calls += 1
        return {
            'fabric_group_id': self.config.fabric_group_id,
            'contract_version': 'v3',
            'running': True,
            'observed_at_ms': 2_000_000,
            'policy_generation': 4,
            'policy_digest': 'a' * 64,
            'lanes': [
                {
                    'pool_id': 'document-small',
                    'request_queue_depth': 3,
                    'oldest_pending_age_seconds': 12.5,
                    'charged_cost': 9,
                    'refunded_cost': 2,
                    'committed_charge': 7,
                    'state_counts': {
                        'ready': 3,
                        'claimed': 1,
                        'executing': 1,
                        'outcome_unknown': 0,
                    },
                    'drain_references': 5,
                }
            ],
            'endpoint_groups': [
                {
                    'group_id': 'document-llm',
                    'revision': 'r4',
                    'selected_replica_id': 'replica-b',
                    'drain_references': 2,
                    'replicas': [
                        {
                            'replica_id': 'replica-b',
                            'enabled': True,
                            'circuit': 'closed',
                            'reserved_items': 1,
                            'reserved_bytes': 4096,
                            'execution_limit': 4,
                            'execution_bytes': 8192,
                            'credential_env': 'SECRET_ENV',
                            'base_url': 'https://secret',
                            'raw_error': 'PHI',
                        }
                    ],
                }
            ],
            'messages': [{'role': 'user', 'content': 'PHI'}],
            'images': ['data:image/png;base64,secret'],
        }


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


def test_runtime_snapshot_reports_bounded_routing_recovery_and_replica_state(
    monkeypatch,
):
    monkeypatch.setattr(registry.time, 'time', lambda: 2_000.5)
    registry.register_dispatcher('routing', RoutingRuntime())

    snapshot = registry.dispatch_runtime_live_state(fabric_group_id='a')

    assert snapshot['policy'] == {
        'observed_generation': 4,
        'observed_digest': 'a' * 64,
    }
    assert snapshot['drr'] == {
        'charged_cost': 9,
        'refunded_cost': 2,
        'committed_charge': 7,
    }
    assert snapshot['pools'][0]['state_counts'] == {
        'ready': 3,
        'claimed': 1,
        'running': 1,
        'unknown': 0,
    }
    assert snapshot['pools'][0]['oldest_ready_age_seconds'] == 12.5
    assert snapshot['endpoint_groups'][0]['selected_replica_id'] == 'replica-b'
    assert snapshot['endpoint_groups'][0]['replicas'][0]['available_items'] == 3
    assert snapshot['observation']['stale'] is False
    encoded = json.dumps(snapshot)
    for forbidden in (
        'messages',
        'images',
        'credential_env',
        'base_url',
        'raw_error',
        'SECRET_ENV',
        'PHI',
    ):
        assert forbidden not in encoded


def test_runtime_snapshot_labels_old_observation_as_stale(monkeypatch):
    monkeypatch.setattr(registry.time, 'time', lambda: 2_020.0)
    registry.register_dispatcher('routing', RoutingRuntime())

    snapshot = registry.dispatch_runtime_live_state(fabric_group_id='a')

    assert snapshot['observation'] == {
        'observed_at_ms': 2_000_000,
        'stale': True,
        'stale_after_ms': registry.OBSERVATION_STALE_AFTER_MS,
    }


def test_policy_observation_is_unknown_when_any_dispatcher_omits_identity():
    registry.register_dispatcher('identified', RoutingRuntime())
    registry.register_dispatcher('unidentified', Runtime())

    snapshot = registry.dispatch_runtime_live_state(fabric_group_id='a')

    assert snapshot['policy'] == {
        'observed_generation': None,
        'observed_digest': None,
    }
