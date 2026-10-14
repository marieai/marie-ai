import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any

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
            'mode': 'queued-dispatch',
            'sampling_available': True,
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
                'state_updated_at': 2,
                'queue_wait_age_seconds': 3.5,
                'inflight_age_seconds': None,
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
            'mode': 'queued-dispatch',
            'sampling_available': True,
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
                    'accepted': 42,
                    'completed': 40,
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
                            'port': 4010,
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
    assert result['live_requests'][0]['state_updated_at'] == 2
    assert result['live_requests'][0]['queue_wait_age_seconds'] == 3.5
    assert result['live_requests'][0]['inflight_age_seconds'] is None
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


def test_queued_sampling_reads_metadata_without_request_payload() -> None:
    from marie.engine.llm_queue.request_dispatcher import RequestDispatcher
    from test_snapshot_read_budget import MetadataStore
    from test_snapshot_read_budget import Runtime as MetadataRuntime

    store = MetadataStore('default', malformed=False)
    store._read = lambda *args: ['request']
    store.request = lambda *args: pytest.fail('payload read')
    runtime = MetadataRuntime(store)

    rows = RequestDispatcher.sample_pending_requests(runtime, 5)

    assert [row['request_id'] for row in rows] == ['request']
    assert store.calls == [('metadata', 'request')]


def test_drr_status_does_not_peek_payload():
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )
    from test_snapshot_read_budget import MetadataStore

    store = MetadataStore('default', malformed=False)
    store.request = lambda *args: pytest.fail('payload read')
    store.route_status = lambda pool: {'reserved_items': 0, 'reserved_bytes': 0}
    runtime = RequestDispatcher(
        store=store,
        endpoints=[RegisteredEndpoint('endpoint', 'https://example.test')],
        lanes=[DispatchLane('default', 'endpoint')],
    )
    try:
        assert runtime.health()['lanes'][0]['head_cost_units'] == 1
    finally:
        for worker in (
            runtime._store_workers,
            runtime._lease_worker,
            runtime._refresh_worker,
        ):
            worker.shutdown()


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


def test_depth_failure_is_unavailable_not_successful_zero():
    runtime = Runtime()
    runtime.health = lambda **kwargs: {
        'mode': 'queued-dispatch',
        'request_queue_depth': None,
        'request_queue_depth_error': 'queue_unavailable',
    }
    registry.register_dispatcher('failed', runtime)
    with pytest.raises(registry.SnapshotUnavailable, match='^runtime_read_failed$'):
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
    assert snapshot['pools'][0]['accepted'] == 42
    assert snapshot['pools'][0]['completed'] == 40
    assert snapshot['endpoint_groups'][0]['selected_replica_id'] == 'replica-b'
    assert snapshot['endpoint_groups'][0]['replicas'][0]['available_items'] == 3
    assert snapshot['endpoint_groups'][0]['replicas'][0]['port'] == 4010
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


def test_empty_registry_describes_queue_scope_without_inventing_metadata() -> None:
    snapshot = registry.dispatch_runtime_live_state(fabric_group_id='a')

    assert snapshot['mode'] == 'queued-dispatch'
    assert 'contract_version' not in snapshot
    summary = snapshot['runtime_summary']
    assert summary['registered_dispatchers'] == 0
    assert summary['running_dispatchers'] == 0
    assert summary['sampling_available'] is False
    assert summary['metadata_available'] is False
    assert summary['pending_request_count'] is None
    assert snapshot['dispatchers'] == []
    assert registry.dispatch_runtime_snapshot()['mode'] == 'queued-dispatch'


@pytest.mark.parametrize('mode', [None, 'direct-batch', 'v3', 'mixed'])
def test_registered_runtime_must_explicitly_declare_queued_dispatch(
    mode: str | None,
) -> None:
    runtime = Runtime()
    health = runtime.health

    def invalid_health(**kwargs: Any) -> dict[str, Any]:
        row = health(**kwargs)
        if mode is None:
            row.pop('mode')
            row['contract_version'] = 'v3'
        else:
            row['mode'] = mode
        return row

    runtime.health = invalid_health
    registry.register_dispatcher('valid', Runtime())
    registry.register_dispatcher('invalid', runtime)

    with pytest.raises(registry.SnapshotUnavailable, match='^runtime_mode_mismatch$'):
        registry.dispatch_runtime_live_state(fabric_group_id='a')


@pytest.mark.asyncio
async def test_async_snapshot_preserves_bounded_mode_mismatch() -> None:
    runtime = Runtime()
    runtime.health = lambda **kwargs: {'mode': 'direct-batch'}
    registry.register_dispatcher('invalid', runtime)

    with pytest.raises(registry.SnapshotUnavailable, match='^runtime_mode_mismatch$'):
        await registry.read_runtime_snapshot(fabric_group_id='a')


@pytest.mark.parametrize('capability', [False, None, 1, 'true'])
def test_sampling_requires_explicit_true_from_every_runtime(
    capability: bool | int | str | None,
) -> None:
    runtime = Runtime()
    health = runtime.health

    def unsupported_health(**kwargs: Any) -> dict[str, Any]:
        row = health(**kwargs)
        if capability is None:
            row.pop('sampling_available')
        else:
            row['sampling_available'] = capability
        return row

    runtime.health = unsupported_health
    registry.register_dispatcher('capable', Runtime())
    registry.register_dispatcher('unsupported', runtime)

    snapshot = registry.dispatch_runtime_live_state(fabric_group_id='a')

    assert snapshot['mode'] == 'queued-dispatch'
    assert snapshot['runtime_summary']['sampling_available'] is False
    assert snapshot['runtime_summary']['metadata_available'] is True


def test_sampling_capability_does_not_imply_pending_metadata_available() -> None:
    runtime = Runtime()
    health = runtime.health

    def unknown_depth(**kwargs: Any) -> dict[str, Any]:
        row = health(**kwargs)
        row['lanes'] = [{'pool_id': 'default'}]
        return row

    runtime.health = unknown_depth
    registry.register_dispatcher('capable', runtime)

    snapshot = registry.dispatch_runtime_live_state(fabric_group_id='a')

    assert snapshot['runtime_summary']['sampling_available'] is True
    assert snapshot['runtime_summary']['metadata_available'] is False
    assert snapshot['runtime_summary']['pending_request_count'] is None
    assert snapshot['dispatchers'][0]['mode'] == 'queued-dispatch'


@pytest.mark.parametrize('capability', [False, None])
def test_unsupported_sampling_preserves_health_without_calling_samplers(
    capability: bool | None,
) -> None:
    runtime = Runtime()
    health = runtime.health

    def unsupported_health(**kwargs: Any) -> dict[str, Any]:
        row = health(**kwargs)
        if capability is None:
            row.pop('sampling_available')
        else:
            row['sampling_available'] = capability
        return row

    def unsupported_sampler(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError('unsupported runtime sampler was invoked')

    runtime.health = unsupported_health
    runtime.sample_pending_requests = unsupported_sampler
    runtime.inflight_requests_snapshot = unsupported_sampler
    registry.register_dispatcher('unsupported', runtime)

    snapshot = registry.dispatch_runtime_live_state(fabric_group_id='a')

    assert snapshot['mode'] == 'queued-dispatch'
    assert snapshot['dispatchers'][0]['running'] is True
    assert snapshot['runtime_summary']['registered_dispatchers'] == 1
    assert snapshot['runtime_summary']['running_dispatchers'] == 1
    assert snapshot['runtime_summary']['sampling_available'] is False
    assert snapshot['runtime_summary']['metadata_available'] is True
    assert snapshot['runtime_summary']['pending_request_count'] == 10
    assert snapshot['runtime_summary']['pending_request_sample_count'] == 0
    assert snapshot['live_requests'] == []


@pytest.mark.parametrize('category', ['dispatch_start_failed', 'runtime_mode_mismatch'])
def test_runtime_observation_preserves_new_bounded_error_categories(
    category: str,
) -> None:
    assert registry._clean({'last_error': category}) == {'last_error': category}
