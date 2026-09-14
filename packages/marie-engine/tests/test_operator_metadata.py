import json
from dataclasses import replace

import pytest
from test_request_store import request_for
from test_request_store import store as store


def test_v3_snapshot_selects_only_metadata_and_preserves_good_rows(store, monkeypatch):
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    request = request_for(store)
    assert store.admit(request).disposition == 'admitted'
    other = replace(request, attempt_id='malformed')
    assert store.admit(other).disposition == 'admitted'
    store.client.hset(store.keys.request('malformed'), 'cost', 'bad')
    original = store._read

    def read(method, *args, **kwargs):
        assert method != 'hgetall'
        assert 'payload' not in args and 'result' not in args
        return original(method, *args, **kwargs)

    monkeypatch.setattr(store, '_read', read)
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                endpoint_id='endpoint',
                base_url='https://example.com',
                execution_limit=2,
                execution_bytes=100000,
            )
        ],
        lanes=[
            DispatchLane(
                pool_id='pool',
                endpoint_id='endpoint',
                execution_limit=2,
                execution_bytes=100000,
            )
        ],
    )
    health = runtime.health()
    assert health['lanes'][0]['request_queue_depth'] == 2
    rows = runtime.sample_pending_requests(10)
    assert len(rows) == 1 and rows[0]['request_id'] == request.attempt_id
    assert rows[0]['state_source'] == 'store'
    assert rows[0]['model'] == 'model'
    assert rows[0]['admitted_at_ms'] > 0
    assert health['lanes'][0]['head_cost_units'] == request.estimated_cost_units
    assert (
        health['lanes'][0]['oldest_pending_admitted_at_ms'] == rows[0]['admitted_at_ms']
    )
    assert 'producer_id' not in json.dumps(rows)
    assert runtime.health()['metadata_unavailable'] == 1
    runtime._store_workers.shutdown()
    runtime._lease_worker.shutdown()


def test_endpoint_diagnostic_reports_cooldown_with_nonprobe_reservation(store):
    store.client.hset(
        store.keys.endpoint('endpoint'),
        mapping={'circuit': 'open', 'reserved_items': 1, 'next_probe': 0},
    )
    assert store.endpoint_status('endpoint')['waiting_reason'] == 'circuit_cooldown'


@pytest.mark.asyncio
async def test_producer_cleanup_and_live_recovery_have_distinct_lifetime_counters(
    store,
):
    import asyncio

    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                endpoint_id='endpoint',
                base_url='https://example.com',
                execution_limit=2,
                execution_bytes=100000,
            )
        ],
        lanes=[
            DispatchLane(
                pool_id='pool',
                endpoint_id='endpoint',
                execution_limit=2,
                execution_bytes=100000,
            )
        ],
    )
    runtime.owner = store.test_owner
    dead = request_for(store)
    live = request_for(store)
    store.admit(dead)
    store.admit(live)
    store.claim(store.test_owner, dead.attempt_id, pool_id='pool', claim_id='deadclaim')
    store.claim(store.test_owner, live.attempt_id, pool_id='pool', claim_id='liveclaim')
    store.close_producer(dead.producer_id)
    await asyncio.sleep(0.12)
    try:
        await runtime._maintenance()
        assert runtime.health()['counters']['producer_expired_requests'] == 1
        assert runtime.health()['counters']['live_claims_recovered'] == 1
        await runtime._maintenance()
        assert runtime.health()['counters']['producer_expired_requests'] == 1
        assert runtime.health()['counters']['live_claims_recovered'] == 1
    finally:
        runtime._store_workers.shutdown()
        runtime._lease_worker.shutdown()


def test_metadata_model_uses_effective_override_and_first_admission_time(store):
    from dataclasses import replace

    request = request_for(store)
    request = replace(
        request,
        call=replace(
            request.call,
            extra_create_kwargs={
                'model': 'middle',
                'extra_body': {'model': 'effective'},
            },
        ),
    )
    store.admit(request)
    first = store.metadata(request.attempt_id)
    assert first.model == 'effective'
    assert first.admitted_at_ms > 0
    store.admit(replace(request, expires_at_ms=request.expires_at_ms + 100))
    assert store.metadata(request.attempt_id).admitted_at_ms == first.admitted_at_ms
    store.client.hdel(store.keys.request(request.attempt_id), 'model', 'admitted_at_ms')
    old = store.metadata(request.attempt_id)
    assert old.model is None and old.admitted_at_ms is None


@pytest.mark.parametrize(
    'model', ['x' * 129, 'data:image/png;base64,PHI', 'has whitespace']
)
def test_invalid_model_alias_rejects_before_store_charge(store, model):
    request = request_for(store)
    before = store.usage()
    with pytest.raises(ValueError):
        store.admit(replace(request, call=replace(request.call, model=model)))
    assert store.usage() == before


def test_status_polling_does_not_fetch_large_inline_image(store, monkeypatch, request):
    import os
    from pathlib import Path
    from uuid import uuid4

    from marie.engine.llm_queue import registry
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )
    from marie.engine.llm_queue.store import RequestStore

    engine = request.node.callspec.params['store']
    url = json.loads(Path(os.environ['MARIE_LLM_QUEUE_TEST_STORES']).read_text())[
        engine
    ]['url']
    large = RequestStore(url, fabric_id='large-' + uuid4().hex, version='v3')
    runtime = None
    try:
        owner = large.acquire_owner('fixture', lease_ms=10000)
        large.configure_route(
            owner,
            'pool',
            'endpoint',
            revision='r1',
            execution_limit=1,
            execution_bytes=64 * 1024 * 1024,
        )
        item = request_for(large)
        item = replace(
            item,
            call=replace(
                item.call,
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': 'data:image/png;base64,'
                                    + 'A' * (15 * 1024 * 1024)
                                },
                            }
                        ],
                    }
                ],
            ),
        )
        assert large.admit(item).disposition == 'admitted'
        original = large._read
        returned_bytes = 0

        def read(method, *args, **kwargs):
            nonlocal returned_bytes
            assert (
                method != 'hgetall' and 'payload' not in args and 'result' not in args
            )
            result = original(method, *args, **kwargs)
            returned_bytes += len(json.dumps(result))
            return result

        monkeypatch.setattr(large, '_read', read)
        runtime = RequestDispatcher(
            store=large,
            endpoints=[RegisteredEndpoint('endpoint', 'https://example.com')],
            lanes=[DispatchLane('pool', 'endpoint')],
        )
        registry.register_dispatcher(runtime.dispatcher_id, runtime)
        snapshot = registry.dispatch_runtime_live_state(
            fabric_group_id=large.keys.fabric_id
        )
        assert snapshot['live_requests'][0]['payload_bytes'] > 15 * 1024 * 1024
        assert returned_bytes < 10000
        assert 'data:image' not in json.dumps(snapshot)
    finally:
        if runtime:
            registry.unregister_dispatcher(runtime.dispatcher_id)
            runtime._store_workers.shutdown()
            runtime._lease_worker.shutdown()
        keys = list(large.client.scan_iter(match=large.keys.prefix + '*'))
        if keys:
            large.client.delete(*keys)
        large.close()


@pytest.mark.asyncio
async def test_execution_failure_counter_and_last_category_survive_success_and_idle(
    store,
):
    import time

    from marie.engine.llm_queue.endpoint import ExecutionOutcome, RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    class Client:
        async def execute(self, call, **kwargs):
            return (
                ExecutionOutcome(category='request_rejected', remote_settled=True)
                if call.model == 'reject'
                else ExecutionOutcome(response={}, remote_settled=True)
            )

    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'endpoint',
                'https://example.com',
                execution_limit=2,
                execution_bytes=100000,
            )
        ],
        lanes=[
            DispatchLane('pool', 'endpoint', execution_limit=2, execution_bytes=100000)
        ],
    )
    runtime.owner = store.test_owner
    runtime._owner_until = time.monotonic() + 10
    runtime._clients['endpoint'] = Client()
    try:
        for index, model in enumerate(['reject', 'success']):
            item = request_for(store)
            item = replace(item, call=replace(item.call, model=model))
            store.admit(item)
            claim = 'claim' + str(index)
            store.claim(
                store.test_owner, item.attempt_id, pool_id='pool', claim_id=claim
            )
            await runtime._execute(item.attempt_id, claim, 'endpoint')
        await runtime._maintenance()
        health = runtime.health()
        assert health['counters']['execution_errors'] == 1
        assert health['last_error'] == 'request_rejected'
        assert health['last_error_at_ms'] > 0
    finally:
        runtime._store_workers.shutdown()
        runtime._lease_worker.shutdown()


@pytest.mark.parametrize('scheme', ['HTTP', 'hTtP', 'HTTPS', 'HtTpS', 'DATA', 'DaTa'])
def test_live_projection_redacts_case_insensitive_url_fields(
    store, monkeypatch, scheme
):
    from types import SimpleNamespace

    from marie.engine.llm_queue import registry

    sentinel = scheme + '://synthetic.invalid/SENTINEL'
    monkeypatch.setattr(registry, '_DISPATCHERS', {})
    runtime = SimpleNamespace(
        store=store,
        health=lambda **kwargs: {
            'contract_version': 'v3',
            'request_queue_depth': 1,
            'model': sentinel,
            'waiting_reason': sentinel,
        },
        sample_pending_requests=lambda *args, **kwargs: [
            {'model': sentinel, 'lifecycle_stage': sentinel}
        ],
        inflight_requests_snapshot=lambda: [],
    )
    registry.register_dispatcher('runtime', runtime)
    result = registry.dispatch_runtime_live_state(fabric_group_id=store.keys.fabric_id)
    assert 'SENTINEL' not in json.dumps(result)
    for alias in ['openai/gpt-4.1', 'provider:model-v1']:
        assert registry._clean({'model': alias})['model'] == alias


@pytest.mark.parametrize('owner', [False, True])
@pytest.mark.parametrize('probe', [False, True])
def test_health_reads_shared_endpoint_and_usage_after_ownership_change(
    store, owner, probe, monkeypatch
):
    from marie.engine.llm_queue import registry
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    request = request_for(store)
    store.admit(request)
    store.client.hset(
        store.keys.endpoint('endpoint'),
        mapping={
            'circuit': 'open',
            'reserved_items': 1,
            'reserved_bytes': 100,
            'probe_claim': 'probe' if probe else '',
        },
    )
    store.client.hset(
        store.keys.prefix + 'usage',
        mapping={'reserved_items': 1, 'reserved_bytes': 100},
    )
    runtime = RequestDispatcher(
        store=store,
        endpoints=[RegisteredEndpoint('endpoint', 'https://example.com')],
        lanes=[DispatchLane('pool', 'endpoint')],
    )
    runtime.owner = store.test_owner if owner else None
    runtime._usage = {'reserved_items': 0}
    runtime._endpoint_states = {'endpoint': {'circuit': 'closed'}}
    monkeypatch.setattr(registry, '_DISPATCHERS', {})
    registry.register_dispatcher('runtime', runtime)
    try:
        result = registry.dispatch_runtime_live_state(
            fabric_group_id=store.keys.fabric_id
        )
        health = result['dispatchers'][0]
        assert health['role'] == ('owner' if owner else 'standby')
        assert health['usage']['reserved_items'] == 1
        assert health['endpoints']['endpoint']['circuit'] == 'open'
        assert health['endpoints']['endpoint']['reserved_items'] == 1
        assert health['lanes'][0]['waiting_reason'] == (
            'probe_unresolved' if probe else 'circuit_cooldown'
        )
        store.client.hset(
            store.keys.endpoint('endpoint'),
            mapping={'circuit': 'closed', 'probe_claim': ''},
        )
        assert runtime.health()['lanes'][0]['waiting_reason'] == (
            'insufficient_credit' if owner else None
        )
    finally:
        for worker in (
            runtime._store_workers,
            runtime._lease_worker,
            runtime._refresh_worker,
        ):
            worker.shutdown()
