"""Fairness and physical reservations against the owned Redis/Valkey stores."""

import asyncio
from dataclasses import replace

import pytest
from marie.engine.llm_queue.endpoint import RegisteredEndpoint
from marie.engine.llm_queue.request_dispatcher import DispatchLane, RequestDispatcher
from marie.engine.llm_queue.store import StaleOwner
from test_request_dispatcher import admit_request
from test_request_dispatcher import store as store


def runtime_for(store, lanes):
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint('endpoint', 'https://example.com/v1', execution_limit=16)
        ],
        lanes=lanes,
    )
    runtime.owner = store.acquire_owner('owner', lease_ms=10000)
    return runtime


async def close_runtime(runtime):
    for task in runtime._tasks.values():
        task.cancel()
    await asyncio.gather(*runtime._tasks.values(), return_exceptions=True)
    runtime._store_workers.shutdown(wait=True)
    runtime._lease_worker.shutdown(wait=True)
    runtime._refresh_worker.shutdown(wait=True)


async def test_weighted_bursts_charge_real_cost_without_payload_reads(store):
    runtime = runtime_for(
        store,
        [
            DispatchLane('a', 'endpoint', quantum=3),
            DispatchLane('b', 'endpoint', quantum=1),
        ],
    )
    await runtime._configure()
    sent = []

    async def execute(attempt, claim, endpoint):
        sent.append(store.metadata(attempt).pool_id)
        await asyncio.Event().wait()

    runtime._execute = execute
    for _ in range(8):
        admit_request(store, pool_id='a')
        admit_request(store, pool_id='b')
    store.fetch_payload = lambda *a, **k: pytest.fail('selection fetched payload')
    try:
        await runtime._dispatch()
        await asyncio.sleep(0)
        assert sent[:8] == ['a', 'a', 'a', 'b', 'a', 'a', 'a', 'b']
        assert store.usage()['reserved_items'] == len(sent)
    finally:
        await close_runtime(runtime)


async def test_expected_cost_mismatch_does_not_claim_or_charge(store):
    runtime = runtime_for(store, [DispatchLane('pool', 'endpoint')])
    await runtime._configure()
    req = admit_request(store)
    reply = store.claim(
        runtime.owner, req.attempt_id, pool_id='pool', claim_id='claim', expected_cost=2
    )
    assert reply.disposition == 'cost_changed'
    assert store.ready_head('pool') == req.attempt_id
    assert store.usage()['reserved_items'] == 0
    await close_runtime(runtime)


async def test_unknown_nonprobe_allows_only_one_probe_and_retains_unknown_probe(store):
    runtime = runtime_for(store, [DispatchLane('pool', 'endpoint', execution_limit=2)])
    await runtime._configure()
    old = admit_request(store)
    store.claim(runtime.owner, old.attempt_id, pool_id='pool', claim_id='old')
    store.authorize_start(runtime.owner, old.attempt_id, claim_id='old')
    store.mark_unknown(
        runtime.owner,
        old.attempt_id,
        claim_id='old',
        execution_seq=1,
        category='read_timeout',
    )
    store.client.hset(
        store.keys.endpoint('endpoint'), mapping={'circuit': 'open', 'next_probe': 0}
    )
    probe = admit_request(store)
    assert (
        store.claim(
            runtime.owner, probe.attempt_id, pool_id='pool', claim_id='probe'
        ).disposition
        == 'claimed'
    )
    store.authorize_start(runtime.owner, probe.attempt_id, claim_id='probe')
    store.record_endpoint_outcome(
        runtime.owner,
        probe.attempt_id,
        claim_id='probe',
        execution_seq=1,
        outcome='unavailable',
        category='read_timeout',
        open_ms=1,
    )
    assert store.endpoint_status('endpoint')['probe_claim'] == 'probe'
    store.mark_unknown(
        runtime.owner,
        probe.attempt_id,
        claim_id='probe',
        execution_seq=1,
        category='read_timeout',
    )
    store.client.hset(store.keys.endpoint('endpoint'), 'next_probe', 0)
    queued = admit_request(store)
    assert (
        store.claim(
            runtime.owner, queued.attempt_id, pool_id='pool', claim_id='next'
        ).disposition
        == 'gated'
    )
    store.close_producer(probe.producer_id)
    store.expire_producer(probe.producer_id)
    assert store.endpoint_status('endpoint')['probe_claim'] == 'probe'
    assert store.usage()['reserved_items'] == 2
    store.release_owner(runtime.owner)
    replacement = store.acquire_owner('replacement', lease_ms=10000)
    with pytest.raises(StaleOwner):
        store.record_endpoint_outcome(
            runtime.owner,
            probe.attempt_id,
            claim_id='probe',
            execution_seq=1,
            outcome='success',
        )
    assert (
        store.claim(
            replacement, queued.attempt_id, pool_id='pool', claim_id='next'
        ).disposition
        == 'gated'
    )
    assert (
        store.settle_remote(
            replacement,
            probe.attempt_id,
            claim_id='probe',
            execution_seq=1,
            evidence='remote_completed',
        ).disposition
        == 'settled'
    )
    assert store.usage()['reserved_items'] == 1
    assert (
        store.claim(
            replacement, queued.attempt_id, pool_id='pool', claim_id='next'
        ).disposition
        == 'claimed'
    )
    await close_runtime(runtime)


@pytest.mark.parametrize('value', [True, '2', 1.5, 0, 17])
def test_prepared_cost_rejects_invalid_override(value):
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.scheduler import prepared_cost_units

    with pytest.raises(ValueError):
        prepared_cost_units(
            CompletionCallParams(model='m', messages=[]),
            {'estimated_cost_units': value},
        )


def test_prepared_cost_uses_effective_ordered_images():
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.scheduler import prepared_cost_units

    call = CompletionCallParams(
        model='m',
        messages=[
            {
                'role': 'user',
                'content': [{'type': 'image_url', 'image_url': {'url': 'data:x'}}] * 3,
            }
        ],
    )
    assert prepared_cost_units(call, {'page_count': 8}) == 8


async def test_protected_slots_borrow_and_return_with_unresolved_reservations(store):
    runtime = runtime_for(
        store,
        [
            DispatchLane('a', 'endpoint', execution_limit=16),
            DispatchLane('b', 'endpoint', execution_limit=16, min_concurrent=2),
        ],
    )
    await runtime._configure()

    async def hold(*args):
        await asyncio.Event().wait()

    runtime._execute = hold
    for _ in range(20):
        admit_request(store, pool_id='a')
    try:
        await runtime._dispatch()
        assert store.usage()['reserved_items'] == 16
        for _ in range(2):
            admit_request(store, pool_id='b')
        ids = list(runtime._tasks)[:2]
        for attempt in ids:
            meta = store.metadata(attempt)
            store.reject_claim(
                runtime.owner,
                attempt,
                claim_id=meta.claim_id,
                category='invalid_request',
            )
            runtime._tasks[attempt].cancel()
        await asyncio.gather(
            *(runtime._tasks[attempt] for attempt in ids), return_exceptions=True
        )
        await runtime._dispatch()
        assert store.route_status('a')['reserved_items'] == 14
        assert store.route_status('b')['reserved_items'] == 2
        assert store.usage()['reserved_items'] == 16
    finally:
        await close_runtime(runtime)


async def test_head_changed_and_lost_claim_reply_preserve_credit_and_recovery(store):
    from marie.engine.llm_queue.store import StoreUnavailable

    runtime = runtime_for(store, [DispatchLane('pool', 'endpoint', quantum=3)])
    await runtime._configure()
    first = admit_request(store)
    second = admit_request(store)
    original = store.claim

    def lost(owner, attempt, **kwargs):
        reply = original(owner, attempt, **kwargs)
        assert reply.disposition == 'claimed'
        raise StoreUnavailable('lost response')

    store.claim = lost
    try:
        with pytest.raises(StoreUnavailable):
            await runtime._dispatch()
        assert store.ready_head('pool') == second.attempt_id
        assert store.usage()['reserved_items'] == 1
        assert runtime.scheduler.lane_metadata('pool')['deficit'] == 3
        store.release_owner(runtime.owner)
        replacement = store.acquire_owner('replacement', lease_ms=10000)
        assert (
            store.recover_claim(replacement, first.attempt_id).disposition == 'requeued'
        )
        assert store.usage()['reserved_items'] == 0
    finally:
        await close_runtime(runtime)


def test_maximum_store_cost_gets_credit_without_a_million_poll_cycles():
    from marie.engine.llm_queue.scheduler import DrrLaneConfig, DrrLaneScheduler

    scheduler = DrrLaneScheduler(
        queue_client=None, lanes=[DrrLaneConfig('a')], total_concurrent_dispatch=1
    )
    scheduler.sync_reservations({}, 0, {'a'})
    assert scheduler.select_metadata({'a': 1_000_000}) == 'a'
    scheduler.claimed('a', 1_000_000)
    assert scheduler.lane_metadata('a')['deficit'] == 0


@pytest.mark.parametrize('active', ['ready', 'unknown', 'empty'])
def test_persisted_endpoint_binding_rejects_intermediate_state(store, active):
    owner = store.acquire_owner('owner', lease_ms=10000)
    store.configure_route(
        owner,
        'pool',
        'endpoint',
        revision='r1',
        execution_limit=2,
        execution_bytes=10000,
    )
    if active != 'empty':
        req = admit_request(store)
        if active == 'unknown':
            store.claim(owner, req.attempt_id, pool_id='pool', claim_id='call')
            store.authorize_start(owner, req.attempt_id, claim_id='call')
            store.mark_unknown(
                owner,
                req.attempt_id,
                claim_id='call',
                execution_seq=1,
                category='read_timeout',
            )
    before = store.usage()
    reply = store.configure_endpoint(
        owner,
        'endpoint',
        execution_limit=2,
        execution_bytes=10000,
        transport_fingerprint='a' * 64,
    )
    assert reply.disposition == (
        'configured' if active == 'empty' else 'binding_migration_required'
    )
    assert store.usage() == before


def test_binding_is_immutable_across_owner_and_capacity_edits(store):
    owner = store.acquire_owner('owner', lease_ms=10000)
    assert (
        store.configure_endpoint(
            owner,
            'endpoint',
            execution_limit=2,
            execution_bytes=10000,
            transport_fingerprint='a' * 64,
        ).disposition
        == 'configured'
    )
    store.release_owner(owner)
    owner = store.acquire_owner('new-owner', lease_ms=10000)
    assert (
        store.configure_endpoint(
            owner,
            'endpoint',
            execution_limit=3,
            execution_bytes=20000,
            gate_open=False,
            transport_fingerprint='a' * 64,
        ).disposition
        == 'configured'
    )
    assert (
        store.configure_endpoint(
            owner,
            'endpoint',
            execution_limit=3,
            execution_bytes=20000,
            transport_fingerprint='b' * 64,
        ).disposition
        == 'binding_conflict'
    )
    assert 'transport_fingerprint' not in store.endpoint_status('endpoint')


async def test_producer_window_bounds_preparation_and_effective_image_cost(store):
    import threading

    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.config import LlmQueueConfig
    from marie.engine.llm_queue.producer import PreparedCalls, V3Producer
    from test_request_dispatcher import eventually

    runtime = runtime_for(store, [DispatchLane('pool', 'endpoint')])
    await runtime._configure()
    producer = V3Producer(
        config=LlmQueueConfig(
            enabled=True,
            queue_url=store.test_url,
            queue_contract_version='v3',
            fabric_group_id=store.keys.fabric_id,
            pool_id='pool',
            max_buffered_requests_per_pool=2,
        )
    )
    prepared = []

    def prepare(index):
        prepared.append(index)
        return CompletionCallParams(
            model='m',
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {'url': 'data:image/png;base64,AAAA'},
                        }
                    ]
                    * 2,
                }
            ],
        )

    cancelled = threading.Event()
    task = asyncio.create_task(
        asyncio.to_thread(
            producer.execute,
            calls=PreparedCalls(
                100, prepare, CompletionCallParams(model='m', messages=[])
            ),
            batch_request_id='large',
            batch_timeout=5,
            cancellation=cancelled,
        )
    )
    try:
        await eventually(lambda: store.usage()['active_items'] == 2)
        await asyncio.sleep(0.1)
        assert prepared == [0, 1]
        assert len(producer._pending) == 2
        head = store.metadata(store.ready_head('pool'))
        assert head.cost == 5
    finally:
        cancelled.set()
        await task
        await asyncio.to_thread(producer.close)
        await close_runtime(runtime)


async def test_refresh_disables_without_reroute_and_preserves_reservations(store):
    from test_request_dispatcher import eventually

    runtime = runtime_for(store, [DispatchLane('pool', 'endpoint')])
    await runtime._configure()
    request = admit_request(store)
    store.claim(runtime.owner, request.attempt_id, pool_id='pool', claim_id='call')
    store.authorize_start(runtime.owner, request.attempt_id, claim_id='call')
    store.mark_unknown(
        runtime.owner,
        request.attempt_id,
        claim_id='call',
        execution_seq=1,
        category='read_timeout',
    )
    before = store.usage()
    policy = dict(
        limits=store.limits,
        endpoints=list(runtime.endpoints.values()),
        lanes=[replace(runtime.lanes[0], enabled=False)],
        policy='drr',
        total_concurrent_dispatch=16,
    )
    runtime.policy_loader = lambda: policy
    try:
        await runtime._refresh_policy()
        await eventually(
            lambda: runtime._refresh_future is None or runtime._refresh_future.done()
        )
        await runtime._refresh_policy()
        assert store.resolve_route('pool') is None
        assert store.usage() == before
        assert runtime.lanes[0].enabled is False
        runtime._next_refresh = 0
        runtime.policy_loader = lambda: (_ for _ in ()).throw(
            ValueError('secret must not escape')
        )
        await runtime._refresh_policy()
        await eventually(
            lambda: runtime._refresh_future is None or runtime._refresh_future.done()
        )
        await runtime._refresh_policy()
        assert runtime._policy_paused
        assert store.usage() == before
        assert runtime._category == 'policy_refresh_unavailable'
    finally:
        runtime._refresh_worker.shutdown(wait=True)
        await close_runtime(runtime)


async def test_slow_refresh_has_one_read_and_does_not_block_maintenance_or_stop(store):
    import threading

    runtime = runtime_for(store, [DispatchLane('pool', 'endpoint')])
    await runtime._configure()
    calls = []
    release = threading.Event()

    def loader():
        calls.append(1)
        release.wait(2)
        raise ValueError('offline')

    runtime.policy_loader = loader
    runtime.refresh_timeout_seconds = 0.01
    try:
        await runtime._refresh_policy()
        await asyncio.sleep(0.03)
        for _ in range(4):
            await runtime._refresh_policy()
            await runtime._maintenance()
        assert calls == [1]
        assert runtime._policy_paused
        assert store.resolve_route('pool') is None
    finally:
        release.set()
        runtime._refresh_worker.shutdown(wait=True)
        await close_runtime(runtime)


@pytest.mark.parametrize(
    'transition',
    [
        'unsent_recover',
        'unknown_recover',
        'feedback_gap',
        'third_success',
        'defer',
        'finish',
    ],
)
async def test_probe_identity_tracks_actual_release_boundary(store, transition):
    runtime = runtime_for(store, [DispatchLane('pool', 'endpoint', execution_limit=4)])
    await runtime._configure()
    store.client.hset(
        store.keys.endpoint('endpoint'),
        mapping={
            'circuit': 'open',
            'next_probe': 0,
            'probe_successes': 2 if transition == 'third_success' else 0,
        },
    )
    probe = admit_request(store)
    queued = admit_request(store)
    assert (
        store.claim(
            runtime.owner, probe.attempt_id, pool_id='pool', claim_id='probe'
        ).disposition
        == 'claimed'
    )
    if transition != 'unsent_recover':
        store.authorize_start(runtime.owner, probe.attempt_id, claim_id='probe')
        store.record_endpoint_outcome(
            runtime.owner,
            probe.attempt_id,
            claim_id='probe',
            execution_seq=1,
            outcome='success' if transition == 'third_success' else 'unavailable',
            category='read_timeout',
            open_ms=1,
        )
    store.client.hset(store.keys.endpoint('endpoint'), 'next_probe', 0)
    if transition in {'third_success', 'feedback_gap', 'unknown_recover'}:
        assert (
            store.claim(
                runtime.owner, queued.attempt_id, pool_id='pool', claim_id='next'
            ).disposition
            == 'gated'
        )
    if transition in {'unsent_recover', 'unknown_recover', 'feedback_gap'}:
        store.release_owner(runtime.owner)
        runtime.owner = store.acquire_owner('nextowner', lease_ms=10000)
        recovered = store.recover_claim(runtime.owner, probe.attempt_id)
        assert recovered.disposition == (
            'requeued' if transition == 'unsent_recover' else 'outcome_unknown'
        )
    elif transition == 'defer':
        assert (
            store.defer(
                runtime.owner,
                probe.attempt_id,
                claim_id='probe',
                execution_seq=1,
                delay_ms=10,
                reason='retry',
                remote_settled=True,
            ).disposition
            == 'deferred'
        )
    else:
        assert (
            store.finish(
                runtime.owner,
                probe.attempt_id,
                claim_id='probe',
                execution_seq=1,
                result={'ok': True},
            ).disposition
            == 'finished'
        )
    retained = transition in {'unknown_recover', 'feedback_gap'}
    assert bool(store.endpoint_status('endpoint')['probe_claim']) == retained
    assert store.claim(
        runtime.owner, queued.attempt_id, pool_id='pool', claim_id='next'
    ).disposition == ('gated' if retained else 'claimed')
    await close_runtime(runtime)


async def test_concurrent_probe_claims_and_full_physical_capacity(store):
    from concurrent.futures import ThreadPoolExecutor

    runtime = runtime_for(
        store,
        [
            DispatchLane('a', 'endpoint', execution_limit=2),
            DispatchLane('b', 'endpoint', execution_limit=2),
        ],
    )
    await runtime._configure()
    store.client.hset(
        store.keys.endpoint('endpoint'), mapping={'circuit': 'open', 'next_probe': 0}
    )
    requests = [admit_request(store, pool_id=pool) for pool in ('a', 'b')]
    with ThreadPoolExecutor(max_workers=2) as workers:
        replies = list(
            workers.map(
                lambda req: store.claim(
                    runtime.owner,
                    req.attempt_id,
                    pool_id=req.pool_id,
                    claim_id=req.pool_id,
                ),
                requests,
            )
        )
    assert sorted(reply.disposition for reply in replies) == ['claimed', 'gated']
    assert store.usage()['reserved_items'] == 1
    await close_runtime(runtime)


def test_burst_limit_and_mixed_cost_credit_remain_bounded():
    from marie.engine.llm_queue.scheduler import DrrLaneConfig, DrrLaneScheduler

    scheduler = DrrLaneScheduler(
        queue_client=None,
        lanes=[
            DrrLaneConfig('a', quantum=8, max_burst_per_visit=2),
            DrrLaneConfig('b', quantum=1),
        ],
        total_concurrent_dispatch=16,
    )
    scheduler.sync_reservations({}, 0, {'a', 'b'})
    order = []
    for _ in range(6):
        pool = scheduler.select_metadata({'a': 1, 'b': 1})
        scheduler.claimed(pool, 1)
        order.append(pool)
    assert order == ['a', 'a', 'b', 'a', 'a', 'b']
    scheduler.sync_reservations({}, 0, set())
    assert scheduler.lane_metadata('a')['deficit'] == 0
    assert scheduler.lane_metadata('b')['deficit'] == 0


async def test_head_mismatch_read_failure_and_selected_sibling_survive(store):
    from marie.engine.llm_queue.store import StoreUnavailable

    runtime = runtime_for(store, [DispatchLane('pool', 'endpoint', quantum=4)])
    await runtime._configure()
    first = admit_request(store)
    second = admit_request(store)
    original_head = store.ready_head
    store.ready_head = lambda pool: second.attempt_id
    try:
        await runtime._dispatch()
        assert store.usage()['reserved_items'] == 0
        assert runtime.scheduler.lane_metadata('pool')['deficit'] == 4
        store.ready_head = original_head
        seen = []

        async def hold(*args):
            seen.append(args[0])
            await asyncio.Event().wait()

        runtime._execute = hold
        original_metadata = store.metadata
        calls = []

        def fail_followup(attempt):
            calls.append(attempt)
            if len(calls) == 2:
                raise StoreUnavailable('read unavailable')
            return original_metadata(attempt)

        store.metadata = fail_followup
        with pytest.raises(StoreUnavailable):
            await runtime._dispatch()
        await asyncio.sleep(0)
        assert seen == [first.attempt_id]
        assert store.usage()['reserved_items'] == 1
        assert original_head('pool') == second.attempt_id
    finally:
        await close_runtime(runtime)


async def test_saturated_shared_endpoint_does_not_hold_protection_against_healthy_endpoint(
    store,
):
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'blocked', 'https://blocked.example/v1', execution_limit=1
            ),
            RegisteredEndpoint(
                'healthy', 'https://healthy.example/v1', execution_limit=1
            ),
        ],
        lanes=[
            DispatchLane('old', 'blocked', execution_limit=1),
            DispatchLane('protected', 'blocked', execution_limit=1, min_concurrent=1),
            DispatchLane('healthy', 'healthy', execution_limit=1),
        ],
        total_concurrent_dispatch=2,
    )
    runtime.owner = store.acquire_owner('owner', lease_ms=10000)
    await runtime._configure()
    old = admit_request(store, pool_id='old', endpoint_id='blocked')
    store.claim(runtime.owner, old.attempt_id, pool_id='old', claim_id='old')
    store.authorize_start(runtime.owner, old.attempt_id, claim_id='old')
    store.mark_unknown(
        runtime.owner,
        old.attempt_id,
        claim_id='old',
        execution_seq=1,
        category='read_timeout',
    )
    admit_request(store, pool_id='protected', endpoint_id='blocked')
    healthy = admit_request(store, pool_id='healthy', endpoint_id='healthy')

    async def hold(*args):
        await asyncio.Event().wait()

    runtime._execute = hold
    try:
        await runtime._dispatch()
        assert store.metadata(healthy.attempt_id).state == 'claimed'
        assert store.usage()['reserved_items'] == 2
        assert store.route_status('protected')['reserved_items'] == 0
    finally:
        await close_runtime(runtime)


@pytest.mark.parametrize('retired', [False, True])
def test_target_alias_rejected_without_policy_writes_across_restart(store, retired):
    owner = store.acquire_owner('owner', lease_ms=10000)
    assert (
        store.configure_endpoint(
            owner,
            'original',
            execution_limit=2,
            execution_bytes=10000,
            transport_fingerprint='a' * 64,
            target_fingerprint='1' * 64,
        ).disposition
        == 'configured'
    )
    if retired:
        store.configure_endpoint(
            owner,
            'original',
            execution_limit=2,
            execution_bytes=10000,
            gate_open=False,
            transport_fingerprint='a' * 64,
            target_fingerprint='1' * 64,
        )
    store.release_owner(owner)
    owner = store.acquire_owner('replacement', lease_ms=10000)
    before = store.client.hgetall(store.keys.endpoint('original'))
    assert (
        store.configure_endpoint(
            owner,
            'alias',
            execution_limit=2,
            execution_bytes=10000,
            transport_fingerprint='b' * 64,
            target_fingerprint='1' * 64,
        ).disposition
        == 'target_conflict'
    )
    assert store.client.hgetall(store.keys.endpoint('original')) == before
    assert not store.client.exists(store.keys.endpoint('alias'))


def test_stale_or_incomplete_endpoint_registry_view_cannot_bind(store):
    owner = store.acquire_owner('owner', lease_ms=10000)
    store.configure_endpoint(
        owner,
        'original',
        execution_limit=2,
        execution_bytes=10000,
        transport_fingerprint='a' * 64,
        target_fingerprint='1' * 64,
    )
    reply = store._change(
        'endpoint',
        owner=owner,
        endpoint_id='alias',
        execution_limit=2,
        execution_bytes=10000,
        gate='open',
        transport_fingerprint='b' * 64,
        target_fingerprint='1' * 64,
        registry_ids=[],
    )
    assert reply.disposition == 'registry_changed'
    assert not store.client.exists(store.keys.endpoint('alias'))


def test_canonical_target_fingerprint_ignores_default_port_case_and_trailing_slash():
    first = RegisteredEndpoint('first', 'https://Example.com:443/v1/')
    second = RegisteredEndpoint('second', 'https://example.com/v1', api_key='changed')
    assert first.target_fingerprint() == second.target_fingerprint()
    assert first.transport_fingerprint() != second.transport_fingerprint()


def test_concurrent_target_registration_cannot_create_aliases(store):
    from concurrent.futures import ThreadPoolExecutor

    owner = store.acquire_owner('owner', lease_ms=10000)

    def register(identity):
        return store.configure_endpoint(
            owner,
            identity,
            execution_limit=2,
            execution_bytes=10000,
            transport_fingerprint=identity * 64,
            target_fingerprint='1' * 64,
        )

    with ThreadPoolExecutor(max_workers=2) as workers:
        replies = list(workers.map(register, ['a', 'b']))
    assert sum(reply.disposition == 'configured' for reply in replies) == 1
    assert all(
        reply.disposition in {'configured', 'registry_changed', 'target_conflict'}
        for reply in replies
    )
    assert store.client.scard(store.keys.prefix + 'endpoints') == 1


@pytest.mark.parametrize('state', ['ready', 'unknown'])
def test_new_target_does_not_bypass_unbound_intermediate_work(store, state):
    owner = store.acquire_owner('owner', lease_ms=10000)
    store.configure_route(
        owner,
        'pool',
        'unbound',
        revision='r1',
        execution_limit=2,
        execution_bytes=10000,
    )
    req = admit_request(store, endpoint_id='unbound')
    if state == 'unknown':
        store.claim(owner, req.attempt_id, pool_id='pool', claim_id='old')
        store.authorize_start(owner, req.attempt_id, claim_id='old')
        store.mark_unknown(
            owner,
            req.attempt_id,
            claim_id='old',
            execution_seq=1,
            category='read_timeout',
        )
    before = store.usage()
    assert (
        store.configure_endpoint(
            owner,
            'new',
            execution_limit=2,
            execution_bytes=10000,
            transport_fingerprint='a' * 64,
            target_fingerprint='1' * 64,
        ).disposition
        == 'binding_migration_required'
    )
    assert store.usage() == before
    assert not store.client.exists(store.keys.endpoint('new'))


@pytest.mark.parametrize('source', ['messages', 'extra_body', 'nested'])
@pytest.mark.parametrize(
    'metadata,expected',
    [
        ({}, 7),
        ({'image_count': 1}, 7),
        ({'image_count': 5}, 11),
        ({'estimated_cost_units': 1}, 7),
        ({'estimated_cost_units': 12}, 12),
    ],
)
async def test_prepared_cost_matches_effective_transport_images(
    source, metadata, expected
):
    import json

    import httpx
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient
    from marie.engine.llm_queue.scheduler import prepared_cost_units

    images = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/png;base64,{index}'},
                }
                for index in range(3)
            ],
        }
    ]
    call = CompletionCallParams(model='model', messages=images)
    if source != 'messages':
        call = replace(
            call,
            messages=[{'role': 'user', 'content': 'text'}],
            extra_body={'messages': images},
        )
    if source == 'nested':
        call = replace(
            call,
            extra_body={'messages': []},
            extra_create_kwargs={'extra_body': {'messages': images}},
        )
    observed = []

    async def respond(request):
        observed.append(json.loads(request.content)['messages'])
        return httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}}]})

    client = EndpointClient(RegisteredEndpoint('endpoint', 'https://example.test/v1'))
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        assert (await client.execute(call, timeout_seconds=1)).remote_settled
        assert observed == [images]
        assert prepared_cost_units(call, metadata) == expected
    finally:
        await client.close()


async def test_producer_effective_cost_is_immutable_through_safe_retry(store):
    import threading

    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.config import LlmQueueConfig
    from marie.engine.llm_queue.producer import V3Producer
    from test_request_dispatcher import eventually

    runtime = runtime_for(store, [DispatchLane('pool', 'endpoint')])
    await runtime._configure()
    producer = V3Producer(
        config=LlmQueueConfig(
            enabled=True,
            queue_url=store.test_url,
            queue_contract_version='v3',
            fabric_group_id=store.keys.fabric_id,
            pool_id='pool',
        )
    )
    call = CompletionCallParams(
        model='m',
        messages=[],
        extra_create_kwargs={
            'extra_body': {
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {'url': 'data:image/png;base64,AAAA'},
                            }
                        ]
                        * 3,
                    }
                ]
            }
        },
    )
    cancelled = threading.Event()
    task = asyncio.create_task(
        asyncio.to_thread(
            producer.execute,
            calls=[call],
            metadata={'estimated_cost_units': 1},
            batch_request_id='cost-retry',
            batch_timeout=5,
            cancellation=cancelled,
        )
    )
    try:
        await eventually(lambda: store.usage()['active_items'] == 1)
        attempt = store.ready_head('pool')
        assert store.metadata(attempt).cost == 7
        assert (
            store.claim(
                runtime.owner,
                attempt,
                pool_id='pool',
                claim_id='first',
                expected_cost=7,
            ).disposition
            == 'claimed'
        )
        started = store.authorize_start(runtime.owner, attempt, claim_id='first')
        assert (
            store.defer(
                runtime.owner,
                attempt,
                claim_id='first',
                execution_seq=started.execution_seq,
                delay_ms=0,
                reason='connect_refused',
                remote_settled=True,
            ).disposition
            == 'deferred'
        )
        assert store.metadata(attempt).cost == 7
        store.promote_due(runtime.owner)
        assert (
            store.claim(
                runtime.owner,
                attempt,
                pool_id='pool',
                claim_id='second',
                expected_cost=7,
            ).disposition
            == 'claimed'
        )
        assert store.metadata(attempt).cost == 7
    finally:
        cancelled.set()
        await task
        await asyncio.to_thread(producer.close)
        await close_runtime(runtime)
