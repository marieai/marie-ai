"""Authoritative V3 transitions on isolated real Redis and Valkey servers."""

import importlib.util
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest


def test_store_boundary_exists():
    assert importlib.util.find_spec('marie.engine.llm_queue.store') is not None


@pytest.fixture(params=['redis', 'valkey'])
def store(request):
    from marie.engine.llm_queue.store import RequestStore, StoreLimits

    config = os.environ.get('MARIE_LLM_QUEUE_TEST_STORES')
    if not config:
        pytest.skip('Set MARIE_LLM_QUEUE_TEST_STORES to an isolated test-stores.json')
    url = json.loads(Path(config).read_text())[request.param]['url']
    result = RequestStore(
        url,
        fabric_id='test-' + uuid4().hex,
        version='v3',
        limits=StoreLimits(
            max_active_items=4,
            max_payload_bytes=100_000,
            max_records=8,
            max_storage_bytes=100_000,
            max_ready_ids=8,
            result_allowance=1024,
            delivery_grace_ms=100,
            claim_lease_ms=100,
            remote_uncertainty_ms=150,
        ),
    )
    owner = result.acquire_owner('dispatcher', lease_ms=10_000)
    result.configure_route(
        owner,
        'pool',
        'endpoint',
        revision='r1',
        execution_limit=2,
        execution_bytes=100_000,
    )
    result.test_owner = owner
    yield result
    keys = list(result.client.scan_iter(match=result.keys.prefix + '*', count=100))
    if keys:
        result.client.delete(*keys)
    result.close()


def request_for(store, producer=None, **changes):
    from marie.engine.completion_contract import (
        CompletionCallParams,
        QueuedCompletionEnvelopeV3,
    )

    return QueuedCompletionEnvelopeV3(
        contract_version='v3',
        fabric_group_id=store.keys.fabric_id,
        producer_id=producer or store.create_producer(lease_ms=10_000),
        attempt_id=changes.pop('attempt_id', uuid4().hex),
        pool_id='pool',
        endpoint_id='endpoint',
        config_revision='r1',
        logical_batch_id='batch',
        logical_task_id='task',
        item_index=0,
        expires_at_ms=store.server_time_ms() + 5000,
        call=CompletionCallParams(
            model='model',
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {'url': 'data:image/png;base64,FIRST'},
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': 'data:image/png;base64,SECOND'},
                        },
                    ],
                }
            ],
        ),
        **changes,
    )


def run_request(store, req):
    assert store.admit(req).disposition == 'admitted'
    token = uuid4().hex
    claim = store.claim(
        store.test_owner, req.attempt_id, pool_id='pool', claim_id=token
    )
    assert claim.disposition == 'claimed'
    payload = store.fetch_payload(store.test_owner, req.attempt_id, claim_id=token)
    assert payload['call']['messages'][0]['content'][0]['image_url']['url'].endswith(
        'FIRST'
    )
    started = store.authorize_start(store.test_owner, req.attempt_id, claim_id=token)
    assert started.disposition == 'started'
    return token, started.execution_seq


def test_admit_is_atomic_bounded_and_digest_fenced(store):
    from marie.engine.llm_queue.store import AdmissionConflict

    req = request_for(store)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: store.admit(req).disposition, range(8)))
    assert results.count('admitted') == 1
    assert results.count('existing') == 7
    assert store.usage()['active_items'] == 1
    assert store.client.lrange(store.keys.ready('pool'), 0, -1) == [req.attempt_id]
    with pytest.raises(AdmissionConflict):
        store.admit(replace(req, producer_id=store.create_producer(lease_ms=10_000)))
    changed = replace(req, call=replace(req.call, model='other'))
    with pytest.raises(AdmissionConflict):
        store.admit(changed)
    assert (
        store.admit(replace(req, expires_at_ms=req.expires_at_ms + 1000)).disposition
        == 'existing'
    )
    assert store.metadata(req.attempt_id).expires_at_ms == req.expires_at_ms
    requests = [request_for(store, producer=req.producer_id) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda req: store.admit(req).disposition, requests))
    assert results.count('admitted') == 3
    assert results.count('backpressure') == 5
    assert store.usage()['active_items'] == 4


def test_terminal_commit_reread_and_original_deadline_retention(store):
    req = request_for(store)
    token, seq = run_request(store, req)
    assert (
        store.finish(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            result={'answer': 'yes'},
        ).disposition
        == 'finished'
    )
    assert not store.client.hexists(store.keys.request(req.attempt_id), 'payload')
    assert (
        store.finish(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            result={'answer': 'different'},
        ).disposition
        == 'existing'
    )
    assert store.read_result(req.producer_id, req.attempt_id) == {'answer': 'yes'}
    assert store.read_result(req.producer_id, req.attempt_id) == {'answer': 'yes'}
    time.sleep(0.12)
    assert (
        store.purge_terminal(store.test_owner, req.attempt_id).disposition == 'not_due'
    )
    assert store.admit(req).disposition == 'existing'
    assert store.usage()['active_items'] == store.usage()['reserved_items'] == 0
    assert store.usage()['records'] == 1


def test_owner_recovery_distinguishes_unsent_and_unknown(store):
    from marie.engine.llm_queue.store import StaleOwner

    req = request_for(store)
    store.admit(req)
    claim_id = uuid4().hex
    store.claim(store.test_owner, req.attempt_id, pool_id='pool', claim_id=claim_id)
    assert store.acquire_owner('standby', lease_ms=1000) is None
    store.client.pexpire(store.keys.owner, 1)
    time.sleep(0.01)
    owner2 = store.acquire_owner('standby', lease_ms=10_000)
    with pytest.raises(StaleOwner):
        store.authorize_start(store.test_owner, req.attempt_id, claim_id=claim_id)
    assert store.recover_claim(owner2, req.attempt_id).disposition == 'requeued'
    store.test_owner = owner2
    token = uuid4().hex
    store.claim(owner2, req.attempt_id, pool_id='pool', claim_id=token)
    seq = store.authorize_start(owner2, req.attempt_id, claim_id=token).execution_seq
    store.client.pexpire(store.keys.owner, 1)
    time.sleep(0.01)
    owner3 = store.acquire_owner('third', lease_ms=10_000)
    assert store.recover_claim(owner3, req.attempt_id).disposition == 'outcome_unknown'
    assert store.usage()['reserved_items'] == 1
    assert (
        store.defer(
            owner3,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            delay_ms=1,
            reason='unavailable',
            remote_settled=True,
        ).disposition
        == 'invalid_state'
    )
    time.sleep(0.16)
    with pytest.raises(ValueError):
        store.settle_remote(
            owner3,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            evidence='uncertainty_elapsed',
        )
    assert store.usage()['reserved_items'] == 1
    assert (
        store.settle_remote(
            owner3,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            evidence='remote_completed',
        ).disposition
        == 'settled'
    )
    assert store.usage()['reserved_items'] == 0


def test_producer_death_discards_all_states_and_retains_only_remote_reservation(store):
    from marie.engine.llm_queue.store import ProducerDead

    producer = store.create_producer(lease_ms=10_000)
    ready, delayed, executing, terminal = [
        request_for(store, producer=producer) for _ in range(4)
    ]
    token, seq = run_request(store, terminal)
    store.finish(
        store.test_owner,
        terminal.attempt_id,
        claim_id=token,
        execution_seq=seq,
        result={'secret': 'result'},
    )
    token, seq = run_request(store, executing)
    store.admit(delayed)
    deferred_token = uuid4().hex
    store.claim(
        store.test_owner, delayed.attempt_id, pool_id='pool', claim_id=deferred_token
    )
    store.defer(
        store.test_owner,
        delayed.attempt_id,
        claim_id=deferred_token,
        execution_seq=0,
        delay_ms=1000,
        reason='unavailable',
    )
    store.admit(ready)
    other = request_for(store)
    store.admit(other)
    store.close_producer(producer)
    assert not store.renew_producer(producer, lease_ms=10_000)
    for _ in range(5):
        store.expire_producer(producer, limit=2)
    for req in [ready, delayed, terminal]:
        assert not store.client.exists(store.keys.request(req.attempt_id))
    assert not store.client.hexists(store.keys.request(executing.attempt_id), 'payload')
    assert not store.client.hexists(store.keys.request(executing.attempt_id), 'result')
    assert store.usage()['reserved_items'] == 1
    assert store.usage()['active_items'] == 1
    with pytest.raises(ProducerDead):
        store.read_result(producer, terminal.attempt_id)
    assert store.finish(
        store.test_owner,
        executing.attempt_id,
        claim_id=token,
        execution_seq=seq,
        result={'late': 'discard'},
    ).disposition in {'producer_dead', 'missing'}
    assert (
        store.settle_remote(
            store.test_owner,
            executing.attempt_id,
            claim_id=token,
            execution_seq=seq,
            evidence='remote_completed',
        ).disposition
        == 'settled'
    )
    assert not store.client.exists(store.keys.request(executing.attempt_id))
    assert (
        store.claim(
            store.test_owner, ready.attempt_id, pool_id='pool', claim_id=uuid4().hex
        ).disposition
        == 'tombstone'
    )
    assert (
        store.claim(
            store.test_owner, other.attempt_id, pool_id='pool', claim_id=uuid4().hex
        ).disposition
        == 'claimed'
    )


def test_cancel_finish_race_and_oversized_result_are_terminal(store):
    req = request_for(store)
    token, seq = run_request(store, req)
    with ThreadPoolExecutor(max_workers=2) as executor:
        finish = executor.submit(
            store.finish,
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            result={'answer': 'ok'},
        )
        cancel = executor.submit(
            store.cancel_or_expire, req.producer_id, req.attempt_id
        )
        finish.result(), cancel.result()
    assert store.metadata(req.attempt_id).state in {'succeeded', 'cancelled'}
    assert not store.client.hexists(store.keys.request(req.attempt_id), 'payload')
    store.settle_remote(
        store.test_owner,
        req.attempt_id,
        claim_id=token,
        execution_seq=seq,
        evidence='remote_completed',
    )
    assert store.usage()['reserved_items'] == 0
    second = request_for(store)
    token, seq = run_request(store, second)
    result = store.finish(
        store.test_owner,
        second.attempt_id,
        claim_id=token,
        execution_seq=seq,
        result={'large': 'x' * 2000},
    )
    assert result.disposition == 'result_too_large'
    assert store.metadata(second.attempt_id).state == 'failed'
    assert store.read_result(second.producer_id, second.attempt_id) == {
        'error': 'result_too_large'
    }
    assert store.usage()['reserved_items'] == 0


def test_wrong_types_transport_failure_and_script_reload_fail_closed(
    store, monkeypatch
):
    from marie.engine.llm_queue.store import StoreUnavailable
    from valkey.exceptions import ConnectionError

    req = request_for(store)
    store.client.set(store.keys.ready('pool'), 'wrong-type')
    with pytest.raises(StoreUnavailable):
        store.admit(req)
    assert store.usage().get('records', 0) == 0
    assert not store.client.exists(store.keys.request(req.attempt_id))
    store.client.delete(store.keys.ready('pool'))
    store.client.script_flush()
    assert store.admit(req).disposition == 'admitted'
    with monkeypatch.context() as patch:
        patch.setattr(
            store.client,
            'evalsha',
            lambda *a, **k: (_ for _ in ()).throw(ConnectionError('test')),
        )
        with pytest.raises(StoreUnavailable):
            store.expire_producer(req.producer_id)
    assert store.metadata(req.attempt_id).state == 'ready'
    assert store.renew_producer(req.producer_id, lease_ms=10_000)


def test_expiry_and_cleanup_release_exactly_once(store):
    req = replace(request_for(store), expires_at_ms=store.server_time_ms() + 80)
    store.admit(req)
    time.sleep(0.09)
    assert (
        store.cancel_or_expire(req.producer_id, req.attempt_id, expire=True).state
        == 'expired'
    )
    time.sleep(0.11)
    assert (
        store.purge_terminal(store.test_owner, req.attempt_id).disposition == 'purged'
    )
    assert (
        store.purge_terminal(store.test_owner, req.attempt_id).disposition == 'missing'
    )
    store.claim(store.test_owner, req.attempt_id, pool_id='pool', claim_id=uuid4().hex)
    assert all(value == 0 for value in store.usage().values())


def test_contract_routes_and_streaming_rejected_before_writes(store):
    from marie.engine.completion_contract import UnsupportedQueueStreaming
    from marie.engine.llm_queue.store import RequestStore

    with pytest.raises(ValueError):
        RequestStore('redis://localhost', fabric_id='bad:{fabric}', version='v3')
    with pytest.raises(ValueError):
        RequestStore('redis://localhost', fabric_id='okay', version='v2')
    req = request_for(store)
    for call in [
        replace(req.call, stream=True),
        replace(req.call, extra_create_kwargs={'stream': True}),
        replace(req.call, extra_body={'stream': True}),
        replace(req.call, extra_create_kwargs={'extra_body': {'stream': True}}),
    ]:
        with pytest.raises(UnsupportedQueueStreaming):
            store.admit(replace(req, call=call))
    with pytest.raises(ValueError):
        store.admit(replace(req, pool_id='bad:pool'))
    assert (
        store.admit(replace(req, config_revision='stale')).disposition
        == 'route_unavailable'
    )
    assert store.usage().get('records', 0) == 0


def test_lost_replies_reconcile_without_duplicate_admission_or_execution(
    store, monkeypatch
):
    from marie.engine.llm_queue.store import StoreUnavailable
    from valkey.exceptions import ConnectionError

    def lose_reply(call):
        original = store.client.evalsha

        def once(*args, **kwargs):
            result = original(*args, **kwargs)
            raise ConnectionError('Injected lost response after real script completion')

        with monkeypatch.context() as patch:
            patch.setattr(store.client, 'evalsha', once)
            with pytest.raises(StoreUnavailable):
                call()

    req = request_for(store)
    # Load the script before injecting a lost EVALSHA response.
    store.renew_producer(req.producer_id, lease_ms=10_000)
    lose_reply(lambda: store.admit(req))
    assert store.admit(req).disposition == 'existing'
    assert store.usage()['active_items'] == 1
    claim_id = uuid4().hex
    lose_reply(
        lambda: store.claim(
            store.test_owner, req.attempt_id, pool_id='pool', claim_id=claim_id
        )
    )
    assert (
        store.claim(
            store.test_owner, req.attempt_id, pool_id='pool', claim_id=claim_id
        ).disposition
        == 'existing'
    )
    lose_reply(
        lambda: store.authorize_start(
            store.test_owner, req.attempt_id, claim_id=claim_id
        )
    )
    started = store.authorize_start(store.test_owner, req.attempt_id, claim_id=claim_id)
    assert started.disposition == 'already_executing'
    assert started.execution_seq == 1
    lose_reply(
        lambda: store.finish(
            store.test_owner,
            req.attempt_id,
            claim_id=claim_id,
            execution_seq=1,
            result={'answer': 'stored'},
        )
    )
    assert (
        store.finish(
            store.test_owner,
            req.attempt_id,
            claim_id=claim_id,
            execution_seq=1,
            result={'answer': 'stored'},
        ).disposition
        == 'existing'
    )
    assert store.read_result(req.producer_id, req.attempt_id) == {'answer': 'stored'}
    lose_reply(lambda: store.renew_producer(req.producer_id, lease_ms=10_000))
    assert store.renew_producer(req.producer_id, lease_ms=10_000)
    store.close_producer(req.producer_id)
    lose_reply(lambda: store.expire_producer(req.producer_id))
    store.expire_producer(req.producer_id)
    assert store.usage()['records'] == 0


def test_expired_lease_cannot_revive_and_gate_does_not_block_cleanup(store):
    producer = store.create_producer(lease_ms=30)
    req = request_for(store, producer=producer)
    store.admit(req)
    store.configure_route(
        store.test_owner,
        'pool',
        'endpoint',
        revision='r1',
        execution_limit=2,
        execution_bytes=100_000,
        gate_open=False,
    )
    time.sleep(0.04)
    assert not store.renew_producer(producer, lease_ms=10_000)
    store.expire_producer(producer)
    assert store.prune_ready(store.test_owner, 'pool') == 1
    assert all(value == 0 for value in store.usage().values())


def test_defer_promote_attempt_budget_and_capacity(store):
    req = request_for(store)
    token, seq = run_request(store, req)
    second = request_for(store)
    token2, seq2 = run_request(store, second)
    third = request_for(store)
    store.admit(third)
    assert (
        store.claim(
            store.test_owner, third.attempt_id, pool_id='pool', claim_id=uuid4().hex
        ).disposition
        == 'capacity'
    )
    assert (
        store.defer(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            delay_ms=0,
            reason='unavailable',
        ).disposition
        == 'unresolved'
    )
    assert store.usage()['reserved_items'] == 2
    assert (
        store.defer(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            delay_ms=0,
            reason='unavailable',
            remote_settled=True,
        ).disposition
        == 'deferred'
    )
    assert store.promote_due(store.test_owner)[0].disposition == 'promoted'
    store.cancel_or_expire(third.producer_id, third.attempt_id)
    store.prune_ready(store.test_owner, 'pool')
    for expected_seq in range(2, 6):
        token = uuid4().hex
        store.claim(store.test_owner, req.attempt_id, pool_id='pool', claim_id=token)
        assert (
            store.authorize_start(
                store.test_owner, req.attempt_id, claim_id=token
            ).execution_seq
            == expected_seq
        )
        result = store.defer(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=expected_seq,
            delay_ms=0,
            reason='unavailable',
            remote_settled=True,
        )
        if expected_seq == 5:
            assert result.disposition == 'attempts_exhausted'
        else:
            assert result.disposition == 'deferred'
            store.promote_due(store.test_owner)
    assert store.metadata(req.attempt_id).execution_seq == 5


def test_fabric_isolation_and_server_configuration_is_authoritative(store):
    from marie.engine.llm_queue.store import RequestStore

    other = RequestStore(
        store.client.connection_pool.connection_kwargs['host']
        and 'redis://127.0.0.1:'
        + str(store.client.connection_pool.connection_kwargs['port']),
        fabric_id=store.keys.fabric_id + '-other',
        version='v3',
        limits=store.limits,
    )
    try:
        producer = other.create_producer(lease_ms=10_000)
        owner = other.acquire_owner('dispatcher', lease_ms=10_000)
        other.configure_route(
            owner,
            'pool',
            'endpoint',
            revision='r1',
            execution_limit=2,
            execution_bytes=100_000,
        )
        req = request_for(store, attempt_id='same')
        store.admit(req)
        other_req = replace(
            req, fabric_group_id=other.keys.fabric_id, producer_id=producer
        )
        assert other.admit(other_req).disposition == 'admitted'
        assert other.metadata('same').producer_id != store.metadata('same').producer_id
        assert store.usage()['records'] == other.usage()['records'] == 1
    finally:
        keys = list(other.client.scan_iter(match=other.keys.prefix + '*'))
        if keys:
            other.client.delete(*keys)
        other.close()
    with pytest.raises(ValueError, match='limits'):
        RequestStore(
            'redis://127.0.0.1:'
            + str(store.client.connection_pool.connection_kwargs['port']),
            fabric_id=store.keys.fabric_id,
            version='v3',
            limits=replace(store.limits, result_allowance=2048),
        )


def test_lua_rejects_invalid_arguments_before_mutation(store):
    from valkey.exceptions import ResponseError

    req = request_for(store)
    store.admit(req)
    before = store.usage()
    original = store._script
    captured = {}

    def capture(*, keys, args):
        captured.update(keys=keys, args=args)
        return original(keys=keys, args=args)

    store._script = capture
    store.renew_owner(store.test_owner, lease_ms=10_000)
    store._script = original
    args = json.loads(captured['args'][0])
    args['lease_ms'] = -1
    with pytest.raises(ResponseError):
        original(keys=captured['keys'], args=[json.dumps(args)])
    assert store.usage() == before
    assert store.renew_producer(req.producer_id, lease_ms=10_000)


def test_pools_share_physical_endpoint_capacity_and_gate(store):
    owner = store.test_owner
    store.configure_route(
        owner,
        'pool',
        'endpoint',
        revision='r1',
        execution_limit=1,
        execution_bytes=100_000,
    )
    store.configure_route(
        owner,
        'pool2',
        'endpoint',
        revision='r1',
        execution_limit=1,
        execution_bytes=100_000,
    )
    store.configure_route(
        owner,
        'pool3',
        'other_endpoint',
        revision='r1',
        execution_limit=1,
        execution_bytes=100_000,
    )
    first = request_for(store)
    token, seq = run_request(store, first)
    second = replace(request_for(store), pool_id='pool2')
    third = replace(request_for(store), pool_id='pool3', endpoint_id='other_endpoint')
    store.admit(second)
    store.admit(third)
    assert (
        store.claim(
            owner, second.attempt_id, pool_id='pool2', claim_id=uuid4().hex
        ).disposition
        == 'capacity'
    )
    assert (
        store.claim(
            owner, third.attempt_id, pool_id='pool3', claim_id=uuid4().hex
        ).disposition
        == 'claimed'
    )
    store.finish(
        owner, first.attempt_id, claim_id=token, execution_seq=seq, result={'ok': True}
    )
    store.configure_endpoint(
        owner, 'endpoint', execution_limit=1, execution_bytes=100_000, gate_open=False
    )
    assert (
        store.claim(
            owner, second.attempt_id, pool_id='pool2', claim_id=uuid4().hex
        ).disposition
        == 'gated'
    )
    store.configure_endpoint(
        owner, 'endpoint', execution_limit=1, execution_bytes=100_000, gate_open=True
    )
    assert (
        store.claim(
            owner, second.attempt_id, pool_id='pool2', claim_id=uuid4().hex
        ).disposition
        == 'claimed'
    )


def test_metadata_and_result_reads_never_fetch_whole_hash_or_input(store, monkeypatch):
    req = request_for(store)
    token, seq = run_request(store, req)
    store.finish(
        store.test_owner,
        req.attempt_id,
        claim_id=token,
        execution_seq=seq,
        result={'ok': True},
    )
    original = store.client.hmget
    selected = []

    def track(key, *fields):
        selected.extend(fields)
        return original(key, *fields)

    monkeypatch.setattr(store.client, 'hmget', track)
    monkeypatch.setattr(
        store.client, 'hgetall', lambda *a: pytest.fail('HGETALL fetched private body')
    )
    assert store.metadata(req.attempt_id).state == 'succeeded'
    assert store.read_result(req.producer_id, req.attempt_id) == {'ok': True}
    assert 'payload' not in selected and 'result' not in selected


def test_record_version_and_corrupt_accounting_fail_before_writes(store):
    from marie.engine.llm_queue.store import StoreUnavailable

    req = request_for(store)
    token, seq = run_request(store, req)
    key = store.keys.request(req.attempt_id)
    store.client.hset(key, 'version', 'v2')
    with pytest.raises(StoreUnavailable):
        store.finish(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            result={'ok': True},
        )
    assert store.client.hexists(key, 'payload')
    store.client.hset(key, 'version', 'v3')
    store.client.hset(store.keys.prefix + 'usage', 'reserved_items', 0)
    with pytest.raises(StoreUnavailable):
        store.finish(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            result={'ok': True},
        )
    assert store.client.hexists(key, 'payload')


def test_deadline_finish_race_never_starts_or_commits_late_success(store):
    req = replace(request_for(store), expires_at_ms=store.server_time_ms() + 60)
    token, seq = run_request(store, req)
    time.sleep(0.07)
    assert (
        store.finish(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            result={'late': 'success'},
        ).disposition
        == 'expired'
    )
    assert (
        store.cancel_or_expire(req.producer_id, req.attempt_id, expire=True).state
        == 'expired'
    )
    assert store.usage()['reserved_items'] == 1
    assert store.read_result(req.producer_id, req.attempt_id) == {'error': 'expired'}
    store.settle_remote(
        store.test_owner,
        req.attempt_id,
        claim_id=token,
        execution_seq=seq,
        evidence='remote_completed',
    )
    assert store.usage()['reserved_items'] == 0


def test_closed_producer_claim_recovery_does_not_requeue(store):
    req = request_for(store)
    store.admit(req)
    store.claim(store.test_owner, req.attempt_id, pool_id='pool', claim_id=uuid4().hex)
    store.close_producer(req.producer_id)
    assert (
        store.recover_claim(store.test_owner, req.attempt_id).disposition
        == 'producer_dead'
    )
    assert store.metadata(req.attempt_id) is None
    assert store.ready_head('pool') is None
    assert store.usage()['reserved_items'] == 0


def test_result_record_limit_and_ready_tombstones_are_bounded(store):
    producer = store.create_producer(lease_ms=10_000)
    requests = []
    for _ in range(8):
        req = request_for(store, producer=producer)
        token, seq = run_request(store, req)
        store.finish(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            result={'ok': True},
        )
        requests.append(req)
    assert (
        store.admit(request_for(store, producer=producer)).disposition == 'backpressure'
    )
    assert store.usage()['records'] == 8
    assert store.usage()['active_items'] == 0
    assert store.usage()['storage_bytes'] <= store.limits.max_storage_bytes
    store.close_producer(producer)
    store.expire_producer(producer)
    assert store.usage()['records'] == 0
    producer = store.create_producer(lease_ms=10_000)
    for _ in range(8):
        req = request_for(store, producer=producer)
        store.admit(req)
        store.cancel_or_expire(producer, req.attempt_id)
        store.close_producer(producer)
        store.expire_producer(producer)
        producer = store.create_producer(lease_ms=10_000)
    assert store.usage()['ready_ids'] == 8
    req = request_for(store, producer=producer)
    assert store.admit(req).disposition == 'backpressure'
    assert store.prune_ready(store.test_owner, 'pool', limit=3) == 3
    assert store.usage()['ready_ids'] == 5
    assert store.admit(req).disposition == 'admitted'


def test_actual_noeviction_oom_rejects_admission_but_cleanup_reclaims(store):
    from marie.engine.llm_queue.store import StoreUnavailable

    req = request_for(store)
    store.admit(req)
    replacement = request_for(store)
    prior = store.client.config_get('maxmemory')['maxmemory']
    assert (
        store.client.config_get('maxmemory-policy')['maxmemory-policy'] == 'noeviction'
    )
    try:
        store.client.config_set('maxmemory', 1)
        with pytest.raises(StoreUnavailable):
            store.admit(replacement)
        assert store.metadata(req.attempt_id).state == 'ready'
        store.close_producer(req.producer_id)
        store.expire_producer(req.producer_id)
        assert store.prune_ready(store.test_owner, 'pool') == 1
        assert store.metadata(req.attempt_id) is None
        assert store.usage()['records'] == 0
    finally:
        store.client.config_set('maxmemory', prior)
        assert store.client.config_get('maxmemory')['maxmemory'] == prior
        assert (
            store.client.config_get('maxmemory-policy')['maxmemory-policy']
            == 'noeviction'
        )


def test_payload_byte_limit_is_atomic_and_inline_limit_rejects(store):
    req = request_for(store)
    req = replace(
        req,
        call=replace(req.call, messages=[{'role': 'user', 'content': 'x' * 60_000}]),
    )
    assert store.admit(req).disposition == 'admitted'
    other = replace(req, attempt_id=uuid4().hex)
    assert store.admit(other).disposition == 'backpressure'
    assert store.usage()['active_items'] == 1
    with pytest.raises(ValueError, match='payload limit'):
        store.admit(
            replace(
                other,
                call=replace(
                    req.call,
                    messages=[{'role': 'user', 'content': 'x' * (16 * 1024 * 1024)}],
                ),
            )
        )
    assert store.usage()['active_items'] == 1


def test_aggregate_result_allowance_and_empty_sessions_are_bounded(store):
    from marie.engine.llm_queue.store import RequestStore, TransitionRejected

    url = 'redis://127.0.0.1:' + str(
        store.client.connection_pool.connection_kwargs['port']
    )
    limited = RequestStore(
        url,
        fabric_id=store.keys.fabric_id + '-limited',
        version='v3',
        limits=replace(
            store.limits,
            max_storage_bytes=5120,
            max_producers=1,
            max_routes=1,
            max_endpoints=1,
        ),
    )
    try:
        owner = limited.acquire_owner('owner', lease_ms=10_000)
        limited.configure_route(
            owner,
            'pool',
            'endpoint',
            revision='r1',
            execution_limit=1,
            execution_bytes=100_000,
        )
        producer = limited.create_producer(lease_ms=10_000)
        with pytest.raises(TransitionRejected, match='backpressure'):
            limited.create_producer(lease_ms=10_000)
        req = request_for(limited, producer=producer)
        assert limited.admit(req).disposition == 'admitted'
        assert (
            limited.admit(replace(req, attempt_id=uuid4().hex)).disposition
            == 'backpressure'
        )
        assert limited.usage()['storage_bytes'] == 5120
        assert (
            limited.configure_route(
                owner,
                'another_pool',
                'endpoint',
                revision='r1',
                execution_limit=1,
                execution_bytes=100_000,
            ).disposition
            == 'backpressure'
        )
        assert (
            limited.configure_endpoint(
                owner, 'another_endpoint', execution_limit=1, execution_bytes=100_000
            ).disposition
            == 'backpressure'
        )
        limited.close_producer(producer)
        limited.expire_producer(producer)
        assert limited.create_producer(lease_ms=10_000) != producer
        assert limited.usage()['storage_bytes'] == 0
    finally:
        keys = list(limited.client.scan_iter(match=limited.keys.prefix + '*'))
        if keys:
            limited.client.delete(*keys)
        limited.close()


def test_valid_underscore_attempt_does_not_alias_control_operations(store):
    req = request_for(store, attempt_id='_')
    assert store.admit(req).disposition == 'admitted'
    store.renew_owner(store.test_owner, lease_ms=10_000)
    assert (
        store.claim(
            store.test_owner, req.attempt_id, pool_id='pool', claim_id='claim'
        ).disposition
        == 'claimed'
    )
    assert (
        store.authorize_start(
            store.test_owner, req.attempt_id, claim_id='claim'
        ).disposition
        == 'started'
    )


def test_unsent_claim_cannot_release_remote_capacity(store):
    owner = store.test_owner
    store.configure_endpoint(
        owner, 'endpoint', execution_limit=1, execution_bytes=100_000
    )
    first, second = request_for(store), request_for(store)
    store.admit(first)
    store.admit(second)
    assert (
        store.claim(owner, first.attempt_id, pool_id='pool', claim_id='c1').disposition
        == 'claimed'
    )
    for evidence in ('remote_completed', 'remote_cancelled'):
        assert (
            store.settle_remote(
                owner,
                first.attempt_id,
                claim_id='c1',
                execution_seq=0,
                evidence=evidence,
            ).disposition
            == 'invalid_state'
        )
    assert store.usage()['reserved_items'] == 1
    assert (
        store.claim(owner, second.attempt_id, pool_id='pool', claim_id='c2').disposition
        == 'capacity'
    )
    assert (
        store.authorize_start(owner, first.attempt_id, claim_id='c1').disposition
        == 'started'
    )
    assert (
        store.settle_remote(
            owner,
            first.attempt_id,
            claim_id='c1',
            execution_seq=1,
            evidence='remote_completed',
        ).disposition
        == 'settled'
    )
    assert (
        store.settle_remote(
            owner,
            first.attempt_id,
            claim_id='c1',
            execution_seq=1,
            evidence='remote_completed',
        ).disposition
        == 'existing'
    )
    assert (
        store.claim(owner, second.attempt_id, pool_id='pool', claim_id='c2').disposition
        == 'claimed'
    )


def test_start_requires_an_intact_claim_reservation(store):
    req = request_for(store)
    store.admit(req)
    store.claim(store.test_owner, req.attempt_id, pool_id='pool', claim_id='claim')
    store.client.hset(
        store.keys.request(req.attempt_id),
        mapping={'reserved': 0, 'reservation_bytes': 0},
    )
    assert (
        store.authorize_start(
            store.test_owner, req.attempt_id, claim_id='claim'
        ).disposition
        == 'invalid_state'
    )
    assert store.metadata(req.attempt_id).state == 'claimed'
    assert store.metadata(req.attempt_id).execution_seq == 0


def _fabric_dump(store):
    readers = {
        'string': store.client.get,
        'hash': store.client.hgetall,
        'list': lambda key: store.client.lrange(key, 0, -1),
        'set': store.client.smembers,
        'zset': lambda key: store.client.zrange(key, 0, -1, withscores=True),
    }
    return {
        key: readers[store.client.type(key)](key)
        for key in store.client.scan_iter(match=store.keys.prefix + '*')
    }


@pytest.mark.parametrize(
    'raw', ['2.0', '02', '+2', '2e0', ' 2', '9223372036854775807', '9007199254740992']
)
def test_invalid_integer_admission_does_not_partially_write(store, raw):
    from marie.engine.llm_queue.store import StoreUnavailable

    req = request_for(store)
    store.client.hset(store.keys.prefix + 'usage', 'active_items', raw)
    before = _fabric_dump(store)
    with pytest.raises(StoreUnavailable):
        store.admit(req)
    assert _fabric_dump(store) == before


@pytest.mark.parametrize('counter', ['usage', 'pool', 'endpoint', 'sequence'])
def test_invalid_mutated_integer_rejects_before_claim_or_start(store, counter):
    from marie.engine.llm_queue.store import StoreUnavailable

    req = request_for(store)
    store.admit(req)
    if counter == 'sequence':
        store.claim(store.test_owner, req.attempt_id, pool_id='pool', claim_id='claim')
        store.client.hset(store.keys.request(req.attempt_id), 'execution_seq', '0.0')
        action = lambda: store.authorize_start(
            store.test_owner, req.attempt_id, claim_id='claim'
        )
    else:
        key = {
            'usage': store.keys.prefix + 'usage',
            'pool': store.keys.route('pool'),
            'endpoint': store.keys.endpoint('endpoint'),
        }[counter]
        store.client.hset(key, 'reserved_items', '0.0')
        action = lambda: store.claim(
            store.test_owner, req.attempt_id, pool_id='pool', claim_id='claim'
        )
    before = _fabric_dump(store)
    with pytest.raises(StoreUnavailable):
        action()
    assert _fabric_dump(store) == before


def test_owner_generation_overflow_is_rejected_before_lease_write(store):
    from marie.engine.llm_queue.store import StoreUnavailable

    store.client.delete(store.keys.owner)
    store.client.set(store.keys.prefix + 'owner-generation', '9007199254740991')
    before = _fabric_dump(store)
    with pytest.raises(StoreUnavailable):
        store.acquire_owner('replacement', lease_ms=10_000)
    assert _fabric_dump(store) == before


def test_retry_claim_keeps_reservation_until_its_own_execution_starts(store):
    owner = store.test_owner
    store.configure_endpoint(
        owner, 'endpoint', execution_limit=1, execution_bytes=100_000
    )
    req = request_for(store)
    token, seq = run_request(store, req)
    assert (
        store.defer(
            owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            delay_ms=0,
            reason='unavailable',
            remote_settled=True,
        ).disposition
        == 'deferred'
    )
    store.promote_due(owner)
    store.claim(owner, req.attempt_id, pool_id='pool', claim_id='retry')
    second = request_for(store)
    store.admit(second)
    assert store.metadata(req.attempt_id).execution_seq == 1
    before = _fabric_dump(store)
    for evidence in ('remote_completed', 'remote_cancelled'):
        assert (
            store.settle_remote(
                owner,
                req.attempt_id,
                claim_id='retry',
                execution_seq=1,
                evidence=evidence,
            ).disposition
            == 'invalid_state'
        )
        assert _fabric_dump(store) == before
    assert (
        store.claim(
            owner, second.attempt_id, pool_id='pool', claim_id='second'
        ).disposition
        == 'capacity'
    )
    assert (
        store.authorize_start(owner, req.attempt_id, claim_id='retry').execution_seq
        == 2
    )


def test_payload_requires_matching_reserved_bytes(store):
    from marie.engine.llm_queue.store import TransitionRejected

    req = request_for(store)
    store.admit(req)
    store.claim(store.test_owner, req.attempt_id, pool_id='pool', claim_id='claim')
    store.client.hset(store.keys.request(req.attempt_id), 'reservation_bytes', 1)
    before = _fabric_dump(store)
    with pytest.raises(TransitionRejected, match='invalid_state'):
        store.fetch_payload(store.test_owner, req.attempt_id, claim_id='claim')
    assert (
        store.authorize_start(
            store.test_owner, req.attempt_id, claim_id='claim'
        ).disposition
        == 'invalid_state'
    )
    assert _fabric_dump(store) == before


def test_sent_settlement_requires_positive_uncertainty_bound(store):
    req = request_for(store)
    token, seq = run_request(store, req)
    store.client.hset(store.keys.request(req.attempt_id), 'uncertainty_until', 0)
    before = _fabric_dump(store)
    assert (
        store.settle_remote(
            store.test_owner,
            req.attempt_id,
            claim_id=token,
            execution_seq=seq,
            evidence='remote_completed',
        ).disposition
        == 'invalid_state'
    )
    assert _fabric_dump(store) == before


@pytest.mark.parametrize('boundary', ['python', 'lua'])
def test_elapsed_uncertainty_never_releases_reservation(store, boundary):
    from marie.engine.llm_queue.store import StoreUnavailable

    req = request_for(store)
    token, seq = run_request(store, req)
    sibling = request_for(store)
    run_request(store, sibling)
    store.client.hset(store.keys.request(req.attempt_id), 'uncertainty_until', 1)
    before = _fabric_dump(store)
    arguments = dict(claim_id=token, execution_seq=seq, evidence='uncertainty_elapsed')
    if boundary == 'python':
        with pytest.raises(ValueError):
            store.settle_remote(store.test_owner, req.attempt_id, **arguments)
    else:
        with pytest.raises(StoreUnavailable):
            store._change(
                'settle',
                owner=store.test_owner,
                attempt_id=req.attempt_id,
                cleanup=True,
                **arguments,
            )
    assert _fabric_dump(store) == before
    assert store.usage()['reserved_items'] == 2
