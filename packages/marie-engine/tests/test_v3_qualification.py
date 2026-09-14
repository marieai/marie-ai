"""Final isolated admission, retention and contract cutover boundaries."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

from test_request_store import request_for, store


def test_atomic_partial_admission_cancel_expire_retention_and_v2_isolation(store):
    from marie.engine.llm_queue.queue_keys import request_queue_key

    legacy_key = request_queue_key('qualification-' + uuid4().hex)
    store.client.rpush(legacy_key, 'synthetic-v2-sentinel')
    try:
        producer = store.create_producer(lease_ms=10_000)
        requests = [request_for(store, producer=producer) for _ in range(6)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            replies = list(pool.map(store.admit, requests))
        assert sum(reply.disposition == 'admitted' for reply in replies) == 4
        assert sum(reply.disposition == 'backpressure' for reply in replies) == 2
        admitted = [
            req
            for req, reply in zip(requests, replies)
            if reply.disposition == 'admitted'
        ]
        first = admitted[0]
        store.cancel_or_expire(producer, first.attempt_id)
        assert store.read_result(producer, first.attempt_id) == {'error': 'cancelled'}
        assert store.read_result(producer, first.attempt_id) == {'error': 'cancelled'}
        assert (
            store.purge_terminal(store.test_owner, first.attempt_id).disposition
            == 'not_due'
        )
        expiring = replace(
            request_for(store, producer=producer),
            expires_at_ms=store.server_time_ms() + 50,
        )
        assert store.admit(expiring).disposition == 'admitted'
        time.sleep(0.06)
        store.cancel_or_expire(producer, expiring.attempt_id, expire=True)
        assert store.read_result(producer, expiring.attempt_id) == {'error': 'expired'}
        assert store.client.lrange(legacy_key, 0, -1) == ['synthetic-v2-sentinel']
        store.close_producer(producer)
        store.expire_producer(producer)
        assert store.usage()['active_items'] == store.usage()['records'] == 0
        # A V2 consumer still sees only the original version's queue after V3 drain.
        assert store.client.lpop(legacy_key) == 'synthetic-v2-sentinel'
    finally:
        store.client.delete(legacy_key)


def test_same_pool_and_attempt_are_isolated_across_fabrics(store):
    from marie.engine.llm_queue.store import RequestStore

    other = RequestStore(
        store.client.connection_pool.connection_kwargs.get('url', '')
        or 'redis://127.0.0.1:'
        + str(store.client.connection_pool.connection_kwargs['port'])
        + '/0',
        fabric_id='other-' + uuid4().hex,
        version='v3',
        limits=store.limits,
    )
    try:
        owner = other.acquire_owner('other', lease_ms=5000)
        other.configure_route(
            owner,
            'pool',
            'endpoint',
            revision='r1',
            execution_limit=2,
            execution_bytes=100_000,
        )
        attempt = uuid4().hex
        first = request_for(store, attempt_id=attempt)
        second = request_for(other, attempt_id=attempt)
        store.admit(first)
        other.admit(second)
        store.cancel_or_expire(first.producer_id, attempt)
        assert store.metadata(attempt).state == 'cancelled'
        assert other.metadata(attempt).state == 'ready'
        assert other.ready_head('pool') == attempt
        assert other.read_result(second.producer_id, attempt) is None
    finally:
        keys = list(other.client.scan_iter(match=other.keys.prefix + '*'))
        if keys:
            other.client.delete(*keys)
        other.close()
