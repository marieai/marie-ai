"""V3 dispatch against isolated real stores and loopback HTTP fixtures."""

import asyncio
import importlib.util
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import pytest


def test_runtime_boundary_exists():
    assert (
        importlib.util.find_spec('marie.engine.llm_queue.request_dispatcher')
        is not None
    )


def test_explicit_contract_and_fabric(monkeypatch):
    from marie.engine.llm_queue.config import LlmQueueConfig, resolve_fabric_id

    assert LlmQueueConfig(enabled=True).queue_contract_version == 'v2'
    monkeypatch.setenv('LLM_QUEUE_CONTRACT_VERSION', 'v3')
    assert LlmQueueConfig.from_env().queue_contract_version == 'v3'
    with pytest.raises(ValueError):
        LlmQueueConfig(enabled=True, queue_contract_version='v4')
    assert resolve_fabric_id('FABRIC', 'fabric', None) == 'fabric'
    for values in [(None, ''), ('one', 'two')]:
        with pytest.raises(ValueError):
            resolve_fabric_id(*values)


@pytest.fixture(params=['redis', 'valkey'])
def store(request):
    from marie.engine.llm_queue.store import RequestStore, StoreLimits

    config = os.environ.get('MARIE_LLM_QUEUE_TEST_STORES')
    if not config:
        pytest.skip('isolated store manifest required')
    url = json.loads(Path(config).read_text())[request.param]['url']
    store = RequestStore(
        url,
        fabric_id='runtime-' + uuid4().hex,
        version='v3',
        limits=StoreLimits(claim_lease_ms=150, remote_uncertainty_ms=200),
    )
    store.test_url = url
    yield store
    keys = list(store.client.scan_iter(match=store.keys.prefix + '*'))
    if keys:
        store.client.delete(*keys)
    store.close()


def test_shared_circuit_is_fenced_and_survives_config(store):
    from marie.engine.llm_queue.store import StaleOwner

    owner = store.acquire_owner('first', lease_ms=5000)
    for pool in ('one', 'two'):
        store.configure_route(
            owner,
            pool,
            'endpoint',
            revision='r1',
            execution_limit=2,
            execution_bytes=10000,
        )
    for index in range(3):
        req = admit_request(store, pool_id='one')
        claim = f'claim{index}'
        store.claim(owner, req.attempt_id, pool_id='one', claim_id=claim)
        started = store.authorize_start(owner, req.attempt_id, claim_id=claim)
        store.record_endpoint_outcome(
            owner,
            req.attempt_id,
            claim_id=claim,
            execution_seq=started.execution_seq,
            outcome='unavailable',
            category='connect_refused',
            open_ms=20,
        )
        store.finish(
            owner,
            req.attempt_id,
            claim_id=claim,
            execution_seq=started.execution_seq,
            result={'error': 'connect_refused'},
            success=False,
        )
    assert store.endpoint_status('endpoint')['circuit'] == 'open'
    store.configure_endpoint(
        owner, 'endpoint', execution_limit=2, execution_bytes=10000
    )
    assert store.endpoint_status('endpoint')['circuit'] == 'open'
    store.release_owner(owner)
    other = store.acquire_owner('second', lease_ms=5000)
    with pytest.raises(StaleOwner):
        store.record_endpoint_outcome(
            owner, req.attempt_id, claim_id=claim, execution_seq=1, outcome='success'
        )
    assert store.endpoint_status('endpoint')['circuit'] == 'open'
    assert other.generation > owner.generation


@pytest.mark.parametrize(
    'url',
    [
        'http://u:p@example.com',
        'http://example.com?a=b',
        'http://example.com/#x',
        'file:///tmp/x',
        'http://169.254.169.254',
        'http://127.0.0.1',
        'http://10.0.0.1',
    ],
)
def test_endpoint_rejects_untrusted_targets(url):
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint

    with pytest.raises(ValueError):
        RegisteredEndpoint('endpoint', url, api_key='sentinel')


async def test_real_refused_connection_is_retryable():
    import socket

    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    client = EndpointClient(
        RegisteredEndpoint(
            'refused', f'http://127.0.0.1:{port}/v1', allow_loopback=True
        )
    )
    result = await client.execute(
        CompletionCallParams(model='model', messages=[]), timeout_seconds=1
    )
    assert result.category == 'connect_refused'
    assert result.retryable and result.remote_settled
    assert not result.availability_success
    await client.close()


def admit_request(
    store,
    *,
    pool_id='pool',
    endpoint_id='endpoint',
    producer_id=None,
    call=None,
    deadline_ms=5000,
):
    from marie.engine.completion_contract import (
        CompletionCallParams,
        QueuedCompletionEnvelopeV3,
    )

    req = QueuedCompletionEnvelopeV3(
        contract_version='v3',
        fabric_group_id=store.keys.fabric_id,
        producer_id=producer_id or store.create_producer(lease_ms=10000),
        attempt_id=uuid4().hex,
        pool_id=pool_id,
        endpoint_id=endpoint_id,
        config_revision='r1',
        logical_batch_id='batch',
        logical_task_id='task',
        item_index=0,
        expires_at_ms=store.server_time_ms() + deadline_ms,
        call=call
        or CompletionCallParams(
            model='model', messages=[{'role': 'user', 'content': 'sentinel'}]
        ),
    )
    assert store.admit(req).disposition == 'admitted'
    return req


@pytest.fixture
async def http_endpoint():
    """An owned HTTP/1.1 keepalive fixture; records only test payloads."""
    received = []
    releases = {}
    active = set()
    connections = []

    async def handle(reader, writer):
        active.add(asyncio.current_task())
        connections.append(writer)
        try:
            while True:
                try:
                    header = await reader.readuntil(b'\r\n\r\n')
                except asyncio.IncompleteReadError:
                    return
                headers = dict(
                    line.split(': ', 1)
                    for line in header.decode().split('\r\n')[1:]
                    if ': ' in line
                )
                body = json.loads(
                    await reader.readexactly(int(headers.get('Content-Length', '0')))
                )
                received.append((body, headers))
                if body['model'] in releases:
                    await releases[body['model']].wait()
                response = json.dumps(
                    {
                        'choices': [{'message': {'content': body['model']}}],
                        'usage': {'total_tokens': 1},
                    }
                ).encode()
                writer.write(
                    b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '
                    + str(len(response)).encode()
                    + b'\r\n\r\n'
                    + response
                )
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            active.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    yield (
        f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1',
        received,
        releases,
        connections,
    )
    server.close()
    for task in list(active):
        task.cancel()
    await asyncio.gather(*tuple(active), return_exceptions=True)
    await server.wait_closed()


@pytest.fixture
async def http_response_endpoint():
    received = []
    responses = []
    active = set()

    async def handle(reader, writer):
        active.add(asyncio.current_task())
        try:
            while True:
                try:
                    header = await reader.readuntil(b'\r\n\r\n')
                except asyncio.IncompleteReadError:
                    return
                headers = dict(
                    line.split(': ', 1)
                    for line in header.decode().split('\r\n')[1:]
                    if ': ' in line
                )
                body = json.loads(
                    await reader.readexactly(int(headers.get('Content-Length', '0')))
                )
                received.append((body, headers, time.monotonic()))
                status, retry_after = responses.pop(0)
                response = (
                    json.dumps(
                        {
                            'choices': [{'message': {'content': body['model']}}],
                            'usage': {'total_tokens': 1},
                        }
                    ).encode()
                    if status == 200
                    else b''
                )
                response_headers = [
                    f'HTTP/1.1 {status} '
                    + ('OK' if status == 200 else 'Too Many Requests'),
                    f'Content-Length: {len(response)}',
                ]
                if retry_after is not None:
                    response_headers.append(f'Retry-After: {retry_after}')
                writer.write(
                    ('\r\n'.join(response_headers) + '\r\n\r\n').encode() + response
                )
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            active.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    yield (
        f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1',
        received,
        responses,
    )
    server.close()
    for task in list(active):
        task.cancel()
    await asyncio.gather(*tuple(active), return_exceptions=True)
    await server.wait_closed()


async def eventually(predicate, seconds=5):
    deadline = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate()


def dispatcher_for(store, url, *, retry_429=False, **kwargs):
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    return RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'endpoint',
                url,
                allow_loopback=True,
                api_key='endpoint-secret',
                retry_429=retry_429,
            )
        ],
        lanes=[DispatchLane('pool', 'endpoint')],
        poll_seconds=0.01,
        **kwargs,
    )


async def test_http_429_is_terminal_by_default(http_response_endpoint):
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    url, received, responses = http_response_endpoint
    responses.append((429, '0'))
    client = EndpointClient(RegisteredEndpoint('endpoint', url, allow_loopback=True))
    try:
        outcome = await client.execute(
            CompletionCallParams(model='model', messages=[]), timeout_seconds=1
        )
        assert outcome.category == 'provider_rejected'
        assert outcome.remote_settled and not outcome.retryable
        assert outcome.retry_after_ms == 0
        assert len(received) == 1
    finally:
        await client.close()


@pytest.mark.parametrize(
    ('retry_after', 'expected_delay_ms'),
    [
        ('0', 0),
        ('30', 30_000),
        ('Tue, 15 Sep 2026 12:00:30 GMT', 30_000),
    ],
)
async def test_http_429_retry_policy_accepts_bounded_delay(
    http_response_endpoint, monkeypatch, retry_after, expected_delay_ms
):
    from types import SimpleNamespace

    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue import endpoint as endpoint_module
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    if retry_after.startswith('Tue'):
        monkeypatch.setattr(
            endpoint_module, 'time', SimpleNamespace(time=lambda: 1_789_473_600)
        )
    url, received, responses = http_response_endpoint
    responses.append((429, retry_after))
    client = EndpointClient(
        RegisteredEndpoint('endpoint', url, allow_loopback=True, retry_429=True)
    )
    try:
        outcome = await client.execute(
            CompletionCallParams(model='model', messages=[]), timeout_seconds=1
        )
        assert outcome.category == 'rate_limited'
        assert outcome.retryable and outcome.remote_settled
        assert outcome.retry_after_ms == expected_delay_ms
        assert len(received) == 1
    finally:
        await client.close()


@pytest.mark.parametrize(
    'retry_after',
    [-1, 31, None, 'later', 'NaN', 'Infinity', '-Infinity'],
)
async def test_http_429_retry_policy_rejects_invalid_delay(
    http_response_endpoint, retry_after
):
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    url, received, responses = http_response_endpoint
    responses.append((429, None if retry_after is None else str(retry_after)))
    client = EndpointClient(
        RegisteredEndpoint('endpoint', url, allow_loopback=True, retry_429=True)
    )
    try:
        outcome = await client.execute(
            CompletionCallParams(model='model', messages=[]), timeout_seconds=1
        )
        assert outcome.category == 'provider_rejected'
        assert outcome.remote_settled and not outcome.retryable
        assert outcome.retry_after_ms == 0
        assert len(received) == 1
    finally:
        await client.close()


async def test_http_429_retry_preserves_logical_attempt_and_deadline(
    store, http_response_endpoint
):
    url, received, responses = http_response_endpoint
    responses.extend([(429, '0.03'), (200, None)])
    runtime = dispatcher_for(
        store,
        url,
        retry_429=True,
        retry_min_ms=5,
        retry_max_ms=5,
    )
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        req = admit_request(store, deadline_ms=2000)
        original_deadline = req.expires_at_ms
        await eventually(lambda: store.metadata(req.attempt_id).state == 'succeeded')
        await eventually(lambda: runtime.health()['counters'].get('completed') == 1)
        metadata = store.metadata(req.attempt_id)
        result = store.read_result(req.producer_id, req.attempt_id)

        assert metadata.attempt_id == req.attempt_id
        assert metadata.execution_seq == 2
        assert metadata.expires_at_ms == original_deadline
        assert len(received) == 2
        assert [entry[0]['model'] for entry in received] == ['model', 'model']
        assert 0.02 <= received[1][2] - received[0][2] < 0.5
        assert result['choices'][0]['message']['content'] == 'model'
        counters = runtime.health()['counters']
        assert counters['provider_starts'] == 2
        assert counters['retries'] == 1
        assert counters['completed'] == 1
        usage = store.usage()
        assert usage['records'] == 1
        assert usage['reserved_items'] == 0
        assert usage['reserved_bytes'] == 0
        assert store.route_status('pool')['reserved_items'] == 0
        assert store.route_status('pool')['reserved_bytes'] == 0
        assert store.endpoint_status('endpoint')['reserved_items'] == '0'
        assert store.endpoint_status('endpoint')['reserved_bytes'] == '0'
    finally:
        await runtime.stop()


async def test_http_429_retry_beyond_budget_expires_without_second_call(
    store, http_response_endpoint
):
    url, received, responses = http_response_endpoint
    responses.append((429, '1'))
    runtime = dispatcher_for(
        store,
        url,
        retry_429=True,
        retry_min_ms=5,
        retry_max_ms=5,
    )
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        began = time.monotonic()
        req = admit_request(store, deadline_ms=120)
        original_deadline = req.expires_at_ms
        await eventually(lambda: store.metadata(req.attempt_id).state == 'expired')
        elapsed = time.monotonic() - began
        metadata = store.metadata(req.attempt_id)
        result = store.read_result(req.producer_id, req.attempt_id)

        assert metadata.execution_seq == 1
        assert metadata.expires_at_ms == original_deadline
        assert len(received) == 1
        assert 0.08 <= elapsed < 0.75
        assert result == {'error': 'expired', 'category': 'rate_limited'}
        counters = runtime.health()['counters']
        assert counters['provider_starts'] == 1
        assert counters['retries'] == 1
        usage = store.usage()
        assert usage['active_items'] == 0
        assert usage['payload_bytes'] == 0
        assert usage['ready_ids'] == 0
        assert usage['reserved_items'] == 0
        assert usage['reserved_bytes'] == 0
        assert store.route_status('pool')['reserved_items'] == 0
        assert store.route_status('pool')['reserved_bytes'] == 0
        assert store.endpoint_status('endpoint')['reserved_items'] == '0'
        assert store.endpoint_status('endpoint')['reserved_bytes'] == '0'
    finally:
        await runtime.stop()


async def test_persistent_calls_commit_independently_and_preserve_images(
    store, http_endpoint
):
    from marie.engine.completion_contract import CompletionCallParams

    url, received, releases, connections = http_endpoint
    runtime = dispatcher_for(store, url)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        releases['slow'] = asyncio.Event()
        slow = admit_request(
            store, call=CompletionCallParams(model='slow', messages=[])
        )
        call = CompletionCallParams(
            model='fast',
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
            extra_body={'seed': 42},
        )
        fast = admit_request(store, call=call)
        await eventually(lambda: store.metadata(fast.attempt_id).state == 'succeeded')
        assert store.metadata(slow.attempt_id).state == 'executing'
        assert store.usage()['reserved_items'] == 1
        sent = next(body for body, _ in received if body['model'] == 'fast')
        assert sent['messages'] == call.messages and sent['seed'] == 42
        assert all(
            headers['Authorization'] == 'Bearer endpoint-secret'
            for _, headers in received
        )
        third = admit_request(store)
        await eventually(lambda: store.metadata(third.attempt_id).state == 'succeeded')
        assert len(connections) == 2
        releases['slow'].set()
        await eventually(lambda: store.metadata(slow.attempt_id).state == 'succeeded')
        assert store.read_result(fast.producer_id, fast.attempt_id)['usage'] == {
            'total_tokens': 1
        }
    finally:
        releases['slow'].set()
        await runtime.stop()
    await runtime.stop()


async def test_post_send_timeout_never_retries_or_releases_unknown(
    store, http_endpoint
):
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    url, received, releases, _ = http_endpoint
    releases['slow'] = asyncio.Event()
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'endpoint', url, allow_loopback=True, call_timeout_seconds=0.05
            )
        ],
        lanes=[DispatchLane('pool', 'endpoint')],
        poll_seconds=0.01,
    )
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        req = admit_request(store, call=CompletionCallParams(model='slow', messages=[]))
        await eventually(
            lambda: store.metadata(req.attempt_id).state == 'outcome_unknown'
        )
        await asyncio.sleep(
            0.3
        )  # Store default uncertainty is deliberately NOT remote proof.
        assert len(received) == 1
        assert store.usage()['reserved_items'] == 1
        store.cancel_or_expire(req.producer_id, req.attempt_id)
        assert store.usage()['reserved_items'] == 1
        releases['slow'].set()
    finally:
        await runtime.stop()


@pytest.mark.parametrize('winner', ['cancelled', 'producer_dead'])
async def test_cancellation_winner_settles_after_actual_completion(
    store, http_endpoint, winner
):
    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    original = store.finish
    req = None

    def terminal_wins(*args, **kwargs):
        if winner == 'cancelled':
            store.cancel_or_expire(req.producer_id, req.attempt_id)
        else:
            store.close_producer(req.producer_id)
        return original(*args, **kwargs)

    store.finish = terminal_wins
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        req = admit_request(store)
        await eventually(lambda: runtime.health()['counters'].get('completed') == 1)
        assert store.usage()['reserved_items'] == 0
        assert len(received) == 1
        if winner == 'cancelled':
            assert store.read_result(req.producer_id, req.attempt_id) == {
                'error': 'cancelled'
            }
    finally:
        await runtime.stop()


async def test_lost_finish_reply_reconciles_without_resend(store, http_endpoint):
    from marie.engine.llm_queue.store import StoreUnavailable

    url, received, _, _ = http_endpoint
    original = store.finish
    count = 0

    def lose_reply(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if count == 1:
            raise StoreUnavailable('injected lost actual reply')
        return result

    store.finish = lose_reply
    runtime = dispatcher_for(store, url)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        req = admit_request(store)
        await eventually(lambda: count >= 2)
        assert len(received) == 1
        assert store.metadata(req.attempt_id).state == 'succeeded'
        assert store.usage()['reserved_items'] == 0
    finally:
        await runtime.stop()


async def test_two_owners_and_lost_start_ack_never_send(store, http_endpoint):
    from marie.engine.llm_queue.store import RequestStore, StoreUnavailable

    url, received, _, _ = http_endpoint
    other_store = RequestStore(
        store.test_url,
        fabric_id=store.keys.fabric_id,
        version='v3',
        limits=store.limits,
    )
    first = dispatcher_for(store, url)
    second = dispatcher_for(other_store, url)
    original = store.authorize_start

    def lose_start(*args, **kwargs):
        original(*args, **kwargs)
        raise StoreUnavailable('lost start acknowledgement')

    store.authorize_start = lose_start
    await first.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        await second.start()
        req = admit_request(store)
        await eventually(
            lambda: store.metadata(req.attempt_id).state == 'outcome_unknown'
        )
        assert first.owner and second.owner is None
        assert received == []
        assert store.usage()['reserved_items'] == 1
        await first.stop()
        await eventually(lambda: second.owner is not None)
        assert second.owner.generation > 1
        assert received == []
        assert store.usage()['reserved_items'] == 1
    finally:
        await first.stop()
        await second.stop()


async def test_producer_death_drops_body_then_completion_settles(store, http_endpoint):
    from marie.engine.completion_contract import CompletionCallParams

    url, received, releases, _ = http_endpoint
    releases['slow'] = asyncio.Event()
    runtime = dispatcher_for(store, url)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        req = admit_request(store, call=CompletionCallParams(model='slow', messages=[]))
        await eventually(lambda: received)
        store.close_producer(req.producer_id)
        await eventually(lambda: store.metadata(req.attempt_id).state == 'abandoned')
        assert store.client.hget(store.keys.request(req.attempt_id), 'payload') is None
        assert store.usage()['reserved_items'] == 1
        await eventually(lambda: not runtime._tasks)
        assert store.usage()['reserved_items'] == 1
        releases['slow'].set()
    finally:
        releases['slow'].set()
        await runtime.stop()


async def test_forced_drain_keeps_remote_reservation_and_repeated_stop(
    store, http_endpoint
):
    from marie.engine.completion_contract import CompletionCallParams

    url, received, releases, _ = http_endpoint
    releases['slow'] = asyncio.Event()
    runtime = dispatcher_for(store, url, drain_seconds=0.03)
    await runtime.start()
    await eventually(lambda: store.resolve_route('pool'))
    req = admit_request(store, call=CompletionCallParams(model='slow', messages=[]))
    await eventually(lambda: received)
    await asyncio.gather(runtime.stop(), runtime.stop())
    assert runtime.health()['running'] is False
    assert runtime._clients == {} and runtime._tasks == {}
    assert store.usage()['reserved_items'] == 1
    assert not releases['slow'].is_set()
    await runtime.stop()


async def test_image_call_released_before_result_commit_retry(store, http_endpoint):
    import gc
    import weakref

    from marie.engine.llm_queue.endpoint import EndpointClient
    from marie.engine.llm_queue.store import StoreUnavailable

    url, received, _, _ = http_endpoint
    refs = []

    class ObservedClient(EndpointClient):
        async def execute(self, call, *, timeout_seconds):
            refs.append(weakref.ref(call))
            return await super().execute(call, timeout_seconds=timeout_seconds)

    original = store.finish
    failures = 0

    def unavailable(*args, **kwargs):
        nonlocal failures
        failures += 1
        gc.collect()
        assert refs and all(ref() is None for ref in refs)
        if failures < 3:
            raise StoreUnavailable('injected outage')
        return original(*args, **kwargs)

    store.finish = unavailable
    runtime = dispatcher_for(store, url, client_factory=ObservedClient)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        req = admit_request(store)
        await eventually(lambda: store.metadata(req.attempt_id).state == 'succeeded')
        assert len(received) == 1 and failures == 3
    finally:
        await runtime.stop()


async def test_refusal_opens_shared_circuit_healthy_endpoint_progresses(
    store, http_endpoint
):
    import socket

    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    url, received, _, _ = http_endpoint
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'endpoint',
                f'http://127.0.0.1:{port}/v1',
                allow_loopback=True,
                execution_limit=1,
            ),
            RegisteredEndpoint('healthy', url, allow_loopback=True),
        ],
        lanes=[
            DispatchLane('pool', 'endpoint'),
            DispatchLane('shared', 'endpoint'),
            DispatchLane('healthy', 'healthy'),
        ],
        poll_seconds=0.01,
        retry_min_ms=10,
        retry_max_ms=10,
        circuit_open_ms=1000,
    )
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('healthy'))
        req = admit_request(store)
        shared = admit_request(store, pool_id='shared')
        healthy = admit_request(store, pool_id='healthy', endpoint_id='healthy')
        await eventually(lambda: store.endpoint_status('endpoint')['circuit'] == 'open')
        await eventually(
            lambda: store.metadata(healthy.attempt_id).state == 'succeeded'
        )
        starts = runtime.health()['counters']['provider_starts']
        await asyncio.sleep(0.15)
        assert runtime.health()['counters']['provider_starts'] == starts
        assert len(received) == 1
        assert (
            store.metadata(req.attempt_id).execution_seq
            + store.metadata(shared.attempt_id).execution_seq
            == 3
        )
        assert store.usage()['reserved_items'] == 0
    finally:
        await runtime.stop()


async def test_queued_transport_logs_are_content_free(http_endpoint, caplog):
    import logging

    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    url, _, _, _ = http_endpoint
    client = EndpointClient(
        RegisteredEndpoint('alias', url, allow_loopback=True, api_key='KEY_SENTINEL')
    )
    try:
        with caplog.at_level(logging.DEBUG):
            await client.execute(
                CompletionCallParams(
                    model='model',
                    messages=[{'role': 'user', 'content': 'PROMPT_SENTINEL'}],
                ),
                timeout_seconds=1,
            )
        assert url not in caplog.text
        assert (
            'PROMPT_SENTINEL' not in caplog.text and 'KEY_SENTINEL' not in caplog.text
        )
    finally:
        await client.close()


async def test_shared_half_open_probe_requires_three_current_probe_successes(store):
    owner = store.acquire_owner('owner', lease_ms=5000)
    for pool in ('pool', 'shared'):
        store.configure_route(
            owner,
            pool,
            'endpoint',
            revision='r1',
            execution_limit=8,
            execution_bytes=100000,
        )
    late = admit_request(store)
    store.claim(owner, late.attempt_id, pool_id='pool', claim_id='late')
    store.authorize_start(owner, late.attempt_id, claim_id='late')
    for index in range(3):
        req = admit_request(store)
        claim = f'failure{index}'
        store.claim(owner, req.attempt_id, pool_id='pool', claim_id=claim)
        store.authorize_start(owner, req.attempt_id, claim_id=claim)
        store.record_endpoint_outcome(
            owner,
            req.attempt_id,
            claim_id=claim,
            execution_seq=1,
            outcome='unavailable',
            category='connect_refused',
            open_ms=20,
        )
        store.finish(owner, req.attempt_id, claim_id=claim, execution_seq=1, result={})
    store.record_endpoint_outcome(
        owner, late.attempt_id, claim_id='late', execution_seq=1, outcome='success'
    )
    store.finish(owner, late.attempt_id, claim_id='late', execution_seq=1, result={})
    assert store.endpoint_status('endpoint')['circuit'] == 'open'
    assert store.endpoint_status('endpoint')['probe_successes'] == '0'
    await asyncio.sleep(0.03)
    for index in range(3):
        req = admit_request(store)
        blocked = admit_request(store, pool_id='shared')
        claim = f'probe{index}'
        assert (
            store.claim(
                owner, req.attempt_id, pool_id='pool', claim_id=claim
            ).disposition
            == 'claimed'
        )
        assert (
            store.claim(
                owner, blocked.attempt_id, pool_id='shared', claim_id='blocked'
            ).disposition
            == 'gated'
        )
        store.authorize_start(owner, req.attempt_id, claim_id=claim)
        assert (
            store.record_endpoint_outcome(
                owner,
                req.attempt_id,
                claim_id=claim,
                execution_seq=1,
                outcome='success',
            ).disposition
            == 'recorded'
        )
        assert (
            store.record_endpoint_outcome(
                owner,
                req.attempt_id,
                claim_id=claim,
                execution_seq=1,
                outcome='success',
            ).disposition
            == 'existing'
        )
        store.finish(owner, req.attempt_id, claim_id=claim, execution_seq=1, result={})
        store.cancel_or_expire(blocked.producer_id, blocked.attempt_id)
        store.prune_ready(owner, 'shared')
        assert store.endpoint_status('endpoint')['circuit'] == (
            'closed' if index == 2 else 'half_open'
        )


def test_queue_stream_guard_precedes_all_producer_writes():
    from marie.engine.completion_contract import (
        CompletionCallParams,
        UnsupportedQueueStreaming,
    )
    from marie.engine.llm_queue.config import LlmQueueConfig
    from marie.engine.llm_queue.submitter import QueuedBatchExecutor

    class NoWrites:
        def __getattr__(self, name):
            raise AssertionError('queue accessed before streaming rejection')

    executor = QueuedBatchExecutor(
        queue_client=NoWrites(), config=LlmQueueConfig(enabled=True), logger=None
    )
    with pytest.raises(UnsupportedQueueStreaming):
        executor.execute(
            calls=[
                CompletionCallParams(model='model', messages=[]),
                CompletionCallParams(
                    model='model',
                    messages=[],
                    extra_create_kwargs={'extra_body': {'stream': True}},
                ),
            ],
            batch_request_id='batch',
            batch_timeout=1,
        )


def test_disabled_and_unknown_routes_fail_closed(store):
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    endpoint = RegisteredEndpoint('known', 'https://example.com/v1')
    with pytest.raises(ValueError):
        RequestDispatcher(
            store=store, endpoints=[endpoint], lanes=[DispatchLane('pool', 'unknown')]
        )
    owner = store.acquire_owner('owner', lease_ms=5000)
    store.configure_route(
        owner,
        'pool',
        'known',
        revision='r1',
        execution_limit=1,
        execution_bytes=10000,
        enabled=False,
    )
    assert store.resolve_route('pool') is None


async def test_gateway_selects_v3_with_effective_fabric_without_global_key(
    store, http_endpoint, monkeypatch
):
    from marie.engine.llm_queue.config import LlmQueueConfig

    from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
        GatewayLlmDispatchRuntime,
    )

    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('LLM_QUEUE_FABRIC_GROUP_ID', raising=False)
    url, _, _, _ = http_endpoint
    # Gateway owns a distinct default-limits fabric and its complete lifecycle.
    fabric = 'gateway-' + uuid4().hex
    config = LlmQueueConfig(
        enabled=True, queue_url=store.test_url, queue_contract_version='v3'
    )
    runtime = GatewayLlmDispatchRuntime(
        queue_config=config,
        config={
            'fabric_group_id': fabric,
            'llm_dispatch': {
                'endpoints': [
                    {'endpoint_id': 'endpoint', 'base_url': url, 'allow_loopback': True}
                ],
                'lanes': [{'pool_id': 'pool', 'endpoint_id': 'endpoint'}],
            },
        },
    )
    await runtime.start()
    owned_store = runtime._queue_client
    try:
        await eventually(lambda: owned_store.resolve_route('pool'))
        assert runtime.health()['contract_version'] == 'v3'
        from marie.engine.llm_queue.registry import dispatch_runtime_snapshot

        assert dispatch_runtime_snapshot()['contract_version'] == 'v3'
        assert runtime.health()['fabric_group_id'] == fabric
        req = admit_request(owned_store)
        await eventually(
            lambda: owned_store.metadata(req.attempt_id).state == 'succeeded'
        )
    finally:
        await runtime.stop()
        keys = list(owned_store.client.scan_iter(match=owned_store.keys.prefix + '*'))
        if keys:
            owned_store.client.delete(*keys)
        owned_store.close()
    with pytest.raises(ValueError):
        GatewayLlmDispatchRuntime(
            queue_config=config,
            fabric_group_id='explicit',
            config={'fabric_group_id': 'yaml'},
        )


async def test_actual_outage_recovers_through_three_shared_probes(store):
    import socket

    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'endpoint',
                f'http://127.0.0.1:{port}/v1',
                allow_loopback=True,
                execution_limit=2,
            )
        ],
        lanes=[DispatchLane('pool', 'endpoint'), DispatchLane('shared', 'endpoint')],
        poll_seconds=0.01,
        retry_min_ms=10,
        retry_max_ms=10,
        circuit_open_ms=100,
    )
    active = 0
    peak = 0
    calls = 0

    async def handle(reader, writer):
        nonlocal active, peak, calls
        try:
            header = await reader.readuntil(b'\r\n\r\n')
            size = next(
                int(line.split(b':')[1])
                for line in header.split(b'\r\n')
                if line.lower().startswith(b'content-length:')
            )
            await reader.readexactly(size)
            active += 1
            peak = max(peak, active)
            calls += 1
            await asyncio.sleep(0.03)
            writer.write(
                b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}'
            )
            await writer.drain()
            active -= 1
        finally:
            writer.close()
            await writer.wait_closed()

    server = None
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('shared'))
        requests = [
            admit_request(store, pool_id=pool) for pool in ('pool', 'shared', 'pool')
        ]
        await eventually(lambda: store.endpoint_status('endpoint')['circuit'] == 'open')
        await eventually(lambda: store.usage()['reserved_items'] == 0)
        failed_starts = runtime.health()['counters']['provider_starts']
        failed_probes = runtime.health()['counters'].get('probe_starts', 0)
        server = await asyncio.start_server(handle, '127.0.0.1', port)
        await eventually(
            lambda: all(
                store.metadata(req.attempt_id).state == 'succeeded' for req in requests
            )
        )
        assert store.endpoint_status('endpoint')['circuit'] == 'closed'
        assert calls == 3 and peak == 1
        assert runtime.health()['counters']['probe_starts'] == failed_probes + 3
        assert runtime.health()['counters']['provider_starts'] == failed_starts + 3
    finally:
        await runtime.stop()
        if server:
            server.close()
            await server.wait_closed()


async def test_store_read_backoff_preserves_head_and_stops_responsively(
    store, http_endpoint
):
    from marie.engine.llm_queue.store import StoreUnavailable

    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    original = store.ready_head
    attempts = []

    def unavailable(*args):
        attempts.append(time.monotonic())
        raise StoreUnavailable('controlled read outage')

    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        store.ready_head = unavailable
        req = admit_request(store)
        await asyncio.sleep(0.45)
        assert 1 <= len(attempts) <= 5
        assert original('pool') == req.attempt_id
        assert received == []
        began = asyncio.get_running_loop().time()
        await runtime.stop()
        assert asyncio.get_running_loop().time() - began < 1
        assert runtime.health()['counters'].get('malformed_requests_dropped', 0) == 0
    finally:
        store.ready_head = original
        await runtime.stop()


async def test_graceful_drain_keeps_lease_until_result_commit(store, http_endpoint):
    from marie.engine.completion_contract import CompletionCallParams

    url, received, releases, _ = http_endpoint
    releases['slow'] = asyncio.Event()
    runtime = dispatcher_for(store, url, owner_lease_ms=300, drain_seconds=2)
    await runtime.start()
    await eventually(lambda: store.resolve_route('pool'))
    req = admit_request(store, call=CompletionCallParams(model='slow', messages=[]))
    await eventually(lambda: received)
    stopping = asyncio.create_task(runtime.stop())
    await asyncio.sleep(0.4)
    assert runtime.health()['draining']
    assert store.client.get(store.keys.owner) == runtime.owner.value
    releases['slow'].set()
    await stopping
    assert store.metadata(req.attempt_id).state == 'succeeded'
    assert store.usage()['reserved_items'] == 0
    assert store.client.get(store.keys.owner) is None


async def test_dns_rebinding_is_rejected_before_body_transmission(
    http_endpoint, monkeypatch
):
    from urllib.parse import urlsplit

    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    url, received, _, _ = http_endpoint
    port = urlsplit(url).port
    loop = asyncio.get_running_loop()

    async def rebinding(*args, **kwargs):
        import socket

        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('169.254.169.254', port))]

    monkeypatch.setattr(loop, 'getaddrinfo', rebinding)
    client = EndpointClient(
        RegisteredEndpoint(
            'endpoint',
            f'http://registered.test:{port}/v1',
            allow_private=True,
            allow_loopback=True,
        )
    )
    try:
        outcome = await client.execute(
            CompletionCallParams(model='model', messages=[]), timeout_seconds=1
        )
        assert outcome.category == 'endpoint_policy'
        assert received == []
    finally:
        await client.close()


async def test_endpoint_credentials_and_transport_overrides(http_endpoint):
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    url, received, _, _ = http_endpoint
    first = EndpointClient(
        RegisteredEndpoint('first', url, api_key='FIRST', allow_loopback=True)
    )
    second = EndpointClient(
        RegisteredEndpoint('second', url, api_key='SECOND', allow_loopback=True)
    )
    try:
        for client in (first, second):
            await client.execute(
                CompletionCallParams(model='model', messages=[]), timeout_seconds=1
            )
        assert [headers['Authorization'] for _, headers in received] == [
            'Bearer FIRST',
            'Bearer SECOND',
        ]
        outcome = await first.execute(
            CompletionCallParams(
                model='model',
                messages=[],
                extra_create_kwargs={'extra_headers': {'Authorization': 'evil'}},
            ),
            timeout_seconds=1,
        )
        assert outcome.category == 'invalid_request'
        assert len(received) == 2
    finally:
        await first.close()
        await second.close()


@pytest.mark.parametrize('lifecycle', ['cancelled', 'dead', 'expired'])
async def test_restart_retires_old_route_and_reclaims_its_ready_limit(
    store, http_endpoint, lifecycle
):
    from dataclasses import replace

    from marie.engine.llm_queue.store import RequestStore

    url, _, _, _ = http_endpoint
    limited = RequestStore(
        store.test_url,
        fabric_id='retired-' + uuid4().hex,
        version='v3',
        limits=replace(store.limits, max_ready_ids=1),
    )
    runtime = dispatcher_for(limited, url)
    try:
        owner = limited.acquire_owner('old', lease_ms=5000)
        endpoint = runtime.endpoints['endpoint']
        limited.configure_endpoint(
            owner,
            'endpoint',
            execution_limit=endpoint.execution_limit,
            execution_bytes=endpoint.execution_bytes,
            transport_fingerprint=endpoint.transport_fingerprint(),
            target_fingerprint=endpoint.target_fingerprint(),
        )
        limited.configure_route(
            owner,
            'retired',
            'endpoint',
            revision='r1',
            execution_limit=1,
            execution_bytes=100000,
        )
        old = admit_request(
            limited,
            pool_id='retired',
            deadline_ms=50 if lifecycle == 'expired' else 5000,
        )
        if lifecycle == 'cancelled':
            limited.cancel_or_expire(old.producer_id, old.attempt_id)
        elif lifecycle == 'dead':
            limited.close_producer(old.producer_id)
        else:
            await asyncio.sleep(0.06)
        limited.release_owner(owner)
        await runtime.start()
        await eventually(lambda: limited.resolve_route('pool') is not None)
        await asyncio.sleep(0.1)
        assert limited.usage()['ready_ids'] == 0
        assert limited.resolve_route('retired') is None
        assert (
            limited.admit(
                replace(
                    old,
                    attempt_id=uuid4().hex,
                    producer_id=limited.create_producer(lease_ms=5000),
                    expires_at_ms=limited.server_time_ms() + 5000,
                )
            ).disposition
            == 'route_unavailable'
        )
        current = admit_request(limited)
        await eventually(
            lambda: limited.metadata(current.attempt_id).state == 'succeeded'
        )
    finally:
        await runtime.stop()
        keys = list(limited.client.scan_iter(match=limited.keys.prefix + '*'))
        if keys:
            limited.client.delete(*keys)
        limited.close()


@pytest.mark.parametrize('phase', ['maintenance', 'configuration'])
async def test_stop_does_not_start_remaining_lane_operations(
    store, http_endpoint, phase
):
    import threading

    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    url, _, _, _ = http_endpoint
    runtime = RequestDispatcher(
        store=store,
        endpoints=[RegisteredEndpoint('endpoint', url, allow_loopback=True)],
        lanes=[DispatchLane(f'lane{i}', 'endpoint') for i in range(20)],
        poll_seconds=0.01,
        drain_seconds=0.01,
    )
    operation = 'prune_ready' if phase == 'maintenance' else 'configure_route'
    original = getattr(store, operation)
    started = threading.Event()
    calls = []

    def delayed(*args, **kwargs):
        calls.append(time.monotonic())
        started.set()
        time.sleep(0.05)
        return original(*args, **kwargs)

    setattr(store, operation, delayed)
    await runtime.start()
    try:
        assert await asyncio.to_thread(started.wait, 2)
        stopped_at = time.monotonic()
        await runtime.stop()
        elapsed = time.monotonic() - stopped_at
        assert len(calls) == 1
        assert not any(began >= stopped_at for began in calls)
        assert elapsed < 0.25
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    'body',
    [
        b'{"number":1e400}',
        b'{"number":-1e400}',
        b'{"number":NaN}',
        b'{"number":Infinity}',
        b'{"broken":',
        b'\xff',
        b'DEEP',
    ],
)
async def test_fully_read_nonfinite_response_is_terminal_invalid_response(store, body):
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    calls = 0

    async def handle(reader, writer):
        nonlocal calls
        try:
            headers = await reader.readuntil(b'\r\n\r\n')
            size = next(
                int(line.split(b':')[1])
                for line in headers.split(b'\r\n')
                if line.lower().startswith(b'content-length:')
            )
            await reader.readexactly(size)
            calls += 1
            response_body = body
            if body == b'DEEP':
                depth = 10000
                response_body = (
                    b'{"nested":' + b'[' * depth + b'0' + b']' * depth + b'}'
                )
            writer.write(
                b'HTTP/1.1 200 OK\r\nContent-Length: '
                + str(len(response_body)).encode()
                + b'\r\nConnection: close\r\n\r\n'
                + response_body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    url = f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1'
    runtime = RequestDispatcher(
        store=store,
        endpoints=[RegisteredEndpoint('endpoint', url, allow_loopback=True)],
        lanes=[DispatchLane('pool', 'endpoint')],
        poll_seconds=0.01,
    )
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        req = admit_request(store)
        await eventually(lambda: calls == 1)
        await asyncio.sleep(0.2)
        assert store.read_result(req.producer_id, req.attempt_id) == {
            'error': 'invalid_response'
        }
        assert store.usage()['reserved_items'] == 0
        assert calls == 1
    finally:
        await runtime.stop()
        server.close()
        await server.wait_closed()


async def test_finish_serialization_failure_preserves_completed_evidence(
    store, http_endpoint
):
    url, received, _, _ = http_endpoint
    original = store.finish
    commits = 0

    def invalid_once(*args, **kwargs):
        nonlocal commits
        commits += 1
        if commits == 1:
            kwargs['result'] = {'number': float('inf')}
        return original(*args, **kwargs)

    store.finish = invalid_once
    runtime = dispatcher_for(store, url)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        req = admit_request(store)
        await eventually(lambda: commits >= 1)
        await asyncio.sleep(0.2)
        assert store.read_result(req.producer_id, req.attempt_id) == {
            'error': 'invalid_response'
        }
        assert store.usage()['reserved_items'] == 0
        assert len(received) == 1
    finally:
        await runtime.stop()


@pytest.mark.parametrize('operation', ['prune_ready', 'expire_producer', 'promote_due'])
def test_compound_maintenance_stops_between_real_transitions(store, operation):
    import threading

    owner = store.acquire_owner('owner', lease_ms=5000)
    store.configure_route(
        owner,
        'pool',
        'endpoint',
        revision='r1',
        execution_limit=8,
        execution_bytes=100000,
    )
    producer = store.create_producer(lease_ms=5000)
    requests = [admit_request(store, producer_id=producer) for _ in range(3)]
    for index, req in enumerate(requests):
        if operation == 'prune_ready':
            store.cancel_or_expire(producer, req.attempt_id)
        elif operation == 'promote_due':
            claim = f'claim{index}'
            store.claim(owner, req.attempt_id, pool_id='pool', claim_id=claim)
            store.defer(
                owner,
                req.attempt_id,
                claim_id=claim,
                execution_seq=0,
                delay_ms=0,
                reason='pool_pressure',
            )
    if operation == 'expire_producer':
        store.close_producer(producer)
    stopped = threading.Event()
    transition = {
        'prune_ready': 'prune',
        'expire_producer': 'discard',
        'promote_due': 'promote',
    }[operation]
    original = store._change
    transitions = []

    def stop_after_first(op, **kwargs):
        reply = original(op, **kwargs)
        if op == transition:
            transitions.append(kwargs['attempt_id'])
            stopped.set()
        return reply

    store._change = stop_after_first
    if operation == 'prune_ready':
        assert store.prune_ready(owner, 'pool', should_stop=stopped.is_set) == 1
        assert store.usage()['ready_ids'] == 2
    elif operation == 'expire_producer':
        assert len(store.expire_producer(producer, should_stop=stopped.is_set)) == 1
        assert store.client.scard(store.keys.members(producer)) == 2
    else:
        assert len(store.promote_due(owner, should_stop=stopped.is_set)) == 1
        assert store.usage()['ready_ids'] == 1
    assert len(transitions) == 1


async def test_retired_sent_reservation_keeps_original_endpoint_and_deadline(
    store, http_endpoint
):
    from marie.engine.llm_queue.store import StaleOwner

    url, received, _, _ = http_endpoint
    owner = store.acquire_owner('retired-owner', lease_ms=5000)
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint

    retired_endpoint = RegisteredEndpoint('old-endpoint', 'https://retired.example/v1')
    store.configure_endpoint(
        owner,
        'old-endpoint',
        execution_limit=2,
        execution_bytes=100000,
        transport_fingerprint=retired_endpoint.transport_fingerprint(),
        target_fingerprint=retired_endpoint.target_fingerprint(),
    )
    store.configure_route(
        owner,
        'retired',
        'old-endpoint',
        revision='r1',
        execution_limit=2,
        execution_bytes=100000,
    )
    sent = admit_request(store, pool_id='retired', endpoint_id='old-endpoint')
    store.claim(owner, sent.attempt_id, pool_id='retired', claim_id='sent')
    store.authorize_start(owner, sent.attempt_id, claim_id='sent')
    store.mark_unknown(
        owner,
        sent.attempt_id,
        claim_id='sent',
        execution_seq=1,
        category='read_timeout',
    )
    waiting = admit_request(
        store, pool_id='retired', endpoint_id='old-endpoint', deadline_ms=300
    )
    store.release_owner(owner)
    runtime = dispatcher_for(store, url)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        assert store.resolve_route('retired') is None
        assert store.metadata(waiting.attempt_id).expires_at_ms == waiting.expires_at_ms
        assert store.metadata(waiting.attempt_id).execution_seq == 0
        await eventually(lambda: store.metadata(waiting.attempt_id).state == 'expired')
        await eventually(lambda: store.usage()['ready_ids'] == 0)
        metadata = store.metadata(sent.attempt_id)
        assert (
            metadata.endpoint_id == 'old-endpoint'
            and metadata.state == 'outcome_unknown'
        )
        assert (
            store.client.hget(store.keys.endpoint('old-endpoint'), 'reserved_items')
            == '1'
        )
        assert store.usage()['reserved_items'] == 1
        current = admit_request(store)
        await eventually(
            lambda: store.metadata(current.attempt_id).state == 'succeeded'
        )
        assert len(received) == 1 and store.usage()['reserved_items'] == 1
        with pytest.raises(StaleOwner):
            store.disable_route(owner, 'pool')
    finally:
        await runtime.stop()


async def test_stop_interrupts_active_metadata_pass(store, http_endpoint):
    import threading

    from marie.engine.completion_contract import CompletionCallParams

    url, received, releases, _ = http_endpoint
    releases['slow'] = asyncio.Event()
    runtime = dispatcher_for(store, url, drain_seconds=0.01)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        for _ in range(3):
            admit_request(store, call=CompletionCallParams(model='slow', messages=[]))
        await eventually(lambda: len(received) == 3)
        started = threading.Event()
        original = store.metadata
        calls = []

        def delayed(*args):
            calls.append(time.monotonic())
            started.set()
            time.sleep(0.05)
            return original(*args)

        store.metadata = delayed
        assert await asyncio.to_thread(started.wait, 2)
        stopped_at = time.monotonic()
        await runtime.stop()
        assert len(calls) == 1 and all(began < stopped_at for began in calls)
        assert time.monotonic() - stopped_at < 0.25
        assert store.usage()['reserved_items'] == 3
    finally:
        await runtime.stop()


async def test_stop_interrupts_running_producer_cleanup_batch(store, http_endpoint):
    import threading

    url, _, _, _ = http_endpoint
    owner = store.acquire_owner('setup', lease_ms=5000)
    runtime = dispatcher_for(store, url, drain_seconds=0.01)
    endpoint = runtime.endpoints['endpoint']
    store.configure_endpoint(
        owner,
        'endpoint',
        execution_limit=endpoint.execution_limit,
        execution_bytes=endpoint.execution_bytes,
        transport_fingerprint=endpoint.transport_fingerprint(),
        target_fingerprint=endpoint.target_fingerprint(),
    )
    store.configure_route(
        owner,
        'pool',
        'endpoint',
        revision='r1',
        execution_limit=8,
        execution_bytes=100000,
    )
    producer = store.create_producer(lease_ms=5000)
    for _ in range(20):
        admit_request(store, producer_id=producer)
    store.close_producer(producer)
    store.release_owner(owner)
    original = store._change
    started = threading.Event()
    discards = []

    def delayed(op, **kwargs):
        if op == 'discard':
            discards.append(time.monotonic())
            started.set()
            time.sleep(0.05)
        return original(op, **kwargs)

    store._change = delayed
    await runtime.start()
    try:
        assert await asyncio.to_thread(started.wait, 2)
        stopped_at = time.monotonic()
        await runtime.stop()
        assert len(discards) == 1 and all(began < stopped_at for began in discards)
        assert time.monotonic() - stopped_at < 0.25
        assert store.client.scard(store.keys.members(producer)) == 19
    finally:
        await runtime.stop()


async def test_oversized_response_abort_keeps_unknown_reservation(store, http_endpoint):
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    url, received, _, _ = http_endpoint
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'endpoint', url, allow_loopback=True, max_response_bytes=8
            )
        ],
        lanes=[DispatchLane('pool', 'endpoint')],
        poll_seconds=0.01,
    )
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        req = admit_request(store)
        await eventually(
            lambda: store.metadata(req.attempt_id).state == 'outcome_unknown'
        )
        assert store.usage()['reserved_items'] == 1
        assert store.read_result(req.producer_id, req.attempt_id) is None
        assert len(received) == 1
    finally:
        await runtime.stop()
