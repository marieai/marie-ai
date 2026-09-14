"""Original caller delivery against real stores and an owned HTTP provider."""

import asyncio
import logging
import threading
import time
from uuid import uuid4

import pytest
from marie.engine.batch_processor import BatchProcessor
from marie.engine.completion_contract import CompletionCallParams
from openai import AsyncOpenAI
from test_request_dispatcher import dispatcher_for, eventually, http_endpoint, store


def processor_for(store, **kwargs):
    return BatchProcessor(
        client=AsyncOpenAI(api_key='test', base_url='http://127.0.0.1:1/v1'),
        model_string='model',
        logger=logging.getLogger('producer-test'),
        queue_enabled=True,
        queue_url=store.test_url,
        queue_pool_id='pool',
        queue_contract_version='v3',
        queue_fabric_group_id=store.keys.fabric_id,
        batch_timeout=kwargs.pop('batch_timeout', 5),
        **kwargs,
    )


def calls(count):
    return [CompletionCallParams(model=f'item-{i}', messages=[]) for i in range(count)]


async def test_batch_processor_retains_success_then_recovers_nine_in_order(
    store, http_endpoint
):
    url, received, releases, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    delivered = []
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        for i in range(1, 9):
            releases[f'item-{i}'] = asyncio.Event()
        task = asyncio.create_task(
            asyncio.to_thread(
                processor.batch_generate_calls,
                calls=calls(9),
                request_id='original',
                on_result=lambda ident, output: delivered.append((ident, output)),
            )
        )
        await eventually(lambda: len(delivered) == 1)
        assert delivered == [('original_task_0', 'item-0')]
        assert not task.done()
        assert store.usage()['active_items'] == 8
        for release in releases.values():
            release.set()
        assert await task == [f'item-{i}' for i in range(9)]
        assert sorted(delivered) == [
            (f'original_task_{i}', f'item-{i}') for i in range(9)
        ]
        assert len(received) == 9
    finally:
        for release in releases.values():
            release.set()
        await asyncio.to_thread(processor.close)
        await runtime.stop()


async def test_same_logical_batch_has_independent_attempts(store, http_endpoint):
    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        result = await asyncio.gather(
            *[
                asyncio.to_thread(
                    processor.batch_generate_calls, calls=calls(1), request_id='same'
                )
                for _ in range(2)
            ]
        )
        assert result == [['item-0'], ['item-0']]
        assert len(received) == 2
        assert store.usage()['records'] == 2
    finally:
        await asyncio.to_thread(processor.close)
        await runtime.stop()


async def test_lost_admission_reply_reconciles_same_attempt(
    store, http_endpoint, monkeypatch
):
    from marie.engine.llm_queue.store import RequestStore, StoreUnavailable

    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    original = RequestStore.admit
    seen = []

    def lose_one(self, envelope):
        reply = original(self, envelope)
        seen.append(envelope.attempt_id)
        if len(seen) == 2:
            raise StoreUnavailable('lost reply')
        return reply

    monkeypatch.setattr(RequestStore, 'admit', lose_one)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        result = await asyncio.to_thread(processor.batch_generate_calls, calls=calls(3))
        assert result == ['item-0', 'item-1', 'item-2']
        assert len(received) == 3
        assert len(set(seen)) == 3
    finally:
        await asyncio.to_thread(processor.close)
        await runtime.stop()


async def test_callback_error_keeps_siblings_and_does_not_resend(store, http_endpoint):
    from marie.engine.exceptions import BatchExecutionError

    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    delivered = []

    def callback(ident, output):
        delivered.append(ident)
        if ident.endswith('_0'):
            raise OSError('write failed')

    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        with pytest.raises(BatchExecutionError) as raised:
            await asyncio.to_thread(
                processor.batch_generate_calls, calls=calls(2), on_result=callback
            )
        assert len(delivered) == len(set(delivered)) == 2
        assert len(received) == 2
        assert raised.value.successful_results[0].response == 'item-1'
        assert raised.value.primary_error.category == 'callback_failed'
    finally:
        await asyncio.to_thread(processor.close)
        await runtime.stop()


async def test_two_images_remain_one_ordered_request(store, http_endpoint):
    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    messages = [
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
    ]
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        assert await asyncio.to_thread(
            processor.batch_generate_calls,
            calls=[CompletionCallParams(model='images', messages=messages)],
        ) == ['images']
        assert len(received) == 1
        assert received[0][0]['messages'] == messages
    finally:
        await asyncio.to_thread(processor.close)
        await runtime.stop()


def test_producer_joins_custom_policy_without_initializing(store):
    from marie.engine.llm_queue.store import RequestStore

    joined = RequestStore.for_producer(store.test_url, fabric_id=store.keys.fabric_id)
    try:
        assert joined.limits.claim_lease_ms == 150
        assert joined.limits.remote_uncertainty_ms == 200
    finally:
        joined.close()


@pytest.mark.parametrize(
    'policy', [None, '{}', '{"max_active_items":true}', 'not json']
)
def test_producer_rejects_missing_or_malformed_policy_without_writes(store, policy):
    from marie.engine.llm_queue.queue_keys import QueueKeys
    from marie.engine.llm_queue.store import RequestStore

    keys = QueueKeys('uninitialized-' + uuid4().hex)
    if policy is not None:
        store.client.set(keys.prefix + 'limits', policy)
    before = set(store.client.scan_iter(match=keys.prefix + '*'))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            RequestStore.for_producer(store.test_url, fabric_id=keys.fabric_id)
        assert set(store.client.scan_iter(match=keys.prefix + '*')) == before
    finally:
        if before:
            store.client.delete(*before)


def engine_for(store, monkeypatch, *, multimodal=False, timeout=5):
    from marie.engine.openai_engine import OpenAIEngine

    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    return OpenAIEngine(
        model_name='model',
        is_multimodal=multimodal,
        base_url='http://127.0.0.1:1/v1',
        queue_enabled=True,
        queue_url=store.test_url,
        queue_pool_id='pool',
        queue_contract_version='v3',
        queue_fabric_group_id=store.keys.fabric_id,
        batch_timeout=timeout,
    )


@pytest.mark.parametrize('multimodal', [False, True])
async def test_async_cancellation_stops_future_admission_without_cancelling_sibling(
    store,
    http_endpoint,
    monkeypatch,
    multimodal,
):
    from marie.engine.llm_ops import LLMCall
    from marie.engine.llm_queue.store import RequestStore
    from marie.engine.multimodal_ops import MultimodalLLMCall
    from PIL import Image

    url, received, releases, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    engine = engine_for(store, monkeypatch, multimodal=multimodal)
    entered = threading.Event()
    release_admission = threading.Event()
    admit = RequestStore.admit
    attempts = []

    def blocked_admission(self, envelope):
        result = admit(self, envelope)
        attempts.append(envelope.attempt_id)
        if len(attempts) == 1:
            entered.set()
            release_admission.wait(3)
        return result

    monkeypatch.setattr(RequestStore, 'admit', blocked_admission)
    wrapper = MultimodalLLMCall(engine) if multimodal else LLMCall(engine)
    inputs = (
        [[Image.new('RGB', (2, 2)), 'prompt'] for _ in range(5)]
        if multimodal
        else ['prompt'] * 5
    )
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        releases['model'] = asyncio.Event()
        task = asyncio.create_task(wrapper.aforward(inputs))
        await eventually(entered.is_set)
        sibling = asyncio.create_task(
            asyncio.to_thread(engine.batch_generate, ['sibling'])
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_admission.set()
        await eventually(lambda: store.metadata(attempts[0]).state == 'cancelled')
        releases['model'].set()
        assert await sibling == ['model']
        assert len(attempts) == 2
        assert engine.batch_processor._queued_executor.producer_id is not None
    finally:
        release_admission.set()
        for release in releases.values():
            release.set()
        await asyncio.to_thread(engine.close)
        await runtime.stop()


async def test_deadline_includes_preparation_and_admits_nothing_after_budget(
    store, http_endpoint, monkeypatch
):
    from marie.engine.exceptions import BatchExecutionError

    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    engine = engine_for(store, monkeypatch, timeout=0.1)
    build = engine._build_completion_calls

    def slow_build(**kwargs):
        time.sleep(0.15)
        return build(**kwargs)

    monkeypatch.setattr(engine, '_build_completion_calls', slow_build)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        with pytest.raises(BatchExecutionError) as raised:
            await asyncio.to_thread(engine.batch_generate, ['slow'])
        assert raised.value.primary_error.category == 'deadline_exceeded'
        assert received == []
        assert store.usage()['records'] == 0
    finally:
        await asyncio.to_thread(engine.close)
        await runtime.stop()


async def test_empty_multimodal_call_has_no_session_or_admission(store, monkeypatch):
    from marie.engine.multimodal_ops import MultimodalLLMCall

    engine = engine_for(store, monkeypatch, multimodal=True)
    try:
        wrapper = MultimodalLLMCall(engine)
        assert await wrapper.aforward([]) == []
        assert wrapper.forward([]) == []
        assert store.usage()['records'] == 0
        assert engine.batch_processor._queued_executor is None
    finally:
        await asyncio.to_thread(engine.close)


async def test_close_wakes_caller_and_deletes_owned_unsent_and_terminal_results(
    store, http_endpoint
):
    from marie.engine.llm_queue.producer import ProducerClosed

    url, _, releases, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        releases['item-1'] = asyncio.Event()
        task = asyncio.create_task(
            asyncio.to_thread(processor.batch_generate_calls, calls=calls(2))
        )
        await eventually(
            lambda: store.usage()['active_items'] == 1 and store.usage()['records'] == 2
        )
        await asyncio.to_thread(processor.close)
        with pytest.raises(ProducerClosed):
            await task
        assert store.usage()['payload_bytes'] == 0
        keys = list(store.client.scan_iter(match=store.keys.prefix + 'request:*'))
        assert all(not store.client.hexists(key, 'result') for key in keys)
        assert all(not store.client.hexists(key, 'payload') for key in keys)
    finally:
        for release in releases.values():
            release.set()
        await runtime.stop()


async def test_prepared_calls_released_after_admission_while_results_wait(
    store, http_endpoint, monkeypatch
):
    import gc
    import weakref

    url, received, releases, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    engine = engine_for(store, monkeypatch)
    build = engine._build_completion_calls
    references = []

    def track_calls(**kwargs):
        built = build(**kwargs)
        references.extend(weakref.ref(call) for call in built)
        return built

    monkeypatch.setattr(engine, '_build_completion_calls', track_calls)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        releases['model'] = asyncio.Event()
        task = asyncio.create_task(
            asyncio.to_thread(engine.batch_generate, ['one', 'two', 'three'])
        )
        await eventually(lambda: len(received) == 3)
        gc.collect()
        assert all(reference() is None for reference in references)
        releases['model'].set()
        assert await task == ['model'] * 3
    finally:
        releases['model'].set()
        await asyncio.to_thread(engine.close)
        await runtime.stop()


async def test_expired_unknown_keeps_original_cause_and_successful_sibling(
    store, http_endpoint
):
    from marie.engine.exceptions import BatchExecutionError
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    url, received, releases, _ = http_endpoint
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'endpoint',
                url,
                allow_loopback=True,
                call_timeout_seconds=0.08,
            )
        ],
        lanes=[DispatchLane('pool', 'endpoint')],
        poll_seconds=0.01,
    )
    processor = processor_for(store, batch_timeout=0.4)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        releases['item-1'] = asyncio.Event()
        with pytest.raises(BatchExecutionError) as raised:
            await asyncio.to_thread(processor.batch_generate_calls, calls=calls(2))
        assert raised.value.primary_error.category == 'call_timeout'
        assert raised.value.successful_results[0].response == 'item-0'
        assert len(received) == 2
    finally:
        releases['item-1'].set()
        await asyncio.to_thread(processor.close)
        await runtime.stop()


async def test_cancel_pending_delivery_recovers_within_original_budget(
    store, http_endpoint, monkeypatch
):
    from marie.engine.exceptions import BatchExecutionError
    from marie.engine.llm_queue.store import RequestStore, StoreUnavailable

    url, _, releases, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    unavailable = threading.Event()
    unavailable.set()
    cancel = RequestStore.cancel_or_expire
    attempts = []
    admit = RequestStore.admit

    def record(self, envelope):
        reply = admit(self, envelope)
        attempts.append(envelope.attempt_id)
        return reply

    def unavailable_cancel(self, producer, attempt, **kwargs):
        if unavailable.is_set():
            raise StoreUnavailable('connection unavailable')
        return cancel(self, producer, attempt, **kwargs)

    monkeypatch.setattr(RequestStore, 'admit', record)
    monkeypatch.setattr(RequestStore, 'cancel_or_expire', unavailable_cancel)
    cancellation = threading.Event()
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        releases['item-0'] = asyncio.Event()
        task = asyncio.create_task(
            asyncio.to_thread(
                processor.batch_generate_calls,
                calls=calls(1),
                cancellation=cancellation,
            )
        )
        await eventually(lambda: len(attempts) == 1)
        cancellation.set()
        with pytest.raises(BatchExecutionError) as raised:
            await task
        assert raised.value.primary_error.category == 'cancellation_requested'
        assert not raised.value.primary_error.confirmed
        assert store.metadata(attempts[0]).state != 'cancelled'
        unavailable.clear()
        await eventually(lambda: store.metadata(attempts[0]).state == 'cancelled')
        assert processor._queued_executor._pending == {}
    finally:
        unavailable.clear()
        releases['item-0'].set()
        await asyncio.to_thread(processor.close)
        await runtime.stop()


async def test_heartbeat_and_polls_survive_transient_store_error_then_confirm_expiry(
    store, http_endpoint, monkeypatch
):
    from marie.engine.llm_queue.store import (
        ProducerDead,
        RequestStore,
        StoreUnavailable,
    )

    monkeypatch.setenv('LLM_QUEUE_PRODUCER_TTL_SECONDS', '1')
    monkeypatch.setenv('LLM_QUEUE_PRODUCER_REFRESH_INTERVAL_SECONDS', '.1')
    url, _, releases, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    renew = RequestStore.renew_producer
    blocked = threading.Event()

    def intermittent(self, producer_id, **kwargs):
        if blocked.is_set():
            raise StoreUnavailable('temporary')
        return renew(self, producer_id, **kwargs)

    monkeypatch.setattr(RequestStore, 'renew_producer', intermittent)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        releases['item-0'] = asyncio.Event()
        task = asyncio.create_task(
            asyncio.to_thread(processor.batch_generate_calls, calls=calls(1))
        )
        await eventually(lambda: store.usage()['active_items'] == 1)
        session = processor._queued_executor.producer_id
        blocked.set()
        await asyncio.sleep(0.2)
        blocked.clear()
        await asyncio.sleep(1.1)
        assert not task.done()
        assert store.client.exists(store.keys.alive(session))
        blocked.set()
        await asyncio.sleep(1.1)
        blocked.clear()
        with pytest.raises(ProducerDead):
            await task
        assert not store.client.exists(store.keys.alive(session))
        await eventually(lambda: store.usage()['payload_bytes'] == 0)
    finally:
        blocked.clear()
        releases['item-0'].set()
        await asyncio.to_thread(processor.close)
        await runtime.stop()


def test_child_does_not_inherit_parent_session_or_locked_mutex(store):
    import multiprocessing

    processor = processor_for(store, batch_timeout=0.1)
    executor = processor._get_queued_executor()
    executor._start(time.monotonic() + 1, threading.Event())
    parent_session = executor.producer_id
    context = multiprocessing.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    processor._queued_executor_lock.acquire()

    def child():
        other = processor._get_queued_executor()
        other._start(time.monotonic() + 1, threading.Event())
        sender.send(other.producer_id)
        other.close()

    worker = context.Process(target=child)
    try:
        worker.start()
        assert receiver.poll(3)
        child_session = receiver.recv()
        worker.join(3)
        assert worker.exitcode == 0
        assert child_session != parent_session
        assert store.client.exists(store.keys.alive(parent_session))
        assert not store.client.exists(store.keys.alive(child_session))
    finally:
        processor._queued_executor_lock.release()
        if worker.is_alive():
            worker.terminate()
            worker.join(3)
        receiver.close()
        sender.close()
        processor.close()


async def test_permanent_invalid_item_preserves_other_original_items(
    store, http_endpoint
):
    from marie.engine.exceptions import BatchExecutionError

    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    inputs = calls(3)
    inputs[1].extra_body = {'invalid': float('nan')}
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        with pytest.raises(BatchExecutionError) as raised:
            await asyncio.to_thread(
                processor.batch_generate_calls, calls=inputs, request_id='validation'
            )
        assert raised.value.primary_task_id == 'validation_task_1'
        assert raised.value.primary_error.category == 'invalid_request'
        assert [result.response for result in raised.value.successful_results] == [
            'item-0',
            'item-2',
        ]
        assert len(received) == 2
    finally:
        await asyncio.to_thread(processor.close)
        await runtime.stop()


def test_engine_cache_tracks_effective_v3_policy_and_normalized_fabric(
    store, monkeypatch
):
    from marie.extract.annotators import util

    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    monkeypatch.setenv('OPENAI_API_BASE', 'http://127.0.0.1:1/v1')
    monkeypatch.setenv('LLM_QUEUE_URL', store.test_url)
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'true')
    monkeypatch.setenv('LLM_QUEUE_CONTRACT_VERSION', 'v3')
    monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', store.keys.fabric_id)
    util.clear_engine_cache()
    try:
        original = util.route_llm_engine('model', False)
        monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', store.keys.fabric_id.upper())
        assert util.route_llm_engine('model', False) is original
        monkeypatch.setenv('LLM_BATCH_TIMEOUT_S', '12')
        timed = util.route_llm_engine('model', False)
        assert timed is not original
        assert timed.batch_processor.batch_timeout == 12
        monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'other-' + uuid4().hex)
        other = util.route_llm_engine('model', False)
        assert other is not timed
        monkeypatch.setenv('LLM_QUEUE_CONTRACT_VERSION', 'v2')
        legacy = util.route_llm_engine('model', False)
        assert legacy is not other
        assert not legacy.batch_processor.uses_v3_queue
    finally:
        util.close_all_engines()


@pytest.mark.parametrize('boundary', ['process_batch', 'executor'])
async def test_process_batch_original_writes_survive_provider_outage(
    store, monkeypatch, tmp_path, boundary
):
    import hashlib
    import json
    from collections import Counter

    import psutil
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )
    from PIL import Image
    from v3_qualification_evidence import record_evidence

    from marie.extract.annotators import util

    recovered = asyncio.Event()
    sent = Counter()
    settled = Counter()
    active = set()

    async def handle(reader, writer):
        active.add(asyncio.current_task())
        try:
            header = await reader.readuntil(b'\r\n\r\n')
            headers = dict(
                line.split(': ', 1)
                for line in header.decode().split('\r\n')
                if ': ' in line
            )
            body = json.loads(await reader.readexactly(int(headers['Content-Length'])))
            content = body['messages'][-1]['content']
            index = int(content[-1]['text'] if isinstance(content, list) else content)
            if boundary == 'executor':
                assert body['temperature'] == 0.25
                assert body['max_tokens'] == 512
                assert body['messages'][0]['content'] == 'fixture-system'
                assert content[0]['min_pixels'] == 784
                assert content[0]['max_pixels'] == 1568
            sent[index] += 1
            ok = index == 0 or recovered.is_set()
            if index == 0 and not recovered.is_set():
                server.close()
            if ok:
                settled[index] += 1
            payload = json.dumps(
                {'choices': [{'message': {'content': f'output-{index}'}}]}
                if ok
                else {'error': 'unavailable'}
            ).encode()
            status = b'200 OK' if ok else b'503 Service Unavailable'
            writer.write(
                b'HTTP/1.1 '
                + status
                + b'\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: '
                + str(len(payload)).encode()
                + b'\r\n\r\n'
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            active.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    url = f'http://127.0.0.1:{port}/v1'
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint('endpoint', url, allow_loopback=True, execution_limit=1)
        ],
        lanes=[DispatchLane('pool', 'endpoint', execution_limit=1)],
        poll_seconds=0.01,
        retry_min_ms=200,
        retry_max_ms=200,
        circuit_open_ms=200,
    )
    engine = engine_for(
        store, monkeypatch, multimodal=boundary == 'executor', timeout=15
    )
    original_write = util._write_single_result
    writes = []

    def record_write(**kwargs):
        writes.append(kwargs['task_id'])
        return original_write(**kwargs)

    monkeypatch.setattr(util, '_write_single_result', record_write)
    batch = [(Image.new('RGB', (2, 2)), str(i), f'page_{i}.png') for i in range(9)]
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        refs, phases = [], []
        if boundary == 'executor':
            from test_v3_image_lifetime import executor_request

            request, refs, phases = executor_request(tmp_path, monkeypatch, engine)
            tmp_path = tmp_path / 'agent-output' / 'test'
        else:
            request = util.process_batch(
                batch, engine, str(tmp_path), expect_output='none'
            )
        task = asyncio.create_task(request)
        await eventually(lambda: (tmp_path / 'page_0.md').exists())
        await eventually(
            lambda: (
                store.usage()['active_items'] == 8
                and store.endpoint_status('endpoint')['category'] == 'connect_refused'
                and store.endpoint_status('endpoint')['circuit'] == 'open'
            )
        )
        assert not task.done()
        assert writes[0].endswith('_task_0')
        assert (tmp_path / 'page_0.md').read_text() == 'output-0'
        outage = {
            'usage': store.usage(),
            'endpoint': store.endpoint_status('endpoint'),
            'writes': list(writes),
            'provider_counts': dict(sent),
            'rss': psutil.Process().memory_info().rss,
            'live_original_arrays': sum(ref() is not None for ref in refs),
        }
        assert all(ref() is None for ref in refs)
        assert not any(p['phase'] == 'annotation_assets' for p in phases)
        recovered.set()
        server = await asyncio.start_server(handle, '127.0.0.1', port)
        result = await task
        if boundary == 'executor':
            assert result['status'] == 'success', result
            assert len([p for p in phases if p['phase'] == 'annotation_assets']) == 1
        else:
            assert result == [f'output-{i}' for i in range(9)]
        assert len(writes) == len(set(writes)) == 9
        prefix = writes[0].rsplit('_task_', 1)[0]
        assert sorted(writes) == [f'{prefix}_task_{i}' for i in range(9)]
        assert settled == Counter({i: 1 for i in range(9)})
        assert [(tmp_path / f'page_{i}.md').read_text() for i in range(9)] == [
            f'output-{i}' for i in range(9)
        ]
        record_evidence(
            'nine-' + boundary,
            store,
            {
                'dimensions': [2525, 3293] if boundary == 'executor' else [2, 2],
                'job': 'same-job',
                'node': 'same-node',
                'phases': phases,
                'outage': outage,
                'ordered_outputs': [f'output-{i}' for i in range(9)],
                'writes': writes,
                'provider_counts': dict(sent),
                'final_usage': store.usage(),
                'output_sha256': {
                    f'page_{i}.md': hashlib.sha256(
                        (tmp_path / f'page_{i}.md').read_bytes()
                    ).hexdigest()
                    for i in range(9)
                },
                'job_event_boundary': 'actual executor request return and annotation asset callback; scheduler worker-history pilot pending',
            },
        )
    finally:
        recovered.set()
        await asyncio.to_thread(engine.close)
        await runtime.stop()
        server.close()
        await server.wait_closed()
        for task in list(active):
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)


async def test_slow_admission_never_starts_after_original_server_deadline(
    store, http_endpoint, monkeypatch
):
    from marie.engine.exceptions import BatchExecutionError
    from marie.engine.llm_queue.store import RequestStore

    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store, batch_timeout=0.12)
    original = RequestStore.admit
    deadlines = []

    def delay(self, envelope):
        deadlines.append(envelope.expires_at_ms)
        time.sleep(0.2)
        return original(self, envelope)

    monkeypatch.setattr(RequestStore, 'admit', delay)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        started = time.monotonic()
        with pytest.raises(BatchExecutionError):
            await asyncio.to_thread(processor.batch_generate_calls, calls=calls(3))
        elapsed = time.monotonic() - started
        assert 0.2 <= elapsed < 0.6
        assert len(deadlines) == 1
        assert deadlines[0] < store.server_time_ms()
        assert store.usage()['records'] == 0
        assert received == []
    finally:
        await asyncio.to_thread(processor.close)
        await runtime.stop()


@pytest.mark.parametrize(
    'overrides',
    [
        {'stream': True},
        {'extra_create_kwargs': {'stream': True}},
        {'extra_body': {'stream': True}},
        {'extra_create_kwargs': {'extra_body': {'stream': True}}},
    ],
)
def test_streaming_batch_rejected_before_any_producer_writes(store, overrides):
    from marie.engine.completion_contract import UnsupportedQueueStreaming

    processor = processor_for(store)
    before = set(store.client.scan_iter(match=store.keys.prefix + '*'))
    try:
        with pytest.raises(UnsupportedQueueStreaming):
            processor.batch_generate_calls(
                calls=[
                    calls(1)[0],
                    CompletionCallParams(model='stream', messages=[], **overrides),
                ]
            )
        assert set(store.client.scan_iter(match=store.keys.prefix + '*')) == before
        assert processor._queued_executor.producer_id is None
    finally:
        processor.close()


async def test_lost_terminal_read_delivers_callback_once(
    store, http_endpoint, monkeypatch
):
    from marie.engine.llm_queue.store import RequestStore, StoreUnavailable

    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    read = RequestStore.read_result
    reads = []
    writes = []

    def lost_read(self, producer_id, attempt_id):
        result = read(self, producer_id, attempt_id)
        if result is not None:
            reads.append(attempt_id)
            if len(reads) == 1:
                raise StoreUnavailable('lost result reply')
        return result

    monkeypatch.setattr(RequestStore, 'read_result', lost_read)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        assert await asyncio.to_thread(
            processor.batch_generate_calls,
            calls=calls(1),
            on_result=lambda *args: writes.append(args),
        ) == ['item-0']
        assert len(reads) == 2 and reads[0] == reads[1]
        assert len(writes) == len(received) == 1
    finally:
        await asyncio.to_thread(processor.close)
        await runtime.stop()


async def test_blocked_callback_does_not_stop_heartbeat_or_other_result_polling(
    store, http_endpoint, monkeypatch
):
    monkeypatch.setenv('LLM_QUEUE_PRODUCER_TTL_SECONDS', '1')
    monkeypatch.setenv('LLM_QUEUE_PRODUCER_REFRESH_INTERVAL_SECONDS', '.1')
    url, _, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    entered = threading.Event()
    released = threading.Event()

    def callback(*args):
        entered.set()
        released.wait(3)

    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        task = asyncio.create_task(
            asyncio.to_thread(
                processor.batch_generate_calls, calls=calls(2), on_result=callback
            )
        )
        await eventually(entered.is_set)
        await asyncio.sleep(1.1)
        producer = processor._queued_executor
        assert not producer._pending
        assert store.client.exists(store.keys.alive(producer.producer_id))
        released.set()
        assert await task == ['item-0', 'item-1']
    finally:
        released.set()
        await asyncio.to_thread(processor.close)
        await runtime.stop()


async def test_empty_process_batch_and_text_wrapper_have_no_admission(
    store, monkeypatch, tmp_path
):
    from marie.engine.llm_ops import LLMCall

    from marie.extract.annotators.util import process_batch

    engine = engine_for(store, monkeypatch)
    try:
        assert (
            await process_batch([], engine, str(tmp_path), expect_output='none') == []
        )
        assert await LLMCall(engine).aforward([]) == []
        assert engine.batch_processor._queued_executor is None
    finally:
        await asyncio.to_thread(engine.close)


async def test_backpressure_uses_original_deadline_without_admitting_later_items(
    store, http_endpoint
):
    from dataclasses import replace

    from marie.engine.exceptions import BatchExecutionError
    from marie.engine.llm_queue.store import RequestStore

    limited = RequestStore(
        store.test_url,
        fabric_id='limited-' + uuid4().hex,
        version='v3',
        limits=replace(store.limits, max_active_items=1),
    )
    limited.test_url = store.test_url
    url, received, releases, _ = http_endpoint
    runtime = dispatcher_for(limited, url)
    processor = processor_for(limited, batch_timeout=0.2)
    await runtime.start()
    try:
        await eventually(lambda: limited.resolve_route('pool'))
        releases['item-0'] = asyncio.Event()
        started = time.monotonic()
        with pytest.raises(BatchExecutionError):
            await asyncio.to_thread(processor.batch_generate_calls, calls=calls(3))
        assert time.monotonic() - started < 0.5
        assert limited.usage()['records'] == 1
        assert len(received) == 1
    finally:
        releases['item-0'].set()
        await asyncio.to_thread(processor.close)
        await runtime.stop()
        keys = list(limited.client.scan_iter(match=limited.keys.prefix + '*'))
        if keys:
            limited.client.delete(*keys)
        limited.close()


def test_producer_capacity_waits_only_within_original_budget(store):
    from dataclasses import replace

    from marie.engine.exceptions import BatchExecutionError
    from marie.engine.llm_queue.store import RequestStore

    limited = RequestStore(
        store.test_url,
        fabric_id='sessions-' + uuid4().hex,
        version='v3',
        limits=replace(store.limits, max_producers=1),
    )
    limited.test_url = store.test_url
    limited.create_producer(lease_ms=5000)
    processor = processor_for(limited, batch_timeout=0.12)
    try:
        started = time.monotonic()
        with pytest.raises(BatchExecutionError) as raised:
            processor.batch_generate_calls(calls=calls(1))
        assert 0.1 <= time.monotonic() - started < 0.6
        assert raised.value.primary_error.category == 'store_unavailable'
        assert limited.usage()['records'] == 0
    finally:
        processor.close()
        keys = list(limited.client.scan_iter(match=limited.keys.prefix + '*'))
        if keys:
            limited.client.delete(*keys)
        limited.close()


@pytest.mark.parametrize('stage', ['join', 'create'])
def test_close_waits_for_inflight_startup_and_reclaims_late_resources(
    store, monkeypatch, stage
):
    from concurrent.futures import ThreadPoolExecutor

    from marie.engine.llm_queue.producer import ProducerClosed
    from marie.engine.llm_queue.store import RequestStore

    entered = threading.Event()
    release = threading.Event()
    joined = []
    sessions = []
    original_join = RequestStore.for_producer
    original_create = RequestStore.create_producer

    def join(cls, *args, **kwargs):
        client = original_join(*args, **kwargs)
        joined.append(client)
        if stage == 'join':
            entered.set()
            assert release.wait(2)
        return client

    def create(self, **kwargs):
        identity = original_create(self, **kwargs)
        sessions.append(identity)
        if stage == 'create':
            entered.set()
            assert release.wait(2)
        return identity

    monkeypatch.setattr(RequestStore, 'for_producer', classmethod(join))
    monkeypatch.setattr(RequestStore, 'create_producer', create)
    processor = processor_for(store)
    producer = processor._get_queued_executor()
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            caller = workers.submit(processor.batch_generate_calls, calls=calls(1))
            assert entered.wait(2)
            shutdown = workers.submit(producer.close)
            until = time.monotonic() + 1
            while not producer._stop.is_set() and time.monotonic() < until:
                time.sleep(0.001)
            assert producer._stop.is_set()
            release.set()
            shutdown.result(timeout=2)
            assert producer._closed
            with pytest.raises(ProducerClosed):
                caller.result(timeout=2)
        assert len(joined) == 1
        connections = joined[0].client.connection_pool._available_connections
        assert connections and all(
            connection._sock is None for connection in connections
        )
        assert all(
            not store.client.exists(store.keys.alive(identity)) for identity in sessions
        )
        assert store.usage()['records'] == 0
        assert not any(thread.is_alive() for thread in producer._threads)
    finally:
        release.set()
        processor.close()
        for identity in sessions:
            store.close_producer(identity)
        for client in joined:
            client.close()


def _enable_test_route(store):
    owner = store.acquire_owner('producer-ordering', lease_ms=5000)
    store.configure_route(
        owner,
        'pool',
        'endpoint',
        revision='r1',
        execution_limit=1,
        execution_bytes=10000,
    )


async def test_missing_cancel_remains_pending_until_delayed_admission_is_visible(
    store, monkeypatch
):
    from marie.engine.exceptions import BatchExecutionError
    from marie.engine.llm_queue.store import RequestStore, StoreUnavailable

    _enable_test_route(store)
    processor = processor_for(store, batch_timeout=2)
    cancellation = threading.Event()
    transmitted = []
    missing_seen = threading.Event()
    admit = RequestStore.admit
    cancel = RequestStore.cancel_or_expire

    def delay_admission_reply(self, envelope):
        transmitted.append(envelope)
        cancellation.set()
        raise StoreUnavailable('Original admission is still in transit')

    def observe_cancel(self, *args, **kwargs):
        reply = cancel(self, *args, **kwargs)
        if reply.disposition == 'missing':
            missing_seen.set()
        return reply

    monkeypatch.setattr(RequestStore, 'admit', delay_admission_reply)
    monkeypatch.setattr(RequestStore, 'cancel_or_expire', observe_cancel)
    try:
        with pytest.raises(BatchExecutionError) as raised:
            await asyncio.to_thread(
                processor.batch_generate_calls,
                calls=calls(1),
                cancellation=cancellation,
            )
        assert raised.value.primary_error.category == 'cancellation_requested'
        assert not raised.value.primary_error.confirmed
        await eventually(missing_seen.is_set)
        envelope = transmitted.pop()
        producer = processor._queued_executor
        assert envelope.attempt_id in producer._pending
        assert store.metadata(envelope.attempt_id) is None
        assert admit(store, envelope).disposition == 'admitted'
        await eventually(
            lambda: store.metadata(envelope.attempt_id).state == 'cancelled'
        )
        assert envelope.attempt_id not in producer._pending
    finally:
        await asyncio.to_thread(processor.close)


async def test_cancellation_tracker_stops_before_new_io_after_original_deadline(
    store, monkeypatch
):
    from marie.engine.exceptions import BatchExecutionError
    from marie.engine.llm_queue.store import RequestStore

    _enable_test_route(store)
    processor = processor_for(store, batch_timeout=0.08)
    admit = RequestStore.admit
    cancel = RequestStore.cancel_or_expire
    cancellation_calls = []

    def slow_admission(self, envelope):
        result = admit(self, envelope)
        time.sleep(0.12)
        return result

    def record_cancel(self, *args, **kwargs):
        cancellation_calls.append(time.monotonic())
        return cancel(self, *args, **kwargs)

    monkeypatch.setattr(RequestStore, 'admit', slow_admission)
    monkeypatch.setattr(RequestStore, 'cancel_or_expire', record_cancel)
    try:
        with pytest.raises(BatchExecutionError):
            await asyncio.to_thread(processor.batch_generate_calls, calls=calls(1))
        await eventually(lambda: not processor._queued_executor._pending)
        assert cancellation_calls == []
    finally:
        await asyncio.to_thread(processor.close)
