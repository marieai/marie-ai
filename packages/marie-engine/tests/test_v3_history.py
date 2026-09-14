"""Actual exported V3 execution history; run with OTEL_SDK_DISABLED=false."""

import json
import os

import pytest
from test_request_dispatcher import (
    admit_request,
    dispatcher_for,
    eventually,
    http_endpoint,
    store,
)
from test_v3_worker_lifetime import events, kill_and_reap, start_dispatcher


@pytest.mark.parametrize('model', ['model', 'https://forbidden.example/path'])
async def test_execution_exports_content_free_history_once(store, http_endpoint, model):
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.store import StoreUnavailable
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    assert os.environ['OTEL_SDK_DISABLED'] == 'false'
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    url, received, _, _ = http_endpoint
    dispatcher = dispatcher_for(store, url)
    original_finish = store.finish
    failures = 0

    def finish(*args, **kwargs):
        nonlocal failures
        reply = original_finish(*args, **kwargs)
        if failures == 0:
            failures += 1
            raise StoreUnavailable('synthetic lost commit reply')
        return reply

    store.finish = finish
    await dispatcher.start()
    try:
        await eventually(lambda: dispatcher._owner_ready)
        req = admit_request(
            store,
            call=CompletionCallParams(
                model=model, messages=[{'role': 'user', 'content': 'sentinel'}]
            ),
        )
        await eventually(lambda: dispatcher._counts['completed'] == 1)
        spans = [
            s
            for s in exporter.get_finished_spans()
            if s.name == 'LLMDispatch.completion'
        ]
        assert len(spans) == 1
        attributes = dict(spans[0].attributes)
        prefix = 'marie.llm_dispatch.'
        assert attributes[prefix + 'fabric_group_id'] == store.keys.fabric_id
        assert attributes[prefix + 'request_id'] == req.attempt_id
        assert attributes[prefix + 'execution_seq'] == 1
        assert (
            attributes[prefix + 'claim_id'] == store.metadata(req.attempt_id).claim_id
        )
        assert attributes[prefix + 'contract_version'] == 'v3'
        assert attributes[prefix + 'status'] == 'ok'
        assert attributes[prefix + 'model'] == (
            '' if model.startswith('https:') else model
        )
        assert attributes['llm.token_count.total'] == 1
        assert attributes[prefix + 'message_count'] == 1
        for field in ('queue_wait_ms', 'execution_ms', 'total_latency_ms'):
            assert attributes[prefix + field] >= 0
        exported = json.dumps(attributes)
        assert all(
            value not in exported
            for value in (
                'sentinel',
                'endpoint-secret',
                url,
                req.producer_id,
                'https://forbidden.example/path',
            )
        )
        assert not spans[0].events
        assert len(received) == 1
    finally:
        await dispatcher.stop()
        exporter.clear()


async def test_unknown_transport_exports_error_without_success_or_exception_content(
    store, http_endpoint
):
    import asyncio
    from dataclasses import replace

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    url, _, releases, _ = http_endpoint
    releases['model'] = asyncio.Event()
    runtime = dispatcher_for(store, url)
    runtime.endpoints['endpoint'] = replace(
        runtime.endpoints['endpoint'], call_timeout_seconds=0.1
    )
    await runtime.start()
    try:
        await eventually(lambda: runtime._owner_ready)
        req = admit_request(store)
        await eventually(
            lambda: store.metadata(req.attempt_id).state == 'outcome_unknown'
        )
        spans = [
            s
            for s in exporter.get_finished_spans()
            if s.name == 'LLMDispatch.completion'
        ]
        assert len(spans) == 1
        attributes = dict(spans[0].attributes)
        assert attributes['marie.llm_dispatch.status'] == 'error'
        assert attributes['marie.llm_dispatch.error_type'] in {
            'call_timeout',
            'read_timeout',
        }
        assert store.usage()['reserved_items'] == 1
        assert not spans[0].events
        assert 'sentinel' not in json.dumps(attributes)
    finally:
        releases['model'].set()
        await runtime.stop()
        exporter.clear()


async def test_killed_child_exports_completed_attempt_but_not_unfinished_response(
    store, http_endpoint, tmp_path
):
    import asyncio

    from marie.engine.completion_contract import CompletionCallParams

    assert os.environ['OTEL_SDK_DISABLED'] == 'false'
    url, received, releases, _ = http_endpoint
    releases['unfinished'] = asyncio.Event()
    path = tmp_path / 'child-history.jsonl'
    child = await start_dispatcher(store, path, url, barrier=None, trace=True)
    producer = store.create_producer(lease_ms=10_000)
    try:
        await eventually(lambda: store.resolve_route('pool'), seconds=15)
        completed = admit_request(
            store,
            producer_id=producer,
            call=CompletionCallParams(
                model='completed', messages=[{'role': 'user', 'content': 'sentinel'}]
            ),
        )
        await eventually(
            lambda: store.metadata(completed.attempt_id).state == 'succeeded',
            seconds=15,
        )
        await eventually(
            lambda: any(
                e['event'] == 'span'
                and e['attributes'].get('marie.llm_dispatch.request_id')
                == completed.attempt_id
                for e in events(path)
            ),
            seconds=5,
        )
        assert store.read_result(producer, completed.attempt_id) is not None

        unfinished = admit_request(
            store,
            producer_id=producer,
            call=CompletionCallParams(
                model='unfinished', messages=[{'role': 'user', 'content': 'sentinel'}]
            ),
        )
        await eventually(
            lambda: any(body['model'] == 'unfinished' for body, _ in received),
            seconds=15,
        )
        await eventually(
            lambda: store.metadata(unfinished.attempt_id).state == 'executing'
        )
        accepted = {
            'event': 'http_request_accepted_response_unfinished',
            'attempt': unfinished.attempt_id,
            'claim_id': store.metadata(unfinished.attempt_id).claim_id,
            'execution_seq': store.metadata(unfinished.attempt_id).execution_seq,
            'pid': child.pid,
        }
        stdout, stderr = await kill_and_reap(child)
        span_events = [e for e in events(path) if e['event'] == 'span']
        completion_spans = [
            e for e in span_events if e['name'] == 'LLMDispatch.completion'
        ]
        completed_spans = [
            e
            for e in completion_spans
            if e['attributes']['marie.llm_dispatch.request_id'] == completed.attempt_id
        ]
        unfinished_spans = [
            e
            for e in completion_spans
            if e['attributes']['marie.llm_dispatch.request_id'] == unfinished.attempt_id
        ]
        assert len(completed_spans) == 1
        assert completed_spans[0]['status'] == 'OK'
        assert completed_spans[0]['attributes']['marie.llm_dispatch.execution_seq'] == 1
        assert unfinished_spans == []
        interrupted = store.metadata(unfinished.attempt_id)
        assert interrupted.state == 'executing'
        assert interrupted.claim_id == accepted['claim_id']
        assert interrupted.execution_seq == accepted['execution_seq'] == 1
        assert store.read_result(producer, unfinished.attempt_id) is None
        assert store.client.exists(store.keys.alive(producer))
        print(
            json.dumps(
                {
                    'scenario': 'accepted-unfinished-history-kill',
                    'barrier': accepted,
                    'positive_span_attempt': completed.attempt_id,
                    'positive_span_status': completed_spans[0]['status'],
                    'interrupted_state': interrupted.state,
                    'unfinished_span_count': len(unfinished_spans),
                    'child_stdout': stdout,
                    'child_stderr': stderr,
                },
                sort_keys=True,
            )
        )
    finally:
        releases['unfinished'].set()
        if child.returncode is None:
            await kill_and_reap(child)
