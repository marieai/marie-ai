"""Actual exported queued-dispatch execution history; run with OTEL_SDK_DISABLED=false."""

import json
import os

import pytest
from test_queued_dispatch_worker_lifetime import events, kill_and_reap, start_dispatcher
from test_request_dispatcher import (
    admit_request,
    dispatcher_for,
    eventually,
    http_endpoint,
    store,
)


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
        assert attributes[prefix + 'mode'] == 'queued-dispatch'
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
        runtime.endpoints['endpoint'], call_timeout_seconds=0.2
    )
    await runtime.start()
    try:
        await eventually(lambda: runtime._owner_ready)
        req = admit_request(store)
        await eventually(lambda: store.metadata(req.attempt_id).state == 'failed')
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


async def test_queue_context_links_concurrent_producers_to_http_and_metrics(
    store, http_endpoint, monkeypatch
):
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue import request_dispatcher
    from marie.engine.llm_queue.metrics import DispatchMetrics
    from marie.instrumentation.context import (
        extract_trace_context,
        inject_trace_context,
    )
    from opentelemetry import trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('dispatch-test')
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(request_dispatcher, '_tracer', tracer)
    monkeypatch.setattr(
        request_dispatcher,
        'dispatch_metrics',
        DispatchMetrics(meters.get_meter('dispatch-test')),
    )
    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    await runtime.start()
    try:
        await eventually(lambda: runtime._owner_ready)
        roots = {}
        requests = []
        for model in ('alpha', 'beta'):
            with tracer.start_as_current_span('producer-' + model) as root:
                headers = inject_trace_context()
                roots[model] = root.get_span_context()
                requests.append(
                    admit_request(
                        store,
                        call=CompletionCallParams(
                            model=model,
                            messages=[{'role': 'user', 'content': 'private-sentinel'}],
                        ),
                        **headers,
                    )
                )
        await eventually(lambda: runtime._counts['completed'] == 2)
        for model, request in zip(('alpha', 'beta'), requests):
            related = [
                s
                for s in exporter.get_finished_spans()
                if s.attributes.get('marie.llm_dispatch.request_id')
                == request.attempt_id
            ]
            dispatch = next(s for s in related if s.name == 'LLMDispatch.dispatch')
            completion = next(s for s in related if s.name == 'LLMDispatch.completion')
            assert dispatch.context.trace_id == roots[model].trace_id
            assert dispatch.parent.span_id == roots[model].span_id
            assert completion.context.trace_id == roots[model].trace_id
            assert completion.parent.span_id == dispatch.context.span_id
            headers = next(
                headers for body, headers in received if body['model'] == model
            )
            wire_parent = trace.get_current_span(
                extract_trace_context(
                    headers.get('traceparent'), headers.get('tracestate')
                )
            ).get_span_context()
            assert wire_parent.trace_id == roots[model].trace_id
            assert wire_parent.span_id == completion.context.span_id
            assert completion.attributes['openinference.span.kind'] == 'LLM'
            assert completion.attributes['llm.model_name'] == model
            assert completion.attributes['marie.llm_dispatch.remote_settled'] is True
            assert not completion.events
            assert 'result_committed' in [event.name for event in dispatch.events]
            assert 'private-sentinel' not in json.dumps(
                [dict(s.attributes) for s in related]
            )
        result = reader.get_metrics_data()
        measurements = {
            m.name: m
            for r in result.resource_metrics
            for s in r.scope_metrics
            for m in s.metrics
        }
        durations = measurements['marie_llm_dispatch_request_seconds'].data.data_points
        assert sum(point.count for point in durations) == 2
        assert all(point.attributes['mode'] == 'queued-dispatch' for point in durations)
        outcomes = measurements['marie_llm_dispatch_request_outcomes'].data.data_points
        assert sum(point.value for point in outcomes) == 2
    finally:
        await runtime.stop()
        provider.shutdown()
        meters.shutdown()


async def test_retry_preserves_original_trace_context_and_emits_deferred_outcome(
    store, http_endpoint, monkeypatch
):
    from marie.engine.llm_queue import request_dispatcher
    from marie.engine.llm_queue.endpoint import EndpointClient, ExecutionOutcome
    from marie.instrumentation.context import inject_trace_context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('retry-test')
    monkeypatch.setattr(request_dispatcher, '_tracer', tracer)
    url, _, _, _ = http_endpoint

    class RetryClient(EndpointClient):
        calls = 0

        async def execute(self, call, *, timeout_seconds):
            self.calls += 1
            if self.calls == 1:
                return ExecutionOutcome(
                    category='provider_unavailable',
                    retryable=True,
                    remote_settled=True,
                    availability_failure=True,
                )
            return await super().execute(call, timeout_seconds=timeout_seconds)

    runtime = dispatcher_for(store, url)
    runtime.client_factory = RetryClient
    await runtime.start()
    try:
        await eventually(lambda: runtime._owner_ready)
        with tracer.start_as_current_span('original-producer') as parent:
            root = parent.get_span_context()
            carrier = inject_trace_context()
            request = admit_request(store, **carrier)
        await eventually(lambda: runtime._counts['completed'] == 1)
        related = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes.get('marie.llm_dispatch.request_id') == request.attempt_id
        ]
        physical = [s for s in related if s.name == 'LLMDispatch.completion']
        dispatches = [s for s in related if s.name == 'LLMDispatch.dispatch']
        assert len(physical) == len(dispatches) == 2
        assert {s.attributes['marie.llm_dispatch.execution_seq'] for s in physical} == {
            1,
            2,
        }
        assert all(s.context.trace_id == root.trace_id for s in related)
        assert all(s.parent.span_id == root.span_id for s in dispatches)
        assert 'deferred' in [e.name for s in dispatches for e in s.events]
        assert store.metadata(request.attempt_id).traceparent == carrier['traceparent']
    finally:
        await runtime.stop()
        provider.shutdown()


async def test_unsampled_context_reaches_provider_without_exporting_spans(
    store, http_endpoint, monkeypatch
):
    from marie.engine.llm_queue import request_dispatcher
    from marie.instrumentation.context import (
        extract_trace_context,
        inject_trace_context,
    )
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        TraceFlags,
        TraceState,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        request_dispatcher, '_tracer', provider.get_tracer('unsampled-test')
    )
    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    await runtime.start()
    try:
        await eventually(lambda: runtime._owner_ready)
        parent = SpanContext(
            trace_id=123,
            span_id=456,
            is_remote=True,
            trace_flags=TraceFlags(0),
            trace_state=TraceState([('vendor', 'value')]),
        )
        with trace.use_span(NonRecordingSpan(parent)):
            request = admit_request(store, **inject_trace_context())
        await eventually(lambda: runtime._counts['completed'] == 1)
        headers = received[0][1]
        context = trace.get_current_span(
            extract_trace_context(headers.get('traceparent'), headers.get('tracestate'))
        ).get_span_context()
        assert context.trace_id == parent.trace_id
        assert not context.trace_flags.sampled
        assert list(context.trace_state.items()) == [('vendor', 'value')]
        assert not exporter.get_finished_spans()
        assert store.metadata(request.attempt_id).traceparent.endswith('-00')
    finally:
        await runtime.stop()
        provider.shutdown()


async def test_actual_producer_manifest_dispatch_and_delivery_share_origin(
    store, http_endpoint, monkeypatch
):
    import asyncio

    from marie.engine import batch_processor
    from marie.engine.completion_contract import CompletionCallParams, RequestContext
    from marie.engine.llm_queue import producer, request_dispatcher
    from marie.engine.llm_queue.store import RoutingManifestProjection
    from marie.instrumentation.context import extract_trace_context
    from marie.instrumentation.processor import OpenInferenceSpanProcessor
    from openinference.instrumentation import OITracer, TraceConfig, using_attributes
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from test_queued_dispatch_producer import processor_for

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(OpenInferenceSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(provider.get_tracer('producer-round-trip'), TraceConfig())
    monkeypatch.setattr(producer, '_tracer', tracer)
    monkeypatch.setattr(request_dispatcher, '_tracer', tracer)
    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    processor = processor_for(store)
    batch_contexts = []
    original_start_span = batch_processor.start_span

    def capture_batch_parent(*args, **kwargs):
        span = original_start_span(*args, **kwargs)
        batch_contexts.append(span.get_span_context())
        return span

    monkeypatch.setattr(batch_processor, 'start_span', capture_batch_parent)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        context = RequestContext(job_id='job', work_unit_id='work')
        store.project_routing_manifest(
            RoutingManifestProjection(
                job_id=context.job_id,
                work_unit_id=context.work_unit_id,
                fabric_group_id=store.keys.fabric_id,
                policy_generation=1,
                route_digest='e' * 64,
                pool_id='pool',
                endpoint_group_id='endpoint',
                endpoint_revision='r1',
            )
        )
        with tracer.start_as_current_span('producer-origin') as origin:
            with using_attributes(
                metadata={'private': 'private-sentinel'},
                prompt_template='private-sentinel',
                prompt_template_variables={'patient': 'private-sentinel'},
            ):
                result = await asyncio.to_thread(
                    processor.batch_generate_calls,
                    calls=[
                        CompletionCallParams(
                            model='model',
                            messages=[{'role': 'user', 'content': 'private-sentinel'}],
                            context=context,
                        )
                    ],
                    request_id='traced-batch',
                )
        assert result == ['model']
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert len(batch_contexts) == 1
        assert batch_contexts[0].trace_id == origin.get_span_context().trace_id
        for name in ('LLMQueue.admit', 'LLMQueue.result', 'LLMDispatch.dispatch'):
            assert spans[name].parent.span_id == batch_contexts[0].span_id
            assert spans[name].context.trace_id == origin.get_span_context().trace_id
        assert (
            spans['LLMDispatch.completion'].parent.span_id
            == spans['LLMDispatch.dispatch'].context.span_id
        )
        headers = received[0][1]
        wire = trace.get_current_span(
            extract_trace_context(headers.get('traceparent'), headers.get('tracestate'))
        ).get_span_context()
        assert wire.trace_id == origin.get_span_context().trace_id
        assert wire.span_id == spans['LLMDispatch.completion'].context.span_id
        assert 'private-sentinel' not in json.dumps(
            [dict(s.attributes) for s in spans.values()]
        )
        assert store.usage()['records'] == 0
    finally:
        await asyncio.to_thread(processor.close)
        await runtime.stop()
        provider.shutdown()
