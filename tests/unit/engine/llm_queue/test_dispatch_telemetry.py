from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
from marie.engine.completion_contract import (
    CompletionCallParams,
    QueuedCompletionEnvelope,
    RequestContext,
)
from marie.engine.llm_queue.config import LlmQueueRuntimeConfig
from marie.engine.llm_queue.endpoint import (
    EndpointClient,
    ExecutionOutcome,
    RegisteredEndpoint,
)
from marie.engine.llm_queue.producer import QueuedDispatchProducer
from marie.engine.llm_queue.result_types import BatchResult
from marie.engine.llm_queue.store import RequestStore, StoreLimits, StoreReply
from marie.instrumentation.context import inject_trace_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState

PARENT = SpanContext(
    trace_id=1,
    span_id=2,
    is_remote=True,
    trace_flags=TraceFlags(1),
    trace_state=TraceState([('vendor', 'value')]),
)


def test_capture_preserves_sampling_and_vendor_state() -> None:
    parent = SpanContext(
        trace_id=1,
        span_id=2,
        is_remote=True,
        trace_flags=TraceFlags(0),
        trace_state=PARENT.trace_state,
    )
    with trace.use_span(NonRecordingSpan(parent)):
        headers = inject_trace_context()
        traceparent, tracestate = headers.get('traceparent'), headers.get('tracestate')
    assert traceparent == '00-00000000000000000000000000000001-0000000000000002-00'
    assert tracestate == 'vendor=value'


def test_producer_captures_context_once_at_admission() -> None:
    producer = QueuedDispatchProducer(
        config=LlmQueueRuntimeConfig(
            enabled=True,
            queue_url='redis://unused:6379/0',
            fabric_group_id='analysis',
        )
    )
    producer.producer_id = 'producer'
    admissions = []

    class Store:
        def server_time_ms(self):
            return 1000

        def admit_from_manifest(self, request):
            admissions.append(request)
            producer._pending[request.attempt_id].result = BatchResult(
                request.logical_task_id, 'done', None
            )
            return StoreReply(disposition='admitted')

        def ack_result(self, *args):
            return StoreReply(disposition='acked')

    producer._store = Store()
    producer._start = lambda *args: producer._store
    producer._check = lambda: None
    with trace.use_span(NonRecordingSpan(PARENT)):
        result = producer.execute(
            calls=[
                CompletionCallParams(
                    model='model',
                    messages=[],
                    context=RequestContext(job_id='job', work_unit_id='work'),
                )
            ],
            batch_request_id='batch',
            batch_timeout=1,
        )
    assert result[0].response == 'done'
    assert admissions[0].traceparent.endswith('-0000000000000002-01')
    assert admissions[0].tracestate == 'vendor=value'
    assert admissions[0].job_id == 'job'


def test_context_round_trip_does_not_change_business_digest() -> None:
    from marie.engine.llm_queue.queue_keys import QueueKeys

    request = QueuedCompletionEnvelope(
        schema_version=1,
        fabric_group_id='analysis',
        producer_id='producer',
        attempt_id='attempt',
        pool_id='pool',
        endpoint_id='endpoint',
        config_revision='r1',
        logical_batch_id='batch',
        logical_task_id='task',
        item_index=0,
        expires_at_ms=1000,
        call=CompletionCallParams(model='model', messages=[]),
    )
    store = object.__new__(RequestStore)
    store.keys = QueueKeys('analysis')
    store.limits = StoreLimits()
    original_json, _, original_digest, _ = store._serialize_admission(request)
    traced = replace(
        request,
        traceparent='00-00000000000000000000000000000001-0000000000000002-01',
        tracestate='vendor=value',
    )
    payload, _, digest, _ = store._serialize_admission(traced)
    restored = QueuedCompletionEnvelope.from_json(payload)
    assert restored.traceparent == traced.traceparent
    assert restored.tracestate == traced.tracestate
    assert digest == original_digest
    assert 'traceparent' not in json.loads(original_json)


async def test_endpoint_injects_active_context_without_forwarding_payload_options() -> (
    None
):
    received = []

    async def handle(request):
        received.append(dict(request.headers))
        return httpx.Response(200, json={'choices': [{'message': {'content': 'done'}}]})

    client = EndpointClient(
        RegisteredEndpoint('endpoint', 'http://127.0.0.1:1/v1', allow_loopback=True)
    )
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        with trace.use_span(NonRecordingSpan(PARENT)):
            outcome = await client.execute(
                CompletionCallParams(model='model', messages=[]), timeout_seconds=1
            )
        assert outcome.response is not None
        assert received[0]['traceparent'].endswith('-0000000000000002-01')
        assert received[0]['tracestate'] == 'vendor=value'
    finally:
        await client.close()


def test_history_uses_explicit_parent_and_llm_semantics(monkeypatch) -> None:
    from marie.engine.llm_queue import request_dispatcher

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(request_dispatcher, '_tracer', provider.get_tracer('test'))
    attributes = {
        'marie.llm_dispatch.queue_wait_ms': 1,
        'marie.llm_dispatch.model': 'model',
    }
    parent = trace.set_span_in_context(NonRecordingSpan(PARENT))
    request_dispatcher._emit_execution_history(
        attributes,
        ExecutionOutcome(response={'usage': {'total_tokens': 3}}, remote_settled=True),
        time.time_ns(),
        time.monotonic(),
        parent_context=parent,
    )
    span = exporter.get_finished_spans()[0]
    assert span.context.trace_id == PARENT.trace_id
    assert span.parent.span_id == PARENT.span_id
    assert span.attributes['openinference.span.kind'] == 'LLM'
    assert span.attributes['llm.model_name'] == 'model'
    assert span.attributes['marie.llm_dispatch.remote_settled'] is True
    assert not span.events
    provider.shutdown()


def test_gateway_exports_dispatcher_metrics_through_its_provider() -> None:
    code = '''
from unittest.mock import patch
from opentelemetry import metrics
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from marie.serve.instrumentation import InstrumentationMixin
reader = InMemoryMetricReader()
with patch('opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter', return_value=object()), patch('opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader', return_value=reader):
    runtime = InstrumentationMixin()
    runtime._setup_instrumentation('test-gateway', metrics=True)
from marie.engine.llm_queue.metrics import dispatch_metrics
dispatch_metrics.record_batch(pool_id='pool', dispatcher_id='dispatcher', batch_size=1, duration_seconds=0.01)
assert metrics.get_meter_provider() is runtime.meter_provider
result = reader.get_metrics_data()
names = [m.name for r in result.resource_metrics for s in r.scope_metrics for m in s.metrics]
assert 'marie_llm_dispatch_batches' in names
runtime.meter_provider.shutdown()
'''
    env = {**os.environ, 'OTEL_SDK_DISABLED': 'false'}
    result = subprocess.run(
        [sys.executable, '-c', code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


async def test_legacy_adapter_injects_context_without_mutating_call() -> None:
    from marie.engine.llm_queue.adapters.openai_compatible import (
        OpenAICompatibleExecutionAdapter,
    )

    recorded = []

    async def create(**kwargs):
        recorded.append(kwargs)
        return {'choices': []}

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    call = CompletionCallParams(
        model='model',
        messages=[],
        extra_create_kwargs={
            'extra_headers': {'Traceparent': 'caller-context', 'custom': 'value'}
        },
    )
    adapter = OpenAICompatibleExecutionAdapter(client=client, logger=None)
    with trace.use_span(NonRecordingSpan(PARENT)):
        await adapter.execute(call)
    headers = recorded[0]['extra_headers']
    assert headers['traceparent'].endswith('-0000000000000002-01')
    assert headers['tracestate'] == 'vendor=value'
    assert headers['custom'] == 'value'
    assert 'Traceparent' not in headers
    assert call.extra_create_kwargs['extra_headers']['Traceparent'] == 'caller-context'


def test_admission_and_delivery_are_content_free_children_of_producer(
    monkeypatch,
) -> None:
    from marie.engine.llm_queue import producer as producer_module
    from marie.engine.llm_queue.producer import QueueTaskError

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        producer_module, '_tracer', provider.get_tracer('admission-test'), raising=False
    )
    producer = QueuedDispatchProducer(
        config=LlmQueueRuntimeConfig(
            enabled=True,
            queue_url='redis://unused:6379/0',
            fabric_group_id='analysis',
        )
    )
    producer.producer_id = 'producer'

    class Store:
        def server_time_ms(self):
            return 1000

        def admit_from_manifest(self, request):
            producer._pending[request.attempt_id].result = BatchResult(
                request.logical_task_id,
                None,
                QueueTaskError('expired', state='expired', confirmed=True),
            )
            return StoreReply(disposition='admitted')

        def ack_result(self, *args):
            return StoreReply(disposition='acked')

    producer._store = Store()
    producer._start = lambda *args: producer._store
    producer._check = lambda: None
    with trace.use_span(NonRecordingSpan(PARENT)):
        result = producer.execute(
            calls=[
                CompletionCallParams(
                    model='model',
                    messages=[{'role': 'user', 'content': 'private-sentinel'}],
                    context=RequestContext(job_id='job', work_unit_id='work'),
                )
            ],
            batch_request_id='batch',
            batch_timeout=1,
        )
    assert result[0].error.category == 'expired'
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {'LLMQueue.admit', 'LLMQueue.result'}
    assert all(s.parent.span_id == PARENT.span_id for s in spans.values())
    assert all(s.context.trace_id == PARENT.trace_id for s in spans.values())
    assert spans['LLMQueue.result'].status.status_code is trace.StatusCode.ERROR
    assert spans['LLMQueue.result'].attributes['marie.llm_dispatch.state'] == 'expired'
    assert 'private-sentinel' not in json.dumps(
        [dict(s.attributes) for s in spans.values()]
    )
    provider.shutdown()


def test_settlement_event_does_not_hide_cancelled_request_status() -> None:
    from marie.engine.llm_queue.request_dispatcher import RequestDispatcher
    from marie.engine.llm_queue.store import ClaimRecord

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    runtime = object.__new__(RequestDispatcher)
    runtime.store = SimpleNamespace(keys=SimpleNamespace(fabric_id='fabric'))
    runtime.dispatcher_id = 'dispatcher'
    claim = ClaimRecord(
        'attempt', 'claim', 1, 1, 'pool', 'endpoint', 1, 1, 'not_refunded'
    )
    span = provider.get_tracer('test').start_span('dispatch')
    runtime._observe_outcome(span, claim, 'remote_settled')
    runtime._observe_outcome(span, claim, 'result_committed', category='cancelled')
    span.end()
    assert exporter.get_finished_spans()[0].status.status_code is trace.StatusCode.ERROR
    provider.shutdown()


async def test_legacy_adapter_accepts_null_headers_and_replaces_foreign_state() -> None:
    from marie.engine.llm_queue.adapters.openai_compatible import (
        OpenAICompatibleExecutionAdapter,
    )

    observed = []

    async def create(**kwargs):
        observed.append(kwargs['extra_headers'])
        return {}

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    adapter = OpenAICompatibleExecutionAdapter(client=client, logger=None)
    parent = SpanContext(
        trace_id=1, span_id=2, is_remote=True, trace_flags=TraceFlags(1)
    )
    with trace.use_span(NonRecordingSpan(parent)):
        for headers in (None, {'Tracestate': 'foreign=value'}):
            await adapter.execute(
                CompletionCallParams(
                    model='model',
                    messages=[],
                    extra_create_kwargs={'extra_headers': headers},
                )
            )
    assert all('traceparent' in headers for headers in observed)
    assert all(
        'tracestate' not in headers and 'Tracestate' not in headers
        for headers in observed
    )


def test_history_does_not_inherit_ambient_openinference_content(monkeypatch) -> None:
    from marie.engine.llm_queue import request_dispatcher
    from marie.instrumentation.processor import OpenInferenceSpanProcessor
    from openinference.instrumentation import OITracer, TraceConfig, using_attributes

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(OpenInferenceSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        request_dispatcher,
        '_tracer',
        OITracer(provider.get_tracer('privacy-test'), TraceConfig()),
    )
    parent = trace.set_span_in_context(NonRecordingSpan(PARENT))
    with using_attributes(
        metadata={'private': 'private-sentinel'},
        prompt_template='private-sentinel',
        prompt_template_variables={'patient': 'private-sentinel'},
    ):
        request_dispatcher._emit_execution_history(
            {'marie.llm_dispatch.queue_wait_ms': 1},
            ExecutionOutcome(response={}, remote_settled=True),
            time.time_ns(),
            time.monotonic(),
            parent_context=parent,
        )
    attributes = dict(exporter.get_finished_spans()[0].attributes)
    assert 'private-sentinel' not in json.dumps(attributes)
    assert 'metadata' not in attributes
    assert 'llm.prompt_template.template' not in attributes
    provider.shutdown()


def test_runtime_teardown_keeps_shared_process_meter_available() -> None:
    code = '''
from unittest.mock import patch
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from marie.serve.instrumentation import InstrumentationMixin
from marie.serve.runtimes.servers import BaseServer
reader = InMemoryMetricReader()
with patch('opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter', return_value=object()), patch('opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader', return_value=reader):
    first = InstrumentationMixin()
    first._setup_instrumentation('first', metrics=True)
    second = InstrumentationMixin()
    second._setup_instrumentation('second', metrics=True)
    assert first.meter_provider is second.meter_provider
    BaseServer._teardown_instrumentation(first)
    second.meter.create_counter('after_teardown').add(1)
    data = reader.get_metrics_data()
    assert data is not None
    assert any(m.name == 'after_teardown' for r in data.resource_metrics for s in r.scope_metrics for m in s.metrics)
    third = InstrumentationMixin()
    third._setup_instrumentation('third', metrics=True)
    third.meter.create_counter('after_restart').add(1)
    restarted = reader.get_metrics_data()
    assert restarted is not None
    assert any(m.name == 'after_restart' for r in restarted.resource_metrics for s in r.scope_metrics for m in s.metrics)
    first.meter_provider.shutdown()
'''
    result = subprocess.run(
        [sys.executable, '-c', code],
        env={**os.environ, 'OTEL_SDK_DISABLED': 'false'},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_producer_creates_trace_when_no_ambient_span_exists(monkeypatch) -> None:
    from marie.engine.llm_queue import producer as producer_module
    from marie.instrumentation.context import extract_trace_context

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        producer_module, '_tracer', provider.get_tracer('producer-test')
    )
    producer = QueuedDispatchProducer(
        config=LlmQueueRuntimeConfig(
            enabled=True,
            queue_url='redis://unused:6379/0',
            fabric_group_id='analysis',
        )
    )
    producer.producer_id = 'producer'
    captured = []

    class Store:
        def server_time_ms(self):
            return 1000

        def admit_from_manifest(self, request):
            captured.append(request)
            producer._pending[request.attempt_id].result = BatchResult(
                request.logical_task_id, 'done', None
            )
            return StoreReply(disposition='admitted')

        def ack_result(self, *args):
            return StoreReply(disposition='acked')

    producer._store = Store()
    producer._start = lambda *args: producer._store
    producer._check = lambda: None
    with trace.use_span(trace.INVALID_SPAN):
        producer.execute(
            calls=[CompletionCallParams(model='model', messages=[])],
            batch_request_id='batch',
            batch_timeout=1,
        )
    context = trace.get_current_span(
        extract_trace_context(captured[0].traceparent, captured[0].tracestate)
    ).get_span_context()
    assert context.is_valid
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans['LLMQueue.produce'].context.trace_id == context.trace_id
    assert (
        spans['LLMQueue.admit'].parent.span_id
        == spans['LLMQueue.produce'].context.span_id
    )
    provider.shutdown()


def test_trace_only_helpers_preserve_instrumentation_suppression() -> None:
    from marie.instrumentation.context import (
        extract_trace_context,
        start_trace_span,
        use_trace_context,
    )
    from openinference.instrumentation import OITracer, TraceConfig, suppress_tracing
    from opentelemetry import context
    from opentelemetry.context import (
        _SUPPRESS_HTTP_INSTRUMENTATION_KEY,
        _SUPPRESS_INSTRUMENTATION_KEY,
    )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(provider.get_tracer('suppression-test'), TraceConfig())
    parent = extract_trace_context(None)
    with suppress_tracing():
        span = start_trace_span(tracer, 'suppressed', context=parent)
        span.end()
        with use_trace_context(parent):
            tracer.start_span('also-suppressed').end()
    assert not exporter.get_finished_spans()
    explicit_suppressed = context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True, parent)
    start_trace_span(tracer, 'explicit-suppressed', context=explicit_suppressed).end()
    assert not exporter.get_finished_spans()
    token = context.attach(context.set_value(_SUPPRESS_HTTP_INSTRUMENTATION_KEY, True))
    try:
        with use_trace_context(parent):
            assert context.get_value(_SUPPRESS_HTTP_INSTRUMENTATION_KEY) is True
    finally:
        context.detach(token)
    provider.shutdown()
