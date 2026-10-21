"""Queued completion spans carry the prompt and reply like the direct batch path.

Run with OTEL_SDK_DISABLED=false and isolated stores (MARIE_LLM_QUEUE_TEST_STORES).
"""

import json

import pytest
from test_request_dispatcher import (
    admit_request,
    dispatcher_for,
    eventually,
    http_endpoint,
    store,
)

PAYLOAD_PREFIXES = ('input.', 'output.', 'llm.input_messages.', 'llm.output_messages.')


def _exporting_tracer(config=None):
    from openinference.instrumentation import OITracer, TraceConfig
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(provider.get_tracer('trace-payload'), config or TraceConfig())
    return provider, exporter, tracer


def _completion_spans(exporter, attempt_id):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.name == 'LLMDispatch.completion'
        and s.attributes.get('marie.llm_dispatch.request_id') == attempt_id
    ]


def _content_free(attributes):
    return {
        key: value
        for key, value in attributes.items()
        if not key.startswith(PAYLOAD_PREFIXES)
    }


async def test_completion_exports_actual_prompt_and_assistant_reply(
    store, http_endpoint, monkeypatch
):
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue import request_dispatcher

    provider, exporter, tracer = _exporting_tracer()
    monkeypatch.setattr(request_dispatcher, '_tracer', tracer)
    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    await runtime.start()
    try:
        await eventually(lambda: runtime._owner_ready)
        request = admit_request(
            store,
            call=CompletionCallParams(
                model='reply-sentinel',
                messages=[{'role': 'user', 'content': 'prompt-sentinel'}],
            ),
        )
        await eventually(lambda: runtime._counts['completed'] == 1)
        spans = _completion_spans(exporter, request.attempt_id)
        assert len(spans) == 1
        attributes = dict(spans[0].attributes)
        assert json.loads(attributes['input.value']) == [
            {'role': 'user', 'content': 'prompt-sentinel'}
        ]
        assert attributes['llm.input_messages.0.message.role'] == 'user'
        assert attributes['llm.input_messages.0.message.content'] == 'prompt-sentinel'
        assert attributes['output.value'] == 'reply-sentinel'
        assert attributes['llm.output_messages.0.message.role'] == 'assistant'
        assert attributes['llm.output_messages.0.message.content'] == 'reply-sentinel'
        assert attributes['marie.llm_dispatch.status'] == 'ok'
        exported = json.dumps(attributes)
        for forbidden in ('endpoint-secret', url, request.producer_id, 'Bearer'):
            assert forbidden not in exported
        assert 'prompt-sentinel' not in json.dumps(_content_free(attributes))
        assert received[0][0]['messages'][0]['content'] == 'prompt-sentinel'
    finally:
        await runtime.stop()
        provider.shutdown()


async def test_inline_media_is_replaced_by_source_reference_without_mutation(
    store, http_endpoint, monkeypatch
):
    import marie.instrumentation as instrumentation
    from marie.engine.completion_contract import CompletionCallParams, RequestContext
    from marie.engine.llm_queue import request_dispatcher

    provider, exporter, tracer = _exporting_tracer()
    monkeypatch.setattr(request_dispatcher, '_tracer', tracer)
    resolved = []

    def resolver(context):
        resolved.append(context)
        return f's3://bucket/{context.ref_type}/{context.ref_id}/{context.page_number}'

    monkeypatch.setattr(instrumentation, '_media_reference_resolver', resolver)
    data_url = 'data:image/png;base64,SU1BR0VTRU5USU5FTA=='
    url, received, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    await runtime.start()
    try:
        await eventually(lambda: runtime._owner_ready)
        request = admit_request(
            store,
            call=CompletionCallParams(
                model='model',
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': 'describe'},
                            {'type': 'image_url', 'image_url': {'url': data_url}},
                        ],
                    }
                ],
                context=RequestContext(ref_id='doc-1', ref_type='doc', page_number=2),
            ),
        )
        await eventually(lambda: runtime._counts['completed'] == 1)
        (span,) = _completion_spans(exporter, request.attempt_id)
        attributes = dict(span.attributes)
        exported = json.dumps(attributes)
        assert 'data:' not in exported
        assert 'SU1BR0VTRU5USU5FTA' not in exported
        assert 's3://bucket/doc/doc-1/2' in attributes['input.value']
        assert (
            attributes[
                'llm.input_messages.0.message.contents.1.message_content.image.image.url'
            ]
            == 's3://bucket/doc/doc-1/2'
        )
        assert [c.ref_id for c in resolved] == ['doc-1']
        sent = received[0][0]['messages'][0]['content'][1]['image_url']['url']
        assert sent == data_url
    finally:
        await runtime.stop()
        provider.shutdown()


async def test_trace_config_masks_queued_prompt_and_reply(
    store, http_endpoint, monkeypatch
):
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue import request_dispatcher
    from openinference.instrumentation import TraceConfig

    provider, exporter, tracer = _exporting_tracer(
        TraceConfig(hide_inputs=True, hide_outputs=True)
    )
    monkeypatch.setattr(request_dispatcher, '_tracer', tracer)
    url, _, _, _ = http_endpoint
    runtime = dispatcher_for(store, url)
    await runtime.start()
    try:
        await eventually(lambda: runtime._owner_ready)
        request = admit_request(
            store,
            call=CompletionCallParams(
                model='reply-sentinel',
                messages=[{'role': 'user', 'content': 'prompt-sentinel'}],
            ),
        )
        await eventually(lambda: runtime._counts['completed'] == 1)
        (span,) = _completion_spans(exporter, request.attempt_id)
        attributes = dict(span.attributes)
        assert attributes['input.value'] == '__REDACTED__'
        assert attributes['output.value'] == '__REDACTED__'
        assert not any(
            key.startswith(('llm.input_messages.', 'llm.output_messages.'))
            for key in attributes
        )
        assert 'prompt-sentinel' not in json.dumps(attributes)
        assert attributes['marie.llm_dispatch.status'] == 'ok'
    finally:
        await runtime.stop()
        provider.shutdown()


async def test_unsampled_completion_never_serializes_payload(
    store, http_endpoint, monkeypatch
):
    from marie.engine.llm_queue import request_dispatcher
    from marie.instrumentation.context import inject_trace_context
    from opentelemetry import trace
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        TraceFlags,
        TraceState,
    )

    provider, exporter, tracer = _exporting_tracer()
    monkeypatch.setattr(request_dispatcher, '_tracer', tracer)
    serialized = []
    monkeypatch.setattr(
        request_dispatcher,
        'set_llm_io',
        lambda span, **kwargs: serialized.append((span, kwargs)),
    )
    url, _, _, _ = http_endpoint
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
            admit_request(store, **inject_trace_context())
        await eventually(lambda: runtime._counts['completed'] == 1)
        assert serialized == []
        assert not exporter.get_finished_spans()
    finally:
        await runtime.stop()
        provider.shutdown()


@pytest.mark.parametrize(
    'outcome_kwargs',
    [
        {
            'category': 'invalid_request',
            'response': {
                'choices': [{'message': {'content': 'error-body-sentinel'}}],
                'error': {'message': 'error-body-sentinel'},
            },
            'remote_settled': True,
        },
        {'category': 'read_timeout', 'remote_settled': True},
    ],
)
async def test_transport_failure_captures_prompt_without_invented_output(
    store, http_endpoint, monkeypatch, outcome_kwargs
):
    from marie.engine.llm_queue import request_dispatcher
    from marie.engine.llm_queue.endpoint import EndpointClient, ExecutionOutcome
    from opentelemetry.trace import StatusCode

    provider, exporter, tracer = _exporting_tracer()
    monkeypatch.setattr(request_dispatcher, '_tracer', tracer)
    url, _, _, _ = http_endpoint

    class FailingClient(EndpointClient):
        async def execute(self, call, *, timeout_seconds):
            return ExecutionOutcome(**outcome_kwargs)

    runtime = dispatcher_for(store, url)
    runtime.client_factory = FailingClient
    await runtime.start()
    try:
        await eventually(lambda: runtime._owner_ready)
        request = admit_request(store)
        # Settlement follows the response presence; the span follows the category.
        await eventually(
            lambda: store.metadata(request.attempt_id).state in {'failed', 'succeeded'}
        )
        await eventually(lambda: _completion_spans(exporter, request.attempt_id))
        (span,) = _completion_spans(exporter, request.attempt_id)
        attributes = dict(span.attributes)
        assert span.status.status_code is StatusCode.ERROR
        assert attributes['marie.llm_dispatch.status'] == 'error'
        assert 'sentinel' in attributes['input.value']
        assert not any(
            key.startswith(('output.', 'llm.output_messages.')) for key in attributes
        )
        exported = json.dumps(attributes)
        assert 'error-body-sentinel' not in exported
        assert 'endpoint-secret' not in exported
    finally:
        await runtime.stop()
        provider.shutdown()
