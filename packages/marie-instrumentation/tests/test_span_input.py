import copy
import json

import marie.instrumentation as instrumentation
import pytest
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import TracerProvider


def test_span_input_redacts_deep_payload_without_truncating_or_mutating_it() -> None:
    leaf = {
        'Token': 'secret-token',
        'url': 'data:image/png;base64,secret-image',
        'prompt': 'Full content ' * 1000,
    }
    payload = leaf
    for _ in range(12):
        payload = {'nested': [payload]}
    original = copy.deepcopy(payload)
    provider = TracerProvider()
    with provider.get_tracer('test').start_as_current_span('invoke') as span:
        instrumentation.set_span_input(span, payload)
        recorded = json.loads(span.attributes[SpanAttributes.INPUT_VALUE])
        assert span.attributes[SpanAttributes.INPUT_MIME_TYPE] == 'application/json'
    for _ in range(12):
        recorded = recorded['nested'][0]
    assert recorded == {'url': 'marie://unresolved-media', 'prompt': leaf['prompt']}
    assert payload == original
    provider.shutdown()


def test_span_input_handles_circular_containers_without_exposing_credentials() -> None:
    payload = {'password': 'secret-password'}
    payload['self'] = payload
    provider = TracerProvider()
    with provider.get_tracer('test').start_as_current_span('invoke') as span:
        instrumentation.set_span_input(span, payload)
        assert json.loads(span.attributes[SpanAttributes.INPUT_VALUE]) == {
            'self': '<circular reference>'
        }
    provider.shutdown()


def test_span_input_skips_serialization_when_span_is_not_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.trace import INVALID_SPAN

    def unexpected_serialization(_value) -> None:
        pytest.fail('Inactive tracing must not serialize request content')

    monkeypatch.setattr(instrumentation, '_serialise', unexpected_serialization)
    instrumentation.set_span_input(INVALID_SPAN, {'parameters': {}})


def test_span_input_honors_openinference_input_masking() -> None:
    provider = TracerProvider()
    tracer = instrumentation.OITracer(
        provider.get_tracer('test'), instrumentation.TraceConfig(hide_inputs=True)
    )
    with tracer.start_as_current_span('invoke') as span:
        instrumentation.set_span_input(span, {'prompt': 'hidden input'})
        assert span.attributes[SpanAttributes.INPUT_VALUE] == '__REDACTED__'
    provider.shutdown()
