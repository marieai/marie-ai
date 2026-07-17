from __future__ import annotations

import copy
import threading
from typing import Sequence

import pytest
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from marie.engine.completion_contract import RequestContext
from marie.instrumentation import set_llm_io
from marie.observability.media import resolve_media_reference
from marie.utils.asset_util import s3_asset_path


class _InMemoryExporter(SpanExporter):
    def __init__(self):
        self._spans = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def get_finished_spans(self):
        with self._lock:
            return list(self._spans)

    def clear(self):
        with self._lock:
            self._spans.clear()


@pytest.fixture(autouse=True)
def _reset_tracer_provider():
    yield
    import opentelemetry.trace as trace_mod

    if hasattr(trace_mod, "_TRACER_PROVIDER_SET_ONCE"):
        trace_mod._TRACER_PROVIDER_SET_ONCE = trace_mod.Once()
    if hasattr(trace_mod, "_TRACER_PROVIDER"):
        trace_mod._TRACER_PROVIDER = None


@pytest.fixture
def otel_setup():
    exporter = _InMemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()


def _image_url_attr(message_index: int, content_index: int) -> str:
    return (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}."
        f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}."
        f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
        f"{ImageAttributes.IMAGE_URL}"
    )


def _output_image_url_attr(message_index: int, content_index: int) -> str:
    return (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}."
        f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}."
        f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
        f"{ImageAttributes.IMAGE_URL}"
    )


def _content_type_attr(message_index: int, content_index: int) -> str:
    return (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}."
        f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}."
        f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"
    )


def test_set_llm_io_does_not_mutate_multimodal_request_messages(otel_setup):
    tracer = trace.get_tracer("test.llm_io")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "extract invoice data"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                    "max_pixels": 2007040,
                },
            ],
        }
    ]
    original = copy.deepcopy(messages)

    with tracer.start_as_current_span("llm-span") as span:
        set_llm_io(
            span,
            input_messages=messages,
            context=RequestContext(
                ref_id="document.tif",
                ref_type="stress",
                page_number=1,
            ),
            media_reference_resolver=resolve_media_reference,
        )

    assert messages == original


def test_set_llm_io_preserves_non_data_remote_image_urls(otel_setup):
    tracer = trace.get_tracer("test.llm_io")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/image.png"},
                },
            ],
        }
    ]

    with tracer.start_as_current_span("llm-span") as span:
        set_llm_io(span, input_messages=messages)

    attrs = otel_setup.get_finished_spans()[0].attributes
    assert attrs[_content_type_attr(0, 1)] == "image"
    assert attrs[_image_url_attr(0, 1)] == "https://example.test/image.png"


def test_set_llm_io_uses_unresolved_media_sentinel_without_source_metadata(
    otel_setup,
):
    tracer = trace.get_tracer("test.llm_io")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                },
            ],
        }
    ]

    with tracer.start_as_current_span("llm-span") as span:
        set_llm_io(span, input_messages=messages)

    attrs = otel_setup.get_finished_spans()[0].attributes
    assert attrs[_image_url_attr(0, 1)] == "marie://unresolved-media"
    assert attrs["marie.otel.media.input.count"] == 1
    assert attrs["marie.otel.media.input.reference_mode"] == "unresolved"
    assert attrs["marie.otel.media.input.reference_error"]
    for value in attrs.values():
        if isinstance(value, str):
            assert "data:image/" not in value
            assert "base64," not in value
            assert "ZmFrZQ==" not in value


def test_set_llm_io_sets_source_asset_reference_from_context(otel_setup):
    tracer = trace.get_tracer("test.llm_io")
    source_ref = s3_asset_path(
        ref_id="PID_2_10832_0_255720425.tif",
        ref_type="stress",
        include_filename=True,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "extract"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                },
            ],
        }
    ]

    with tracer.start_as_current_span("llm-span") as span:
        set_llm_io(
            span,
            input_messages=messages,
            context=RequestContext(
                ref_id="PID_2_10832_0_255720425.tif",
                ref_type="stress",
                page_number=2,
            ),
            media_reference_resolver=resolve_media_reference,
        )

    attrs = otel_setup.get_finished_spans()[0].attributes
    assert attrs[_image_url_attr(0, 1)] == source_ref
    assert attrs["marie.otel.media.input.count"] == 1
    assert attrs["marie.otel.media.input.reference_mode"] == "s3_asset_path"
    assert attrs["marie.otel.media.input.0.1.ref_id"] == "PID_2_10832_0_255720425.tif"
    assert attrs["marie.otel.media.input.0.1.ref_type"] == "stress"
    assert attrs["marie.otel.media.input.0.1.page_number"] == 2


def test_set_llm_io_normalizes_multimodal_output_messages(otel_setup):
    tracer = trace.get_tracer("test.llm_io")
    source_ref = s3_asset_path(
        ref_id="PID_2_10832_0_255720425.tif",
        ref_type="stress",
        include_filename=True,
    )
    output_messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "marked page"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                },
            ],
        }
    ]
    original = copy.deepcopy(output_messages)

    with tracer.start_as_current_span("llm-span") as span:
        set_llm_io(
            span,
            output_messages=output_messages,
            context=RequestContext(
                ref_id="PID_2_10832_0_255720425.tif",
                ref_type="stress",
                page_number=3,
            ),
            media_reference_resolver=resolve_media_reference,
        )

    attrs = otel_setup.get_finished_spans()[0].attributes
    assert output_messages == original
    assert attrs[_output_image_url_attr(0, 1)] == source_ref
    assert attrs["marie.otel.media.output.count"] == 1
    assert attrs["marie.otel.media.output.reference_mode"] == "s3_asset_path"
    assert attrs["marie.otel.media.output.0.1.ref_id"] == "PID_2_10832_0_255720425.tif"
    assert attrs["marie.otel.media.output.0.1.ref_type"] == "stress"
    assert attrs["marie.otel.media.output.0.1.page_number"] == 3
