from openinference.instrumentation import TracerProvider
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import marie.instrumentation.tracker as tracker_module
from marie.instrumentation import configure, get_tracker
from marie.instrumentation.config import reset_settings
from marie.instrumentation.exporters.base import AbstractExporter
from marie.instrumentation.exporters.otel import OTelExporter
from marie.instrumentation.tracker import LLMTracker


def test_tracker_exports_nested_otel_spans() -> None:
    span_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace_api.set_tracer_provider(provider)

    reset_settings()
    LLMTracker._instance = None
    tracker_module._tracker = None
    configure({"enabled": True, "exporter": "otel", "project_id": "tests"})
    tracker = get_tracker()

    with tracker.trace("request") as trace:
        parent_id = tracker.span(trace.id, "agent")
        child_id = tracker.generation(
            trace_id=trace.id,
            name="completion",
            model="gpt-4o",
            parent_observation_id=parent_id,
        )
        tracker.end(child_id, output="answer")
        tracker.end(parent_id, output="done")

    provider.force_flush()
    spans = {span.name: span for span in span_exporter.get_finished_spans()}

    assert isinstance(tracker._exporter, AbstractExporter)
    assert isinstance(tracker._exporter, OTelExporter)
    assert spans["agent"].parent.span_id == spans["request"].context.span_id
    assert spans["completion"].parent.span_id == spans["agent"].context.span_id

    tracker.stop()
    reset_settings()
    LLMTracker._instance = None
    tracker_module._tracker = None
