"""
Marie Instrumentation - Unified LLM + Infrastructure Observability.

This module provides comprehensive observability for the Marie AI ecosystem
using OpenInference semantic conventions on top of OpenTelemetry.

Two APIs are available:

1. Decorator API (preferred for new code):
    from marie.instrumentation import get_tracer

    tracer = get_tracer()

    @tracer.agent
    async def my_agent(input: str) -> str: ...

    @tracer.llm
    async def call_llm(messages: list) -> str: ...

2. Imperative API (backward compatible):
    from marie.instrumentation import get_tracker

    tracker = get_tracker()
    with tracker.trace("my-request", user_id="user-123") as trace:
        gen_id = tracker.generation(
            trace_id=trace.id, name="openai_completion",
            model="gpt-4", input=messages,
        )
        response = openai.chat.completions.create(model="gpt-4", messages=messages)
        tracker.end(gen_id, output=response.content, usage=response.usage)

Setup:
    from marie.instrumentation import register
    provider = register(project_name="marie-prod", batch=True)
"""

import os

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    TracerProvider,
    capture_span_context,
    suppress_tracing,
)

from .config import configure_from_yaml
from .tracker import get_tracker


def register(
    *,
    project_name: str | None = None,
    endpoint: str | None = None,
    batch: bool = True,
    trace_config: TraceConfig | None = None,
    set_global_tracer_provider: bool = True,
    console_export: bool = False,
) -> TracerProvider:
    """
    One-liner setup for Marie instrumentation with OpenInference.

    Creates an OI TracerProvider with Marie's SpanProcessor, configures
    the OTLP exporter, and sets the global tracer provider.

    Args:
        endpoint: OTLP gRPC endpoint. Falls back to OTEL_EXPORTER_OTLP_ENDPOINT env var.
        console_export: If True, also prints spans to stdout via ConsoleSpanExporter.
            Useful for local debugging when no OTel Collector is running.

    Usage:
        from marie.instrumentation import register
        provider = register(project_name="marie-prod", batch=True)
    """
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    from .processor import OpenInferenceSpanProcessor

    config = trace_config or TraceConfig()
    project = project_name or os.environ.get("MARIE_PROJECT_NAME", "default")

    resource = Resource.create(
        {
            ResourceAttributes.PROJECT_NAME: project,
            "service.name": "marie-ai",
        }
    )

    provider = TracerProvider(config=config, resource=resource)
    provider.add_span_processor(OpenInferenceSpanProcessor())

    otlp_endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        processor = (
            BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
        )
        provider.add_span_processor(processor)

    if console_export:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    if set_global_tracer_provider:
        trace_api.set_tracer_provider(provider)

    return provider


def get_tracer(
    name: str = "marie.instrumentation",
    config: TraceConfig | None = None,
) -> OITracer:
    """
    Get an OITracer from the global provider.

    Returns an OITracer with decorator methods:
        @tracer.agent, @tracer.chain, @tracer.tool, @tracer.llm

    Usage:
        tracer = get_tracer()

        @tracer.agent
        async def my_agent(input: str) -> str: ...

        @tracer.llm
        async def call_llm(messages: list) -> str: ...
    """
    from opentelemetry import trace

    return OITracer(trace.get_tracer(name), config or TraceConfig())


import json as _json

from openinference.semconv.trace import SpanAttributes

_OI_SPAN_KIND_ATTR = SpanAttributes.OPENINFERENCE_SPAN_KIND


def start_as_current_span(
    tracer,
    name: str,
    *,
    span_kind: str | None = None,
    **kwargs,
):
    """Create an OTel span with OI span kind, compatible with any tracer.

    With an OI tracer the returned span has ``set_input`` / ``set_output``
    natively.  With a vanilla SDK tracer the span is wrapped so those
    convenience methods still work (they fall back to setting the
    ``input.value`` / ``output.value`` attributes directly).

    Args:
        tracer: An OTel or OITracer instance.
        name: Span name.
        span_kind: Lowercase OI kind (e.g. "chain", "agent", "llm", "tool").
        **kwargs: Forwarded to tracer.start_as_current_span().

    Returns:
        A context-manager span.
    """
    try:
        return tracer.start_as_current_span(
            name, openinference_span_kind=span_kind, **kwargs
        )
    except TypeError:
        cm = tracer.start_as_current_span(name, **kwargs)
        return _FallbackSpanContextManager(cm, span_kind)


def start_span(
    tracer,
    name: str,
    *,
    span_kind: str | None = None,
    **kwargs,
):
    """Create an OTel span with OI span kind, compatible with any tracer.

    With an OI tracer the returned span has ``set_input`` / ``set_output``
    natively.  With a vanilla SDK tracer the span is wrapped so those
    convenience methods still work.

    Args:
        tracer: An OTel or OITracer instance.
        name: Span name.
        span_kind: Lowercase OI kind (e.g. "chain", "agent", "llm", "tool").
        **kwargs: Forwarded to tracer.start_span().

    Returns:
        A Span object (or a thin wrapper that adds set_input/set_output).
    """
    try:
        return tracer.start_span(name, openinference_span_kind=span_kind, **kwargs)
    except TypeError:
        span = tracer.start_span(name, **kwargs)
        if span_kind:
            span.set_attribute(_OI_SPAN_KIND_ATTR, span_kind.upper())
        return _ensure_oi_api(span)


# ---------------------------------------------------------------------------
# Fallback helpers — give vanilla SDK spans set_input / set_output
# ---------------------------------------------------------------------------


def _serialise(value):
    """Serialize a value for span attributes, auto-detecting mime type."""
    if isinstance(value, str):
        return value, "text/plain"
    try:
        return _json.dumps(value), "application/json"
    except (TypeError, ValueError):
        return str(value), "text/plain"


def _ensure_oi_api(span):
    """Add set_input / set_output if the span doesn't already have them."""
    if hasattr(span, "set_input"):
        return span
    return _FallbackSpan(span)


class _FallbackSpan:
    """Thin proxy that adds set_input/set_output to a vanilla OTel Span."""

    def __init__(self, span):
        self._span = span

    # --- OI convenience API ------------------------------------------------

    def set_input(self, value, *, mime_type=None):
        text, detected = _serialise(value)
        self._span.set_attribute(SpanAttributes.INPUT_VALUE, text)
        self._span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, mime_type or detected)

    def set_output(self, value, *, mime_type=None):
        text, detected = _serialise(value)
        self._span.set_attribute(SpanAttributes.OUTPUT_VALUE, text)
        self._span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, mime_type or detected)

    # --- Forward everything else to the real span --------------------------

    def __getattr__(self, name):
        return getattr(self._span, name)


class _FallbackSpanContextManager:
    """Wraps start_as_current_span to inject OI span kind + set_input/set_output."""

    def __init__(self, cm, kind):
        self._cm = cm
        self._kind = kind

    def _prepare(self, span):
        if self._kind:
            span.set_attribute(_OI_SPAN_KIND_ATTR, self._kind.upper())
        return _ensure_oi_api(span)

    def __enter__(self):
        return self._prepare(self._cm.__enter__())

    def __exit__(self, *args):
        return self._cm.__exit__(*args)

    async def __aenter__(self):
        return self._prepare(await self._cm.__aenter__())

    async def __aexit__(self, *args):
        return await self._cm.__aexit__(*args)


__all__ = [
    # Setup
    "register",
    "get_tracer",
    "get_tracker",
    "configure_from_yaml",
    # Span helpers
    "start_span",
    "start_as_current_span",
    # OI primitives
    "TracerProvider",
    "OITracer",
    "TraceConfig",
    "suppress_tracing",
    "capture_span_context",
]
