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


def get_tracer(name: str = "marie.instrumentation") -> OITracer:
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

    return trace.get_tracer(name)


__all__ = [
    # Setup
    "register",
    "get_tracer",
    "get_tracker",
    "configure_from_yaml",
    # OI primitives
    "TracerProvider",
    "OITracer",
    "TraceConfig",
    "suppress_tracing",
    "capture_span_context",
]
