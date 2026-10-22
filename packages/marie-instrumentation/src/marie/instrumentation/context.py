"""
Context propagation for OpenInference attributes.

Usage:
    from marie.instrumentation.context import using_attributes, using_session

    with using_attributes(session_id="sess-123", user_id="user-456", tags=["prod"]):
        # All spans created in this block inherit these attributes
        response = await agent.run(input)
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Iterator

# Re-export from openinference.instrumentation
from openinference.instrumentation import (
    capture_span_context,
    get_attributes_from_context,
    suppress_tracing,
    using_attributes,
    using_metadata,
    using_prompt_template,
    using_session,
    using_tags,
    using_user,
)
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import (
    _SUPPRESS_HTTP_INSTRUMENTATION_KEY,
    _SUPPRESS_INSTRUMENTATION_KEY,
    Context,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def inject_trace_context(context: Context | None = None) -> dict[str, str]:
    """Serialize W3C trace identity, sampling flags, and vendor state."""
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier, context=context)
    return carrier


def extract_trace_context(
    traceparent: str | None, tracestate: str | None = None
) -> Context:
    """Extract a queued parent without inheriting the dispatcher's ambient context."""
    carrier = {
        key: value
        for key, value in (("traceparent", traceparent), ("tracestate", tracestate))
        if isinstance(value, str) and len(value) <= 512
    }
    return TraceContextTextMapPropagator().extract(carrier, context=Context())


def submission_trace_links(carrier: Mapping[str, str] | None) -> list[trace.Link]:
    """Link delayed execution to a persisted submission without activating it."""
    if not isinstance(carrier, Mapping):
        return []
    context = extract_trace_context(
        carrier.get('traceparent'), carrier.get('tracestate')
    )
    origin = trace.get_current_span(context).get_span_context()
    if not origin.is_valid:
        return []
    return [trace.Link(origin, attributes={'marie.link.kind': 'submission'})]


@contextmanager
def use_trace_context(context: Context) -> Iterator[None]:
    """Activate trace identity while excluding ambient OpenInference content."""
    parent = trace.set_span_in_context(trace.get_current_span(context), Context())
    for key in (_SUPPRESS_INSTRUMENTATION_KEY, _SUPPRESS_HTTP_INSTRUMENTATION_KEY):
        if context_api.get_value(key) or context_api.get_value(key, context):
            parent = context_api.set_value(key, True, parent)
    token = context_api.attach(parent)
    try:
        yield
    finally:
        context_api.detach(token)


def start_trace_span(
    tracer: trace.Tracer, name: str, *, context: Context | None = None, **kwargs: Any
) -> trace.Span:
    """Create a content-free span using only the selected trace context."""
    with use_trace_context(
        context if context is not None else context_api.get_current()
    ):
        return tracer.start_span(name, context=context_api.get_current(), **kwargs)


__all__ = [
    "using_attributes",
    "using_session",
    "using_user",
    "using_metadata",
    "using_tags",
    "using_prompt_template",
    "suppress_tracing",
    "capture_span_context",
    "get_attributes_from_context",
    "inject_trace_context",
    "extract_trace_context",
    "submission_trace_links",
    "use_trace_context",
    "start_trace_span",
]
