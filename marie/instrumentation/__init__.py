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
import logging as _logging

from openinference.semconv.trace import MessageAttributes, SpanAttributes

from marie.utils.json import EnhancedJSONEncoder

_log = _logging.getLogger(__name__)

_OI_SPAN_KIND_ATTR = SpanAttributes.OPENINFERENCE_SPAN_KIND

# ---------------------------------------------------------------------------
# Limits — used by agent/tool spans for summary fields only.
# Full I/O is NOT truncated (users need to see complete input/output).
# ---------------------------------------------------------------------------

MAX_FIELD_BYTES = 4_096  # Limit for preview/summary fields (tool args, agent query)

_SENSITIVE_KEYS = frozenset(
    {"password", "secret", "token", "api_key", "apikey", "authorization"}
)


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


def _redact_for_span(value, *, _depth=0):
    """Walk a value and redact sensitive/non-serializable content.

    - Replaces bytes/bytearray/memoryview with placeholder
    - Replaces numpy arrays and torch tensors with shape summaries
    - Strips keys matching _SENSITIVE_KEYS
    - Strings are passed through unmodified (no truncation)
    - Max 8 levels deep (messages have 4+ levels of nesting)
    """
    if _depth > 8:
        return value

    # Raw binary objects → placeholder
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary {len(value)} bytes>"

    # numpy arrays → shape summary
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return f"<ndarray shape={value.shape} dtype={value.dtype}>"
    except ImportError:
        pass

    # torch tensors → shape summary
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return f"<tensor shape={tuple(value.shape)} dtype={value.dtype}>"
    except ImportError:
        pass

    # Strings pass through — no truncation so users see full I/O.
    # Truncation can be made opt-in later via TraceConfig.
    if isinstance(value, str):
        return value

    # Dicts → strip sensitive keys, recurse values
    if isinstance(value, dict):
        return {
            k: _redact_for_span(v, _depth=_depth + 1)
            for k, v in value.items()
            if k.lower() not in _SENSITIVE_KEYS
        }

    # Lists → recurse items (no length cap)
    if isinstance(value, (list, tuple)):
        return [_redact_for_span(item, _depth=_depth + 1) for item in value]

    return value


def _serialise(value):
    """Serialize a value for span attributes with redaction.

    Uses EnhancedJSONEncoder (from marie.utils.json) because span payloads
    frequently contain numpy scalars, numpy arrays, and dataclass instances
    that the stdlib encoder cannot handle.

    No truncation is applied — users need full I/O visibility.
    Only sensitive keys and non-serializable types (bytes, numpy, torch)
    are redacted.  The OTLP/ClickHouse pipeline handles large attribute
    values natively (gRPC default 4 MiB).
    """
    if isinstance(value, str):
        return value, "text/plain"

    redacted = _redact_for_span(value)
    try:
        text = _json.dumps(redacted, cls=EnhancedJSONEncoder, ensure_ascii=False)
        return text, "application/json"
    except (TypeError, ValueError):
        return str(redacted), "text/plain"


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


# ---------------------------------------------------------------------------
# LLM message attribute helpers — dual representation per OpenInference spec
# ---------------------------------------------------------------------------
# Real OI instrumenters (openai, langchain) set BOTH:
#   1. input.value / output.value  — JSON blob for generic display
#   2. llm.input_messages.{i}.message.role/content — per-message for chat cards
# They use span.set_attribute() directly, not set_input()/set_output().
# We follow the same pattern using only public semconv constants.


def set_llm_io(span, *, input_messages=None, output_messages=None):
    """Set both input.value/output.value AND per-message attributes on a span.

    Follows the same dual-representation pattern as the official OpenInference
    instrumenters (openai, langchain): sets the JSON blob via INPUT_VALUE /
    OUTPUT_VALUE and expands per-message attributes via set_attribute().

    Args:
        span: An OI or _FallbackSpan with set_input/set_output/set_attribute.
        input_messages: List of {"role": str, "content": str} dicts.
        output_messages: List of {"role": str, "content": str} dicts,
            or a plain string (wrapped as assistant message).
    """
    if input_messages is not None:
        # Blob representation
        text, mime = _serialise(input_messages)
        try:
            span.set_attribute(SpanAttributes.INPUT_VALUE, text)
            if mime != "text/plain":
                span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, mime)
        except Exception:
            _log.debug("Failed to set input attributes", exc_info=True)
        # Per-message attributes
        _set_message_attributes(span, input_messages, SpanAttributes.LLM_INPUT_MESSAGES)

    if output_messages is not None:
        if isinstance(output_messages, str):
            # Plain string → set output.value directly and wrap as assistant msg
            text, mime = _serialise(output_messages)
            try:
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, text)
                if mime != "text/plain":
                    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, mime)
            except Exception:
                _log.debug("Failed to set output attributes", exc_info=True)
            _set_message_attributes(
                span,
                [{"role": "assistant", "content": output_messages}],
                SpanAttributes.LLM_OUTPUT_MESSAGES,
            )
        else:
            # List of message dicts
            text, mime = _serialise(output_messages)
            try:
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, text)
                if mime != "text/plain":
                    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, mime)
            except Exception:
                _log.debug("Failed to set output attributes", exc_info=True)
            _set_message_attributes(
                span, output_messages, SpanAttributes.LLM_OUTPUT_MESSAGES
            )


def _set_message_attributes(span, messages, base_key: str):
    """Expand a list of {role, content} dicts into per-message span attributes.

    Mirrors the pattern in openinference-instrumentation-openai's
    _request_attributes_extractor.py — iterate messages and call
    span.set_attribute() with the full attribute path.
    """
    if not isinstance(messages, (list, tuple)):
        return
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role is not None:
            try:
                span.set_attribute(
                    f"{base_key}.{i}.{MessageAttributes.MESSAGE_ROLE}", str(role)
                )
            except Exception:
                pass
        content = msg.get("content")
        content_text = _message_content_to_text(content)
        if content_text is not None:
            try:
                span.set_attribute(
                    f"{base_key}.{i}.{MessageAttributes.MESSAGE_CONTENT}",
                    content_text,
                )
            except Exception:
                pass
        # Tool calls (if present)
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, (list, tuple)):
            for j, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if tc_id is not None:
                    try:
                        span.set_attribute(
                            f"{base_key}.{i}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{j}.tool_call.id",
                            str(tc_id),
                        )
                    except Exception:
                        pass
                func = tc.get("function")
                if isinstance(func, dict):
                    name = func.get("name")
                    if name is not None:
                        try:
                            span.set_attribute(
                                f"{base_key}.{i}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{j}.tool_call.function.name",
                                str(name),
                            )
                        except Exception:
                            pass
                    args = func.get("arguments")
                    if args is not None:
                        try:
                            span.set_attribute(
                                f"{base_key}.{i}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{j}.tool_call.function.arguments",
                                str(args) if not isinstance(args, str) else args,
                            )
                        except Exception:
                            pass


def _message_content_to_text(content) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            item_text = _content_part_to_text(item)
            if item_text:
                parts.append(item_text)
        if parts:
            return "\n".join(parts)

    text, _ = _serialise(content)
    return text


def _content_part_to_text(item) -> str | None:
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        text, _ = _serialise(item)
        return text

    item_type = item.get("type")
    if item_type in {"text", "input_text"}:
        text = item.get("text") or item.get("content")
        return str(text) if text is not None else None
    if item_type in {"image", "image_url", "input_image"}:
        return _image_content_part_to_text(item)

    text = item.get("text") or item.get("content")
    if text is not None:
        return str(text)

    serialized, _ = _serialise(item)
    return serialized


def _image_content_part_to_text(item: dict) -> str:
    image = item.get("image_url") or item.get("image")
    detail = None
    url = None
    if isinstance(image, dict):
        detail = image.get("detail")
        url = image.get("url") or image.get("image_url")
    elif isinstance(image, str):
        url = image

    suffix = f" detail={detail}" if detail else ""
    if url:
        return f"[image_url: {url}{suffix}]"
    return f"[image_url{suffix}]"


__all__ = [
    # Setup
    "register",
    "get_tracer",
    "get_tracker",
    "configure_from_yaml",
    # Span helpers
    "start_span",
    "start_as_current_span",
    "set_llm_io",
    # Constants
    "MAX_FIELD_BYTES",
    # OI primitives
    "TracerProvider",
    "OITracer",
    "TraceConfig",
    "suppress_tracing",
    "capture_span_context",
]
