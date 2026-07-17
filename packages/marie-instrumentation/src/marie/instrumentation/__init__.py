"""
Marie Instrumentation - Unified LLM + Infrastructure Observability.

This module provides comprehensive observability for the Marie AI ecosystem
using OpenInference semantic conventions on top of OpenTelemetry. Marie keeps
provider payloads unchanged and writes sanitized span attributes for telemetry.

Common usage:

1. Setup:
    from marie.instrumentation import register

    provider = register(project_name="marie-prod")

2. Decorator API:
    from marie.instrumentation import get_tracer

    tracer = get_tracer()

    @tracer.agent
    async def my_agent(input: str) -> str: ...

    @tracer.llm
    async def call_llm(messages: list) -> str: ...

3. LLM span I/O attributes:
    from opentelemetry import trace

    from types import SimpleNamespace

    from marie.instrumentation import set_llm_io

    tracer = trace.get_tracer("marie.engine")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "extract"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        }
    ]
    context = SimpleNamespace(
        ref_id="PID_2_10832_0_255720425.tif",
        ref_type="stress",
        page_number=1,
    )

    with tracer.start_as_current_span("LLM.completion") as span:
        set_llm_io(
            span,
            input_messages=messages,
            context=context,
            media_reference_resolver=resolve_media_reference,
        )
        # Call the provider with the original messages. set_llm_io only writes
        # telemetry attributes and replaces inline image data in the span view.

4. Legacy tracker API:
    from marie.instrumentation import get_tracker

    tracker = get_tracker()
    with tracker.trace("my-request", user_id="user-123") as trace:
        gen_id = tracker.generation(
            trace_id=trace.id, name="openai_completion",
            model="gpt-4", input=messages,
        )
        response = openai.chat.completions.create(model="gpt-4", messages=messages)
        tracker.end(gen_id, output=response.content, usage=response.usage)
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    TracerProvider,
    capture_span_context,
    suppress_tracing,
)

from .config import configure
from .tracker import get_tracker

_DEFAULT_OTEL_GRPC_MAX_MESSAGE_BYTES = 128 * 1024 * 1024
_DEFAULT_OTEL_MAX_EXPORT_BATCH_SIZE = 32
_HTTP_PROTOBUF_PROTOCOLS = {"http", "http/protobuf"}


def _read_positive_int_env(name: str, default: int | None) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


def _otlp_trace_protocol() -> str:
    return (
        (
            os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
            or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
            or "grpc"
        )
        .strip()
        .lower()
    )


def _http_trace_endpoint(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/v1/traces"):
        return normalized
    return f"{normalized}/v1/traces"


def _grpc_channel_options() -> tuple[tuple[str, int], ...]:
    max_message_bytes = _read_positive_int_env(
        "MARIE_OTEL_GRPC_MAX_MESSAGE_BYTES",
        _DEFAULT_OTEL_GRPC_MAX_MESSAGE_BYTES,
    )
    return (
        ("grpc.max_send_message_length", int(max_message_bytes)),
        ("grpc.max_receive_message_length", int(max_message_bytes)),
    )


def _create_otlp_span_exporter(endpoint: str):
    protocol = _otlp_trace_protocol()
    if protocol in _HTTP_PROTOBUF_PROTOCOLS:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=_http_trace_endpoint(endpoint))

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    try:
        return OTLPSpanExporter(
            endpoint=endpoint,
            insecure=True,
            channel_options=_grpc_channel_options(),
        )
    except TypeError as exc:
        if "channel_options" not in str(exc):
            raise
        return OTLPSpanExporter(endpoint=endpoint, insecure=True)


def _batch_span_processor_kwargs() -> dict[str, int]:
    max_queue_size = _read_positive_int_env("OTEL_BSP_MAX_QUEUE_SIZE", None)
    max_export_batch_size = _read_positive_int_env(
        "OTEL_BSP_MAX_EXPORT_BATCH_SIZE",
        _DEFAULT_OTEL_MAX_EXPORT_BATCH_SIZE,
    )
    if max_queue_size is not None:
        max_export_batch_size = min(max_export_batch_size, max_queue_size)
        return {
            "max_queue_size": max_queue_size,
            "max_export_batch_size": max_export_batch_size,
        }
    return {"max_export_batch_size": max_export_batch_size}


def register(
    *,
    project_name: str | None = None,
    service_name: str | None = None,
    resource_attributes: Mapping[str, Any] | None = None,
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
        service_name: Service identity. Falls back to OTEL_SERVICE_NAME.
        endpoint: OTLP gRPC endpoint. Falls back to OTEL_EXPORTER_OTLP_ENDPOINT env var.
        resource_attributes: Additional OpenTelemetry resource attributes.
        console_export: If True, also prints spans to stdout via ConsoleSpanExporter.
            Useful for local debugging when no OTel Collector is running.

    Usage:
        from marie.instrumentation import register
        provider = register(project_name="marie-prod")
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

    resource_values = dict(resource_attributes or {})
    resource_values.setdefault(ResourceAttributes.PROJECT_NAME, project)
    resource_values.setdefault(
        "service.name",
        service_name or os.environ.get("OTEL_SERVICE_NAME", "marie"),
    )
    resource = Resource.create(resource_values)

    provider = TracerProvider(config=config, resource=resource)
    provider.add_span_processor(OpenInferenceSpanProcessor())

    otlp_endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        exporter = _create_otlp_span_exporter(otlp_endpoint)
        processor = (
            BatchSpanProcessor(exporter, **_batch_span_processor_kwargs())
            if batch
            else SimpleSpanProcessor(exporter)
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


import copy as _copy
import json as _json
import logging as _logging

from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

from .attributes import MarieSpanAttributes

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

_UNRESOLVED_MEDIA_URL = "marie://unresolved-media"
_DATA_URL_PREFIX = "data:"

MediaReferenceResolver = Callable[[Any], str | tuple[str, str] | None]
_media_reference_resolver: MediaReferenceResolver | None = None


def configure_media_reference_resolver(
    resolver: MediaReferenceResolver | None,
) -> None:
    """Configure the process-wide host adapter for telemetry media references."""
    global _media_reference_resolver
    _media_reference_resolver = resolver


def _is_data_url(value) -> bool:
    return isinstance(value, str) and value.startswith(_DATA_URL_PREFIX)


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
    - Strings are passed through unmodified except inline data URLs
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
    # Inline data URLs are replaced so raw image bytes never land in traces.
    if isinstance(value, str):
        if _is_data_url(value):
            return _UNRESOLVED_MEDIA_URL
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


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass

    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _serialise(value):
    """Serialize a value for span attributes with redaction.

    Handles dataclasses, Pydantic models, and optional NumPy values without
    depending on the Marie server package.

    No truncation is applied — users need full I/O visibility.
    Only sensitive keys and non-serializable types (bytes, numpy, torch)
    are redacted.  The OTLP/ClickHouse pipeline handles large attribute
    values natively (gRPC default 4 MiB).
    """
    if isinstance(value, str):
        return value, "text/plain"

    redacted = _redact_for_span(value)
    try:
        text = _json.dumps(redacted, default=_json_default, ensure_ascii=False)
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
#   2. llm.input_messages.{i}.message.role/content(s) — chat message fields
# They use span.set_attribute() directly, not set_input()/set_output().
# We follow the same pattern using only public semconv constants.


def set_llm_io(
    span,
    *,
    input_messages=None,
    output_messages=None,
    context=None,
    media_reference_resolver: MediaReferenceResolver | None = None,
):
    """Set both input.value/output.value AND per-message attributes on a span.

    Follows the same dual-representation pattern as the official OpenInference
    instrumenters (openai, langchain): sets the JSON blob via INPUT_VALUE /
    OUTPUT_VALUE and expands per-message attributes via set_attribute().

    Args:
        span: An OI or _FallbackSpan with set_input/set_output/set_attribute.
        input_messages: List of {"role": str, "content": str | list} dicts.
        output_messages: List of {"role": str, "content": str | list} dicts,
            or a plain string wrapped as an assistant message.
        context: Optional request provenance for replacing inline image data
            URLs in telemetry. Expected fields are ref_id, ref_type, and
            page_number.
        media_reference_resolver: Optional host callback that converts the
            request context into a durable media URL. It may return a URL or a
            ``(URL, reference_mode)`` tuple.
    """
    media_reference_resolver = media_reference_resolver or _media_reference_resolver

    if input_messages is not None:
        input_messages, media_refs = _normalise_messages_for_telemetry(
            input_messages,
            context=context,
            direction="input",
            media_reference_resolver=media_reference_resolver,
        )
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
        _set_media_attributes(span, media_refs)

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
            output_messages, media_refs = _normalise_messages_for_telemetry(
                output_messages,
                context=context,
                direction="output",
                media_reference_resolver=media_reference_resolver,
            )
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
            _set_media_attributes(span, media_refs)


def _normalise_messages_for_telemetry(
    messages,
    *,
    context,
    direction: str,
    media_reference_resolver: MediaReferenceResolver | None,
):
    """Return a trace-safe copy of chat messages and any media reference attrs."""
    if not isinstance(messages, (list, tuple)):
        return messages, []

    media_refs = []
    normalised_messages = []
    for message_index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            normalised_messages.append(msg)
            continue

        normalised_msg = _copy.deepcopy(msg)
        content = normalised_msg.get("content")
        if isinstance(content, (list, tuple)):
            normalised_content = []
            for content_index, part in enumerate(content):
                normalised_part, media_ref = _normalise_content_part(
                    part,
                    context=context,
                    direction=direction,
                    message_index=message_index,
                    content_index=content_index,
                    media_reference_resolver=media_reference_resolver,
                )
                normalised_content.append(normalised_part)
                if media_ref is not None:
                    media_refs.append(media_ref)
            normalised_msg["content"] = normalised_content
        normalised_messages.append(normalised_msg)

    return normalised_messages, media_refs


def _normalise_content_part(
    part,
    *,
    context,
    direction: str,
    message_index: int,
    content_index: int,
    media_reference_resolver: MediaReferenceResolver | None,
):
    if not isinstance(part, dict):
        return part, None

    url = _extract_image_url(part)
    if not _is_data_url(url):
        return _copy.deepcopy(part), None

    source_ref, media_attrs, error, reference_mode = _source_media_reference(
        context,
        media_reference_resolver,
    )
    normalised_part = _replace_image_url(
        part,
        source_ref or _UNRESOLVED_MEDIA_URL,
    )

    media_ref = {
        "direction": direction,
        "message_index": message_index,
        "content_index": content_index,
    }
    media_ref.update(media_attrs)
    if error:
        media_ref["reference_error"] = error
    else:
        media_ref["reference_mode"] = reference_mode

    return normalised_part, media_ref


def _source_media_reference(
    context,
    resolver: MediaReferenceResolver | None,
):
    if context is None:
        return None, {}, "missing source context", "unresolved"

    ref_id = getattr(context, "ref_id", None)
    ref_type = getattr(context, "ref_type", None)
    page_number = _coerce_page_number(getattr(context, "page_number", None))
    missing = []
    if not ref_id:
        missing.append("ref_id")
    if not ref_type:
        missing.append("ref_type")
    if page_number is None:
        missing.append("page_number")
    if missing:
        return None, {}, f"missing {','.join(missing)}", "unresolved"

    media_attrs = {
        "ref_id": str(ref_id),
        "ref_type": str(ref_type),
        "page_number": page_number,
    }

    if resolver is None:
        return None, media_attrs, "missing media reference resolver", "unresolved"

    try:
        resolved = resolver(context)
        if resolved is None:
            return None, media_attrs, "media reference unresolved", "unresolved"
        if isinstance(resolved, tuple):
            source_ref, reference_mode = resolved
        else:
            source_ref, reference_mode = resolved, "resolved"
        if not source_ref:
            return None, media_attrs, "media reference unresolved", "unresolved"
        return str(source_ref), media_attrs, None, str(reference_mode)
    except Exception as exc:
        return (
            None,
            media_attrs,
            f"source reference error: {exc}",
            "unresolved",
        )


def _coerce_page_number(value) -> int | None:
    try:
        page_number = int(value)
    except (TypeError, ValueError):
        return None
    if page_number < 1:
        return None
    return page_number


def _extract_image_url(part):
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        return image_url.get("url")
    if isinstance(image_url, str):
        return image_url

    image = part.get("image")
    if isinstance(image, dict):
        return image.get("url")
    if isinstance(image, str):
        return image

    return part.get("url")


def _replace_image_url(part, url: str):
    normalised = _copy.deepcopy(part)
    image_url = normalised.get("image_url")
    if isinstance(image_url, dict):
        image_url["url"] = url
        return normalised
    if isinstance(image_url, str):
        normalised["image_url"] = url
        return normalised

    image = normalised.get("image")
    if isinstance(image, dict):
        image["url"] = url
        return normalised
    if isinstance(image, str):
        normalised["image"] = url
        return normalised

    normalised["url"] = url
    return normalised


def _set_media_attributes(span, media_refs):
    if not media_refs:
        return

    direction_counts: dict[str, int] = {}
    direction_errors: dict[str, list[str]] = {}
    direction_modes: dict[str, set[str]] = {}
    for media_ref in media_refs:
        direction = media_ref["direction"]
        message_index = media_ref["message_index"]
        content_index = media_ref["content_index"]
        direction_counts[direction] = direction_counts.get(direction, 0) + 1

        for key in ("ref_id", "ref_type", "page_number"):
            value = media_ref.get(key)
            if value is None:
                continue
            try:
                span.set_attribute(
                    MarieSpanAttributes.media_reference(
                        direction,
                        message_index,
                        content_index,
                        key,
                    ),
                    value,
                )
            except Exception:
                pass

        error = media_ref.get("reference_error")
        if error:
            error_text = str(error)
            direction_errors.setdefault(direction, []).append(error_text)
            direction_modes.setdefault(direction, set()).add("unresolved")
            try:
                span.set_attribute(
                    MarieSpanAttributes.media_reference(
                        direction,
                        message_index,
                        content_index,
                        "reference_error",
                    ),
                    error_text,
                )
            except Exception:
                pass
        else:
            direction_modes.setdefault(direction, set()).add(
                media_ref.get("reference_mode", "resolved")
            )

    for direction, count in direction_counts.items():
        try:
            span.set_attribute(MarieSpanAttributes.media_count(direction), count)
        except Exception:
            pass

        modes = direction_modes.get(direction, set())
        if modes:
            reference_mode = modes.pop() if len(modes) == 1 else "mixed"
            try:
                span.set_attribute(
                    MarieSpanAttributes.media_reference_mode(direction),
                    reference_mode,
                )
            except Exception:
                pass

    for direction, errors in direction_errors.items():
        try:
            span.set_attribute(
                MarieSpanAttributes.media_reference_error(direction),
                "; ".join(errors),
            )
        except Exception:
            pass


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
        if isinstance(content, (list, tuple)):
            _set_message_content_attributes(span, content, f"{base_key}.{i}")
        else:
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

    text, _ = _serialise(content)
    return text


def _set_message_content_attributes(span, content_parts, message_base_key: str):
    for content_index, part in enumerate(content_parts):
        if not isinstance(part, dict):
            text, _ = _serialise(part)
            if text is not None:
                _set_message_content_text(span, message_base_key, content_index, text)
            continue

        url = _extract_image_url(part)
        if url is not None:
            _set_message_content_image(span, message_base_key, content_index, str(url))
            continue

        text = part.get("text")
        if text is not None:
            _set_message_content_text(span, message_base_key, content_index, str(text))
            continue

        text, _ = _serialise(part)
        _set_message_content_text(span, message_base_key, content_index, text)


def _message_content_base(message_base_key: str, content_index: int) -> str:
    return (
        f"{message_base_key}.{MessageAttributes.MESSAGE_CONTENTS}." f"{content_index}"
    )


def _set_message_content_text(
    span,
    message_base_key: str,
    content_index: int,
    text: str,
):
    base = _message_content_base(message_base_key, content_index)
    try:
        span.set_attribute(
            f"{base}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
            "text",
        )
        span.set_attribute(
            f"{base}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
            text,
        )
    except Exception:
        pass


def _set_message_content_image(
    span,
    message_base_key: str,
    content_index: int,
    image_url: str,
):
    base = _message_content_base(message_base_key, content_index)
    try:
        span.set_attribute(
            f"{base}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
            "image",
        )
        span.set_attribute(
            f"{base}.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
            f"{ImageAttributes.IMAGE_URL}",
            image_url,
        )
    except Exception:
        pass


__all__ = [
    # Setup
    "register",
    "get_tracer",
    "get_tracker",
    "configure",
    "configure_media_reference_resolver",
    # Span helpers
    "start_span",
    "start_as_current_span",
    "set_llm_io",
    # Constants
    "MAX_FIELD_BYTES",
    "MarieSpanAttributes",
    # OI primitives
    "TracerProvider",
    "OITracer",
    "TraceConfig",
    "suppress_tracing",
    "capture_span_context",
]
