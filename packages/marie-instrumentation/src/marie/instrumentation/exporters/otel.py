from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from marie.instrumentation.exporters.base import AbstractExporter
from marie.instrumentation.openinference import (
    infer_span_kind,
    observation_to_span_attributes,
)
from marie.instrumentation.types import Observation, ObservationLevel, Score, Trace

logger = logging.getLogger(__name__)


def _to_otel_time(dt: Optional[datetime]) -> Optional[int]:
    """Convert a datetime to OTel nanosecond timestamp."""
    if dt is None:
        return None
    return int(dt.timestamp() * 1e9)


def safe_json_dumps(obj: Any) -> str:
    """Serialize obj to JSON string, falling back to str() on failure."""
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return str(obj)


class OTelExporter(AbstractExporter):
    """
    Exports LLMTracker observations as OTel spans with correct parent/child nesting.

    Span ownership:
      _trace_root_spans[trace_id] -> the root span for a tracker trace (stays open)
      _trace_root_tokens[trace_id] -> OTel context token for the attached trace root
      _active_spans[observation_id] -> span for each observation (open until end())
      _span_context_tokens[observation_id] -> OTel context token for the attached observation span

    Parent resolution order for observations:
      1. parent_observation_id -> look up in _active_spans
      2. Current OTel Context (ambient gRPC/FastAPI span, or currently-attached tracker/dispatcher span)
      3. trace_id -> fall back to _trace_root_spans
    """

    def __init__(self, tracer_name: str = "marie.instrumentation"):
        self._tracer = None
        self._tracer_name = tracer_name
        self._active_spans: dict[str, trace.Span] = {}
        self._trace_root_spans: dict[str, trace.Span] = {}
        self._trace_root_tokens: dict[str, object] = {}
        self._span_context_tokens: dict[str, object] = {}

    @property
    def tracer(self):
        if self._tracer is None:
            self._tracer = trace.get_tracer(self._tracer_name)
        return self._tracer

    # -- Trace lifecycle -------------------------------------------------------

    def export_trace(self, trace_obj: Trace) -> None:
        """
        Called FIRST -- when tracker.trace() is entered.
        Creates the trace root span as a child of the current OTel context
        (e.g., gRPC interceptor span). Stays open until finalize_trace().
        """
        root_span = self.tracer.start_span(
            name=trace_obj.name or "trace",
            start_time=_to_otel_time(trace_obj.timestamp),
            openinference_span_kind=OpenInferenceSpanKindValues.CHAIN,
        )

        if trace_obj.session_id:
            root_span.set_attribute(SpanAttributes.SESSION_ID, trace_obj.session_id)
        if trace_obj.user_id:
            root_span.set_attribute(SpanAttributes.USER_ID, trace_obj.user_id)
        if trace_obj.metadata:
            root_span.set_attribute(
                SpanAttributes.METADATA, safe_json_dumps(trace_obj.metadata)
            )
        if trace_obj.tags:
            root_span.set_attribute(SpanAttributes.TAG_TAGS, trace_obj.tags)

        root_token = otel_context.attach(trace.set_span_in_context(root_span))
        self._trace_root_spans[trace_obj.id] = root_span
        self._trace_root_tokens[trace_obj.id] = root_token

    def finalize_trace(self, trace_obj: Trace) -> None:
        """
        Called LAST -- when tracker.trace() context manager exits.
        Ends the trace root span after all child observations are done.
        """
        root_span = self._trace_root_spans.pop(trace_obj.id, None)
        root_token = self._trace_root_tokens.pop(trace_obj.id, None)
        if root_span is not None:
            if root_token is not None:
                otel_context.detach(root_token)
            if trace_obj.output is not None:
                root_span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE, str(trace_obj.output)
                )
            if trace_obj.metadata:
                root_span.set_attribute(
                    SpanAttributes.METADATA, safe_json_dumps(trace_obj.metadata)
                )
            if trace_obj.tags:
                root_span.set_attribute(SpanAttributes.TAG_TAGS, trace_obj.tags)
            root_span.end(end_time=_to_otel_time(trace_obj.updated_at))

    # -- Observation lifecycle -------------------------------------------------

    def start_span(self, observation: Observation) -> None:
        """
        Called when tracker.generation() / tracker.span() is called.
        Creates an OTel span as a CHILD of the resolved parent.
        """
        parent_ctx = self._resolve_parent_context(observation)
        kind = infer_span_kind(observation.type, observation.name, observation.metadata)

        span = self.tracer.start_span(
            name=observation.name,
            context=parent_ctx,
            start_time=_to_otel_time(observation.start_time),
            openinference_span_kind=kind,
        )
        self._active_spans[observation.id] = span
        self._span_context_tokens[observation.id] = otel_context.attach(
            trace.set_span_in_context(span)
        )

    def export_observation(self, observation: Observation) -> None:
        """
        Called when tracker.end() is called.
        Sets final OI attributes and ends the span.
        """
        span = self._active_spans.pop(observation.id, None)
        if span is None:
            parent_ctx = self._resolve_parent_context(observation)
            kind = infer_span_kind(
                observation.type, observation.name, observation.metadata
            )
            span = self.tracer.start_span(
                name=observation.name,
                context=parent_ctx,
                start_time=_to_otel_time(observation.start_time),
                openinference_span_kind=kind,
            )

        attrs = observation_to_span_attributes(observation)
        for k, v in attrs.items():
            span.set_attribute(k, v)

        if observation.level == ObservationLevel.ERROR:
            span.set_status(StatusCode.ERROR, observation.status_message or "")

        token = self._span_context_tokens.pop(observation.id, None)
        if token is not None:
            otel_context.detach(token)
        span.end(
            end_time=(
                _to_otel_time(observation.end_time) if observation.end_time else None
            )
        )

    def export_score(self, score: Score) -> None:
        pass  # Deferred

    # -- Parent resolution -----------------------------------------------------

    def _resolve_parent_context(self, observation: Observation) -> otel_context.Context:
        """
        Resolve the OTel Context that should be the parent of this observation's span.

        Resolution order:
          1. parent_observation_id -> look up sibling span in _active_spans
          2. current ambient OTel span -> picks up the currently-attached dispatcher/span/root
          3. trace_id -> fall back to trace root span in _trace_root_spans
        """
        # 1. Explicit parent observation
        if observation.parent_observation_id:
            parent_span = self._active_spans.get(observation.parent_observation_id)
            if parent_span is not None:
                return trace.set_span_in_context(parent_span)

        # 2. Ambient current span (dispatcher span, current tracker span, or gRPC/FastAPI)
        current_span = trace.get_current_span()
        if current_span is not None and current_span.get_span_context().is_valid:
            return otel_context.get_current()

        # 3. Trace root span
        if observation.trace_id:
            root_span = self._trace_root_spans.get(observation.trace_id)
            if root_span is not None:
                return trace.set_span_in_context(root_span)

        # No parent found -> ambient OTel context
        return otel_context.get_current()

    # -- Cleanup ---------------------------------------------------------------

    def shutdown(self) -> None:
        """End any orphaned spans on shutdown."""
        for span in self._active_spans.values():
            span.end()
        self._active_spans.clear()
        for token in self._span_context_tokens.values():
            otel_context.detach(token)
        self._span_context_tokens.clear()
        for span in self._trace_root_spans.values():
            span.end()
        self._trace_root_spans.clear()
        for token in self._trace_root_tokens.values():
            otel_context.detach(token)
        self._trace_root_tokens.clear()

    def stop(self) -> None:
        """Close any tracker spans left open by interrupted requests."""
        self.shutdown()
