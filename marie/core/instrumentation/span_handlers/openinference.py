import inspect
from typing import Any, Dict, Optional

from opentelemetry import context as otel_context, trace
from opentelemetry.trace import StatusCode, Status
from openinference.semconv.trace import OpenInferenceSpanKindValues

from marie.core.bridge.pydantic import PrivateAttr
from marie.core.instrumentation.span_handlers.base import BaseSpanHandler


class OpenInferenceSpanHandler(BaseSpanHandler):
    """
    Creates OTel spans with OI attributes from Dispatcher lifecycle.
    Preserves parent/child relationships via OTel Context propagation.
    """

    _tracer: Optional[Any] = PrivateAttr(default=None)
    _context_tokens: dict = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._tracer = None
        self._context_tokens = {}

    @property
    def tracer(self):
        if self._tracer is None:
            self._tracer = trace.get_tracer("marie.dispatcher")
        return self._tracer

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Create an OTel span as a child of the resolved parent.

        Args:
            parent_span_id: Dispatcher span ID of the parent (from active_span_id ContextVar).
                           Maps to a key in self.open_spans where the value is an OTel Span.
        """
        kind = self._infer_kind(instance)
        parent_ctx = self._resolve_parent_context(parent_span_id)

        span = self.tracer.start_span(
            name=self._span_name(instance, bound_args),
            context=parent_ctx,
            openinference_span_kind=kind,
        )

        if tags:
            for k, v in tags.items():
                span.set_attribute(f"marie.tag.{k}", str(v))

        # Make this dispatcher span the CURRENT OTel span for the wrapped function body.
        self._context_tokens[id_] = otel_context.attach(trace.set_span_in_context(span))
        return span  # stored in open_spans[id_] by BaseSpanHandler.span_enter()

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ):
        span = self.open_spans.get(id_)
        if span is None:
            return None
        if hasattr(span, 'is_recording') and span.is_recording():
            if result is not None:
                from openinference.instrumentation import get_output_attributes
                for k, v in get_output_attributes(result).items():
                    span.set_attribute(k, v)
            span.set_status(Status(StatusCode.OK))
            token = self._context_tokens.pop(id_, None)
            if token is not None:
                otel_context.detach(token)
            span.end()
        return span  # MUST return span so BaseSpanHandler.span_exit() removes it from open_spans

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ):
        span = self.open_spans.get(id_)
        if span is None:
            return None
        if hasattr(span, 'is_recording') and span.is_recording():
            span.set_status(Status(StatusCode.ERROR, str(err) if err else ""))
            if err:
                span.record_exception(err)
            token = self._context_tokens.pop(id_, None)
            if token is not None:
                otel_context.detach(token)
            span.end()
        return span  # MUST return span so BaseSpanHandler.span_drop() removes it from open_spans

    def _resolve_parent_context(self, parent_span_id: Optional[str] = None):
        """
        Resolve OTel Context for parent span.

        If parent_span_id is set, look up the parent's OTel span from open_spans
        (where BaseSpanHandler stored it) and create a Context carrying it.
        Otherwise, use the ambient OTel context (picks up gRPC/FastAPI span).
        """
        if parent_span_id is not None:
            parent_otel_span = self.open_spans.get(parent_span_id)
            if parent_otel_span is not None:
                return trace.set_span_in_context(parent_otel_span)
        # No explicit parent -> inherit from ambient OTel context
        return otel_context.get_current()

    @staticmethod
    def _span_name(instance, bound_args) -> str:
        if instance is not None:
            return type(instance).__name__
        if hasattr(bound_args, 'args') and bound_args.args:
            func = bound_args.args[0] if callable(bound_args.args[0]) else None
            if func and hasattr(func, '__qualname__'):
                return func.__qualname__
        return "unknown"

    @staticmethod
    def _infer_kind(instance) -> OpenInferenceSpanKindValues:
        cls_name = type(instance).__name__.lower() if instance else ""
        if "agent" in cls_name:
            return OpenInferenceSpanKindValues.AGENT
        if "llm" in cls_name:
            return OpenInferenceSpanKindValues.LLM
        if "tool" in cls_name:
            return OpenInferenceSpanKindValues.TOOL
        if "retriev" in cls_name:
            return OpenInferenceSpanKindValues.RETRIEVER
        if "embed" in cls_name:
            return OpenInferenceSpanKindValues.EMBEDDING
        return OpenInferenceSpanKindValues.CHAIN
