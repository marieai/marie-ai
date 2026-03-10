from __future__ import annotations

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry.sdk.trace import SpanProcessor


class OpenInferenceSpanProcessor(SpanProcessor):
    """Enriches non-OI spans with inferred OpenInference attributes."""

    def on_start(self, span, parent_context=None):
        if not span.is_recording():
            return
        # Skip if already has OI span kind (set by OITracer or OTelExporter)
        if (
            span.attributes
            and SpanAttributes.OPENINFERENCE_SPAN_KIND in span.attributes
        ):
            return
        # Infer OI kind from span name for infra spans
        kind = self._infer_from_span_name(span.name)
        if kind:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind.value)

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        return True

    @staticmethod
    def _infer_from_span_name(name: str) -> OpenInferenceSpanKindValues | None:
        lower = name.lower()
        # gRPC service methods
        if "/" in lower:
            return OpenInferenceSpanKindValues.CHAIN
        # FastAPI endpoints
        if lower.startswith(("get ", "post ", "put ", "delete ", "patch ")):
            return OpenInferenceSpanKindValues.CHAIN
        return None
