"""
Regression test: OTel context propagation across thread boundaries.

Verifies that spans created in:
  1. run_in_executor (LLMCall.aforward)
  2. batch_generate (detached span → attached)
  3. run_coroutine_in_current_loop (new thread)
all share the same trace_id and have correct parent linkage.
"""

import asyncio
import contextvars
import threading
from typing import List, Optional, Sequence

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from marie.engine.async_helper import run_coroutine_in_current_loop


class _InMemoryExporter(SpanExporter):
    """Minimal in-memory span exporter for testing."""

    def __init__(self):
        self._spans: List[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def get_finished_spans(self) -> List[ReadableSpan]:
        with self._lock:
            return list(self._spans)


def test_trace_context_across_thread_boundaries():
    """All spans across run_in_executor + run_coroutine_in_current_loop
    share the same trace_id and form a correct parent chain."""
    exporter = _InMemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test")

    async def _async_leaf():
        """Simulates completion_non_streaming creating a leaf span."""
        with tracer.start_as_current_span("leaf_span"):
            pass

    async def _run():
        # Simulate __acall_endpoint__ creating the root executor span
        with tracer.start_as_current_span("root_span"):
            # Simulate LLMCall.aforward() → run_in_executor with ctx propagation
            loop = asyncio.get_running_loop()
            ctx = contextvars.copy_context()

            def sync_work():
                # Simulate batch_generate() → attach batch span
                batch_span = tracer.start_span("batch_span")
                token = otel_context.attach(
                    otel_trace.set_span_in_context(batch_span)
                )
                try:
                    # Simulate run_coroutine_in_current_loop
                    run_coroutine_in_current_loop(_async_leaf())
                finally:
                    otel_context.detach(token)
                    batch_span.end()

            await loop.run_in_executor(None, ctx.run, sync_work)

    asyncio.run(_run())

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "root_span" in spans, f"Missing root_span, got: {list(spans.keys())}"
    assert "batch_span" in spans, f"Missing batch_span, got: {list(spans.keys())}"
    assert "leaf_span" in spans, f"Missing leaf_span, got: {list(spans.keys())}"

    root = spans["root_span"]
    batch = spans["batch_span"]
    leaf = spans["leaf_span"]

    # All spans share the same trace_id
    assert root.context.trace_id == batch.context.trace_id, (
        f"batch trace_id mismatch: root={root.context.trace_id:#x} "
        f"batch={batch.context.trace_id:#x}"
    )
    assert root.context.trace_id == leaf.context.trace_id, (
        f"leaf trace_id mismatch: root={root.context.trace_id:#x} "
        f"leaf={leaf.context.trace_id:#x}"
    )

    # Parent chain: leaf → batch → root
    assert batch.parent.span_id == root.context.span_id, (
        f"batch parent mismatch: expected={root.context.span_id:#x} "
        f"got={batch.parent.span_id:#x}"
    )
    assert leaf.parent.span_id == batch.context.span_id, (
        f"leaf parent mismatch: expected={batch.context.span_id:#x} "
        f"got={leaf.parent.span_id:#x}"
    )
