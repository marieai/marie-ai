from contextvars import ContextVar
from typing import Optional

from marie.core.instrumentation.span.base import BaseSpan
from marie.core.instrumentation.span.simple import SimpleSpan

# ContextVar for managing active spans
active_span_id: ContextVar[Optional[str]] = ContextVar("active_span_id", default=None)
active_span_id.set(None)

__all__ = ["BaseSpan", "SimpleSpan", "active_span_id"]
