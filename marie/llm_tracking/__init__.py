"""
Marie LLM Tracking - Langfuse-inspired LLM Observability.

This module provides comprehensive LLM observability for the Marie AI ecosystem.
It captures detailed telemetry from LLM calls including:
- Traces: Request-level containers
- Observations: Spans, Generations, Events within traces
- Scores: Evaluation metrics for traces/observations
- Token usage and cost tracking

Usage:
    from marie.llm_tracking import get_tracker

    tracker = get_tracker()

    # Create a trace for a request
    with tracker.trace("my-request", user_id="user-123") as trace:
        # Start a generation observation
        gen_id = tracker.generation(
            trace_id=trace.id,
            name="openai_completion",
            model="gpt-4",
            input=messages,
        )

        # Make the API call
        response = openai.chat.completions.create(model="gpt-4", messages=messages)

        # End with output and usage
        tracker.end(gen_id, output=response.content, usage=response.usage)

    # Add a score
    tracker.score(trace.id, "quality", 0.9)

Configuration:
    Set environment variables to configure:
    - MARIE_LLM_TRACKING_ENABLED=true/false
    - MARIE_LLM_TRACKING_EXPORTER=console/rabbitmq
    - MARIE_LLM_TRACKING_PROJECT_ID=your-project

    See config.py for full list of options.
"""

from marie.llm_tracking.config import ExporterType, LLMTrackingSettings, get_settings
from marie.llm_tracking.tracker import LLMTracker, TraceContext, get_tracker
from marie.llm_tracking.types import (
    Cost,
    EventType,
    Observation,
    ObservationLevel,
    ObservationType,
    QueueMessage,
    RawEvent,
    Score,
    ScoreDataType,
    Trace,
    Usage,
)


# Lazy imports for optional components (to avoid import errors if deps not installed)
def get_clickhouse_writer():
    """Get the ClickHouse writer (lazy import)."""
    from marie.llm_tracking.clickhouse.writer import (
        get_clickhouse_writer as _get_writer,
    )

    return _get_writer()


def get_normalizer():
    """Get the event normalizer (lazy import)."""
    from marie.llm_tracking.normalizer import get_normalizer as _get_normalizer

    return _get_normalizer()


__all__ = [
    # Main tracker
    "LLMTracker",
    "TraceContext",
    "get_tracker",
    # Configuration
    "LLMTrackingSettings",
    "ExporterType",
    "get_settings",
    # Types
    "Trace",
    "Observation",
    "Score",
    "Usage",
    "Cost",
    "RawEvent",
    "QueueMessage",
    "EventType",
    "ObservationType",
    "ObservationLevel",
    "ScoreDataType",
    # Worker components (lazy)
    "get_clickhouse_writer",
    "get_normalizer",
]

__version__ = "0.1.0"
