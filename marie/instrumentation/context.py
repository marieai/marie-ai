"""
Context propagation for OpenInference attributes.

Usage:
    from marie.instrumentation.context import using_attributes, using_session

    with using_attributes(session_id="sess-123", user_id="user-456", tags=["prod"]):
        # All spans created in this block inherit these attributes
        response = await agent.run(input)
"""

from __future__ import annotations

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
]
