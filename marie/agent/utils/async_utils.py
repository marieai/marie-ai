"""Async utilities for Marie agent framework.

This module re-exports async utilities from marie.helper for convenience,
providing a consistent import path within the agent framework.
"""

from marie.helper import get_or_reuse_loop, run_async

# Alias for backward compatibility and naming consistency
asyncio_run = run_async

__all__ = [
    "run_async",
    "asyncio_run",
    "get_or_reuse_loop",
]
