"""Task state mapping for A2A protocol.

This module provides utilities for mapping between Marie job states
and A2A task states.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from a2a.types import TaskState

logger = logging.getLogger(__name__)

# Terminal task states (no further transitions possible)
TERMINAL_STATES: frozenset[str] = frozenset(
    {"completed", "canceled", "failed", "rejected"}
)


class TaskStateMapper:
    """Maps between Marie job states and A2A task states.

    Provides bidirectional mapping for interoperability between
    Marie's internal job management and the A2A protocol.
    """

    # Marie JobStatus -> A2A TaskState string
    _MARIE_TO_A2A: dict[str, str] = {
        "pending": "submitted",
        "running": "working",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "canceled",
        "queued": "submitted",
        "processing": "working",
        "success": "completed",
        "error": "failed",
    }

    # A2A TaskState string -> Marie JobStatus
    _A2A_TO_MARIE: dict[str, str] = {
        "submitted": "pending",
        "working": "running",
        "input-required": "waiting",
        "completed": "completed",
        "canceled": "cancelled",
        "failed": "failed",
        "rejected": "failed",
        "auth-required": "waiting",
        "unknown": "unknown",
    }

    @classmethod
    def to_a2a(cls, marie_status: str) -> str:
        """Convert Marie job status to A2A task state.

        Args:
            marie_status: Marie job status string.

        Returns:
            Corresponding A2A task state string.
        """
        normalized = marie_status.lower().strip()
        return cls._MARIE_TO_A2A.get(normalized, "unknown")

    @classmethod
    def to_marie(cls, a2a_state: str) -> str:
        """Convert A2A task state to Marie job status.

        Args:
            a2a_state: A2A task state string.

        Returns:
            Corresponding Marie job status string.
        """
        normalized = a2a_state.lower().strip()
        return cls._A2A_TO_MARIE.get(normalized, "unknown")

    @classmethod
    def is_terminal(cls, state: str) -> bool:
        """Check if a task state is terminal (no further transitions)."""
        return state.lower() in TERMINAL_STATES
