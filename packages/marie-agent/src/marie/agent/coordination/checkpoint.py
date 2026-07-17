"""Checkpoint store for workflow state persistence.

Provides PostgreSQL-based checkpointing using the existing AsyncPostgresPool,
plus an in-memory implementation for testing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from marie.agent.coordination.state import AgentWorkflowState

logger = logging.getLogger("marie.agent.coordination.checkpoint")


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for workflow state checkpoint storage."""

    async def save(self, workflow_id: str, state: "AgentWorkflowState") -> None:
        """Save workflow state checkpoint."""
        ...

    async def load(self, workflow_id: str) -> Optional["AgentWorkflowState"]:
        """Load workflow state from checkpoint."""
        ...

    async def delete(self, workflow_id: str) -> None:
        """Delete workflow checkpoint."""
        ...

    async def list_checkpoints(self, prefix: Optional[str] = None) -> list[str]:
        """List all checkpoint workflow IDs."""
        ...


class InMemoryCheckpointStore:
    """In-memory checkpoint storage for testing.

    Not suitable for production - checkpoints are lost on restart.
    """

    def __init__(self):
        self._checkpoints: Dict[str, Dict[str, Any]] = {}

    async def save(self, workflow_id: str, state: "AgentWorkflowState") -> None:
        """Save workflow state checkpoint to memory."""
        now = datetime.now(timezone.utc).isoformat()
        state_dict = state.to_dict()

        if workflow_id in self._checkpoints:
            self._checkpoints[workflow_id]["state"] = state_dict
            self._checkpoints[workflow_id]["updated_at"] = now
        else:
            self._checkpoints[workflow_id] = {
                "state": state_dict,
                "created_at": now,
                "updated_at": now,
            }
        logger.debug(f"Saved checkpoint for workflow {workflow_id}")

    async def load(self, workflow_id: str) -> Optional["AgentWorkflowState"]:
        """Load workflow state from memory."""
        from marie.agent.coordination.state import AgentWorkflowState

        checkpoint = self._checkpoints.get(workflow_id)
        if checkpoint is None:
            return None

        return AgentWorkflowState.from_dict(checkpoint["state"])

    async def delete(self, workflow_id: str) -> None:
        """Delete workflow checkpoint from memory."""
        self._checkpoints.pop(workflow_id, None)
        logger.debug(f"Deleted checkpoint for workflow {workflow_id}")

    async def list_checkpoints(self, prefix: Optional[str] = None) -> list[str]:
        """List all checkpoint workflow IDs."""
        if prefix:
            return [wid for wid in self._checkpoints.keys() if wid.startswith(prefix)]
        return list(self._checkpoints.keys())

    def clear(self) -> None:
        """Clear all checkpoints (for testing)."""
        self._checkpoints.clear()
