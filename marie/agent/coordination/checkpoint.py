"""Checkpoint store for workflow state persistence.

Provides PostgreSQL-based checkpointing using the existing AsyncPostgresPool,
plus an in-memory implementation for testing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, runtime_checkable

from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.agent.coordination.state import AgentWorkflowState

logger = MarieLogger("marie.agent.coordination.checkpoint")


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


class PostgresCheckpointStore:
    """PostgreSQL-based checkpoint storage using AsyncPostgresPool.

    Uses the existing Marie PostgreSQL infrastructure for persistence.
    Table is created automatically on first use.

    Example:
        ```python
        from marie.storage.database.asyncpg_pool import AsyncPostgresPool

        # Initialize pool (usually done at app startup)
        pool = AsyncPostgresPool.get_instance()
        await pool.initialize(config)

        # Create checkpoint store
        store = PostgresCheckpointStore(pool)
        await store.initialize()

        # Use store
        await store.save(workflow_id, state)
        ```
    """

    TABLE_NAME = "workflow_checkpoints"

    def __init__(
        self,
        pool: Optional[Any] = None,
        schema: str = "public",
        table_name: Optional[str] = None,
    ):
        """Initialize PostgreSQL checkpoint store.

        Args:
            pool: AsyncPostgresPool instance. If None, uses singleton.
            schema: Database schema (default: public)
            table_name: Custom table name (default: workflow_checkpoints)
        """
        self._pool = pool
        self._schema = schema
        self._table_name = table_name or self.TABLE_NAME
        self._initialized = False

    @property
    def qualified_table(self) -> str:
        """Get fully qualified table name."""
        return f"{self._schema}.{self._table_name}"

    async def _get_pool(self):
        """Get the connection pool, using singleton if not provided."""
        if self._pool is not None:
            return self._pool

        from marie.storage.database.asyncpg_pool import AsyncPostgresPool

        pool = AsyncPostgresPool.get_instance()
        if not pool.is_initialized:
            raise RuntimeError(
                "AsyncPostgresPool not initialized. "
                "Call pool.initialize(config) first or pass an initialized pool."
            )
        return pool

    async def initialize(self) -> None:
        """Create the checkpoint table if it doesn't exist."""
        if self._initialized:
            return

        pool = await self._get_pool()
        await pool.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.qualified_table} (
                workflow_id VARCHAR(256) PRIMARY KEY,
                state_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create index on updated_at for listing recent checkpoints
        await pool.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_updated
            ON {self.qualified_table} (updated_at DESC)
        """
        )

        self._initialized = True
        logger.info(f"Initialized checkpoint table: {self.qualified_table}")

    async def save(self, workflow_id: str, state: "AgentWorkflowState") -> None:
        """Save workflow state checkpoint to PostgreSQL."""
        await self.initialize()
        pool = await self._get_pool()

        state_dict = state.to_dict()
        state_json = json.dumps(state_dict, default=str)

        await pool.execute(
            f"""
            INSERT INTO {self.qualified_table} (workflow_id, state_json, created_at, updated_at)
            VALUES ($1, $2, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (workflow_id) DO UPDATE SET
                state_json = EXCLUDED.state_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            workflow_id,
            state_json,
        )
        logger.debug(f"Saved checkpoint for workflow {workflow_id}")

    async def load(self, workflow_id: str) -> Optional["AgentWorkflowState"]:
        """Load workflow state from PostgreSQL."""
        from marie.agent.coordination.state import AgentWorkflowState

        await self.initialize()
        pool = await self._get_pool()

        row = await pool.fetchrow(
            f"SELECT state_json FROM {self.qualified_table} WHERE workflow_id = $1",
            workflow_id,
        )

        if row is None:
            return None

        state_dict = json.loads(row["state_json"])
        return AgentWorkflowState.from_dict(state_dict)

    async def delete(self, workflow_id: str) -> None:
        """Delete workflow checkpoint from PostgreSQL."""
        await self.initialize()
        pool = await self._get_pool()

        await pool.execute(
            f"DELETE FROM {self.qualified_table} WHERE workflow_id = $1",
            workflow_id,
        )
        logger.debug(f"Deleted checkpoint for workflow {workflow_id}")

    async def list_checkpoints(self, prefix: Optional[str] = None) -> list[str]:
        """List all checkpoint workflow IDs."""
        await self.initialize()
        pool = await self._get_pool()

        if prefix:
            rows = await pool.fetch(
                f"SELECT workflow_id FROM {self.qualified_table} WHERE workflow_id LIKE $1 ORDER BY updated_at DESC",
                f"{prefix}%",
            )
        else:
            rows = await pool.fetch(
                f"SELECT workflow_id FROM {self.qualified_table} ORDER BY updated_at DESC"
            )

        return [row["workflow_id"] for row in rows]

    async def cleanup_old_checkpoints(
        self,
        max_age_hours: int = 24,
        keep_completed: bool = False,
    ) -> int:
        """Delete checkpoints older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours
            keep_completed: If True, only delete non-completed workflows

        Returns:
            Number of checkpoints deleted
        """
        await self.initialize()
        pool = await self._get_pool()

        if keep_completed:
            # Only delete non-completed workflows
            result = await pool.execute(
                f"""
                DELETE FROM {self.qualified_table}
                WHERE updated_at < CURRENT_TIMESTAMP - INTERVAL '{max_age_hours} hours'
                AND state_json->>'status' NOT IN ('completed', 'failed')
                """,
            )
        else:
            result = await pool.execute(
                f"""
                DELETE FROM {self.qualified_table}
                WHERE updated_at < CURRENT_TIMESTAMP - INTERVAL '{max_age_hours} hours'
                """,
            )

        # Parse "DELETE N" result
        count = int(result.split()[-1]) if result else 0
        logger.info(f"Cleaned up {count} old checkpoints")
        return count
