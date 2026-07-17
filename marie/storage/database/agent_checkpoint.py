"""PostgreSQL checkpoint storage for reusable agent workflows."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marie.agent.coordination.state import AgentWorkflowState

logger = logging.getLogger(__name__)


class PostgresCheckpointStore:
    """Persist agent workflow checkpoints through Marie's PostgreSQL pool."""

    TABLE_NAME = "workflow_checkpoints"

    def __init__(
        self,
        pool: Any | None = None,
        schema: str = "public",
        table_name: str | None = None,
    ) -> None:
        self._pool = pool
        self._schema = schema
        self._table_name = table_name or self.TABLE_NAME
        self._initialized = False

    @property
    def qualified_table(self) -> str:
        return f"{self._schema}.{self._table_name}"

    async def _get_pool(self) -> Any:
        if self._pool is not None:
            return self._pool

        from marie.storage.database.asyncpg_pool import AsyncPostgresPool

        pool = AsyncPostgresPool.get_instance()
        if not pool.is_initialized:
            raise RuntimeError(
                "AsyncPostgresPool is not initialized; initialize it or pass a pool"
            )
        return pool

    async def initialize(self) -> None:
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
        await pool.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_updated
            ON {self.qualified_table} (updated_at DESC)
            """
        )
        self._initialized = True

    async def save(self, workflow_id: str, state: AgentWorkflowState) -> None:
        await self.initialize()
        pool = await self._get_pool()
        await pool.execute(
            f"""
            INSERT INTO {self.qualified_table}
                (workflow_id, state_json, created_at, updated_at)
            VALUES ($1, $2, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (workflow_id) DO UPDATE SET
                state_json = EXCLUDED.state_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            workflow_id,
            json.dumps(state.to_dict(), default=str),
        )

    async def load(self, workflow_id: str) -> AgentWorkflowState | None:
        from marie.agent.coordination.state import AgentWorkflowState

        await self.initialize()
        pool = await self._get_pool()
        row = await pool.fetchrow(
            f"SELECT state_json FROM {self.qualified_table} WHERE workflow_id = $1",
            workflow_id,
        )
        if row is None:
            return None
        return AgentWorkflowState.from_dict(json.loads(row["state_json"]))

    async def delete(self, workflow_id: str) -> None:
        await self.initialize()
        pool = await self._get_pool()
        await pool.execute(
            f"DELETE FROM {self.qualified_table} WHERE workflow_id = $1",
            workflow_id,
        )

    async def list_checkpoints(self, prefix: str | None = None) -> list[str]:
        await self.initialize()
        pool = await self._get_pool()
        if prefix:
            rows = await pool.fetch(
                f"SELECT workflow_id FROM {self.qualified_table} "
                "WHERE workflow_id LIKE $1 ORDER BY updated_at DESC",
                f"{prefix}%",
            )
        else:
            rows = await pool.fetch(
                f"SELECT workflow_id FROM {self.qualified_table} "
                "ORDER BY updated_at DESC"
            )
        return [row["workflow_id"] for row in rows]

    async def cleanup_old_checkpoints(
        self,
        max_age_hours: int = 24,
        keep_completed: bool = False,
    ) -> int:
        await self.initialize()
        pool = await self._get_pool()
        completed_filter = (
            "AND state_json->>'status' NOT IN ('completed', 'failed')"
            if keep_completed
            else ""
        )
        result = await pool.execute(
            f"""
            DELETE FROM {self.qualified_table}
            WHERE updated_at < CURRENT_TIMESTAMP - INTERVAL '{max_age_hours} hours'
            {completed_filter}
            """
        )
        count = int(result.split()[-1]) if result else 0
        logger.info("Cleaned up %s old workflow checkpoints", count)
        return count
