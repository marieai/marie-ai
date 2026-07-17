"""Base class for database-backed agent tools using asyncpg."""

from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Dict, Optional

from marie.agent.tools import AgentTool, ToolOutput
from marie.helper import run_async
from marie.storage.database.asyncpg_pool import AsyncPostgresPool


class AsyncDatabaseTool(AgentTool):
    """Base class for database-backed tools using asyncpg.

    Provides:
    - Automatic pool initialization
    - Schema and table creation
    - Sync/async bridging via run_async
    - Common output helpers

    Subclasses must implement:
    - metadata property: Return ToolMetadata
    - _create_tables: Create tool-specific tables
    - acall: Async execution logic
    """

    SCHEMA = "agent_tools"

    def __init__(
        self,
        config: Dict[str, Any],
        pool: Optional[AsyncPostgresPool] = None,
    ):
        """Initialize the tool.

        Args:
            config: Database configuration matching PostgresqlMixin pattern
            pool: Optional existing pool instance (for sharing across tools)
        """
        self._config = config
        self._pool = pool
        self._schema_ensured = False

    @property
    def name(self) -> str:
        return self.metadata.name

    def call(self, **kwargs) -> ToolOutput:
        """Sync execution via run_async."""
        return run_async(self.acall(**kwargs))

    async def _get_pool(self) -> AsyncPostgresPool:
        """Get or initialize the pool."""
        if self._pool is None:
            self._pool = AsyncPostgresPool.get_instance()
            await self._pool.initialize(self._config)
        return self._pool

    async def _ensure_schema(self) -> None:
        """Create schema and tables if they don't exist."""
        if self._schema_ensured:
            return
        pool = await self._get_pool()
        await pool.execute(f"CREATE SCHEMA IF NOT EXISTS {self.SCHEMA}")
        await self._create_tables(pool)
        self._schema_ensured = True

    @abstractmethod
    async def _create_tables(self, pool: AsyncPostgresPool) -> None:
        """Create tool-specific tables. Override in subclasses."""
        pass

    def _create_output(
        self,
        raw_input: Dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> ToolOutput:
        """Helper to create consistent ToolOutput.

        Args:
            raw_input: Original input parameters
            result: The result to include
            is_error: Whether this is an error response

        Returns:
            Formatted ToolOutput
        """
        if is_error:
            content = str(result)
        else:
            content = json.dumps(result, default=str, ensure_ascii=False)

        return ToolOutput(
            content=content,
            tool_name=self.name,
            raw_input=raw_input,
            raw_output=result,
            is_error=is_error,
        )

    def _error_output(self, raw_input: Dict[str, Any], message: str) -> ToolOutput:
        """Create an error ToolOutput."""
        return self._create_output(raw_input, message, is_error=True)
