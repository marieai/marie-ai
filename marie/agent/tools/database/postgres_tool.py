"""PostgreSQL database tool for direct SQL execution."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.agent.tools import ToolMetadata, ToolOutput
from marie.agent.tools.database.base import AsyncDatabaseTool
from marie.storage.database.asyncpg_pool import AsyncPostgresPool


class PostgresAction(str, Enum):
    """Available actions for PostgresTool."""

    EXECUTE_SQL = "execute_sql"
    GET_SCHEMA = "get_schema"


class PostgresInput(BaseModel):
    """Input schema for PostgresTool."""

    action: PostgresAction = Field(
        ...,
        description="Action to perform: 'execute_sql' to run a query, 'get_schema' to get table/database schema",
    )
    query: Optional[str] = Field(
        None,
        description="SQL query to execute (required for execute_sql)",
    )
    table_name: Optional[str] = Field(
        None,
        description="Table name for get_schema (if omitted, returns all tables)",
    )
    schema_name: Optional[str] = Field(
        "public",
        description="Schema name for get_schema (default: public)",
    )


class PostgresTool(AsyncDatabaseTool):
    """Direct PostgreSQL database access tool.

    Provides SQL execution and schema inspection capabilities.
    No user scoping - this is for direct database access.

    Actions:
        - execute_sql: Execute arbitrary SQL queries (SELECT, INSERT, UPDATE, DELETE)
        - get_schema: Get database or table schema information

    Safety:
        - Results are limited to MAX_ROWS to prevent memory issues
        - Use for trusted contexts only (no user-facing SQL injection protection)
    """

    MAX_ROWS = 1000

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="postgres",
            description=(
                "Execute SQL queries and inspect PostgreSQL database schema. "
                "Actions: execute_sql (run queries), get_schema (inspect tables)."
            ),
            fn_schema=PostgresInput,
        )

    async def _create_tables(self, pool: AsyncPostgresPool) -> None:
        """No tables needed for PostgresTool."""
        pass

    async def acall(
        self,
        action: PostgresAction,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        schema_name: str = "public",
        **kwargs,
    ) -> ToolOutput:
        """Execute the requested action.

        Args:
            action: The action to perform
            query: SQL query for execute_sql action
            table_name: Table name for get_schema (optional)
            schema_name: Schema name for get_schema
            **kwargs: Additional arguments (ignored)

        Returns:
            ToolOutput with query results or schema information
        """
        raw_input = {
            "action": action,
            "query": query,
            "table_name": table_name,
            "schema_name": schema_name,
        }

        pool = await self._get_pool()

        if action == PostgresAction.EXECUTE_SQL:
            return await self._execute_sql(pool, query, raw_input)
        elif action == PostgresAction.GET_SCHEMA:
            return await self._get_schema(pool, table_name, schema_name, raw_input)
        else:
            return self._error_output(raw_input, f"Unknown action: {action}")

    async def _execute_sql(
        self,
        pool: AsyncPostgresPool,
        query: Optional[str],
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Execute a SQL query."""
        if not query:
            return self._error_output(raw_input, "query is required for execute_sql")

        query_lower = query.strip().lower()

        if query_lower.startswith("select"):
            rows = await pool.fetch(query)
            result = self._records_to_dicts(rows[: self.MAX_ROWS])
            truncated = len(rows) > self.MAX_ROWS
            output = {
                "rows": result,
                "row_count": len(result),
                "truncated": truncated,
            }
            if truncated:
                output["message"] = f"Results limited to {self.MAX_ROWS} rows"
            return self._create_output(raw_input, output)
        else:
            status = await pool.execute(query)
            return self._create_output(raw_input, {"status": status})

    async def _get_schema(
        self,
        pool: AsyncPostgresPool,
        table_name: Optional[str],
        schema_name: str,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Get schema information."""
        if table_name:
            return await self._get_table_schema(
                pool, table_name, schema_name, raw_input
            )
        else:
            return await self._get_all_tables(pool, schema_name, raw_input)

    async def _get_table_schema(
        self,
        pool: AsyncPostgresPool,
        table_name: str,
        schema_name: str,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Get schema for a specific table."""
        query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
        """
        rows = await pool.fetch(query, schema_name, table_name)

        if not rows:
            return self._error_output(
                raw_input,
                f"Table '{schema_name}.{table_name}' not found",
            )

        columns = self._records_to_dicts(rows)

        indexes = await self._get_table_indexes(pool, table_name, schema_name)
        constraints = await self._get_table_constraints(pool, table_name, schema_name)

        return self._create_output(
            raw_input,
            {
                "table": f"{schema_name}.{table_name}",
                "columns": columns,
                "indexes": indexes,
                "constraints": constraints,
            },
        )

    async def _get_table_indexes(
        self,
        pool: AsyncPostgresPool,
        table_name: str,
        schema_name: str,
    ) -> List[Dict[str, Any]]:
        """Get indexes for a table."""
        query = """
            SELECT
                indexname as index_name,
                indexdef as definition
            FROM pg_indexes
            WHERE schemaname = $1 AND tablename = $2
        """
        rows = await pool.fetch(query, schema_name, table_name)
        return self._records_to_dicts(rows)

    async def _get_table_constraints(
        self,
        pool: AsyncPostgresPool,
        table_name: str,
        schema_name: str,
    ) -> List[Dict[str, Any]]:
        """Get constraints for a table."""
        query = """
            SELECT
                tc.constraint_name,
                tc.constraint_type,
                kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = $1 AND tc.table_name = $2
        """
        rows = await pool.fetch(query, schema_name, table_name)
        return self._records_to_dicts(rows)

    async def _get_all_tables(
        self,
        pool: AsyncPostgresPool,
        schema_name: str,
        raw_input: Dict[str, Any],
    ) -> ToolOutput:
        """Get all tables in a schema."""
        query = """
            SELECT
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema = $1
            ORDER BY table_name
        """
        rows = await pool.fetch(query, schema_name)
        tables = self._records_to_dicts(rows)

        return self._create_output(
            raw_input,
            {
                "schema": schema_name,
                "tables": tables,
                "table_count": len(tables),
            },
        )

    @staticmethod
    def _records_to_dicts(records: list) -> List[Dict[str, Any]]:
        """Convert asyncpg Records to dictionaries."""
        return [dict(r) for r in records]
