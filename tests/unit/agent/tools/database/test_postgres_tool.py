"""Tests for PostgresTool."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from marie.agent.tools import ToolOutput
from marie.agent.tools.database.postgres_tool import PostgresAction, PostgresTool

from .conftest import MockRecord


class TestPostgresToolMetadata:
    """Tests for PostgresTool metadata."""

    def test_name(self, db_config):
        """Tool should have correct name."""
        tool = PostgresTool(db_config)
        assert tool.name == "postgres"

    def test_description(self, db_config):
        """Tool should have description."""
        tool = PostgresTool(db_config)
        assert "SQL" in tool.description
        assert "execute_sql" in tool.description
        assert "get_schema" in tool.description

    def test_metadata_has_schema(self, db_config):
        """Tool metadata should have fn_schema."""
        tool = PostgresTool(db_config)
        assert tool.metadata.fn_schema is not None


class TestPostgresToolExecuteSql:
    """Tests for execute_sql action."""

    @pytest.mark.asyncio
    async def test_execute_select_returns_rows(self, db_config, mock_pool):
        """execute_sql with SELECT should return rows."""
        mock_pool._mock_fetch = lambda q, a: [
            MockRecord({"id": 1, "name": "Alice"}),
            MockRecord({"id": 2, "name": "Bob"}),
        ]

        tool = PostgresTool(db_config, pool=mock_pool)
        result = await tool.acall(action=PostgresAction.EXECUTE_SQL, query="SELECT * FROM users")

        assert isinstance(result, ToolOutput)
        assert result.is_error is False
        data = json.loads(result.content)
        assert data["row_count"] == 2
        assert data["rows"][0]["name"] == "Alice"
        assert data["truncated"] is False

    @pytest.mark.asyncio
    async def test_execute_select_truncates_large_results(self, db_config, mock_pool):
        """execute_sql should truncate results exceeding MAX_ROWS."""
        # Create more rows than MAX_ROWS
        large_result = [MockRecord({"id": i}) for i in range(PostgresTool.MAX_ROWS + 100)]
        mock_pool._mock_fetch = lambda q, a: large_result

        tool = PostgresTool(db_config, pool=mock_pool)
        result = await tool.acall(action=PostgresAction.EXECUTE_SQL, query="SELECT * FROM large_table")

        data = json.loads(result.content)
        assert data["row_count"] == PostgresTool.MAX_ROWS
        assert data["truncated"] is True
        assert "limited" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_insert_returns_status(self, db_config, mock_pool):
        """execute_sql with INSERT should return status."""
        tool = PostgresTool(db_config, pool=mock_pool)
        result = await tool.acall(
            action=PostgresAction.EXECUTE_SQL,
            query="INSERT INTO users (name) VALUES ('Charlie')"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert "status" in data
        assert "INSERT" in data["status"]

    @pytest.mark.asyncio
    async def test_execute_without_query_returns_error(self, db_config, mock_pool):
        """execute_sql without query should return error."""
        tool = PostgresTool(db_config, pool=mock_pool)
        result = await tool.acall(action=PostgresAction.EXECUTE_SQL)

        assert result.is_error is True
        assert "query is required" in result.content.lower()

    @pytest.mark.asyncio
    async def test_execute_update_returns_status(self, db_config, mock_pool):
        """execute_sql with UPDATE should return status."""
        mock_pool._mock_execute = lambda q, a: "UPDATE 5"

        tool = PostgresTool(db_config, pool=mock_pool)
        result = await tool.acall(
            action=PostgresAction.EXECUTE_SQL,
            query="UPDATE users SET active = true WHERE id > 10"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["status"] == "UPDATE 5"


class TestPostgresToolGetSchema:
    """Tests for get_schema action."""

    @pytest.mark.asyncio
    async def test_get_all_tables(self, db_config, mock_pool):
        """get_schema without table_name should return all tables."""
        mock_pool._mock_fetch = lambda q, a: [
            MockRecord({"table_name": "users", "table_type": "BASE TABLE"}),
            MockRecord({"table_name": "orders", "table_type": "BASE TABLE"}),
        ]

        tool = PostgresTool(db_config, pool=mock_pool)
        result = await tool.acall(action=PostgresAction.GET_SCHEMA)

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["schema"] == "public"
        assert data["table_count"] == 2
        assert data["tables"][0]["table_name"] == "users"

    @pytest.mark.asyncio
    async def test_get_table_schema(self, db_config, mock_pool):
        """get_schema with table_name should return table details."""
        call_count = [0]

        def mock_fetch(query, args):
            call_count[0] += 1
            if "information_schema.columns" in query:
                return [
                    MockRecord({
                        "column_name": "id",
                        "data_type": "integer",
                        "is_nullable": "NO",
                        "column_default": "nextval('users_id_seq')",
                        "character_maximum_length": None,
                    }),
                    MockRecord({
                        "column_name": "name",
                        "data_type": "character varying",
                        "is_nullable": "YES",
                        "column_default": None,
                        "character_maximum_length": 255,
                    }),
                ]
            elif "pg_indexes" in query:
                return [
                    MockRecord({
                        "index_name": "users_pkey",
                        "definition": "CREATE UNIQUE INDEX users_pkey ON public.users USING btree (id)",
                    }),
                ]
            elif "table_constraints" in query:
                return [
                    MockRecord({
                        "constraint_name": "users_pkey",
                        "constraint_type": "PRIMARY KEY",
                        "column_name": "id",
                    }),
                ]
            return []

        mock_pool._mock_fetch = mock_fetch

        tool = PostgresTool(db_config, pool=mock_pool)
        result = await tool.acall(
            action=PostgresAction.GET_SCHEMA,
            table_name="users",
            schema_name="public"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["table"] == "public.users"
        assert len(data["columns"]) == 2
        assert data["columns"][0]["column_name"] == "id"
        assert len(data["indexes"]) == 1
        assert len(data["constraints"]) == 1

    @pytest.mark.asyncio
    async def test_get_schema_table_not_found(self, db_config, mock_pool):
        """get_schema should return error for non-existent table."""
        mock_pool._mock_fetch = lambda q, a: []

        tool = PostgresTool(db_config, pool=mock_pool)
        result = await tool.acall(
            action=PostgresAction.GET_SCHEMA,
            table_name="nonexistent"
        )

        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_get_schema_custom_schema(self, db_config, mock_pool):
        """get_schema should use custom schema_name."""
        mock_pool._mock_fetch = lambda q, a: [
            MockRecord({"table_name": "custom_table", "table_type": "BASE TABLE"}),
        ]

        tool = PostgresTool(db_config, pool=mock_pool)
        result = await tool.acall(
            action=PostgresAction.GET_SCHEMA,
            schema_name="custom_schema"
        )

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["schema"] == "custom_schema"


class TestPostgresToolSyncCall:
    """Tests for synchronous call method."""

    def test_call_bridges_to_acall(self, db_config, mock_pool):
        """call() should bridge to acall() via run_async."""
        mock_pool._mock_fetch = lambda q, a: [
            MockRecord({"table_name": "test", "table_type": "BASE TABLE"}),
        ]

        tool = PostgresTool(db_config, pool=mock_pool)
        result = tool.call(action=PostgresAction.GET_SCHEMA)

        assert isinstance(result, ToolOutput)
        assert result.is_error is False


class TestPostgresToolErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self, db_config, mock_pool):
        """Unknown action should return error."""
        tool = PostgresTool(db_config, pool=mock_pool)
        # Force an invalid action by bypassing Pydantic validation
        result = await tool.acall(action="invalid_action")

        assert result.is_error is True
        assert "unknown action" in result.content.lower()
