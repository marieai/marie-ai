"""Tests for AsyncDatabaseTool base class."""
import json
from typing import Any, Dict

import pytest
from pydantic import BaseModel, Field

from marie.agent.tools import ToolMetadata, ToolOutput
from marie.agent.tools.database.base import AsyncDatabaseTool
from marie.storage.database.asyncpg_pool import AsyncPostgresPool

from .conftest import MockAsyncPostgresPool


class TestInput(BaseModel):
    """Test input schema."""
    value: str = Field(..., description="Test value")


class ConcreteAsyncDatabaseTool(AsyncDatabaseTool):
    """Concrete implementation for testing base class."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test_tool",
            description="A test database tool",
            fn_schema=TestInput,
        )

    async def _create_tables(self, pool: AsyncPostgresPool) -> None:
        await pool.execute(
            f"CREATE TABLE IF NOT EXISTS {self.SCHEMA}.test_table (id SERIAL PRIMARY KEY)"
        )

    async def acall(self, value: str, **kwargs) -> ToolOutput:
        await self._ensure_schema()
        return self._create_output({"value": value}, {"result": f"processed: {value}"})


class TestAsyncDatabaseToolBase:
    """Tests for AsyncDatabaseTool base class."""

    def test_schema_constant(self, db_config):
        """Tools should use agent_tools schema."""
        tool = ConcreteAsyncDatabaseTool(db_config)
        assert tool.SCHEMA == "agent_tools"

    def test_name_property(self, db_config):
        """name property should return metadata name."""
        tool = ConcreteAsyncDatabaseTool(db_config)
        assert tool.name == "test_tool"

    @pytest.mark.asyncio
    async def test_get_pool_initializes(self, db_config, mock_pool):
        """_get_pool should initialize pool if not set."""
        tool = ConcreteAsyncDatabaseTool(db_config, pool=mock_pool)

        pool = await tool._get_pool()

        assert pool is mock_pool
        assert mock_pool._initialized

    @pytest.mark.asyncio
    async def test_ensure_schema_creates_tables(self, db_config, mock_pool):
        """_ensure_schema should create schema and tables."""
        tool = ConcreteAsyncDatabaseTool(db_config, pool=mock_pool)

        await tool._ensure_schema()

        # Check that CREATE SCHEMA was called
        schema_calls = [
            call for call in mock_pool._execute_log
            if "CREATE SCHEMA" in call[1]
        ]
        assert len(schema_calls) == 1
        assert "agent_tools" in schema_calls[0][1]

        # Check that _create_tables was called (CREATE TABLE)
        table_calls = [
            call for call in mock_pool._execute_log
            if "CREATE TABLE" in call[1]
        ]
        assert len(table_calls) == 1

    @pytest.mark.asyncio
    async def test_ensure_schema_idempotent(self, db_config, mock_pool):
        """_ensure_schema should only run once."""
        tool = ConcreteAsyncDatabaseTool(db_config, pool=mock_pool)

        await tool._ensure_schema()
        call_count_1 = len(mock_pool._execute_log)

        await tool._ensure_schema()
        call_count_2 = len(mock_pool._execute_log)

        assert call_count_1 == call_count_2

    def test_create_output_success(self, db_config):
        """_create_output should create proper ToolOutput for success."""
        tool = ConcreteAsyncDatabaseTool(db_config)

        output = tool._create_output(
            raw_input={"key": "value"},
            result={"data": "test"},
            is_error=False,
        )

        assert isinstance(output, ToolOutput)
        assert output.tool_name == "test_tool"
        assert output.is_error is False
        assert output.raw_input == {"key": "value"}
        assert output.raw_output == {"data": "test"}

        # Content should be JSON
        content = json.loads(output.content)
        assert content == {"data": "test"}

    def test_create_output_error(self, db_config):
        """_create_output should create proper ToolOutput for error."""
        tool = ConcreteAsyncDatabaseTool(db_config)

        output = tool._create_output(
            raw_input={"key": "value"},
            result="Something went wrong",
            is_error=True,
        )

        assert output.is_error is True
        assert output.content == "Something went wrong"

    def test_error_output_helper(self, db_config):
        """_error_output should create error ToolOutput."""
        tool = ConcreteAsyncDatabaseTool(db_config)

        output = tool._error_output(
            raw_input={"action": "test"},
            message="Error message",
        )

        assert output.is_error is True
        assert output.content == "Error message"
        assert output.tool_name == "test_tool"

    @pytest.mark.asyncio
    async def test_acall_implementation(self, db_config, mock_pool):
        """Concrete acall should work correctly."""
        tool = ConcreteAsyncDatabaseTool(db_config, pool=mock_pool)

        result = await tool.acall(value="test_value")

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["result"] == "processed: test_value"

    def test_call_bridges_to_acall(self, db_config, mock_pool):
        """call() should bridge to acall() via run_async."""
        tool = ConcreteAsyncDatabaseTool(db_config, pool=mock_pool)

        result = tool.call(value="sync_test")

        assert isinstance(result, ToolOutput)
        assert result.is_error is False
        data = json.loads(result.content)
        assert data["result"] == "processed: sync_test"


class TestAsyncDatabaseToolPoolSharing:
    """Tests for pool sharing across tools."""

    @pytest.mark.asyncio
    async def test_shared_pool(self, db_config, mock_pool):
        """Multiple tools should be able to share a pool."""
        tool1 = ConcreteAsyncDatabaseTool(db_config, pool=mock_pool)
        tool2 = ConcreteAsyncDatabaseTool(db_config, pool=mock_pool)

        pool1 = await tool1._get_pool()
        pool2 = await tool2._get_pool()

        assert pool1 is pool2
        assert pool1 is mock_pool
