"""Integration tests for the Tool system.

Tests tool registration, resolution, calling, and the tool registry.
"""

import json
from typing import Dict, Union

import pytest

from marie.agent import (
    TOOL_REGISTRY,
    AgentTool,
    FunctionTool,
    ToolMetadata,
    ToolOutput,
    get_tool,
    list_tools,
    register_tool,
    resolve_tools,
)


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_register_tool_instance(self, clean_tool_registry, mock_search_tool):
        """Test registering a tool instance."""
        clean_tool_registry.register("test_search", mock_search_tool)

        assert clean_tool_registry.has("test_search")
        tool = clean_tool_registry.get("test_search")
        assert tool is mock_search_tool

    def test_register_tool_class(self, clean_tool_registry):
        """Test registering a tool class."""
        from tests.integration.agent.conftest import MockCalculatorTool

        clean_tool_registry.register("calc_class", MockCalculatorTool)

        assert clean_tool_registry.has("calc_class")
        tool = clean_tool_registry.get("calc_class")
        assert isinstance(tool, MockCalculatorTool)

    def test_register_callable(self, clean_tool_registry):
        """Test registering a plain callable."""

        def my_func(query: str) -> str:
            """Search for something."""
            return f"Results for: {query}"

        clean_tool_registry.register("my_func", my_func)

        assert clean_tool_registry.has("my_func")
        tool = clean_tool_registry.get("my_func")
        assert isinstance(tool, FunctionTool)

    def test_get_nonexistent_tool(self, clean_tool_registry):
        """Test getting a tool that doesn't exist."""
        with pytest.raises(KeyError, match="not registered"):
            clean_tool_registry.get("nonexistent")

    def test_get_or_none(self, clean_tool_registry, mock_search_tool):
        """Test get_or_none method."""
        clean_tool_registry.register("exists", mock_search_tool)

        assert clean_tool_registry.get_or_none("exists") is not None
        assert clean_tool_registry.get_or_none("doesnt_exist") is None

    def test_unregister_tool(self, clean_tool_registry, mock_search_tool):
        """Test unregistering a tool."""
        clean_tool_registry.register("to_remove", mock_search_tool)
        assert clean_tool_registry.has("to_remove")

        result = clean_tool_registry.unregister("to_remove")
        assert result is True
        assert not clean_tool_registry.has("to_remove")

    def test_unregister_nonexistent(self, clean_tool_registry):
        """Test unregistering a tool that doesn't exist."""
        result = clean_tool_registry.unregister("doesnt_exist")
        assert result is False

    def test_list_tools(self, clean_tool_registry, mock_search_tool, mock_calculator_tool):
        """Test listing all registered tools."""
        clean_tool_registry.register("tool_a", mock_search_tool)
        clean_tool_registry.register("tool_b", mock_calculator_tool)

        tools = clean_tool_registry.list_tools()
        assert "tool_a" in tools
        assert "tool_b" in tools

    def test_clear_registry(self, clean_tool_registry, mock_search_tool):
        """Test clearing all tools."""
        clean_tool_registry.register("tool", mock_search_tool)
        assert len(clean_tool_registry.list_tools()) > 0

        clean_tool_registry.clear()
        # Note: This test may affect other tests if not properly isolated
        # The fixture should restore the original state

    def test_no_overwrite_by_default(self, clean_tool_registry, mock_search_tool, mock_calculator_tool):
        """Test that duplicate registration doesn't overwrite by default."""
        clean_tool_registry.register("tool", mock_search_tool)
        clean_tool_registry.register("tool", mock_calculator_tool)  # Should warn, not overwrite

        # First registration should be preserved
        tool = clean_tool_registry.get("tool")
        assert tool is mock_search_tool

    def test_overwrite_explicit(self, clean_tool_registry, mock_search_tool, mock_calculator_tool):
        """Test explicit overwrite of registration."""
        clean_tool_registry.register("tool", mock_search_tool)
        clean_tool_registry.register("tool", mock_calculator_tool, overwrite=True)

        tool = clean_tool_registry.get("tool")
        assert tool is mock_calculator_tool


class TestRegisterToolDecorator:
    """Test the @register_tool decorator."""

    def test_decorator_without_args(self, clean_tool_registry):
        """Test decorator without arguments."""

        @register_tool
        def simple_search(query: str) -> str:
            """Search for documents."""
            return f"Results for: {query}"

        assert clean_tool_registry.has("simple_search")
        tool = clean_tool_registry.get("simple_search")
        assert tool.name == "simple_search"

    def test_decorator_with_name(self, clean_tool_registry):
        """Test decorator with custom name."""

        @register_tool("custom_name")
        def my_function(query: str) -> str:
            """Search for documents."""
            return f"Results for: {query}"

        assert clean_tool_registry.has("custom_name")
        assert not clean_tool_registry.has("my_function")

    def test_decorator_with_description(self, clean_tool_registry):
        """Test decorator with custom description."""

        @register_tool("described_tool", description="Custom description")
        def some_tool(x: int) -> str:
            """Original docstring."""
            return str(x)

        tool = clean_tool_registry.get("described_tool")
        assert tool.metadata.description == "Custom description"

    def test_decorator_preserves_function(self, clean_tool_registry):
        """Test that decorator preserves the original function."""

        @register_tool
        def preserved_func(x: int) -> int:
            """Double the input."""
            return x * 2

        # Original function should still work
        assert preserved_func(5) == 10


class TestResolveTools:
    """Test the resolve_tools function."""

    def test_resolve_from_registry_name(self, clean_tool_registry, mock_search_tool):
        """Test resolving tools by name from registry."""
        clean_tool_registry.register("search", mock_search_tool)

        resolved = resolve_tools(["search"])
        assert "mock_search" in resolved
        assert resolved["mock_search"] is mock_search_tool

    def test_resolve_from_config_dict(self, clean_tool_registry, mock_calculator_tool):
        """Test resolving tools from config dicts."""
        clean_tool_registry.register("calculator", mock_calculator_tool)

        resolved = resolve_tools([{"name": "calculator"}])
        assert "mock_calculator" in resolved

    def test_resolve_tool_instance(self, mock_search_tool):
        """Test resolving tool instances directly."""
        resolved = resolve_tools([mock_search_tool])
        assert "mock_search" in resolved
        assert resolved["mock_search"] is mock_search_tool

    def test_resolve_callable(self):
        """Test resolving callable functions."""

        def my_tool(x: str) -> str:
            """A simple tool."""
            return x.upper()

        resolved = resolve_tools([my_tool])
        assert "my_tool" in resolved
        assert isinstance(resolved["my_tool"], FunctionTool)

    def test_resolve_mixed_specs(self, clean_tool_registry, mock_search_tool):
        """Test resolving mixed tool specifications."""
        clean_tool_registry.register("search", mock_search_tool)

        def inline_tool(x: str) -> str:
            """Inline tool."""
            return x

        resolved = resolve_tools([
            "search",
            inline_tool,
        ])

        assert len(resolved) == 2
        assert "mock_search" in resolved
        assert "inline_tool" in resolved

    def test_resolve_invalid_spec(self):
        """Test resolving invalid tool specification."""
        with pytest.raises(TypeError):
            resolve_tools([12345])  # Invalid type


class TestFunctionTool:
    """Test FunctionTool creation and usage."""

    def test_from_defaults_basic(self):
        """Test creating FunctionTool from a function."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = FunctionTool.from_defaults(fn=add)

        assert tool.name == "add"
        assert "Add two numbers" in tool.metadata.description

    def test_from_defaults_with_custom_name(self):
        """Test creating FunctionTool with custom name."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = FunctionTool.from_defaults(fn=add, name="addition")
        assert tool.name == "addition"

    def test_function_tool_call(self):
        """Test calling a FunctionTool."""

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        tool = FunctionTool.from_defaults(fn=multiply)
        result = tool.call(a=3, b=4)

        # FunctionTool returns ToolOutput
        assert isinstance(result, ToolOutput)
        assert result.content == 12 or result.raw_output == 12

    def test_function_tool_call_with_json_string(self):
        """Test calling FunctionTool with JSON string via safe_call."""

        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        tool = FunctionTool.from_defaults(fn=greet)
        # Use safe_call for JSON string input
        result = tool.safe_call('{"name": "World"}')

        assert isinstance(result, ToolOutput)
        assert "World" in str(result.content)

    def test_function_tool_get_definition(self):
        """Test getting OpenAI-compatible function definition."""

        def search(query: str, limit: int = 10) -> str:
            """Search for documents matching the query."""
            return json.dumps({"results": []})

        tool = FunctionTool.from_defaults(fn=search)
        definition = tool.get_function_definition()

        assert definition["name"] == "search"
        assert "description" in definition
        assert "parameters" in definition


class TestToolCalling:
    """Test tool calling and error handling."""

    def test_tool_call_success(self, mock_calculator_tool):
        """Test successful tool call."""
        result = mock_calculator_tool.call(expression="2 + 2")
        assert isinstance(result, ToolOutput)
        data = json.loads(result.content)
        assert data["result"] == 4

    def test_tool_call_with_string_params(self, mock_search_tool):
        """Test tool call with string parameters via safe_call."""
        result = mock_search_tool.safe_call('{"query": "test"}')
        assert isinstance(result, ToolOutput)
        data = json.loads(result.content)
        assert data["query"] == "test"
        assert "results" in data

    def test_tool_safe_call_error_handling(self, failing_tool):
        """Test safe_call handles errors gracefully."""
        result = failing_tool.safe_call({})

        assert result.is_error is True
        assert "error" in result.content.lower() or "fail" in result.content.lower()

    def test_tool_output_structure(self, mock_search_tool):
        """Test ToolOutput structure."""
        output = mock_search_tool.safe_call('{"query": "test"}')

        assert isinstance(output, ToolOutput)
        assert output.is_error is False
        assert output.content is not None
        assert output.tool_name == "mock_search"


class TestToolMetadata:
    """Test ToolMetadata class."""

    def test_metadata_creation(self):
        """Test creating ToolMetadata."""
        metadata = ToolMetadata(
            name="test_tool",
            description="A test tool",
            fn_schema=None,
            return_direct=False,
        )

        assert metadata.name == "test_tool"
        assert metadata.description == "A test tool"
        assert metadata.return_direct is False

    def test_metadata_with_schema(self):
        """Test ToolMetadata with Pydantic schema."""
        from pydantic import BaseModel

        class SearchInput(BaseModel):
            query: str
            limit: int = 10

        metadata = ToolMetadata(
            name="search",
            description="Search tool",
            fn_schema=SearchInput,
        )

        assert metadata.fn_schema is SearchInput

    def test_metadata_from_tool(self, mock_search_tool):
        """Test getting metadata from a tool."""
        metadata = mock_search_tool.metadata

        assert metadata.name == "mock_search"
        assert len(metadata.description) > 0


class TestToolDefinitionGeneration:
    """Test OpenAI-compatible tool definition generation."""

    def test_function_definition_format(self, mock_search_tool):
        """Test function definition has correct format."""
        definition = mock_search_tool.get_function_definition()

        assert "name" in definition
        assert "description" in definition
        assert "parameters" in definition

    def test_openai_tool_format(self, mock_search_tool):
        """Test OpenAI tool format."""
        tool_def = mock_search_tool.to_openai_tool()

        assert tool_def["type"] == "function"
        assert "function" in tool_def
        assert tool_def["function"]["name"] == "mock_search"
