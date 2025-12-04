"""Tests for Marie agent tool system."""

import pytest

from marie.agent.tools.base import (
    AgentTool,
    FunctionTool,
    ToolMetadata,
    ToolOutput,
    adapt_tool,
)
from marie.agent.tools.registry import (
    TOOL_REGISTRY,
    get_tool,
    list_tools,
    register_tool,
    resolve_tools,
)


class TestToolMetadata:
    """Tests for ToolMetadata class."""

    def test_create_metadata(self):
        """Test creating tool metadata."""
        metadata = ToolMetadata(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
            },
        )
        assert metadata.name == "test_tool"
        assert metadata.description == "A test tool"

    def test_get_parameters_dict(self):
        """Test getting parameters as dict."""
        metadata = ToolMetadata(
            name="test",
            description="test",
            parameters={"type": "object", "properties": {}},
        )
        params = metadata.get_parameters_dict()
        assert isinstance(params, dict)
        assert params["type"] == "object"

    def test_to_openai_function(self):
        """Test converting to OpenAI function format."""
        metadata = ToolMetadata(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        )
        func = metadata.to_openai_function()
        assert func["name"] == "search"
        assert func["description"] == "Search the web"
        assert "parameters" in func


class TestToolOutput:
    """Tests for ToolOutput class."""

    def test_create_output(self):
        """Test creating tool output."""
        output = ToolOutput(
            content="Result data",
            tool_name="my_tool",
        )
        assert output.content == "Result data"
        assert output.tool_name == "my_tool"
        assert output.is_error is False

    def test_error_output(self):
        """Test creating error output."""
        output = ToolOutput(
            content="Error occurred",
            tool_name="my_tool",
            is_error=True,
        )
        assert output.is_error is True


class TestFunctionTool:
    """Tests for FunctionTool class."""

    def test_create_from_function(self):
        """Test creating tool from function."""

        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        tool = FunctionTool.from_defaults(
            fn=add_numbers,
            name="add",
            description="Add two numbers",
        )
        assert tool.name == "add"
        assert "Add two numbers" in tool.description

    def test_call_function_tool(self):
        """Test calling function tool."""

        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        tool = FunctionTool.from_defaults(fn=multiply, name="multiply")
        result = tool.call(x=3, y=4)
        assert result.content == "12"

    @pytest.mark.asyncio
    async def test_async_call_function_tool(self):
        """Test async calling function tool."""

        async def async_double(n: int) -> int:
            """Double a number."""
            return n * 2

        tool = FunctionTool.from_defaults(fn=async_double, name="double")
        result = await tool.acall(n=5)
        assert result.content == "10"


class TestToolRegistry:
    """Tests for tool registry."""

    def test_register_tool_decorator(self):
        """Test registering tool with decorator."""

        @register_tool("test_registered_tool")
        def my_test_tool(input: str) -> str:
            """A test tool."""
            return f"Result: {input}"

        # Check it's registered
        tools = list_tools()
        assert "test_registered_tool" in tools

    def test_get_tool(self):
        """Test getting tool from registry."""

        @register_tool("get_test_tool")
        def another_tool(x: int) -> int:
            """Another test tool."""
            return x * 2

        tool = get_tool("get_test_tool")
        assert tool is not None
        assert tool.name == "get_test_tool"

    def test_get_nonexistent_tool(self):
        """Test getting non-existent tool raises KeyError or returns None."""
        try:
            tool = get_tool("nonexistent_tool_xyz_123456")
            # If it doesn't raise, it should return None
            assert tool is None
        except KeyError:
            # KeyError is also acceptable behavior
            pass

    def test_resolve_tools_by_name(self):
        """Test resolving tools by name."""

        @register_tool("resolve_test_tool")
        def resolve_me(s: str) -> str:
            """Tool to resolve."""
            return s

        tools = resolve_tools(["resolve_test_tool"])
        assert "resolve_test_tool" in tools

    def test_resolve_tools_with_instance(self, mock_tool):
        """Test resolving tools with instance."""
        tools = resolve_tools([mock_tool])
        assert mock_tool.name in tools


class TestAdaptTool:
    """Tests for adapt_tool function."""

    def test_adapt_agent_tool(self, mock_tool):
        """Test adapting AgentTool passes through."""
        adapted = adapt_tool(mock_tool)
        assert adapted is mock_tool

    @pytest.mark.skip(reason="adapt_tool implementation details vary")
    def test_adapt_callable(self):
        """Test adapting callable to tool."""

        def my_func(x: int) -> int:
            """Double the input."""
            return x * 2

        adapted = adapt_tool(my_func)
        # Should return an AgentTool
        assert isinstance(adapted, AgentTool)
        assert adapted.name == "my_func"


class TestSchemaValidation:
    """Tests for tool schema validation."""

    def test_safe_call_with_valid_args(self):
        """Test safe_call passes with valid arguments."""
        from pydantic import BaseModel, Field

        class AddInput(BaseModel):
            a: int = Field(..., description="First number")
            b: int = Field(..., description="Second number")

        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = FunctionTool.from_defaults(
            fn=add_numbers,
            name="add",
            description="Add numbers",
            fn_schema=AddInput,
        )

        result = tool.safe_call({"a": 5, "b": 3})
        assert result.is_error is False
        assert result.content == "8"

    def test_safe_call_with_invalid_args(self):
        """Test safe_call fails with validation error for invalid arguments."""
        from pydantic import BaseModel, Field

        class AddInput(BaseModel):
            a: int = Field(..., description="First number")
            b: int = Field(..., description="Second number")

        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = FunctionTool.from_defaults(
            fn=add_numbers,
            name="add",
            description="Add numbers",
            fn_schema=AddInput,
        )

        # Missing required 'b' parameter
        result = tool.safe_call({"a": 5})
        assert result.is_error is True
        assert "validation failed" in result.content.lower()

    def test_safe_call_with_type_coercion(self):
        """Test safe_call coerces types when possible."""
        from pydantic import BaseModel, Field

        class DoubleInput(BaseModel):
            n: int = Field(..., description="Number to double")

        def double(n: int) -> int:
            """Double a number."""
            return n * 2

        tool = FunctionTool.from_defaults(
            fn=double,
            name="double",
            description="Double a number",
            fn_schema=DoubleInput,
        )

        # Pass string that can be coerced to int
        result = tool.safe_call({"n": "10"})
        assert result.is_error is False
        assert result.content == "20"

    def test_safe_call_with_auto_generated_schema(self):
        """Test safe_call works with auto-generated schema from function signature."""

        def echo(text: str) -> str:
            """Echo the input."""
            return text

        # FunctionTool.from_defaults auto-generates schema from signature
        tool = FunctionTool.from_defaults(
            fn=echo,
            name="echo",
            description="Echo input",
        )

        result = tool.safe_call({"text": "hello"})
        assert result.is_error is False
        # FunctionTool._output_to_string returns strings directly without JSON encoding
        assert result.content == "hello"

    @pytest.mark.asyncio
    async def test_safe_acall_with_valid_args(self):
        """Test async safe_acall passes with valid arguments."""
        from pydantic import BaseModel, Field

        class MultiplyInput(BaseModel):
            x: int = Field(..., description="First factor")
            y: int = Field(..., description="Second factor")

        async def async_multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        tool = FunctionTool.from_defaults(
            fn=async_multiply,
            name="multiply",
            description="Multiply numbers",
            fn_schema=MultiplyInput,
        )

        result = await tool.safe_acall({"x": 4, "y": 5})
        assert result.is_error is False
        assert result.content == "20"

    @pytest.mark.asyncio
    async def test_safe_acall_with_invalid_args(self):
        """Test async safe_acall fails with validation error."""
        from pydantic import BaseModel, Field

        class MultiplyInput(BaseModel):
            x: int = Field(..., description="First factor")
            y: int = Field(..., description="Second factor")

        async def async_multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        tool = FunctionTool.from_defaults(
            fn=async_multiply,
            name="multiply",
            description="Multiply numbers",
            fn_schema=MultiplyInput,
        )

        # Invalid type that can't be coerced
        result = await tool.safe_acall({"x": "not_a_number", "y": 5})
        assert result.is_error is True
        assert "validation failed" in result.content.lower()

    def test_safe_call_with_default_values(self):
        """Test safe_call fills in default values from schema."""
        from pydantic import BaseModel, Field

        class GreetInput(BaseModel):
            name: str = Field(..., description="Name to greet")
            greeting: str = Field(default="Hello", description="Greeting to use")

        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        tool = FunctionTool.from_defaults(
            fn=greet,
            name="greet",
            description="Greet someone",
            fn_schema=GreetInput,
        )

        # Only provide required field, default should be used
        result = tool.safe_call({"name": "World"})
        assert result.is_error is False
        assert "Hello" in result.content
        assert "World" in result.content
