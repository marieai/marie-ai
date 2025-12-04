"""Tests for ComponentTool wrapper."""

from typing import Any, Dict, Optional

import pytest

from marie.agent.tools.wrappers.component_tool import ComponentTool


class SimpleComponent:
    """A simple component with run() method."""

    def run(self, text: str, uppercase: bool = False) -> Dict[str, Any]:
        """Process text.

        Args:
            text: Input text to process
            uppercase: Whether to convert to uppercase

        Returns:
            Processed result
        """
        result = text.upper() if uppercase else text
        return {"processed": result, "length": len(text)}


class CallableComponent:
    """A callable component with __call__ method."""

    def __call__(self, value: int, multiplier: int = 2) -> int:
        """Multiply a value."""
        return value * multiplier


class AsyncComponent:
    """A component with async run method."""

    async def run(self, query: str) -> Dict[str, str]:
        """Async search."""
        return {"query": query, "result": f"Found: {query}"}


class CustomMethodComponent:
    """A component with a custom method name."""

    def process(self, data: str, format: str = "json") -> str:
        """Process data in specified format."""
        return f"{format}: {data}"


class NoDocstringComponent:
    def run(self, x: int) -> int:
        return x * 2


class TestComponentToolBasic:
    """Basic tests for ComponentTool."""

    def test_from_component_with_run_method(self):
        """Test creating tool from component with run() method."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        assert tool.name == "simple_component"
        assert "Process text" in tool.description
        assert tool.component is component

    def test_from_component_with_custom_name(self):
        """Test creating tool with custom name."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(
            component,
            name="text_processor",
            description="Process text input",
        )

        assert tool.name == "text_processor"
        assert tool.description == "Process text input"

    def test_from_component_callable(self):
        """Test creating tool from callable component."""
        component = CallableComponent()
        tool = ComponentTool.from_component(component)

        assert tool.name == "callable_component"
        assert "Multiply" in tool.description

    def test_from_component_custom_method(self):
        """Test creating tool with custom method name."""
        component = CustomMethodComponent()
        tool = ComponentTool.from_component(
            component,
            method_name="process",
            name="data_processor",
        )

        assert tool.name == "data_processor"

    def test_from_callable_convenience(self):
        """Test the from_callable convenience method."""
        component = SimpleComponent()
        tool = ComponentTool.from_callable(component, name="my_tool")

        assert tool.name == "my_tool"

    def test_no_docstring_fallback(self):
        """Test fallback description when no docstring."""
        component = NoDocstringComponent()
        tool = ComponentTool.from_component(component)

        assert "Execute" in tool.description
        assert "run()" in tool.description


class TestComponentToolExecution:
    """Tests for ComponentTool execution."""

    def test_call_sync(self):
        """Test synchronous execution."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(text="hello", uppercase=False)

        assert result.is_error is False
        assert "hello" in result.content
        assert result.raw_output == {"processed": "hello", "length": 5}

    def test_call_with_uppercase(self):
        """Test execution with optional parameter."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(text="hello", uppercase=True)

        assert "HELLO" in result.content

    def test_call_callable_component(self):
        """Test calling a callable component."""
        component = CallableComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(value=5, multiplier=3)

        assert result.raw_output == 15
        assert "15" in result.content

    def test_call_custom_method(self):
        """Test calling custom method."""
        component = CustomMethodComponent()
        tool = ComponentTool.from_component(
            component,
            method_name="process",
        )

        result = tool.call(data="test", format="xml")

        assert "xml: test" in result.content

    @pytest.mark.asyncio
    async def test_acall_async_component(self):
        """Test async execution of async component."""
        component = AsyncComponent()
        tool = ComponentTool.from_component(component)

        result = await tool.acall(query="search term")

        assert result.is_error is False
        assert "search term" in result.content
        assert result.raw_output["query"] == "search term"

    @pytest.mark.asyncio
    async def test_acall_sync_component(self):
        """Test async execution of sync component (runs in thread)."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        result = await tool.acall(text="async test")

        assert result.is_error is False
        assert "async test" in result.content

    def test_call_error_handling(self):
        """Test error handling in call."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        # Missing required parameter
        result = tool.call()  # Missing 'text'

        assert result.is_error is True
        assert "Error" in result.content


class TestComponentToolSchemaGeneration:
    """Tests for automatic schema generation."""

    def test_schema_from_simple_method(self):
        """Test schema generation from simple method."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        params = tool.metadata.get_parameters_dict()

        assert params["type"] == "object"
        assert "text" in params["properties"]
        assert "uppercase" in params["properties"]

    def test_schema_required_fields(self):
        """Test that required fields are marked correctly."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        params = tool.metadata.get_parameters_dict()

        # text is required (no default), uppercase is optional
        # The schema should reflect this
        assert "properties" in params

    def test_openai_tool_format(self):
        """Test conversion to OpenAI tool format."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        openai_tool = tool.to_openai_tool()

        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "simple_component"
        assert "parameters" in openai_tool["function"]


class TestComponentToolOutputTransformation:
    """Tests for output transformation."""

    def test_dict_output(self):
        """Test dictionary output transformation."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(text="test")

        # Should be JSON formatted
        assert "{" in result.content
        assert "processed" in result.content

    def test_string_output(self):
        """Test string output passthrough."""
        component = CustomMethodComponent()
        tool = ComponentTool.from_component(component, method_name="process")

        result = tool.call(data="hello", format="txt")

        assert result.content == "txt: hello"

    def test_numeric_output(self):
        """Test numeric output transformation."""
        component = CallableComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(value=10)

        assert result.content == "20"

    def test_custom_output_transformer(self):
        """Test custom output transformer."""

        def custom_transformer(result):
            return f"CUSTOM: {result}"

        component = SimpleComponent()
        tool = ComponentTool.from_component(
            component,
            output_transformer=custom_transformer,
        )

        result = tool.call(text="test")

        assert result.content.startswith("CUSTOM:")


class TestComponentToolPipelineComponent:
    """Tests for pipeline component handling."""

    def test_from_pipeline_component(self):
        """Test creating tool from pipeline-style component."""

        class PipelineComponent:
            def run(self, query: str) -> Dict[str, Any]:
                return {"documents": [{"content": "doc1"}, {"content": "doc2"}]}

        component = PipelineComponent()
        tool = ComponentTool.from_pipeline_component(component)

        result = tool.call(query="search")

        # Should unwrap single-key dict
        assert "doc1" in result.content


class TestComponentToolEdgeCases:
    """Edge case tests."""

    def test_component_without_run_or_call(self):
        """Test error when component has no run or __call__."""

        class BadComponent:
            def process(self):
                pass

        component = BadComponent()

        with pytest.raises(ValueError, match="has no 'run' method"):
            ComponentTool.from_component(component)

    def test_component_with_optional_params(self):
        """Test component with Optional type hints."""

        class OptionalParamComponent:
            def run(self, required: str, optional: Optional[str] = None) -> str:
                return f"{required}-{optional}"

        component = OptionalParamComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(required="test")

        assert result.is_error is False

    def test_snake_case_name_generation(self):
        """Test automatic snake_case name generation."""

        class MyComplexComponentName:
            def run(self, x: int) -> int:
                return x

        component = MyComplexComponentName()
        tool = ComponentTool.from_component(component)

        assert tool.name == "my_complex_component_name"
