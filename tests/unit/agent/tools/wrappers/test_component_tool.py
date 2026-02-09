"""Tests for ComponentTool."""
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest
from pydantic import BaseModel

from marie.agent.tools import ToolOutput
from marie.agent.tools.wrappers.component_tool import ComponentTool


# Test components
class SimpleComponent:
    """Component with run() method."""

    def run(self, text: str, uppercase: bool = False) -> str:
        """Process text."""
        if uppercase:
            return text.upper()
        return text


class CallableComponent:
    """Component that is directly callable."""

    def __call__(self, value: int) -> int:
        """Double the value."""
        return value * 2


class AsyncComponent:
    """Component with async run() method."""

    async def run(self, query: str) -> Dict[str, Any]:
        """Async search."""
        return {"query": query, "results": ["result1", "result2"]}


class NoMethodComponent:
    """Component with no callable method."""
    pass


class CustomMethodComponent:
    """Component with custom method name."""

    def process(self, data: str, format: str = "json") -> str:
        """Process data."""
        return f"Processed ({format}): {data}"


class DictOutputComponent:
    """Component that returns dictionary."""

    def run(self, key: str) -> Dict[str, Any]:
        return {"key": key, "nested": {"a": 1, "b": 2}}


class PydanticOutputComponent:
    """Component that returns Pydantic model."""

    class Output(BaseModel):
        value: str
        count: int

    def run(self, text: str) -> "PydanticOutputComponent.Output":
        return self.Output(value=text, count=len(text))


@dataclass
class DataclassOutput:
    """Dataclass for testing output transformation."""
    name: str
    value: int


class DataclassOutputComponent:
    """Component that returns dataclass."""

    def run(self, name: str) -> DataclassOutput:
        return DataclassOutput(name=name, value=42)


class FailingComponent:
    """Component that raises an error."""

    def run(self, trigger: str) -> str:
        raise ValueError(f"Intentional failure: {trigger}")


class TestComponentToolFromComponent:
    """Tests for ComponentTool.from_component factory."""

    def test_wrap_simple_component(self):
        """Should wrap component with run() method."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        assert tool.name == "simple_component"
        assert tool.component is component

    def test_wrap_callable_component(self):
        """Should wrap callable component."""
        component = CallableComponent()
        tool = ComponentTool.from_component(component)

        assert tool.name == "callable_component"

    def test_wrap_with_custom_name(self):
        """Should accept custom name."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component, name="my_tool")

        assert tool.name == "my_tool"

    def test_wrap_with_custom_description(self):
        """Should accept custom description."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(
            component,
            description="Custom description"
        )

        assert tool.description == "Custom description"

    def test_wrap_with_custom_method(self):
        """Should wrap custom method name."""
        component = CustomMethodComponent()
        tool = ComponentTool.from_component(
            component,
            method_name="process"
        )

        result = tool.call(data="test", format="xml")
        assert "Processed (xml): test" in result.content

    def test_wrap_raises_for_no_method(self):
        """Should raise error for component without callable method."""
        component = NoMethodComponent()

        with pytest.raises(ValueError, match="has no"):
            ComponentTool.from_component(component)

    def test_auto_generates_schema(self):
        """Should auto-generate schema from method signature."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        schema = tool.metadata.fn_schema
        assert schema is not None

        # Check schema has expected fields
        schema_fields = schema.model_fields
        assert "text" in schema_fields
        assert "uppercase" in schema_fields

    def test_uses_docstring_for_description(self):
        """Should use method docstring for description."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        assert "Process text" in tool.description


class TestComponentToolCall:
    """Tests for ComponentTool.call() execution."""

    def test_call_simple_component(self):
        """call() should execute component method."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(text="hello")

        assert isinstance(result, ToolOutput)
        assert result.is_error is False
        assert result.content == "hello"
        assert result.raw_output == "hello"

    def test_call_with_optional_param(self):
        """call() should handle optional parameters."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(text="hello", uppercase=True)

        assert result.content == "HELLO"

    def test_call_callable_component(self):
        """call() should work with callable components."""
        component = CallableComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(value=21)

        assert result.raw_output == 42

    def test_call_returns_tool_output(self):
        """call() should always return ToolOutput."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(text="test")

        assert isinstance(result, ToolOutput)
        assert result.tool_name == "simple_component"
        assert result.raw_input == {"text": "test"}

    def test_call_error_handling(self):
        """call() should handle errors gracefully."""
        component = FailingComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(trigger="error")

        assert result.is_error is True
        assert "Error" in result.content
        assert "Intentional failure" in result.content


class TestComponentToolAcall:
    """Tests for ComponentTool.acall() async execution."""

    @pytest.mark.asyncio
    async def test_acall_sync_component(self):
        """acall() should work with sync components via thread."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        result = await tool.acall(text="async test")

        assert result.is_error is False
        assert result.content == "async test"

    @pytest.mark.asyncio
    async def test_acall_async_component(self):
        """acall() should work with async components natively."""
        component = AsyncComponent()
        tool = ComponentTool.from_component(component)

        result = await tool.acall(query="search term")

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["query"] == "search term"
        assert "results" in data

    @pytest.mark.asyncio
    async def test_acall_error_handling(self):
        """acall() should handle errors gracefully."""
        component = FailingComponent()
        tool = ComponentTool.from_component(component)

        result = await tool.acall(trigger="async error")

        assert result.is_error is True
        assert "Error" in result.content


class TestComponentToolOutputTransformation:
    """Tests for output transformation."""

    def test_string_output(self):
        """String output should be returned as-is."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(text="hello")
        assert result.content == "hello"

    def test_dict_output(self):
        """Dict output should be JSON serialized."""
        component = DictOutputComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(key="test")

        data = json.loads(result.content)
        assert data["key"] == "test"
        assert data["nested"]["a"] == 1

    def test_pydantic_output(self):
        """Pydantic model output should be JSON serialized."""
        component = PydanticOutputComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(text="hello")

        data = json.loads(result.content)
        assert data["value"] == "hello"
        assert data["count"] == 5

    def test_dataclass_output(self):
        """Dataclass output should be JSON serialized."""
        component = DataclassOutputComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call(name="test")

        data = json.loads(result.content)
        assert data["name"] == "test"
        assert data["value"] == 42

    def test_custom_output_transformer(self):
        """Should use custom output transformer."""
        component = SimpleComponent()

        def custom_transformer(result):
            return f"<<{result}>>"

        tool = ComponentTool.from_component(
            component,
            output_transformer=custom_transformer
        )

        result = tool.call(text="hello")
        assert result.content == "<<hello>>"

    def test_none_output(self):
        """None output should become empty string."""

        class NoneComponent:
            def run(self) -> None:
                return None

        component = NoneComponent()
        tool = ComponentTool.from_component(component)

        result = tool.call()
        assert result.content == ""


class TestComponentToolFromCallable:
    """Tests for ComponentTool.from_callable factory."""

    def test_from_callable_with_run(self):
        """from_callable should prefer run() method."""
        component = SimpleComponent()
        tool = ComponentTool.from_callable(component)

        result = tool.call(text="test")
        assert result.content == "test"

    def test_from_callable_with_call(self):
        """from_callable should use __call__ if no run()."""
        component = CallableComponent()
        tool = ComponentTool.from_callable(component)

        result = tool.call(value=5)
        assert result.raw_output == 10

    def test_from_callable_raises_for_non_callable(self):
        """from_callable should raise for non-callable objects."""
        component = NoMethodComponent()

        with pytest.raises(ValueError, match="not callable"):
            ComponentTool.from_callable(component)


class TestComponentToolFromPipelineComponent:
    """Tests for ComponentTool.from_pipeline_component factory."""

    def test_pipeline_component_output(self):
        """Pipeline component output should be transformed."""

        class PipelineComponent:
            def run(self, text: str):
                return {"output": [{"content": text}]}

        component = PipelineComponent()
        tool = ComponentTool.from_pipeline_component(component)

        result = tool.call(text="pipeline test")

        # Should be JSON
        data = json.loads(result.content)
        assert isinstance(data, list)

    def test_pipeline_single_value_unwrap(self):
        """Single-key dict results should be unwrapped."""

        class SingleOutputPipeline:
            def run(self, x: int):
                return {"result": x * 2}

        component = SingleOutputPipeline()
        tool = ComponentTool.from_pipeline_component(component)

        result = tool.call(x=21)
        assert "42" in result.content


class TestComponentToolSchemaGeneration:
    """Tests for automatic schema generation."""

    def test_schema_with_defaults(self):
        """Schema should capture default values."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        schema = tool.metadata.fn_schema
        fields = schema.model_fields

        # 'uppercase' has default=False
        assert fields["uppercase"].default is False

    def test_schema_required_fields(self):
        """Schema should mark required fields."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        schema = tool.metadata.fn_schema
        fields = schema.model_fields

        # 'text' is required (no default)
        assert fields["text"].is_required()

    def test_schema_with_no_params(self):
        """Schema should have default input for no-param methods."""

        class NoParamComponent:
            def run(self) -> str:
                return "result"

        component = NoParamComponent()
        tool = ComponentTool.from_component(component)

        schema = tool.metadata.fn_schema
        fields = schema.model_fields

        # Should have default 'input' field
        assert "input" in fields

    def test_custom_schema_override(self):
        """Should accept custom schema."""

        class CustomInput(BaseModel):
            custom_field: str

        component = SimpleComponent()
        tool = ComponentTool.from_component(
            component,
            fn_schema=CustomInput
        )

        assert tool.metadata.fn_schema is CustomInput


class TestComponentToolMetadata:
    """Tests for tool metadata generation."""

    def test_openai_tool_format(self):
        """Should generate valid OpenAI tool format."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        openai_tool = tool.to_openai_tool()

        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "simple_component"
        assert "parameters" in openai_tool["function"]

    def test_function_definition(self):
        """Should generate function definition."""
        component = SimpleComponent()
        tool = ComponentTool.from_component(component)

        definition = tool.get_function_definition()

        assert definition["name"] == "simple_component"
        assert "description" in definition
        assert "parameters" in definition
