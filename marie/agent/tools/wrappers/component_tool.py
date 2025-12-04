"""ComponentTool - Wrap component-style classes as agent tools.

This module provides a tool wrapper for classes that follow the component pattern
(having a `run()` or `__call__` method), with automatic schema generation from
method signatures.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.tools.wrappers.component")


class ComponentTool(AgentTool):
    """Wrap a component-style class as an agent tool.

    Supports classes with:
    - `run()` method (common component pattern)
    - `__call__()` method (callable classes)
    - Custom method specified by name

    Automatically generates input schemas from method type hints.

    Example:
        ```python
        # Wrap a class with run() method
        class TextAnalyzer:
            def run(self, text: str, language: str = "en") -> Dict[str, Any]:
                return {"sentiment": "positive", "language": language}


        analyzer = TextAnalyzer()
        tool = ComponentTool.from_component(analyzer)


        # Wrap a callable class
        class Calculator:
            def __call__(self, expression: str) -> float:
                return eval(expression)


        calc = Calculator()
        tool = ComponentTool.from_component(calc, name="calculator")


        # Wrap with custom method
        class DataProcessor:
            def process(self, data: str, format: str = "json") -> str:
                return f"Processed: {data}"


        processor = DataProcessor()
        tool = ComponentTool.from_component(
            processor,
            method_name="process",
            name="data_processor",
        )
        ```
    """

    def __init__(
        self,
        component: Any,
        metadata: ToolMetadata,
        method_name: str = "run",
        output_transformer: Optional[Callable[[Any], str]] = None,
    ):
        """Initialize ComponentTool.

        Args:
            component: The component instance to wrap
            metadata: Tool metadata
            method_name: Name of the method to call (default: "run")
            output_transformer: Optional function to transform output to string
        """
        self._component = component
        self._metadata = metadata
        self._method_name = method_name
        self._output_transformer = (
            output_transformer or self._default_output_transformer
        )
        self._method = self._resolve_method()

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @property
    def component(self) -> Any:
        """Get the wrapped component."""
        return self._component

    def _resolve_method(self) -> Callable:
        """Resolve the callable method from the component.

        Returns:
            The method to call

        Raises:
            ValueError: If method not found
        """
        # Try specified method name
        if hasattr(self._component, self._method_name):
            method = getattr(self._component, self._method_name)
            if callable(method):
                return method

        # Fallback to __call__ if component is callable
        if callable(self._component):
            return self._component

        raise ValueError(
            f"Component {type(self._component).__name__} has no callable method "
            f"'{self._method_name}' and is not callable"
        )

    def _default_output_transformer(self, result: Any) -> str:
        """Transform output to string.

        Args:
            result: Raw output from component

        Returns:
            String representation
        """
        if result is None:
            return ""

        if isinstance(result, str):
            return result

        # Handle Pydantic models
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

        # Handle dataclasses
        if hasattr(result, "__dataclass_fields__"):
            import dataclasses

            return json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2)

        # Handle dict-like results
        if isinstance(result, dict):
            # Extract relevant content if nested with single key
            if len(result) == 1:
                single_value = next(iter(result.values()))
                if isinstance(single_value, (list, str)):
                    return json.dumps(single_value, ensure_ascii=False, indent=2)
            return json.dumps(result, ensure_ascii=False, indent=2)

        # Handle lists
        if isinstance(result, (list, tuple)):
            return json.dumps(result, ensure_ascii=False, indent=2)

        # Generic fallback
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    def call(self, **kwargs: Any) -> ToolOutput:
        """Execute the component method synchronously.

        Args:
            **kwargs: Arguments matching the method signature

        Returns:
            ToolOutput with result
        """
        try:
            # Call the method
            if asyncio.iscoroutinefunction(self._method):
                # Handle async methods in sync context
                from marie.helper import run_async

                result = run_async(self._method(**kwargs))
            else:
                result = self._method(**kwargs)

            # Transform output
            content = self._output_transformer(result)

            return ToolOutput(
                content=content,
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
            )

        except Exception as e:
            logger.error(f"ComponentTool '{self.name}' failed: {e}")
            return ToolOutput(
                content=f"Error: {str(e)}",
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=None,
                is_error=True,
            )

    async def acall(self, **kwargs: Any) -> ToolOutput:
        """Execute the component method asynchronously.

        Args:
            **kwargs: Arguments matching the method signature

        Returns:
            ToolOutput with result
        """
        try:
            # Call the method
            if asyncio.iscoroutinefunction(self._method):
                result = await self._method(**kwargs)
            else:
                result = await asyncio.to_thread(self._method, **kwargs)

            # Transform output
            content = self._output_transformer(result)

            return ToolOutput(
                content=content,
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
            )

        except Exception as e:
            logger.error(f"ComponentTool '{self.name}' failed: {e}")
            return ToolOutput(
                content=f"Error: {str(e)}",
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=None,
                is_error=True,
            )

    @classmethod
    def from_component(
        cls,
        component: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        method_name: str = "run",
        fn_schema: Optional[Type[BaseModel]] = None,
        output_transformer: Optional[Callable[[Any], str]] = None,
        return_direct: bool = False,
    ) -> "ComponentTool":
        """Create a ComponentTool from a component instance.

        Args:
            component: The component instance (must have run() or __call__)
            name: Tool name (defaults to class name in snake_case)
            description: Tool description (defaults to docstring)
            method_name: Method to call (default: "run")
            fn_schema: Input schema (auto-generated if not provided)
            output_transformer: Custom output transformer
            return_direct: Whether to return output directly

        Returns:
            Configured ComponentTool instance
        """
        # Determine method to use
        method = None
        actual_method_name = method_name

        if hasattr(component, method_name):
            method = getattr(component, method_name)
        elif callable(component):
            method = component.__call__
            actual_method_name = "__call__"
        else:
            raise ValueError(
                f"Component {type(component).__name__} has no '{method_name}' method "
                f"and is not callable"
            )

        # Generate name from class name if not provided
        tool_name = name
        if tool_name is None:
            tool_name = type(component).__name__
            # Convert CamelCase to snake_case
            import re

            tool_name = re.sub(r'(?<!^)(?=[A-Z])', '_', tool_name).lower()

        # Extract description from docstring
        tool_description = description
        if tool_description is None:
            # Try method docstring first, then class docstring
            tool_description = method.__doc__ or component.__class__.__doc__
            if tool_description:
                tool_description = tool_description.split("\n\n")[0].strip()
            else:
                tool_description = (
                    f"Execute {type(component).__name__}.{actual_method_name}()"
                )

        # Auto-generate schema from method signature if not provided
        if fn_schema is None:
            fn_schema = cls._generate_schema_from_method(method, tool_name)

        metadata = ToolMetadata(
            name=tool_name,
            description=tool_description,
            fn_schema=fn_schema,
            return_direct=return_direct,
        )

        return cls(
            component=component,
            metadata=metadata,
            method_name=actual_method_name,
            output_transformer=output_transformer,
        )

    @staticmethod
    def _generate_schema_from_method(
        method: Callable, tool_name: str
    ) -> Type[BaseModel]:
        """Generate a Pydantic schema from method signature.

        Args:
            method: The method to analyze
            tool_name: Name for the generated schema

        Returns:
            A dynamically created Pydantic model class
        """
        sig = inspect.signature(method)

        # Get type hints, handling potential errors
        try:
            hints = get_type_hints(method)
        except Exception:
            hints = {}

        # Build field definitions
        fields: Dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            # Skip self, cls, and **kwargs/**args
            if param_name in ("self", "cls", "kwargs", "args"):
                continue
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue

            # Get type hint or default to str
            param_type = hints.get(param_name, str)

            # Handle Optional types - extract inner type
            origin = getattr(param_type, "__origin__", None)
            if origin is Union:
                args = getattr(param_type, "__args__", ())
                # Check if it's Optional (Union[X, None])
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1:
                    param_type = non_none_args[0]

            # Create field with or without default
            if param.default is inspect.Parameter.empty:
                # Required field
                fields[param_name] = (param_type, ...)
            else:
                # Optional field with default
                fields[param_name] = (param_type, param.default)

        # Handle empty fields - create a default input schema
        if not fields:
            fields["input"] = (str, Field(..., description="Input string"))

        # Create dynamic Pydantic model
        schema_name = f"{tool_name.title().replace('_', '')}Input"

        # Build field definitions for create_model
        model_fields = {}
        for field_name, (field_type, field_default) in fields.items():
            if field_default is ...:
                model_fields[field_name] = (field_type, ...)
            else:
                model_fields[field_name] = (field_type, field_default)

        return create_model(schema_name, **model_fields)

    @classmethod
    def from_callable(
        cls,
        obj: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "ComponentTool":
        """Create a ComponentTool from any callable object.

        This is a convenience method that automatically detects the
        appropriate method to wrap (__call__, run, etc.).

        Args:
            obj: Callable object (class instance with __call__ or run)
            name: Tool name
            description: Tool description

        Returns:
            ComponentTool configured for the callable
        """
        # Determine the best method to use
        if hasattr(obj, "run") and callable(getattr(obj, "run")):
            method_name = "run"
        elif callable(obj):
            method_name = "__call__"
        else:
            raise ValueError(f"Object {type(obj).__name__} is not callable")

        return cls.from_component(
            component=obj,
            name=name,
            description=description,
            method_name=method_name,
        )

    @classmethod
    def from_pipeline_component(
        cls,
        component: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "ComponentTool":
        """Create a ComponentTool optimized for pipeline-style components.

        Pipeline components typically return dictionaries with output names
        as keys. This method configures appropriate output transformation.

        Args:
            component: Pipeline component instance
            name: Tool name (defaults to component class name)
            description: Tool description

        Returns:
            ComponentTool configured for pipeline components
        """

        def pipeline_output_transformer(result: Any) -> str:
            """Transform pipeline component output."""
            if isinstance(result, dict):
                # Pipeline components return {output_name: value}
                # Unwrap single-key results for cleaner output
                if len(result) == 1:
                    value = next(iter(result.values()))
                    if isinstance(value, list):
                        # List of documents/results - extract content if available
                        if value and hasattr(value[0], "content"):
                            return json.dumps(
                                [
                                    {
                                        "content": d.content,
                                        "meta": getattr(d, "meta", {}),
                                    }
                                    for d in value
                                ],
                                ensure_ascii=False,
                                indent=2,
                            )
                        return json.dumps(
                            value, ensure_ascii=False, indent=2, default=str
                        )
                    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
                return json.dumps(result, ensure_ascii=False, indent=2, default=str)
            return str(result)

        return cls.from_component(
            component=component,
            name=name,
            description=description,
            method_name="run",
            output_transformer=pipeline_output_transformer,
        )
