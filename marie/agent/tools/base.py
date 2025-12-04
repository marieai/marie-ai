"""Base tool interface for Marie agent framework.

This module provides tool abstractions compatible with Qwen-Agent patterns
while bridging to the existing marie.core.tools infrastructure.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_type_hints,
)

if TYPE_CHECKING:
    from marie.core.tools.types import BaseTool as CoreBaseTool

from pydantic import BaseModel, Field

from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.tools")


class DefaultToolSchema(BaseModel):
    """Default tool function schema when none is provided."""

    input: str = Field(..., description="Input query string")


@dataclass
class ToolMetadata:
    """Metadata describing a tool's interface.

    Attributes:
        name: Tool name (used for registration and function calling)
        description: Human-readable description of what the tool does
        fn_schema: Pydantic model defining the input parameters
        return_direct: If True, return tool output directly without further LLM processing
    """

    name: str
    description: str
    fn_schema: Optional[Type[BaseModel]] = None
    return_direct: bool = False
    parameters: Optional[Dict[str, Any]] = field(default=None)

    def get_parameters_dict(self) -> Dict[str, Any]:
        """Get OpenAI-compatible parameters schema."""
        if self.parameters is not None:
            return self.parameters

        if self.fn_schema is None:
            return {
                "type": "object",
                "properties": {
                    "input": {
                        "title": "input",
                        "type": "string",
                        "description": "Input query string",
                    }
                },
                "required": ["input"],
            }

        schema = self.fn_schema.model_json_schema()
        # Filter to only include relevant schema keys
        return {
            k: v
            for k, v in schema.items()
            if k in ["type", "properties", "required", "definitions", "$defs"]
        }

    def to_openai_tool(self, skip_length_check: bool = False) -> Dict[str, Any]:
        """Convert to OpenAI tool format.

        Args:
            skip_length_check: Skip the 1024 character description limit check

        Returns:
            OpenAI-compatible tool definition
        """
        if not skip_length_check and len(self.description) > 1024:
            raise ValueError(
                "Tool description exceeds maximum length of 1024 characters. "
                "Please shorten your description or move it to the prompt."
            )
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_dict(),
            },
        }

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function format (deprecated, use to_openai_tool)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_dict(),
        }


class ToolOutput(BaseModel):
    """Output from a tool execution.

    Attributes:
        content: String representation of the output
        tool_name: Name of the tool that produced this output
        raw_input: Original input parameters
        raw_output: Raw output before string conversion
        is_error: Whether the execution resulted in an error
    """

    content: str = Field(..., description="String content of the output")
    tool_name: str = Field(..., description="Name of the tool")
    raw_input: Dict[str, Any] = Field(
        default_factory=dict, description="Raw input parameters"
    )
    raw_output: Any = Field(default=None, description="Raw output before conversion")
    is_error: bool = Field(default=False, description="Whether this is an error output")

    def __str__(self) -> str:
        return self.content


class ToolError(Exception):
    """Exception raised when a tool execution fails."""

    def __init__(
        self,
        message: str,
        tool_name: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(message)


class AgentTool(ABC):
    """Base class for agent tools.

    This is the primary tool interface for the Marie agent framework.
    Tools can be synchronous or asynchronous and support both simple
    string inputs and structured Pydantic model inputs.

    Subclasses must implement:
        - metadata property: Return ToolMetadata describing the tool
        - call method: Synchronous execution
        - acall method: Asynchronous execution (optional, defaults to sync)

    Example:
        ```python
        class SearchTool(AgentTool):
            @property
            def metadata(self) -> ToolMetadata:
                return ToolMetadata(
                    name="search",
                    description="Search for information",
                    fn_schema=SearchInput,
                )

            def call(self, query: str, **kwargs) -> ToolOutput:
                results = perform_search(query)
                return ToolOutput(
                    content=json.dumps(results),
                    tool_name=self.name,
                    raw_input={"query": query},
                    raw_output=results,
                )
        ```
    """

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Return the tool's metadata."""
        pass

    @property
    def name(self) -> str:
        """Get the tool name."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Get the tool description."""
        return self.metadata.description

    @abstractmethod
    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the tool synchronously.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments matching fn_schema

        Returns:
            ToolOutput with the result
        """
        pass

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the tool asynchronously.

        Default implementation runs the sync call in a thread pool.
        Override for true async implementations.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments matching fn_schema

        Returns:
            ToolOutput with the result
        """
        return await asyncio.to_thread(self.call, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Make the tool callable directly."""
        return self.call(*args, **kwargs)

    def _parse_input(self, tool_args: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse tool arguments to a dictionary.

        Args:
            tool_args: Either a JSON string or a dictionary

        Returns:
            Parsed arguments as a dictionary
        """
        if isinstance(tool_args, str):
            try:
                return json.loads(tool_args)
            except json.JSONDecodeError:
                # If it's not JSON, treat as simple input
                return {"input": tool_args}
        return tool_args

    def _validate_args(
        self,
        args: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Validate arguments against the tool's schema.

        Args:
            args: Parsed arguments dictionary

        Returns:
            Tuple of (validated_args, error_message).
            If validation succeeds, error_message is None.
            If validation fails, validated_args is the original args
            and error_message contains the validation error.
        """
        fn_schema = self.metadata.fn_schema
        if fn_schema is None:
            # No schema defined, pass through
            return args, None

        try:
            # Validate using Pydantic schema
            validated = fn_schema(**args)
            # Return validated args with defaults filled in
            return validated.model_dump(), None
        except Exception as e:
            # Validation failed
            error_msg = f"Argument validation failed: {type(e).__name__}: {e}"
            return args, error_msg

    def safe_call(
        self, tool_args: Union[str, Dict[str, Any]] = "{}", **kwargs: Any
    ) -> ToolOutput:
        """Safely execute the tool with error handling and schema validation.

        Args:
            tool_args: Tool arguments (string or dict)
            **kwargs: Additional keyword arguments

        Returns:
            ToolOutput, with is_error=True if validation or execution failed
        """
        parsed_args = self._parse_input(tool_args)
        parsed_args.update(kwargs)

        # Validate arguments against schema
        validated_args, validation_error = self._validate_args(parsed_args)
        if validation_error:
            logger.warning(f"Tool {self.name} validation failed: {validation_error}")
            return ToolOutput(
                content=validation_error,
                tool_name=self.name,
                raw_input=parsed_args,
                raw_output=None,
                is_error=True,
            )

        try:
            return self.call(**validated_args)
        except Exception as ex:
            error_message = self._format_error(ex)
            logger.warning(f"Tool {self.name} failed: {error_message}")
            return ToolOutput(
                content=error_message,
                tool_name=self.name,
                raw_input=validated_args,
                raw_output=None,
                is_error=True,
            )

    async def safe_acall(
        self, tool_args: Union[str, Dict[str, Any]] = "{}", **kwargs: Any
    ) -> ToolOutput:
        """Safely execute the tool asynchronously with error handling and schema validation.

        Args:
            tool_args: Tool arguments (string or dict)
            **kwargs: Additional keyword arguments

        Returns:
            ToolOutput, with is_error=True if validation or execution failed
        """
        parsed_args = self._parse_input(tool_args)
        parsed_args.update(kwargs)

        # Validate arguments against schema
        validated_args, validation_error = self._validate_args(parsed_args)
        if validation_error:
            logger.warning(f"Tool {self.name} validation failed: {validation_error}")
            return ToolOutput(
                content=validation_error,
                tool_name=self.name,
                raw_input=parsed_args,
                raw_output=None,
                is_error=True,
            )

        try:
            return await self.acall(**validated_args)
        except Exception as ex:
            error_message = self._format_error(ex)
            logger.warning(f"Tool {self.name} failed: {error_message}")
            return ToolOutput(
                content=error_message,
                tool_name=self.name,
                raw_input=validated_args,
                raw_output=None,
                is_error=True,
            )

    def _format_error(self, ex: Exception) -> str:
        """Format an exception into an error message."""
        exception_type = type(ex).__name__
        exception_message = str(ex)
        traceback_info = "".join(traceback.format_tb(ex.__traceback__))
        return (
            f"An error occurred when calling tool `{self.name}`:\n"
            f"{exception_type}: {exception_message}\n"
            f"Traceback:\n{traceback_info}"
        )

    def to_openai_tool(self, skip_length_check: bool = False) -> Dict[str, Any]:
        """Convert to OpenAI tool format."""
        return self.metadata.to_openai_tool(skip_length_check=skip_length_check)

    def get_function_definition(self) -> Dict[str, Any]:
        """Get function definition for LLM function calling."""
        return self.metadata.to_openai_function()


class FunctionTool(AgentTool):
    """Tool created from a Python function.

    Wraps a regular Python function as an AgentTool, automatically
    extracting the schema from type hints and docstrings.

    Example:
        ```python
        def search(query: str, limit: int = 10) -> str:
            '''Search for documents.

            Args:
                query: Search query string
                limit: Maximum results to return
            '''
            return json.dumps(perform_search(query, limit))


        tool = FunctionTool.from_defaults(fn=search)
        ```
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        metadata: ToolMetadata,
        async_fn: Optional[Callable[..., Any]] = None,
    ):
        """Initialize a FunctionTool.

        Args:
            fn: The synchronous function to wrap
            metadata: Tool metadata
            async_fn: Optional async version of the function
        """
        self._fn = fn
        self._async_fn = async_fn
        self._metadata = metadata

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the wrapped function."""
        raw_output = self._fn(*args, **kwargs)
        content = self._output_to_string(raw_output)
        return ToolOutput(
            content=content,
            tool_name=self.name,
            raw_input=kwargs,
            raw_output=raw_output,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the wrapped function asynchronously."""
        if self._async_fn is not None:
            raw_output = await self._async_fn(*args, **kwargs)
        elif asyncio.iscoroutinefunction(self._fn):
            raw_output = await self._fn(*args, **kwargs)
        else:
            raw_output = await asyncio.to_thread(self._fn, *args, **kwargs)

        content = self._output_to_string(raw_output)
        return ToolOutput(
            content=content,
            tool_name=self.name,
            raw_input=kwargs,
            raw_output=raw_output,
        )

    def _output_to_string(self, output: Any) -> str:
        """Convert output to string."""
        if isinstance(output, str):
            return output
        if isinstance(output, BaseModel):
            return output.model_dump_json()
        try:
            return json.dumps(output, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(output)

    @classmethod
    def from_defaults(
        cls,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        fn_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
        async_fn: Optional[Callable[..., Any]] = None,
    ) -> "FunctionTool":
        """Create a FunctionTool from a function with sensible defaults.

        Args:
            fn: The function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            fn_schema: Input schema (auto-generated if not provided)
            return_direct: Whether to return output directly
            async_fn: Optional async version of the function

        Returns:
            A configured FunctionTool instance
        """
        tool_name = name or fn.__name__

        # Extract description from docstring if not provided
        tool_description = description
        if tool_description is None:
            tool_description = fn.__doc__ or f"Tool: {tool_name}"
            # Take only the first paragraph
            tool_description = tool_description.split("\n\n")[0].strip()

        # Auto-generate schema from type hints if not provided
        if fn_schema is None:
            fn_schema = cls._generate_schema(fn, tool_name)

        metadata = ToolMetadata(
            name=tool_name,
            description=tool_description,
            fn_schema=fn_schema,
            return_direct=return_direct,
        )

        return cls(fn=fn, metadata=metadata, async_fn=async_fn)

    @staticmethod
    def _generate_schema(fn: Callable, name: str) -> Type[BaseModel]:
        """Generate a Pydantic schema from function signature.

        Args:
            fn: The function to analyze
            name: Name for the generated schema class

        Returns:
            A dynamically created Pydantic model class
        """
        sig = inspect.signature(fn)
        hints = get_type_hints(fn) if hasattr(fn, "__annotations__") else {}

        # Build field definitions
        fields: Dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls", "kwargs", "args"):
                continue

            # Get type hint or default to Any
            param_type = hints.get(param_name, Any)

            # Handle default values
            if param.default is inspect.Parameter.empty:
                fields[param_name] = (param_type, ...)
            else:
                fields[param_name] = (param_type, param.default)

        # Create dynamic Pydantic model
        schema_name = f"{name.title().replace('_', '')}Input"
        return type(
            schema_name,
            (BaseModel,),
            {"__annotations__": {k: v[0] for k, v in fields.items()}},
        )


def adapt_tool(tool: Any) -> AgentTool:
    """Adapt various tool types to AgentTool.

    Supports:
        - AgentTool instances (returned as-is)
        - marie.core.tools.types.BaseTool instances
        - Callable functions

    Args:
        tool: The tool to adapt

    Returns:
        An AgentTool instance
    """
    if isinstance(tool, AgentTool):
        return tool

    # Adapt from marie.core.tools.types.BaseTool
    from marie.core.tools.types import BaseTool as CoreBaseTool

    if isinstance(tool, CoreBaseTool):
        return _adapt_core_tool(tool)

    # Adapt from callable
    if callable(tool):
        return FunctionTool.from_defaults(fn=tool)

    raise TypeError(f"Cannot adapt {type(tool)} to AgentTool")


def _adapt_core_tool(core_tool: "CoreBaseTool") -> AgentTool:
    """Adapt a marie.core.tools.types.BaseTool to AgentTool."""
    from marie.core.tools.types import BaseTool as CoreBaseTool

    class CoreToolAdapter(AgentTool):
        """Adapter for marie.core.tools.types.BaseTool."""

        def __init__(self, tool: CoreBaseTool):
            self._tool = tool

        @property
        def metadata(self) -> ToolMetadata:
            core_meta = self._tool.metadata
            return ToolMetadata(
                name=core_meta.name or "unknown",
                description=core_meta.description,
                fn_schema=core_meta.fn_schema,
                return_direct=core_meta.return_direct,
            )

        def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
            result = self._tool(*args, **kwargs)
            # Convert from core ToolOutput if needed
            if hasattr(result, "content"):
                return ToolOutput(
                    content=result.content,
                    tool_name=result.tool_name,
                    raw_input=result.raw_input,
                    raw_output=result.raw_output,
                    is_error=getattr(result, "is_error", False),
                )
            return ToolOutput(
                content=str(result),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
            )

    return CoreToolAdapter(core_tool)
