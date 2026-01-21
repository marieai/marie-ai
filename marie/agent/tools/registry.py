"""Tool registry for Marie agent framework.

Provides a global registry for tools and a decorator for easy registration.
Follows Qwen-Agent patterns with thread-safe operations.
"""

from __future__ import annotations

import threading
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    overload,
)

from pydantic import BaseModel

from marie.agent.tools.base import (
    AgentTool,
    FunctionTool,
    ToolMetadata,
    adapt_tool,
)
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.tools.registry")


class ToolRegistry:
    """Thread-safe registry for agent tools.

    Provides centralized tool management with registration, lookup,
    and instantiation capabilities.

    Example:
        ```python
        registry = ToolRegistry()

        # Register a tool class
        registry.register("search", SearchTool)

        # Register a tool instance
        registry.register("calculator", CalculatorTool())

        # Get tool
        search_tool = registry.get("search")
        ```
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Union[Type[AgentTool], AgentTool, Callable]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        tool: Union[Type[AgentTool], AgentTool, Callable],
        overwrite: bool = False,
    ) -> None:
        """Register a tool with the registry.

        Args:
            name: Unique tool name
            tool: Tool class, instance, or callable function
            overwrite: If True, overwrite existing registration

        Raises:
            ValueError: If name already exists and overwrite is False
        """
        with self._lock:
            if name in self._tools and not overwrite:
                logger.warning(
                    f"Tool '{name}' already registered. Use overwrite=True to replace."
                )
                return

            self._tools[name] = tool
            logger.debug(f"Registered tool: {name}")

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                logger.debug(f"Unregistered tool: {name}")
                return True
            return False

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AgentTool:
        """Get a tool instance by name.

        Args:
            name: Tool name
            config: Optional configuration dict for tool instantiation
            **kwargs: Additional arguments for instantiation

        Returns:
            AgentTool instance

        Raises:
            KeyError: If tool is not registered
        """
        with self._lock:
            if name not in self._tools:
                raise KeyError(f"Tool '{name}' is not registered")

            tool = self._tools[name]

            # Already an instance
            if isinstance(tool, AgentTool):
                return tool

            # Class - instantiate
            if isinstance(tool, type) and issubclass(tool, AgentTool):
                init_kwargs = config or {}
                init_kwargs.update(kwargs)
                return tool(**init_kwargs)

            # Callable - wrap as FunctionTool
            if callable(tool):
                return FunctionTool.from_defaults(fn=tool, name=name)

            raise TypeError(f"Cannot instantiate tool '{name}' of type {type(tool)}")

    def get_or_none(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[AgentTool]:
        """Get a tool instance, returning None if not found.

        Args:
            name: Tool name
            config: Optional configuration dict
            **kwargs: Additional arguments

        Returns:
            AgentTool instance or None
        """
        try:
            return self.get(name, config, **kwargs)
        except KeyError:
            return None

    def has(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name to check

        Returns:
            True if tool exists
        """
        with self._lock:
            return name in self._tools

    def list_tools(self) -> List[str]:
        """Get list of all registered tool names.

        Returns:
            List of tool names
        """
        with self._lock:
            return list(self._tools.keys())

    def get_all_metadata(self) -> Dict[str, ToolMetadata]:
        """Get metadata for all registered tools.

        Returns:
            Dict mapping tool names to their metadata
        """
        result = {}
        for name in self.list_tools():
            try:
                tool = self.get(name)
                result[name] = tool.metadata
            except Exception as e:
                logger.warning(f"Could not get metadata for tool '{name}': {e}")
        return result

    def clear(self) -> None:
        """Remove all registered tools."""
        with self._lock:
            self._tools.clear()
            logger.debug("Cleared all tools from registry")


# Global tool registry instance
TOOL_REGISTRY = ToolRegistry()


@overload
def register_tool(
    tool_or_name: Callable[..., Any],
) -> Callable[..., Any]: ...


@overload
def register_tool(
    tool_or_name: Optional[str] = None,
    *,
    description: Optional[str] = None,
    fn_schema: Optional[Type[BaseModel]] = None,
    return_direct: bool = False,
    overwrite: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def register_tool(
    tool_or_name: Optional[Union[str, Callable[..., Any]]] = None,
    *,
    description: Optional[str] = None,
    fn_schema: Optional[Type[BaseModel]] = None,
    return_direct: bool = False,
    overwrite: bool = True,
) -> Union[Callable[..., Any], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """Decorator to register a function or class as a tool.

    Can be used with or without arguments:

    ```python
    # Without arguments - uses function name
    @register_tool
    def search(query: str) -> str:
        '''Search for documents.'''
        return json.dumps(results)


    # With arguments
    @register_tool("my_search", description="Advanced search")
    def search(query: str, limit: int = 10) -> str:
        '''Search for documents.'''
        return json.dumps(results)


    # With schema
    @register_tool("calc", fn_schema=CalculatorInput)
    def calculate(expression: str) -> str:
        '''Evaluate a math expression.'''
        return str(eval(expression))
    ```

    Args:
        tool_or_name: Tool name string, or the function/class being decorated
        description: Tool description (defaults to docstring)
        fn_schema: Pydantic model for input validation
        return_direct: If True, return output directly to user
        overwrite: If True, overwrite existing registration

    Returns:
        Decorated function or decorator
    """

    def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
        # Determine tool name
        if isinstance(tool_or_name, str):
            tool_name = tool_or_name
        else:
            tool_name = obj.__name__

        # Handle class registration
        if isinstance(obj, type):
            if issubclass(obj, AgentTool):
                TOOL_REGISTRY.register(tool_name, obj, overwrite=overwrite)
            else:
                raise TypeError(f"Class {obj.__name__} must be a subclass of AgentTool")
            return obj

        # Handle function registration
        tool = FunctionTool.from_defaults(
            fn=obj,
            name=tool_name,
            description=description,
            fn_schema=fn_schema,
            return_direct=return_direct,
        )
        TOOL_REGISTRY.register(tool_name, tool, overwrite=overwrite)

        # Return original function for chaining
        return obj

    # Handle @register_tool without parentheses
    if callable(tool_or_name):
        return decorator(tool_or_name)

    return decorator


def get_tool(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentTool:
    """Get a tool from the global registry.

    Convenience function for TOOL_REGISTRY.get().

    Args:
        name: Tool name
        config: Optional configuration dict
        **kwargs: Additional arguments

    Returns:
        AgentTool instance

    Raises:
        KeyError: If tool is not registered
    """
    return TOOL_REGISTRY.get(name, config, **kwargs)


def list_tools() -> List[str]:
    """List all registered tools.

    Returns:
        List of registered tool names
    """
    return TOOL_REGISTRY.list_tools()


def resolve_tools(
    tool_specs: List[Union[str, Dict[str, Any], AgentTool, Callable]],
) -> Dict[str, AgentTool]:
    """Resolve a list of tool specifications to AgentTool instances.

    Supports:
        - Tool name strings (looked up from registry)
        - Configuration dicts with 'name' key
        - AgentTool instances (used directly)
        - Callable functions (wrapped as FunctionTool)

    Args:
        tool_specs: List of tool specifications

    Returns:
        Dict mapping tool names to AgentTool instances

    Example:
        ```python
        tools = resolve_tools(
            [
                "search",  # Lookup from registry
                {"name": "calculator", "precision": 2},  # With config
                MyCustomTool(),  # Direct instance
                my_function,  # Wrap callable
            ]
        )
        ```
    """
    result: Dict[str, AgentTool] = {}

    for spec in tool_specs:
        if isinstance(spec, str):
            # Lookup from registry
            tool = get_tool(spec)
            result[tool.name] = tool

        elif isinstance(spec, dict):
            # Configuration dict
            name = spec.get("name")
            if name is None:
                raise ValueError("Tool config dict must have 'name' key")
            config = {k: v for k, v in spec.items() if k != "name"}
            tool = get_tool(name, config=config)
            result[tool.name] = tool

        elif isinstance(spec, AgentTool):
            # Direct instance
            result[spec.name] = spec

        elif callable(spec):
            # Wrap callable
            tool = adapt_tool(spec)
            result[tool.name] = tool

        else:
            raise TypeError(f"Cannot resolve tool specification: {type(spec)}")

    return result
