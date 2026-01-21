"""Marie agent tools module."""

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
from marie.agent.tools.wrappers import ComponentTool

__all__ = [
    # Base classes
    "AgentTool",
    "FunctionTool",
    "ToolMetadata",
    "ToolOutput",
    "adapt_tool",
    # Component tools
    "ComponentTool",
    # Registry
    "TOOL_REGISTRY",
    "register_tool",
    "get_tool",
    "list_tools",
    "resolve_tools",
]
