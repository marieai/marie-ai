"""Resolve Marie server tool specifications for ``AgentExecutor``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from marie.agent.tools import AgentTool, resolve_tools


def resolve_executor_tools(
    tool_specs: list[str | dict[str, Any] | AgentTool | Callable[..., Any]],
) -> dict[str, AgentTool]:
    """Resolve portable tools plus Marie MCP and EmbeddedPlugin tools."""
    resolved: dict[str, AgentTool] = {}

    for spec in tool_specs:
        if isinstance(spec, dict):
            from marie.mcp.runtime import MCPToolSpec, is_mcp_tool_spec
            from marie.plugins.agent_tool import (
                PluginTool,
                PluginToolSpec,
                is_plugin_tool_spec,
            )

            if is_mcp_tool_spec(spec):
                from marie.mcp.agent_tool import MCPRemoteTool

                tool = MCPRemoteTool(MCPToolSpec.model_validate(spec))
                resolved[tool.name] = tool
                continue

            if is_plugin_tool_spec(spec):
                tool = PluginTool(PluginToolSpec.model_validate(spec))
                resolved[tool.name] = tool
                continue

        resolved.update(resolve_tools([spec]))

    return resolved
