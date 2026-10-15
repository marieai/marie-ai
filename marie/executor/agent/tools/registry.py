"""Resolve Marie server tool specifications for ``AgentExecutor``."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from marie.agent.tools import AgentTool, resolve_tools


@contextmanager
def plugin_tool_runtime(parameters: dict[str, Any]) -> Iterator[None]:
    """Bind plugin invocations to the current request's tenant and host config."""
    from marie.plugins.agent_tool import PluginTool
    from marie.plugins.daemon_client import PluginDaemonClient

    organization_id = parameters.get('organization_id')
    workspace_id = parameters.get('workspace_id')
    user_id = parameters.get('user_id')

    def client_factory() -> PluginDaemonClient:
        if not organization_id or not workspace_id:
            raise ValueError('Plugin tools require organization_id and workspace_id')
        key_id = os.environ.get('MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID')
        secret = os.environ.get('MARIE_PLUGIN_DAEMON_SIGNING_SECRET')
        if bool(key_id) != bool(secret):
            raise ValueError(
                'Plugin daemon signing key ID and secret must be configured together'
            )
        return PluginDaemonClient(
            organization_id=str(organization_id),
            workspace_id=str(workspace_id),
            user_id=str(user_id) if user_id else None,
            signing_key_id=key_id,
            signing_secret=secret,
        )

    with PluginTool.runtime(client_factory):
        yield


def resolve_executor_tools(
    tool_specs: list[str | dict[str, Any] | AgentTool | Callable[..., Any]],
) -> dict[str, AgentTool]:
    """Resolve native, MCP, and extension tools for server executors."""
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
