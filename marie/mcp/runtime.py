from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marie.mcp.models import MCPAuthType, MCPServer, MCPTool
from marie.mcp.repository import MCPServerRepository


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class MCPToolSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)

    type: str = Field(default="mcp")
    tool_name: str | None = None
    remote_tool_name: str | None = None
    name: str | None = None
    server_id: str | None = None
    source_server_id: str | None = None
    server_name: str | None = None
    source_server_name: str | None = None
    server_url: str | None = None
    source_server_url: str | None = None
    workspace_id: str | None = None
    description: str | None = None
    tool_description: str | None = None
    input_schema: dict[str, Any] | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    auth_type: MCPAuthType = MCPAuthType.NONE

    @model_validator(mode="after")
    def normalize(self) -> "MCPToolSpec":
        if self.type != "mcp":
            raise ValueError("MCP tool specs must set type='mcp'")

        tool_name = self.remote_tool_name or self.tool_name or self.name
        if not tool_name:
            raise ValueError("MCP tool spec must include a tool name")

        self.tool_name = tool_name
        self.remote_tool_name = tool_name
        self.server_id = self.server_id or self.source_server_id
        self.server_name = self.server_name or self.source_server_name
        self.server_url = self.server_url or self.source_server_url
        self.description = self.description or self.tool_description

        if not self.server_id and not self.server_url:
            raise ValueError("MCP tool spec must include server_id or server_url")

        if self.auth_type == MCPAuthType.STATIC_HEADERS and not self.headers:
            raise ValueError("Static header auth requires headers")

        return self


def is_mcp_tool_spec(spec: dict[str, Any]) -> bool:
    spec_type = spec.get("type")
    if spec_type == "mcp":
        return True
    return any(
        key in spec
        for key in (
            "server_id",
            "serverId",
            "source_server_id",
            "sourceServerId",
            "remote_tool_name",
            "remoteToolName",
            "server_url",
            "serverUrl",
            "source_server_url",
            "sourceServerUrl",
        )
    )


async def hydrate_mcp_tool_spec(
    spec: MCPToolSpec,
    repo: MCPServerRepository | None = None,
) -> MCPToolSpec:
    if not spec.server_id:
        return spec

    repo = repo or MCPServerRepository()
    server = await repo.get_server(spec.server_id)
    if server is None:
        raise ValueError(f"MCP server {spec.server_id} not found")

    discovered = _find_discovered_tool(server, spec.tool_name or "")

    return spec.model_copy(
        update={
            "server_name": spec.server_name or server.name,
            "server_url": spec.server_url or server.url,
            "auth_type": spec.auth_type if spec.headers else server.auth_type,
            "headers": spec.headers or server.headers,
            "description": spec.description
            or (discovered.description if discovered else None),
            "input_schema": spec.input_schema
            or (discovered.input_schema if discovered else None),
        }
    )


def _find_discovered_tool(server: MCPServer, tool_name: str) -> MCPTool | None:
    for tool in server.discovered_tools:
        if tool.name == tool_name or tool.slug == tool_name:
            return tool
    return None
