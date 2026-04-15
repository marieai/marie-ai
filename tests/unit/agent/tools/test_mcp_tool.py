from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from marie.agent.config import AgentConfig
from marie.agent.tools.mcp_tool import MCPRemoteTool
from marie.agent.tools.registry import resolve_tools
from marie.mcp.models import (
    MCPAuthType,
    MCPServer,
    MCPServerStatus,
    MCPTool,
    MCPTransport,
)
from marie.mcp.runtime import MCPToolSpec


class StubMCPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict[str, object], dict[str, str] | None]] = []

    async def call_tool(
        self,
        server_id: str,
        url: str,
        tool_name: str,
        arguments: dict[str, object],
        headers: dict[str, str] | None = None,
    ):
        from marie.mcp.models import MCPToolCallResult

        self.calls.append((server_id, url, tool_name, arguments, headers))
        return MCPToolCallResult(
            server_id=server_id,
            tool_name=tool_name,
            result={"content": [{"type": "text", "text": "ok"}]},
        )


class StubRepo:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.server = MCPServer(
            id="server-1",
            workspace_id="workspace-1",
            name="Docs MCP",
            url="https://mcp.example.com",
            transport=MCPTransport.STREAMABLE_HTTP,
            auth_type=MCPAuthType.STATIC_HEADERS,
            headers={"Authorization": "Bearer token"},
            status=MCPServerStatus.ACTIVE,
            last_tested_at=None,
            last_error=None,
            tool_count=1,
            discovered_tools=[
                MCPTool(
                    name="search_docs",
                    slug="mcp--docs--search",
                    server_id="server-1",
                    server_name="Docs MCP",
                    description="Search internal docs",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                )
            ],
            last_discovery_at=now,
            is_enabled=True,
            tags=[],
            created_by_id=None,
            updated_by_id=None,
            created_at=now,
            updated_at=now,
        )

    async def get_server(self, server_id: str) -> MCPServer | None:
        if server_id == self.server.id:
            return self.server
        return None


def test_resolve_tools_supports_mcp_specs() -> None:
    resolved = resolve_tools(
        [
            {
                "type": "mcp",
                "toolName": "search_docs",
                "serverUrl": "https://mcp.example.com",
                "description": "Search docs",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
    )

    assert "search_docs" in resolved
    tool = resolved["search_docs"]
    assert isinstance(tool, MCPRemoteTool)
    assert tool.metadata.get_parameters_dict()["properties"]["query"]["type"] == "string"


@pytest.mark.asyncio
async def test_mcp_remote_tool_hydrates_server_and_executes() -> None:
    client = StubMCPClient()
    repo = StubRepo()
    tool = MCPRemoteTool(MCPToolSpec(server_id="server-1", tool_name="search_docs"), client=client, repo=repo)

    assert tool.metadata.description == "Search internal docs"
    result = await tool.acall(query="billing")

    assert result.content == "ok"
    assert client.calls == [
        (
            "server-1",
            "https://mcp.example.com",
            "search_docs",
            {"query": "billing"},
            {"Authorization": "Bearer token"},
        )
    ]


def test_agent_config_emits_mcp_tool_specs() -> None:
    config = AgentConfig.model_validate(
        {
            "tools": ["search"],
            "mcp": {
                "enabled": True,
                "servers": [
                    {
                        "name": "Docs MCP",
                        "server_id": "server-1",
                        "auth_type": "static_headers",
                        "headers": {"Authorization": "Bearer token"},
                        "tools": [
                            "search_docs",
                            {
                                "name": "lookup_policy",
                                "description": "Look up a policy",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"policy_id": {"type": "string"}},
                                },
                            },
                        ],
                    }
                ],
            },
        }
    )

    tools = config.get_tool_list()

    assert tools[0] == "search"
    assert tools[1] == {
        "type": "mcp",
        "tool_name": "search_docs",
        "server_id": "server-1",
        "server_name": "Docs MCP",
        "server_url": None,
        "auth_type": "static_headers",
        "headers": {"Authorization": "Bearer token"},
    }
    assert json.dumps(tools[2])  # sanity check the dict is JSON-serializable
    assert tools[2]["tool_name"] == "lookup_policy"
    assert tools[2]["input_schema"]["properties"]["policy_id"]["type"] == "string"
