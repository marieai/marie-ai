from __future__ import annotations

from datetime import datetime, timezone

import pytest

from marie.mcp.models import (
    MCPAuthType,
    MCPServer,
    MCPServerCreate,
    MCPServerStatus,
    MCPServerTestRequest,
    MCPTool,
    MCPToolTestResult,
    MCPTransport,
)
from marie.mcp.service import MCPServerService


def _server() -> MCPServer:
    now = datetime.now(timezone.utc)
    return MCPServer(
        id="server-1",
        workspace_id="workspace-1",
        name="Docs MCP",
        url="http://127.0.0.1:8000/mcp",
        transport=MCPTransport.STREAMABLE_HTTP,
        auth_type=MCPAuthType.NONE,
        headers={},
        status=MCPServerStatus.PENDING,
        last_tested_at=None,
        last_error=None,
        tool_count=0,
        discovered_tools=[],
        last_discovery_at=None,
        is_enabled=True,
        tags=[],
        created_by_id=None,
        updated_by_id=None,
        created_at=now,
        updated_at=now,
    )


class StubRepo:
    def __init__(self) -> None:
        self.server = _server()
        self.status_updates: list[tuple[str, MCPServerStatus, str | None]] = []
        self.discovered_tools: list[MCPTool] | None = None

    async def create_server(self, payload: MCPServerCreate, created_by_id: str | None = None) -> MCPServer:
        return self.server.model_copy(
            update={
                "workspace_id": payload.workspace_id,
                "name": payload.name,
                "url": str(payload.url),
                "auth_type": payload.auth_type,
                "headers": payload.headers,
                "tags": payload.tags,
                "is_enabled": payload.is_enabled,
                "created_by_id": created_by_id,
                "updated_by_id": created_by_id,
            }
        )

    async def get_server(self, server_id: str) -> MCPServer | None:
        return self.server if self.server.id == server_id else None

    async def update_status(
        self,
        server_id: str,
        status: MCPServerStatus,
        error: str | None = None,
    ) -> None:
        self.status_updates.append((server_id, status, error))

    async def update_discovered_tools(self, server_id: str, tools: list[MCPTool]) -> None:
        self.discovered_tools = tools


class StubClient:
    def __init__(self) -> None:
        self.test_calls: list[tuple[str, str, dict[str, str] | None]] = []
        self.discover_calls: list[tuple[str, dict[str, str] | None]] = []

    async def test_server(
        self,
        server_id: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> MCPToolTestResult:
        self.test_calls.append((server_id, url, headers))
        return MCPToolTestResult(success=True, server_id=server_id, tool_count=2)

    async def list_tools(self, url: str, headers: dict[str, str] | None = None) -> list[dict[str, object]]:
        self.discover_calls.append((url, headers))
        return [
            {
                "name": "search_docs",
                "title": "Search docs",
                "description": "Searches internal docs",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]


@pytest.mark.asyncio
async def test_create_server_blocks_private_hosts_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARIE_MCP_ALLOW_PRIVATE_HOSTS", raising=False)
    service = MCPServerService(repo=StubRepo(), client=StubClient())

    with pytest.raises(ValueError, match="private or local network address"):
        await service.create_server(
            MCPServerCreate(
                name="Local MCP",
                url="http://127.0.0.1:8000/mcp",
                workspace_id="workspace-1",
            )
        )


@pytest.mark.asyncio
async def test_test_draft_returns_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIE_MCP_ALLOW_PRIVATE_HOSTS", "true")
    client = StubClient()
    service = MCPServerService(repo=StubRepo(), client=client)

    result = await service.test_draft(
        MCPServerTestRequest(
            name="Docs MCP",
            url="http://127.0.0.1:8000/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
            auth_type=MCPAuthType.STATIC_HEADERS,
            headers={"Authorization": "Bearer token"},
        )
    )

    assert result.success is True
    assert result.tool_count == 2
    assert result.message == "Connection test passed. Found 2 tools."
    assert client.test_calls == [
        ("draft", "http://127.0.0.1:8000/mcp", {"Authorization": "Bearer token"})
    ]


@pytest.mark.asyncio
async def test_discover_tools_normalizes_and_persists_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIE_MCP_ALLOW_PRIVATE_HOSTS", "true")
    repo = StubRepo()
    client = StubClient()
    service = MCPServerService(repo=repo, client=client)

    result = await service.discover_tools("server-1")

    assert len(result.tools) == 1
    assert result.tools[0].name == "search_docs"
    assert result.tools[0].server_id == "server-1"
    assert result.tools[0].slug.startswith("mcp--Docs MCP--search_docs--")
    assert repo.discovered_tools is not None
    assert repo.discovered_tools[0].name == "search_docs"
