from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

from marie.logging_core.logger import MarieLogger
from marie.mcp.client import MCPClient
from marie.mcp.models import (
    MCPAuthType,
    MCPServer,
    MCPServerCreate,
    MCPServerStatus,
    MCPServerTestRequest,
    MCPServerUpdate,
    MCPToolCallResult,
    MCPToolDiscoveryResult,
    MCPToolTestResult,
)
from marie.mcp.repository import MCPServerRepository
from marie.mcp.sync import normalize_tools

logger = MarieLogger("marie.mcp.service")


class MCPServerService:
    def __init__(
        self,
        repo: MCPServerRepository,
        client: MCPClient | None = None,
    ):
        self.repo = repo
        self.client = client or MCPClient()

    async def list_servers(self, workspace_id: str) -> list[MCPServer]:
        return await self.repo.list_servers(workspace_id)

    async def get_server(self, server_id: str) -> MCPServer:
        server = await self.repo.get_server(server_id)
        if server is None:
            raise ValueError(f"MCP server {server_id} not found")
        return server

    async def create_server(
        self,
        payload: MCPServerCreate,
        user_id: str | None = None,
    ) -> MCPServer:
        self._validate_payload(payload.url, payload.auth_type, payload.headers)
        return await self.repo.create_server(payload, created_by_id=user_id)

    async def test_draft(self, payload: MCPServerTestRequest) -> MCPToolTestResult:
        self._validate_payload(payload.url, payload.auth_type, payload.headers)
        result = await self.client.test_server(
            server_id="draft",
            url=str(payload.url),
            headers=payload.headers or None,
        )
        return result.model_copy(
            update={
                "message": f"Connection test passed. Found {result.tool_count} tools."
            }
        )

    async def update_server(
        self,
        server_id: str,
        payload: MCPServerUpdate,
        user_id: str | None = None,
    ) -> MCPServer:
        current = await self.get_server(server_id)
        url = payload.url if payload.url is not None else current.url
        auth_type = (
            payload.auth_type if payload.auth_type is not None else current.auth_type
        )
        headers = payload.headers if payload.headers is not None else current.headers
        self._validate_payload(url, auth_type, headers)
        return await self.repo.update_server(server_id, payload, updated_by_id=user_id)

    async def delete_server(self, server_id: str) -> bool:
        await self.get_server(server_id)
        return await self.repo.delete_server(server_id)

    async def test_connection(self, server_id: str) -> MCPToolTestResult:
        server = await self.get_server(server_id)
        try:
            result = await self.client.test_server(
                server_id=server.id,
                url=server.url,
                headers=self._request_headers(server),
            )
        except Exception as exc:
            await self.repo.update_status(server.id, MCPServerStatus.ERROR, str(exc))
            raise

        await self.repo.update_status(server.id, MCPServerStatus.ACTIVE)
        return result.model_copy(
            update={
                "message": f"Connection test passed. Found {result.tool_count} tools."
            }
        )

    async def discover_tools(self, server_id: str) -> MCPToolDiscoveryResult:
        server = await self.get_server(server_id)
        try:
            tools = await self.client.list_tools(
                server.url, headers=self._request_headers(server)
            )
            normalized = normalize_tools(server, tools)
        except Exception as exc:
            await self.repo.update_status(server.id, MCPServerStatus.ERROR, str(exc))
            raise

        await self.repo.update_discovered_tools(server.id, normalized)
        return MCPToolDiscoveryResult(server_id=server.id, tools=normalized)

    async def get_cached_tools(self, server_id: str) -> MCPToolDiscoveryResult:
        server = await self.get_server(server_id)
        return MCPToolDiscoveryResult(
            server_id=server.id, tools=server.discovered_tools
        )

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict,
    ) -> MCPToolCallResult:
        server = await self.get_server(server_id)
        if not server.is_enabled:
            raise ValueError(f"MCP server {server.name} is disabled")

        return await self.client.call_tool(
            server_id=server.id,
            url=server.url,
            tool_name=tool_name,
            arguments=arguments,
            headers=self._request_headers(server),
        )

    def _request_headers(self, server: MCPServer) -> dict[str, str] | None:
        if server.auth_type == MCPAuthType.NONE:
            return None
        return server.headers

    def _validate_payload(
        self,
        url: str,
        auth_type: MCPAuthType,
        headers: dict[str, str],
    ) -> None:
        self._validate_url(url)
        if auth_type == MCPAuthType.STATIC_HEADERS and not headers:
            raise ValueError("Static headers auth requires at least one header")

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(str(url))
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("MCP server URL must use http or https")
        if not parsed.hostname:
            raise ValueError("MCP server URL must include a host")
        if parsed.username or parsed.password:
            raise ValueError("Credentials in the MCP server URL are not allowed")

        if _allow_private_hosts():
            return

        try:
            addresses = {
                ipaddress.ip_address(info[4][0])
                for info in socket.getaddrinfo(
                    parsed.hostname, parsed.port or 80, type=socket.SOCK_STREAM
                )
            }
        except socket.gaierror as exc:
            raise ValueError(
                f"Unable to resolve MCP server host: {parsed.hostname}"
            ) from exc

        for address in addresses:
            if (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_reserved
                or address.is_multicast
            ):
                raise ValueError(
                    "MCP server URL resolves to a private or local network address"
                )


def _allow_private_hosts() -> bool:
    return os.getenv("MARIE_MCP_ALLOW_PRIVATE_HOSTS", "").lower() in {
        "1",
        "true",
        "yes",
    }
