from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx

from marie.mcp.models import MCPToolCallResult, MCPToolTestResult


class MCPClient:
    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds

    @asynccontextmanager
    async def session(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[Any]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        http_client = httpx.AsyncClient(
            headers=headers or None, timeout=self.timeout_seconds
        )
        async with streamable_http_client(url, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def test_server(
        self,
        server_id: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> MCPToolTestResult:
        async with self.session(url, headers=headers) as session:
            tools = await session.list_tools()
            return MCPToolTestResult(
                success=True,
                server_id=server_id,
                tool_count=len(tools.tools),
            )

    async def list_tools(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> list[Any]:
        async with self.session(url, headers=headers) as session:
            tools = await session.list_tools()
            return list(tools.tools)

    async def call_tool(
        self,
        server_id: str,
        url: str,
        tool_name: str,
        arguments: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> MCPToolCallResult:
        async with self.session(url, headers=headers) as session:
            result = await session.call_tool(tool_name, arguments=arguments)
            if hasattr(result, "model_dump"):
                payload = result.model_dump(by_alias=True, mode="json")
            else:
                payload = dict(result)
            return MCPToolCallResult(
                server_id=server_id, tool_name=tool_name, result=payload
            )
