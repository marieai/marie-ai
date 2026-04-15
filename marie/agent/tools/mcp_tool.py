from __future__ import annotations

import json
from typing import Any

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.helper import run_async
from marie.mcp.client import MCPClient
from marie.mcp.models import MCPToolCallResult
from marie.mcp.repository import MCPServerRepository
from marie.mcp.runtime import MCPToolSpec, hydrate_mcp_tool_spec


class MCPRemoteTool(AgentTool):
    def __init__(
        self,
        spec: MCPToolSpec,
        client: MCPClient | None = None,
        repo: MCPServerRepository | None = None,
    ):
        self._spec = spec
        self._client = client or MCPClient()
        self._repo = repo

    @property
    def metadata(self) -> ToolMetadata:
        if self._spec.server_id and (
            self._spec.server_url is None
            or self._spec.description is None
            or self._spec.input_schema is None
        ):
            self._spec = run_async(hydrate_mcp_tool_spec(self._spec, repo=self._repo))

        return ToolMetadata(
            name=self._spec.tool_name or "mcp_tool",
            description=self._spec.description or self._default_description(),
            parameters=self._spec.input_schema
            or {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        )

    def call(self, **kwargs: Any) -> ToolOutput:
        return run_async(self.acall(**kwargs))

    async def acall(self, **kwargs: Any) -> ToolOutput:
        spec = await self._get_spec()

        if spec.server_id:
            result = await self._client.call_tool(
                server_id=spec.server_id,
                url=spec.server_url or "",
                tool_name=spec.remote_tool_name or spec.tool_name or "",
                arguments=kwargs,
                headers=spec.headers or None,
            )
        else:
            result = await self._client.call_tool(
                server_id="direct",
                url=spec.server_url or "",
                tool_name=spec.remote_tool_name or spec.tool_name or "",
                arguments=kwargs,
                headers=spec.headers or None,
            )

        return ToolOutput(
            content=self._format_content(result),
            tool_name=self.name,
            raw_input=kwargs,
            raw_output=result.result,
        )

    async def _get_spec(self) -> MCPToolSpec:
        hydrated = await hydrate_mcp_tool_spec(self._spec, repo=self._repo)
        self._spec = hydrated
        return hydrated

    def _default_description(self) -> str:
        if self._spec.server_name:
            return f"Call the {self.name} MCP tool on {self._spec.server_name}."
        return f"Call the remote MCP tool {self.name}."

    def _format_content(self, result: MCPToolCallResult) -> str:
        payload = result.result
        if isinstance(payload, dict):
            content = payload.get("content")
            if isinstance(content, list):
                texts = [
                    item.get("text")
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and item.get("text")
                ]
                if texts:
                    return "\n".join(str(text) for text in texts)
        return json.dumps(payload, ensure_ascii=True)
