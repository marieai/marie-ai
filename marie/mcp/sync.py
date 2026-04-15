from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from marie.mcp.models import MCPServer, MCPTool


def normalize_tools(server: MCPServer, tools: Iterable[Any]) -> list[MCPTool]:
    return [normalize_tool(server, tool) for tool in tools]


def normalize_tool(server: MCPServer, tool: Any) -> MCPTool:
    payload = _coerce_tool(tool)
    name = str(payload["name"])
    title = _get_title(payload)
    input_schema = payload.get("inputSchema") or payload.get("input_schema") or {}
    annotations = payload.get("annotations") or {}
    slug_seed = json.dumps(
        {
            "server_id": server.id,
            "name": name,
            "schema": input_schema,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(slug_seed.encode("utf-8")).hexdigest()[:10]

    return MCPTool(
        name=name,
        slug=f"mcp--{server.name}--{name}--{digest}",
        server_id=server.id,
        server_name=server.name,
        title=title,
        description=payload.get("description"),
        input_schema=input_schema,
        annotations=annotations,
    )


def _coerce_tool(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return tool
    if hasattr(tool, "model_dump"):
        return tool.model_dump(by_alias=True, mode="json")
    if hasattr(tool, "dict"):
        return tool.dict()

    payload: dict[str, Any] = {}
    for key in (
        "name",
        "title",
        "description",
        "inputSchema",
        "input_schema",
        "annotations",
    ):
        if hasattr(tool, key):
            payload[key] = getattr(tool, key)
    if "name" not in payload:
        raise ValueError("Discovered MCP tool is missing a name")
    return payload


def _get_title(payload: dict[str, Any]) -> str | None:
    if payload.get("title"):
        return payload["title"]
    annotations = payload.get("annotations") or {}
    title = annotations.get("title")
    return str(title) if title else None
