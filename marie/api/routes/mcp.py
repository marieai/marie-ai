"""MCP server management routes."""

from __future__ import annotations

from typing import Optional

from marie.logging_core.logger import MarieLogger
from marie.mcp.models import (
    CallToolRequest,
    MCPServer,
    MCPServerCreate,
    MCPServerTestRequest,
    MCPServerUpdate,
    MCPToolCallResult,
    MCPToolDiscoveryResult,
    MCPToolTestResult,
)
from marie.mcp.repository import MCPServerRepository
from marie.mcp.service import MCPServerService

logger = MarieLogger("marie.api.routes.mcp").logger


def create_fastapi_router(prefix: str = "/api/mcp"):
    try:
        from fastapi import APIRouter, Depends, HTTPException, Query

        api_router = APIRouter(prefix=prefix, tags=["mcp"])
        service = MCPServerService(MCPServerRepository())

        async def get_current_user() -> str:
            return "default_user"

        @api_router.get("/servers", response_model=list[MCPServer])
        async def list_servers(
            workspace_id: str = Query(..., alias="workspaceId"),
            _user_id: str = Depends(get_current_user),
        ):
            return await service.list_servers(workspace_id)

        @api_router.post("/servers", response_model=MCPServer)
        async def create_server(
            request: MCPServerCreate,
            user_id: str = Depends(get_current_user),
        ):
            try:
                return await service.create_server(request, user_id=user_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        @api_router.post("/servers/test", response_model=MCPToolTestResult)
        async def test_server_draft(
            request: MCPServerTestRequest,
            _user_id: str = Depends(get_current_user),
        ):
            try:
                return await service.test_draft(request)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc))

        @api_router.get("/servers/{server_id}", response_model=MCPServer)
        async def get_server(
            server_id: str,
            _user_id: str = Depends(get_current_user),
        ):
            try:
                return await service.get_server(server_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc))

        @api_router.put("/servers/{server_id}", response_model=MCPServer)
        async def update_server(
            server_id: str,
            request: MCPServerUpdate,
            user_id: str = Depends(get_current_user),
        ):
            try:
                return await service.update_server(server_id, request, user_id=user_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        @api_router.delete("/servers/{server_id}")
        async def delete_server(
            server_id: str,
            _user_id: str = Depends(get_current_user),
        ):
            try:
                deleted = await service.delete_server(server_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            return {"deleted": deleted}

        @api_router.post("/servers/{server_id}/test", response_model=MCPToolTestResult)
        async def test_server(
            server_id: str,
            _user_id: str = Depends(get_current_user),
        ):
            try:
                return await service.test_connection(server_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc))

        @api_router.post(
            "/servers/{server_id}/discover", response_model=MCPToolDiscoveryResult
        )
        async def discover_tools(
            server_id: str,
            _user_id: str = Depends(get_current_user),
        ):
            try:
                return await service.discover_tools(server_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc))

        @api_router.get(
            "/servers/{server_id}/tools", response_model=MCPToolDiscoveryResult
        )
        async def get_cached_tools(
            server_id: str,
            _user_id: str = Depends(get_current_user),
        ):
            try:
                return await service.get_cached_tools(server_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc))

        @api_router.post("/call", response_model=MCPToolCallResult)
        async def call_tool(
            request: CallToolRequest,
            _user_id: str = Depends(get_current_user),
        ):
            try:
                return await service.call_tool(
                    server_id=request.server_id,
                    tool_name=request.tool_name,
                    arguments=request.arguments,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc))

        return api_router
    except ImportError:
        logger.warning("FastAPI not installed, cannot create MCP router")
        return None
