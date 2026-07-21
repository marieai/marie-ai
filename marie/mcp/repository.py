from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from marie.mcp.models import (
    MCPAuthType,
    MCPServer,
    MCPServerCreate,
    MCPServerStatus,
    MCPServerUpdate,
    MCPTool,
    MCPTransport,
)
from marie.storage.database.postgres_pool import AsyncPostgresPool


class MCPServerRepository:
    def __init__(
        self,
        pool: Optional[AsyncPostgresPool] = None,
        schema: str = "marie_scheduler",
        db_config: Optional[dict[str, Any]] = None,
    ):
        self.pool = pool or AsyncPostgresPool.get_instance()
        self.schema = schema
        self.db_config = db_config or _default_db_config()

    async def initialize(self) -> None:
        await self.pool.initialize(self.db_config)

    async def list_servers(self, workspace_id: str) -> list[MCPServer]:
        await self.initialize()
        rows = await self.pool.fetch(
            f"""
            SELECT *
            FROM {self.schema}.mcp_server_registrations
            WHERE workspace_id = $1
            ORDER BY name ASC
            """,
            workspace_id,
        )
        return [self._row_to_server(row) for row in rows]

    async def get_server(self, server_id: str) -> MCPServer | None:
        await self.initialize()
        row = await self.pool.fetchrow(
            f"""
            SELECT *
            FROM {self.schema}.mcp_server_registrations
            WHERE id = $1
            """,
            UUID(server_id),
        )
        return self._row_to_server(row) if row else None

    async def create_server(
        self,
        payload: MCPServerCreate,
        created_by_id: str | None = None,
    ) -> MCPServer:
        await self.initialize()
        row = await self.pool.fetchrow(
            f"""
            INSERT INTO {self.schema}.mcp_server_registrations (
                workspace_id,
                name,
                url,
                transport,
                auth_type,
                headers,
                tags,
                is_enabled,
                created_by_id,
                updated_by_id
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::text[], $8, $9, $10)
            RETURNING *
            """,
            payload.workspace_id,
            payload.name,
            str(payload.url),
            payload.transport.value,
            payload.auth_type.value,
            json.dumps(payload.headers),
            payload.tags,
            payload.is_enabled,
            created_by_id,
            created_by_id,
        )
        return self._row_to_server(row)

    async def update_server(
        self,
        server_id: str,
        payload: MCPServerUpdate,
        updated_by_id: str | None = None,
    ) -> MCPServer:
        await self.initialize()
        current = await self.get_server(server_id)
        if current is None:
            raise ValueError(f"MCP server {server_id} not found")

        next_name = payload.name if payload.name is not None else current.name
        next_url = str(payload.url) if payload.url is not None else current.url
        next_transport = (
            payload.transport.value
            if payload.transport is not None
            else current.transport.value
        )
        next_auth_type = (
            payload.auth_type.value
            if payload.auth_type is not None
            else current.auth_type.value
        )
        next_headers = (
            payload.headers if payload.headers is not None else current.headers
        )
        next_tags = payload.tags if payload.tags is not None else current.tags
        next_enabled = (
            payload.is_enabled if payload.is_enabled is not None else current.is_enabled
        )

        row = await self.pool.fetchrow(
            f"""
            UPDATE {self.schema}.mcp_server_registrations
            SET
                name = $2,
                url = $3,
                transport = $4,
                auth_type = $5,
                headers = $6::jsonb,
                tags = $7::text[],
                is_enabled = $8,
                updated_by_id = $9,
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            """,
            UUID(server_id),
            next_name,
            next_url,
            next_transport,
            next_auth_type,
            json.dumps(next_headers),
            next_tags,
            next_enabled,
            updated_by_id,
        )
        return self._row_to_server(row)

    async def delete_server(self, server_id: str) -> bool:
        await self.initialize()
        result = await self.pool.execute(
            f"DELETE FROM {self.schema}.mcp_server_registrations WHERE id = $1",
            UUID(server_id),
        )
        return result == "DELETE 1"

    async def update_status(
        self,
        server_id: str,
        status: MCPServerStatus,
        error: str | None = None,
    ) -> None:
        await self.initialize()
        await self.pool.execute(
            f"""
            UPDATE {self.schema}.mcp_server_registrations
            SET
                status = $2,
                last_error = $3,
                last_tested_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            """,
            UUID(server_id),
            status.value,
            error,
        )

    async def update_discovered_tools(
        self,
        server_id: str,
        tools: list[MCPTool],
    ) -> None:
        await self.initialize()
        payload = [tool.model_dump(mode="json") for tool in tools]
        await self.pool.execute(
            f"""
            UPDATE {self.schema}.mcp_server_registrations
            SET
                discovered_tools = $2::jsonb,
                tool_count = $3,
                last_discovery_at = NOW(),
                status = $4,
                last_error = NULL,
                updated_at = NOW()
            WHERE id = $1
            """,
            UUID(server_id),
            json.dumps(payload),
            len(tools),
            MCPServerStatus.ACTIVE.value,
        )

    def _row_to_server(self, row: Any) -> MCPServer:
        discovered = row["discovered_tools"] or []
        return MCPServer(
            id=str(row["id"]),
            workspace_id=row["workspace_id"],
            name=row["name"],
            url=row["url"],
            transport=MCPTransport(row["transport"]),
            auth_type=MCPAuthType(row["auth_type"]),
            headers=row["headers"] or {},
            status=_coerce_status(row["status"]),
            last_tested_at=_to_datetime(row["last_tested_at"]),
            last_error=row["last_error"],
            tool_count=row["tool_count"] or 0,
            discovered_tools=[MCPTool.model_validate(tool) for tool in discovered],
            last_discovery_at=_to_datetime(row["last_discovery_at"]),
            is_enabled=row["is_enabled"],
            tags=list(row["tags"] or []),
            created_by_id=row["created_by_id"],
            updated_by_id=row["updated_by_id"],
            created_at=_to_datetime(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=_to_datetime(row["updated_at"]) or datetime.now(timezone.utc),
        )


def _default_db_config() -> dict[str, Any]:
    return {
        "hostname": os.getenv(
            "POSTGRES_HOSTNAME", os.getenv("MARIE_STUDIO_DB_HOST", "localhost")
        ),
        "port": int(
            os.getenv("POSTGRES_PORT", os.getenv("MARIE_STUDIO_DB_PORT", "5432"))
        ),
        "username": os.getenv(
            "POSTGRES_USER", os.getenv("MARIE_STUDIO_DB_USER", "postgres")
        ),
        "password": os.getenv(
            "POSTGRES_PASSWORD", os.getenv("MARIE_STUDIO_DB_PASSWORD", "")
        ),
        "database": os.getenv(
            "POSTGRES_DB", os.getenv("MARIE_STUDIO_DB_NAME", "postgres")
        ),
        "min_connections": int(os.getenv("MARIE_MCP_DB_MIN_CONNECTIONS", "1")),
        "max_connections": int(os.getenv("MARIE_MCP_DB_MAX_CONNECTIONS", "10")),
    }


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _coerce_status(value: Any) -> MCPServerStatus:
    normalized = str(value)
    if normalized == "pending_validation":
        return MCPServerStatus.PENDING
    return MCPServerStatus(normalized)
