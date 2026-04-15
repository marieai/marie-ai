from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class MCPBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)


class MCPTransport(str, Enum):
    STREAMABLE_HTTP = "streamable_http"


class MCPAuthType(str, Enum):
    NONE = "none"
    STATIC_HEADERS = "static_headers"


class MCPServerStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


class MCPServerCreate(MCPBaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    url: HttpUrl
    workspace_id: str = Field(..., min_length=1, max_length=255)
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP
    auth_type: MCPAuthType = MCPAuthType.NONE
    headers: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_enabled: bool = True


class MCPServerTestRequest(MCPBaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    url: HttpUrl
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP
    auth_type: MCPAuthType = MCPAuthType.NONE
    headers: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_enabled: bool = True


class MCPServerUpdate(MCPBaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    url: Optional[HttpUrl] = None
    transport: Optional[MCPTransport] = None
    auth_type: Optional[MCPAuthType] = None
    headers: Optional[Dict[str, str]] = None
    tags: Optional[List[str]] = None
    is_enabled: Optional[bool] = None


class MCPServer(MCPBaseModel):
    id: str
    workspace_id: str
    name: str
    url: str
    transport: MCPTransport
    auth_type: MCPAuthType
    headers: Dict[str, str] = Field(default_factory=dict)
    status: MCPServerStatus = MCPServerStatus.PENDING
    last_tested_at: Optional[datetime] = None
    last_error: Optional[str] = None
    tool_count: int = 0
    discovered_tools: List["MCPTool"] = Field(default_factory=list)
    last_discovery_at: Optional[datetime] = None
    is_enabled: bool = True
    tags: List[str] = Field(default_factory=list)
    created_by_id: Optional[str] = None
    updated_by_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class MCPTool(MCPBaseModel):
    name: str
    slug: str
    server_id: str
    server_name: str
    title: Optional[str] = None
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    annotations: Dict[str, Any] = Field(default_factory=dict)


class MCPToolTestResult(MCPBaseModel):
    success: bool
    server_id: str
    tool_count: int = 0
    message: Optional[str] = None
    error: Optional[str] = None


class MCPToolDiscoveryResult(MCPBaseModel):
    server_id: str
    tools: List[MCPTool] = Field(default_factory=list)


class MCPToolCallResult(MCPBaseModel):
    server_id: str
    tool_name: str
    result: Dict[str, Any] = Field(default_factory=dict)


class CallToolRequest(MCPBaseModel):
    server_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
