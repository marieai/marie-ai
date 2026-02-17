"""Pydantic models for A2A protocol types.

This module provides simplified Pydantic models for A2A protocol data
structures. For full SDK compatibility, use the a2a.types module directly.
These models provide a Marie-native interface with camelCase serialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(s: str) -> str:
    """Convert snake_case to camelCase."""
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


class A2ABaseModel(BaseModel):
    """Base model with camelCase serialization."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=_to_camel,
        serialize_by_alias=True,
    )


class Role(str, Enum):
    """Message role."""

    USER = "user"
    AGENT = "agent"


class TaskState(str, Enum):
    """Task lifecycle state."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
    UNKNOWN = "unknown"


class TextPart(A2ABaseModel):
    """Text content part."""

    text: str
    kind: Literal["text"] = "text"
    metadata: Optional[dict[str, Any]] = None


class FilePart(A2ABaseModel):
    """File content part."""

    file: FileContent
    kind: Literal["file"] = "file"
    metadata: Optional[dict[str, Any]] = None


class FileContent(A2ABaseModel):
    """File content (bytes or URI)."""

    name: Optional[str] = None
    mime_type: Optional[str] = None
    bytes_: Optional[str] = Field(None, alias="bytes")
    uri: Optional[str] = None


class DataPart(A2ABaseModel):
    """Structured data part."""

    data: dict[str, Any]
    kind: Literal["data"] = "data"
    metadata: Optional[dict[str, Any]] = None


# Part is a discriminated union
Part = Union[TextPart, FilePart, DataPart]


class Message(A2ABaseModel):
    """A2A message."""

    role: Role
    parts: list[Part]
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: Optional[str] = None
    context_id: Optional[str] = None
    reference_task_ids: Optional[list[str]] = None
    extensions: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class TaskStatus(A2ABaseModel):
    """Task status information."""

    state: TaskState
    message: Optional[Message] = None
    timestamp: Optional[datetime] = None


class Artifact(A2ABaseModel):
    """Task output artifact."""

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    parts: list[Part]
    name: Optional[str] = None
    description: Optional[str] = None
    extensions: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class Task(A2ABaseModel):
    """A2A task."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    context_id: str = Field(default_factory=lambda: str(uuid4()))
    status: TaskStatus
    history: Optional[list[Message]] = None
    artifacts: Optional[list[Artifact]] = None
    kind: Literal["task"] = "task"
    metadata: Optional[dict[str, Any]] = None


class AgentSkill(A2ABaseModel):
    """Agent capability/skill."""

    id: str
    name: str
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    examples: Optional[list[str]] = None
    input_modes: Optional[list[str]] = None
    output_modes: Optional[list[str]] = None


class AgentCapabilities(A2ABaseModel):
    """Agent capabilities."""

    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False
    extensions: Optional[list[str]] = None


class AgentProvider(A2ABaseModel):
    """Agent provider information."""

    organization: str
    url: Optional[str] = None


class AgentCard(A2ABaseModel):
    """A2A agent card (self-description)."""

    name: str
    description: Optional[str] = None
    url: str
    version: str = "1.0.0"
    preferred_transport: Optional[str] = None
    additional_interfaces: Optional[list[dict[str, Any]]] = None
    skills: Optional[list[AgentSkill]] = None
    capabilities: Optional[AgentCapabilities] = None
    default_input_modes: Optional[list[str]] = None
    default_output_modes: Optional[list[str]] = None
    provider: Optional[AgentProvider] = None
    documentation_url: Optional[str] = None
    icon_url: Optional[str] = None
    security_schemes: Optional[dict[str, Any]] = None
    security: Optional[list[dict[str, list[str]]]] = None


class MessageSendConfiguration(A2ABaseModel):
    """Configuration for message send requests."""

    blocking: Optional[bool] = None
    history_length: Optional[int] = None
    accepted_output_modes: Optional[list[str]] = None
    push_notification_config: Optional[PushNotificationConfig] = None
    extensions: Optional[list[str]] = None


class PushNotificationConfig(A2ABaseModel):
    """Push notification configuration."""

    url: str
    id: Optional[str] = None
    token: Optional[str] = None
    authentication: Optional[dict[str, Any]] = None


class MessageSendParams(A2ABaseModel):
    """Parameters for message/send request."""

    message: Message
    configuration: Optional[MessageSendConfiguration] = None
    metadata: Optional[dict[str, Any]] = None


class TaskQueryParams(A2ABaseModel):
    """Parameters for tasks/get request."""

    id: str
    history_length: Optional[int] = None


class TaskIdParams(A2ABaseModel):
    """Parameters for task operations requiring only ID."""

    id: str


class JSONRPCRequest(A2ABaseModel):
    """JSON-RPC 2.0 request."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[dict[str, Any]] = None


class JSONRPCSuccessResponse(A2ABaseModel):
    """JSON-RPC 2.0 success response."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    result: Any


class JSONRPCError(A2ABaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCErrorResponse(A2ABaseModel):
    """JSON-RPC 2.0 error response."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    error: JSONRPCError


class TaskStatusUpdateEvent(A2ABaseModel):
    """SSE event for task status update."""

    status: TaskStatus
    task_id: str
    context_id: str
    final: bool = False
    kind: Literal["status-update"] = "status-update"


class TaskArtifactUpdateEvent(A2ABaseModel):
    """SSE event for task artifact update."""

    artifact: Artifact
    task_id: str
    context_id: str
    append: bool = False
    last_chunk: bool = False
    kind: Literal["artifact-update"] = "artifact-update"


# Update forward references
FilePart.model_rebuild()
