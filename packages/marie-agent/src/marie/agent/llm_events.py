"""LLM/Chat model event definitions.

Provides event dataclasses for LLM call lifecycle and streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from marie.agent.message import Message


@dataclass
class ChatModelStartEvent:
    """Emitted when a chat model call starts."""

    model_name: Optional[str]
    messages: List[Message]
    has_functions: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "llm.start",
            "model_name": self.model_name,
            "message_count": len(self.messages),
            "has_functions": self.has_functions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ChatModelNewTokenEvent:
    """Emitted for each new token during streaming."""

    token: str
    accumulated_content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "llm.new_token",
            "token": self.token,
            "accumulated_length": len(self.accumulated_content),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ChatModelSuccessEvent:
    """Emitted when a chat model call completes successfully."""

    model_name: Optional[str]
    response: Message
    duration_ms: float
    has_tool_calls: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "llm.success",
            "model_name": self.model_name,
            "has_tool_calls": self.has_tool_calls,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ChatModelErrorEvent:
    """Emitted when a chat model call fails."""

    model_name: Optional[str]
    error: Exception
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "llm.error",
            "model_name": self.model_name,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ChatModelFinishEvent:
    """Emitted when a chat model call finishes (success or error)."""

    model_name: Optional[str]
    success: bool
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "llm.finish",
            "model_name": self.model_name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }
