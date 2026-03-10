"""Tool-level event definitions.

Provides event dataclasses for tool execution lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class ToolStartEvent:
    """Emitted when a tool starts execution."""

    tool_name: str
    arguments: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tool.start",
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToolSuccessEvent:
    """Emitted when a tool completes successfully."""

    tool_name: str
    result: str
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tool.success",
            "tool_name": self.tool_name,
            "result_length": len(self.result),
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToolErrorEvent:
    """Emitted when a tool encounters an error."""

    tool_name: str
    error: str
    arguments: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tool.error",
            "tool_name": self.tool_name,
            "error": self.error,
            "arguments": self.arguments,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToolFinishEvent:
    """Emitted when a tool finishes (success or error)."""

    tool_name: str
    success: bool
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tool.finish",
            "tool_name": self.tool_name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }
