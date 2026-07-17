"""Agent-level event definitions.

Provides event dataclasses for agent lifecycle events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from marie.agent.message import Message


@dataclass
class AgentStartEvent:
    """Emitted when an agent starts execution."""

    agent_name: Optional[str]
    messages: List[Message]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "agent.start",
            "agent_name": self.agent_name,
            "message_count": len(self.messages),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentSuccessEvent:
    """Emitted when an agent completes successfully."""

    agent_name: Optional[str]
    result: List[Message]
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "agent.success",
            "agent_name": self.agent_name,
            "result_count": len(self.result),
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentErrorEvent:
    """Emitted when an agent encounters an error."""

    agent_name: Optional[str]
    error: Exception
    messages: List[Message]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "agent.error",
            "agent_name": self.agent_name,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentFinishEvent:
    """Emitted when an agent finishes (success or error)."""

    agent_name: Optional[str]
    success: bool
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "agent.finish",
            "agent_name": self.agent_name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }
