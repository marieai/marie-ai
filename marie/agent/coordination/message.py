"""Agent message types for coordination layer.

This module provides message schemas for inter-agent communication within
workflows. These are separate from A2A messages (external protocol) and
conversation Messages (LLM context).

Design Decision: Message-driven routing - the receiver field determines
the next agent to execute, following production patterns from AutoGen,
OpenAI Swarm, and A2A Protocol.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentMessageType(str, Enum):
    """Types of messages exchanged between agents."""

    TASK = "task"  # Task assignment to an agent
    RESULT = "result"  # Result from agent execution
    VALIDATION = "validation"  # Validation outcome
    ERROR = "error"  # Error report
    CONTROL = "control"  # Control signal (pause, resume, cancel)


class ReservedReceiver(str, Enum):
    """Special receiver values for routing control.

    These replace a separate StepSignal enum - routing is message-driven.
    Matches BeeAI Workflow routing signals for compatibility.
    """

    END = "__end__"  # Terminate workflow
    SELF = "__self__"  # Re-execute sender (retry)
    PREV = "__prev__"  # Go back to previous agent in step history
    START = "__start__"  # Go back to first agent in workflow
    NEXT = "__next__"  # Go to next agent in sequence (for SequentialRoutingPolicy)
    COORDINATOR = "__coord__"  # Return to coordinator for LLM routing decision
    BROADCAST = "*"  # Send to all agents (fan-out)


class AgentMessage(BaseModel):
    """Message for inter-agent communication within workflows.

    This is distinct from:
    - `Message` (marie.agent.message): LLM conversation context
    - A2A `Message` (marie.agent.a2a.types): External protocol

    AgentMessage is for internal coordination with explicit sender/receiver
    addressing and message typing.

    Example:
        ```python
        # Agent posts result to next agent
        msg = AgentMessage(
            sender="extractor",
            receiver="analyzer",
            msg_type=AgentMessageType.RESULT,
            content="Extracted 5 entities from document",
            metadata={"entity_count": 5},
        )
        state.post_message(msg)

        # Signal workflow completion
        msg = AgentMessage(
            sender="validator",
            receiver=ReservedReceiver.END,
            msg_type=AgentMessageType.VALIDATION,
            content="APPROVED",
        )

        # Request retry
        msg = AgentMessage(
            sender="validator",
            receiver="planner",  # Route back to planner
            msg_type=AgentMessageType.VALIDATION,
            content="REJECTED: needs more detail",
        )
        ```
    """

    msg_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique message identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message creation timestamp (UTC)",
    )
    sender: str = Field(
        ...,
        description="Name of the sending agent",
    )
    receiver: str = Field(
        ...,
        description="Target agent name or ReservedReceiver value",
    )
    msg_type: AgentMessageType = Field(
        default=AgentMessageType.RESULT,
        description="Message type classification",
    )
    content: str = Field(
        default="",
        description="Message content/payload",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata",
    )
    trace: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tracing information (run_id, group_id, latency_ms, etc.)",
    )

    def is_terminal(self) -> bool:
        """Check if this message signals workflow termination."""
        return self.receiver == ReservedReceiver.END or self.receiver == "__end__"

    def is_retry(self) -> bool:
        """Check if this message signals a retry of the sender."""
        return self.receiver == ReservedReceiver.SELF or self.receiver == "__self__"

    def is_broadcast(self) -> bool:
        """Check if this message should be broadcast to all agents."""
        return self.receiver == ReservedReceiver.BROADCAST or self.receiver == "*"

    def is_coordinator_decision(self) -> bool:
        """Check if routing should be delegated to coordinator (LLM)."""
        return (
            self.receiver == ReservedReceiver.COORDINATOR
            or self.receiver == "__coord__"
        )

    def is_prev(self) -> bool:
        """Check if this message signals going back to previous agent."""
        return self.receiver == ReservedReceiver.PREV or self.receiver == "__prev__"

    def is_start(self) -> bool:
        """Check if this message signals going back to first agent."""
        return self.receiver == ReservedReceiver.START or self.receiver == "__start__"

    def is_next(self) -> bool:
        """Check if this message signals going to next agent in sequence."""
        return self.receiver == ReservedReceiver.NEXT or self.receiver == "__next__"

    def with_trace(self, **trace_fields: Any) -> "AgentMessage":
        """Return a copy with additional trace fields."""
        new_trace = {**self.trace, **trace_fields}
        return self.model_copy(update={"trace": new_trace})


def create_task_message(
    sender: str,
    receiver: str,
    content: str,
    **metadata: Any,
) -> AgentMessage:
    """Helper to create a task message."""
    return AgentMessage(
        sender=sender,
        receiver=receiver,
        msg_type=AgentMessageType.TASK,
        content=content,
        metadata=metadata,
    )


def create_result_message(
    sender: str,
    receiver: str,
    content: str,
    **metadata: Any,
) -> AgentMessage:
    """Helper to create a result message."""
    return AgentMessage(
        sender=sender,
        receiver=receiver,
        msg_type=AgentMessageType.RESULT,
        content=content,
        metadata=metadata,
    )


def create_error_message(
    sender: str,
    error: str,
    receiver: str = ReservedReceiver.COORDINATOR,
    **metadata: Any,
) -> AgentMessage:
    """Helper to create an error message."""
    return AgentMessage(
        sender=sender,
        receiver=receiver,
        msg_type=AgentMessageType.ERROR,
        content=error,
        metadata=metadata,
    )
