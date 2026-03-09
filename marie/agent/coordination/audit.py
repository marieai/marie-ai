"""Audit logging for agent communications."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.coordination.audit")


class AuditEventType(str, Enum):
    """Types of audit events."""

    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    ROUTING_DECISION = "routing_decision"


@dataclass
class AuditEvent:
    """A single audit event."""

    event_type: AuditEventType
    workflow_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: Optional[str] = None
    message_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "message_id": self.message_id,
            "details": self.details,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }


@runtime_checkable
class AuditLogger(Protocol):
    """Protocol for audit logging implementations."""

    async def log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        ...

    async def query(
        self,
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query audit events."""
        ...


class InMemoryAuditLogger:
    """In-memory audit logger for testing."""

    def __init__(self, max_events: int = 10000):
        self._events: List[AuditEvent] = []
        self._max_events = max_events

    async def log(self, event: AuditEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]
        logger.debug(f"Audit: {event.event_type.value} workflow={event.workflow_id}")

    async def query(
        self,
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        results = self._events

        if workflow_id:
            results = [e for e in results if e.workflow_id == workflow_id]
        if agent_name:
            results = [e for e in results if e.agent_name == agent_name]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        return results[-limit:]

    def clear(self) -> None:
        self._events.clear()


class StructuredAuditLogger:
    """Structured JSON audit logger that writes to MarieLogger."""

    def __init__(self, logger_name: str = "marie.agent.audit"):
        self._logger = MarieLogger(logger_name)
        self._in_memory = InMemoryAuditLogger()

    async def log(self, event: AuditEvent) -> None:
        await self._in_memory.log(event)
        self._logger.info(f"AUDIT: {json.dumps(event.to_dict(), default=str)}")

    async def query(
        self,
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        return await self._in_memory.query(
            workflow_id=workflow_id,
            agent_name=agent_name,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )


def create_audit_logger(enabled: bool = True) -> AuditLogger:
    """Factory to create appropriate audit logger."""
    if enabled:
        return StructuredAuditLogger()
    return InMemoryAuditLogger(max_events=0)
