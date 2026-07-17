"""Audit middleware bridging emitter events to AuditLogger.

Translates agent/tool/llm events to the existing AuditLogger protocol
for compliance and observability.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

from marie.agent.coordination.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    InMemoryAuditLogger,
)
from marie.agent.middleware.protocol import BaseMiddleware

if TYPE_CHECKING:
    from marie.agent.emitter import Emitter


class AuditMiddleware(BaseMiddleware):
    """Middleware that logs events to an AuditLogger.

    Bridges the emitter event system to the existing AuditLogger protocol
    for compliance logging and audit trails.
    """

    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
        workflow_id: Optional[str] = None,
    ) -> None:
        """Initialize audit middleware.

        Args:
            audit_logger: AuditLogger implementation to use
            workflow_id: Workflow ID for audit events
        """
        super().__init__(name="AuditMiddleware", priority=50)
        self._audit_logger = audit_logger or InMemoryAuditLogger()
        self._workflow_id = workflow_id or "default"

    def bind(self, emitter: "Emitter") -> None:
        """Bind audit logging to emitter events."""
        # Use group_id as workflow_id if available
        if emitter.group_id:
            self._workflow_id = emitter.group_id

        # Agent events
        self._listener_ids.append(
            emitter.on("agent.start", self._on_agent_start, priority=self.priority)
        )
        self._listener_ids.append(
            emitter.on("agent.success", self._on_agent_success, priority=self.priority)
        )
        self._listener_ids.append(
            emitter.on("agent.error", self._on_agent_error, priority=self.priority)
        )

    async def _log_event(
        self,
        event_type: AuditEventType,
        data: Dict[str, Any],
        agent_name: Optional[str] = None,
    ) -> None:
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            workflow_id=self._workflow_id,
            agent_name=agent_name,
            details=data,
        )
        await self._audit_logger.log(event)

    def _on_agent_start(self, data: Dict[str, Any]) -> None:
        """Handle agent start event."""
        asyncio.create_task(
            self._log_event(
                AuditEventType.AGENT_STARTED,
                data,
                agent_name=data.get("agent_name"),
            )
        )

    def _on_agent_success(self, data: Dict[str, Any]) -> None:
        """Handle agent success event."""
        asyncio.create_task(
            self._log_event(
                AuditEventType.AGENT_COMPLETED,
                data,
                agent_name=data.get("agent_name"),
            )
        )

    def _on_agent_error(self, data: Dict[str, Any]) -> None:
        """Handle agent error event."""
        asyncio.create_task(
            self._log_event(
                AuditEventType.AGENT_FAILED,
                data,
                agent_name=data.get("agent_name"),
            )
        )
