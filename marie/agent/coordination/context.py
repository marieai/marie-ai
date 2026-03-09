"""Execution context for agent coordination.

Provides context hierarchy for distributed tracing and parent-child
agent execution tracking, inspired by BeeAI Framework's RunContext.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_current_context: ContextVar[Optional["AgentExecutionContext"]] = ContextVar(
    "agent_execution_context", default=None
)


@dataclass
class AgentExecutionContext:
    """Execution context for tracking agent runs within workflows.

    Provides:
    - Unique run_id per execution
    - group_id inherited from parent (for distributed tracing)
    - Parent-child relationships for nested agent calls
    - Arbitrary context data storage

    Usage:
        ```python
        # Create root context for workflow
        with AgentExecutionContext(workflow_id="wf-123") as ctx:
            # Execute agents with context tracking
            with ctx.child(agent_name="planner") as agent_ctx:
                # agent_ctx.parent_id == ctx.run_id
                # agent_ctx.group_id == ctx.group_id (inherited)
                result = await execute_agent(...)

        # Access current context anywhere
        current = AgentExecutionContext.current()
        if current:
            print(f"Running in workflow: {current.workflow_id}")
        ```
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    group_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    agent_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _token: Any = field(default=None, repr=False)

    def __enter__(self) -> "AgentExecutionContext":
        """Enter context and set as current."""
        self._token = _current_context.set(self)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore previous."""
        if self._token is not None:
            _current_context.reset(self._token)

    @classmethod
    def current(cls) -> Optional["AgentExecutionContext"]:
        """Get the current execution context, if any."""
        return _current_context.get()

    def child(
        self,
        agent_name: Optional[str] = None,
        **extra_context: Any,
    ) -> "AgentExecutionContext":
        """Create a child context inheriting group_id.

        Args:
            agent_name: Name of the agent being executed
            **extra_context: Additional context to merge

        Returns:
            New child context
        """
        child_context = {**self.context, **extra_context}
        return AgentExecutionContext(
            group_id=self.group_id,  # Inherited for tracing
            parent_id=self.run_id,
            workflow_id=self.workflow_id,
            agent_name=agent_name,
            context=child_context,
        )

    def set(self, key: str, value: Any) -> None:
        """Set a context value."""
        self.context[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self.context.get(key, default)

    def to_trace_headers(self) -> Dict[str, str]:
        """Get trace headers for propagation (e.g., to HTTP requests)."""
        return {
            "X-Run-ID": self.run_id,
            "X-Group-ID": self.group_id,
            "X-Parent-ID": self.parent_id or "",
            "X-Workflow-ID": self.workflow_id or "",
        }

    @classmethod
    def from_trace_headers(cls, headers: Dict[str, str]) -> "AgentExecutionContext":
        """Create context from trace headers."""
        return cls(
            run_id=headers.get("X-Run-ID", str(uuid.uuid4())),
            group_id=headers.get("X-Group-ID", str(uuid.uuid4())),
            parent_id=headers.get("X-Parent-ID") or None,
            workflow_id=headers.get("X-Workflow-ID") or None,
        )
