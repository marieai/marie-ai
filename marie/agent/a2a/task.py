"""Task state machine and status mapping for A2A protocol.

This module provides utilities for managing A2A task lifecycle,
including state transitions and mapping between Marie job states
and A2A task states.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from marie.agent.a2a.constants import TERMINAL_STATES, TaskState
from marie.agent.a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskStatus,
    TextPart,
)

logger = logging.getLogger(__name__)


class TaskStateMapper:
    """Maps between Marie job states and A2A task states.

    Provides bidirectional mapping for interoperability between
    Marie's internal job management and the A2A protocol.
    """

    # Marie JobStatus -> A2A TaskState
    _MARIE_TO_A2A: dict[str, TaskState] = {
        "pending": TaskState.SUBMITTED,
        "running": TaskState.WORKING,
        "completed": TaskState.COMPLETED,
        "failed": TaskState.FAILED,
        "cancelled": TaskState.CANCELED,
        "queued": TaskState.SUBMITTED,
        "processing": TaskState.WORKING,
        "success": TaskState.COMPLETED,
        "error": TaskState.FAILED,
    }

    # A2A TaskState -> Marie JobStatus
    _A2A_TO_MARIE: dict[TaskState, str] = {
        TaskState.SUBMITTED: "pending",
        TaskState.WORKING: "running",
        TaskState.INPUT_REQUIRED: "waiting",
        TaskState.COMPLETED: "completed",
        TaskState.CANCELED: "cancelled",
        TaskState.FAILED: "failed",
        TaskState.REJECTED: "failed",
        TaskState.AUTH_REQUIRED: "waiting",
        TaskState.UNKNOWN: "unknown",
    }

    @classmethod
    def to_a2a(cls, marie_status: str) -> TaskState:
        """Convert Marie job status to A2A task state.

        Args:
            marie_status: Marie job status string.

        Returns:
            Corresponding A2A TaskState.
        """
        normalized = marie_status.lower().strip()
        return cls._MARIE_TO_A2A.get(normalized, TaskState.UNKNOWN)

    @classmethod
    def to_marie(cls, a2a_state: TaskState) -> str:
        """Convert A2A task state to Marie job status.

        Args:
            a2a_state: A2A task state.

        Returns:
            Corresponding Marie job status string.
        """
        return cls._A2A_TO_MARIE.get(a2a_state, "unknown")

    @classmethod
    def is_terminal(cls, state: TaskState) -> bool:
        """Check if a task state is terminal (no further transitions)."""
        return state in TERMINAL_STATES


class TaskManager:
    """Manages A2A task lifecycle and state transitions.

    Provides methods for creating, updating, and querying tasks
    with proper state machine enforcement.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def create_task(
        self,
        message: Optional[Message] = None,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Task:
        """Create a new task.

        Args:
            message: Initial user message.
            context_id: Context ID (generated if not provided).
            task_id: Task ID (generated if not provided).

        Returns:
            The created Task.
        """
        task = Task(
            id=task_id or str(uuid4()),
            context_id=context_id or str(uuid4()),
            status=TaskStatus(
                state=TaskState.SUBMITTED,
                timestamp=datetime.utcnow(),
            ),
            history=[message] if message else [],
            artifacts=[],
        )
        self._tasks[task.id] = task
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def update_status(
        self,
        task_id: str,
        state: TaskState,
        message: Optional[Message] = None,
    ) -> Optional[Task]:
        """Update task status.

        Args:
            task_id: Task ID.
            state: New task state.
            message: Optional status message.

        Returns:
            Updated task or None if not found.
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        if TaskStateMapper.is_terminal(task.status.state):
            logger.warning(
                f"Attempted to update terminal task {task_id} "
                f"from {task.status.state} to {state}"
            )
            return task

        task.status = TaskStatus(
            state=state,
            message=message,
            timestamp=datetime.utcnow(),
        )

        if message:
            if task.history is None:
                task.history = []
            task.history.append(message)

        return task

    def add_message(self, task_id: str, message: Message) -> Optional[Task]:
        """Add a message to task history.

        Args:
            task_id: Task ID.
            message: Message to add.

        Returns:
            Updated task or None if not found.
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        if task.history is None:
            task.history = []
        task.history.append(message)
        return task

    def add_artifact(
        self,
        task_id: str,
        parts: list[Part],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[Task]:
        """Add an artifact to a task.

        Args:
            task_id: Task ID.
            parts: Artifact content parts.
            name: Optional artifact name.
            description: Optional artifact description.

        Returns:
            Updated task or None if not found.
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        artifact = Artifact(
            artifact_id=str(uuid4()),
            parts=parts,
            name=name,
            description=description,
        )

        if task.artifacts is None:
            task.artifacts = []
        task.artifacts.append(artifact)

        return task

    def complete_task(
        self,
        task_id: str,
        result: Optional[str] = None,
    ) -> Optional[Task]:
        """Mark a task as completed.

        Args:
            task_id: Task ID.
            result: Optional result text.

        Returns:
            Updated task or None if not found.
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        task.status = TaskStatus(
            state=TaskState.COMPLETED,
            timestamp=datetime.utcnow(),
        )

        if result:
            self.add_artifact(
                task_id,
                parts=[TextPart(text=result)],
                name="result",
            )

        return task

    def fail_task(
        self,
        task_id: str,
        error: str,
    ) -> Optional[Task]:
        """Mark a task as failed.

        Args:
            task_id: Task ID.
            error: Error message.

        Returns:
            Updated task or None if not found.
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        error_message = Message(
            role=Role.AGENT,
            parts=[TextPart(text=f"Error: {error}")],
        )

        task.status = TaskStatus(
            state=TaskState.FAILED,
            message=error_message,
            timestamp=datetime.utcnow(),
        )

        return task

    def cancel_task(self, task_id: str) -> Optional[Task]:
        """Cancel a task.

        Args:
            task_id: Task ID.

        Returns:
            Updated task or None if not found.
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        if TaskStateMapper.is_terminal(task.status.state):
            logger.warning(f"Cannot cancel terminal task {task_id}")
            return task

        task.status = TaskStatus(
            state=TaskState.CANCELED,
            timestamp=datetime.utcnow(),
        )

        return task

    def delete_task(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Task ID.

        Returns:
            True if deleted, False if not found.
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False


def create_task_from_message(message: Message) -> Task:
    """Create a new task from an incoming message.

    Convenience function for creating tasks with proper defaults.

    Args:
        message: The incoming user message.

    Returns:
        A new Task in SUBMITTED state.
    """
    return Task(
        id=str(uuid4()),
        context_id=message.context_id or str(uuid4()),
        status=TaskStatus(
            state=TaskState.SUBMITTED,
            timestamp=datetime.utcnow(),
        ),
        history=[message],
        artifacts=[],
    )


def create_text_message(
    text: str,
    role: Role = Role.AGENT,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> Message:
    """Create a text message.

    Args:
        text: Message text content.
        role: Message role (user or agent).
        task_id: Optional task ID.
        context_id: Optional context ID.

    Returns:
        A new Message with text content.
    """
    return Message(
        role=role,
        parts=[TextPart(text=text)],
        message_id=str(uuid4()),
        task_id=task_id,
        context_id=context_id,
    )
