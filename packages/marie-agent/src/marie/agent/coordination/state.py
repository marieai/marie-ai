"""Workflow state management for agent coordination.

This module provides the AgentWorkflowState class which maintains
the state of a multi-agent workflow including message mailbox,
execution history, and routing logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.agent.coordination.message import (
    AgentMessage,
    AgentMessageType,
    ReservedReceiver,
)


class AgentWorkflowStatus(str, Enum):
    """Status of a workflow execution."""

    PENDING = "pending"  # Not yet started
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed with error
    PAUSED = "paused"  # Temporarily paused
    CANCELLED = "cancelled"  # Cancelled by user/system


class AgentWorkflowState(BaseModel):
    """State container for multi-agent workflow execution.

    Implements message-driven routing where the receiver field in
    AgentMessage determines the next agent to execute.

    Key features:
    - Mailbox pattern for agent communication
    - Message threading (agent output → next agent input)
    - Communication graph tracking for visualization
    - Step history for routing (SELF, PREV support)

    Example:
        ```python
        state = AgentWorkflowState(
            workflow_id="wf-123",
            goal="Process and validate document",
        )

        # Post message from planner
        msg = AgentMessage(
            sender="planner",
            receiver="executor",
            content="Execute extraction task",
        )
        state.post_message(msg)

        # Get next agent to run
        next_agent = state.next_agent()  # Returns "executor"

        # Build messages for next agent
        messages = state.build_messages_for_agent("executor", base_messages)
        ```
    """

    workflow_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique workflow identifier",
    )
    goal: str = Field(
        default="",
        description="Workflow objective/goal description",
    )
    status: AgentWorkflowStatus = Field(
        default=AgentWorkflowStatus.PENDING,
        description="Current workflow status",
    )
    mailbox: List[AgentMessage] = Field(
        default_factory=list,
        description="Ordered list of agent messages (message bus)",
    )
    communication_edges: List[tuple] = Field(
        default_factory=list,
        description="Communication graph edges: (sender, receiver, msg_type)",
    )
    active_agent: Optional[str] = Field(
        default=None,
        description="Currently executing agent name",
    )
    step: int = Field(
        default=0,
        description="Current step counter",
    )
    step_history: List[str] = Field(
        default_factory=list,
        description="History of executed agent names (for PREV/SELF routing)",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Accumulated error messages",
    )
    shared_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary shared context between agents",
    )
    accumulated_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Accumulated LLM messages for threading (agent output → next input)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Workflow creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )

    def post_message(self, msg: AgentMessage) -> None:
        """Post a message to the mailbox.

        Updates communication graph, step counter, and timestamp.

        Args:
            msg: Message to post
        """
        self.mailbox.append(msg)
        self.communication_edges.append(
            (
                msg.sender,
                msg.receiver,
                (
                    msg.msg_type.value
                    if isinstance(msg.msg_type, AgentMessageType)
                    else msg.msg_type
                ),
            )
        )
        self.step += 1
        self.updated_at = datetime.now(timezone.utc)

    def next_agent(self) -> Optional[str]:
        """Determine the next agent based on the last message's receiver.

        Implements message-driven routing (BeeAI-compatible):
        - Direct agent name → return that agent
        - __end__ → return None (workflow complete)
        - __self__ → return last sender (retry)
        - __prev__ → return previous agent from step_history
        - __start__ → return first agent from step_history
        - __next__ → return None (let SequentialRoutingPolicy decide)
        - __coord__ → return None (let coordinator decide via LLM)
        - * → return "__broadcast__" (signal for fan-out)

        Returns:
            Agent name to execute, or None if workflow should end/pause
        """
        if not self.mailbox:
            return None

        last_msg = self.mailbox[-1]
        receiver = last_msg.receiver

        # Handle reserved receivers
        if receiver == ReservedReceiver.END or receiver == "__end__":
            self.status = AgentWorkflowStatus.COMPLETED
            return None
        elif receiver == ReservedReceiver.SELF or receiver == "__self__":
            return last_msg.sender  # Retry same agent
        elif receiver == ReservedReceiver.PREV or receiver == "__prev__":
            # Go back to previous agent in step history
            return self.previous_agent
        elif receiver == ReservedReceiver.START or receiver == "__start__":
            # Go back to first agent in step history
            if self.step_history:
                return self.step_history[0]
            return None
        elif receiver == ReservedReceiver.NEXT or receiver == "__next__":
            # Let routing policy decide next in sequence
            return None
        elif receiver == ReservedReceiver.COORDINATOR or receiver == "__coord__":
            return None  # Let coordinator decide via LLM
        elif receiver == ReservedReceiver.BROADCAST or receiver == "*":
            return "__broadcast__"  # Signal for fan-out
        else:
            return receiver  # Direct agent name

    @property
    def previous_agent(self) -> Optional[str]:
        """Get the previously executed agent name."""
        if len(self.step_history) >= 2:
            return self.step_history[-2]
        return None

    @property
    def last_message(self) -> Optional[AgentMessage]:
        """Get the last message in the mailbox."""
        return self.mailbox[-1] if self.mailbox else None

    def get_messages_for(
        self,
        agent: str,
        msg_type: Optional[AgentMessageType] = None,
    ) -> List[AgentMessage]:
        """Get messages addressed to a specific agent.

        Args:
            agent: Agent name
            msg_type: Optional filter by message type

        Returns:
            List of matching messages
        """
        return [
            m
            for m in self.mailbox
            if (m.receiver == agent or m.receiver == "*")
            and (msg_type is None or m.msg_type == msg_type)
        ]

    def thread_agent_output(self, agent_name: str, output: str) -> None:
        """Add agent output to accumulated messages for threading.

        This enables the BeeAI pattern where agent output becomes
        the next agent's input context.

        Args:
            agent_name: Name of the agent that produced the output
            output: Output content to thread
        """
        self.accumulated_messages.append(
            {
                "role": "assistant",
                "content": output,
                "name": agent_name,
            }
        )
        self.updated_at = datetime.now(timezone.utc)

    def build_messages_for_agent(
        self,
        agent_name: str,
        base_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build message list for an agent including accumulated context.

        Combines base messages with accumulated outputs from previous
        agents in the workflow.

        Args:
            agent_name: Name of the agent to build messages for
            base_messages: Base messages (e.g., user input)

        Returns:
            Combined message list for agent execution
        """
        # Start with base messages
        messages = list(base_messages)

        # Add accumulated messages from workflow
        messages.extend(self.accumulated_messages)

        return messages

    def record_agent_start(self, agent_name: str) -> None:
        """Record that an agent has started executing.

        Args:
            agent_name: Name of the agent starting
        """
        self.active_agent = agent_name
        self.step_history.append(agent_name)
        self.status = AgentWorkflowStatus.RUNNING
        self.updated_at = datetime.now(timezone.utc)

    def record_agent_complete(self, agent_name: str, output: str) -> None:
        """Record that an agent has completed.

        Args:
            agent_name: Name of the completed agent
            output: Agent's output
        """
        self.thread_agent_output(agent_name, output)
        self.active_agent = None
        self.updated_at = datetime.now(timezone.utc)

    def record_error(self, error: str) -> None:
        """Record an error.

        Args:
            error: Error message
        """
        self.errors.append(error)
        self.updated_at = datetime.now(timezone.utc)

    def fail(self, error: str) -> None:
        """Mark workflow as failed.

        Args:
            error: Error message
        """
        self.record_error(error)
        self.status = AgentWorkflowStatus.FAILED

    def complete(self) -> None:
        """Mark workflow as completed."""
        self.status = AgentWorkflowStatus.COMPLETED
        self.active_agent = None
        self.updated_at = datetime.now(timezone.utc)

    def is_terminal(self) -> bool:
        """Check if workflow is in a terminal state."""
        return self.status in (
            AgentWorkflowStatus.COMPLETED,
            AgentWorkflowStatus.FAILED,
            AgentWorkflowStatus.CANCELLED,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for checkpointing."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentWorkflowState":
        """Deserialize state from dictionary."""
        return cls.model_validate(data)


def create_workflow_state(
    goal: str,
    workflow_id: Optional[str] = None,
    initial_agent: Optional[str] = None,
) -> AgentWorkflowState:
    """Helper to create a new workflow state.

    Args:
        goal: Workflow objective
        workflow_id: Optional workflow ID (generated if not provided)
        initial_agent: Optional first agent to execute

    Returns:
        New AgentWorkflowState instance
    """
    state = AgentWorkflowState(
        workflow_id=workflow_id or str(uuid.uuid4()),
        goal=goal,
    )

    # If initial agent specified, create a task message
    if initial_agent:
        from marie.agent.coordination.message import create_task_message

        msg = create_task_message(
            sender="coordinator",
            receiver=initial_agent,
            content=goal,
        )
        state.post_message(msg)

    return state
