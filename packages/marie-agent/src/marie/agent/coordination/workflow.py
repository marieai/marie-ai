"""WorkflowCoordinator with message-driven routing.

This module provides the WorkflowCoordinator which uses AgentWorkflowState
and message-based routing to determine execution order dynamically, unlike
FanOutCoordinator (parallel) or ChainCoordinator (sequential).
"""

from __future__ import annotations

import contextlib
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from openinference.semconv.trace import SpanAttributes
from opentelemetry import context as otel_context
from opentelemetry import trace as trace_api
from opentelemetry.trace import StatusCode

from marie.agent.coordination.audit import AuditEvent, AuditEventType
from marie.agent.coordination.execution import execute_agent_with_timeout
from marie.agent.coordination.message import (
    AgentMessage,
    AgentMessageType,
    ReservedReceiver,
    create_result_message,
    create_task_message,
)
from marie.agent.coordination.state import (
    AgentWorkflowState,
    create_workflow_state,
)
from marie.agent.coordination.topology import (
    AgentResult,
    BaseCoordinator,
    CoordinationResult,
)
from marie.instrumentation import start_span

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent
    from marie.agent.coordination.audit import AuditLogger
    from marie.agent.coordination.checkpoint import CheckpointStore
    from marie.agent.coordination.config import CoordinationConfig
    from marie.agent.message import Message

logger = logging.getLogger("marie.agent.coordination.workflow")


@runtime_checkable
class RoutingPolicy(Protocol):
    """Protocol for agent routing decisions."""

    async def select_next_agent(
        self,
        state: AgentWorkflowState,
        available_agents: List[str],
    ) -> Optional[str]:
        """Select the next agent to execute based on workflow state.

        Args:
            state: Current workflow state
            available_agents: List of available agent names

        Returns:
            Agent name to execute, or None if workflow should end
        """
        ...


class SequentialRoutingPolicy:
    """Routes through agents in a predefined sequence.

    Agents execute in the order specified during initialization.
    Once an agent completes, the next agent in the sequence executes.
    """

    def __init__(self, sequence: List[str]):
        """Initialize with agent execution sequence.

        Args:
            sequence: Ordered list of agent names to execute
        """
        self._sequence = sequence

    async def select_next_agent(
        self,
        state: AgentWorkflowState,
        available_agents: List[str],
    ) -> Optional[str]:
        """Select the next agent in the sequence.

        Args:
            state: Current workflow state
            available_agents: List of available agent names

        Returns:
            Next agent name in sequence, or None if sequence complete
        """
        completed = set(state.step_history)

        for agent_name in self._sequence:
            if agent_name not in completed and agent_name in available_agents:
                return agent_name

        return None


class MessageDrivenRoutingPolicy:
    """Routes based on AgentMessage.receiver field.

    The default routing policy that uses the receiver field from
    the last message in the workflow state mailbox to determine
    the next agent to execute.

    Handles __next__ by finding the next agent after the last executed
    agent in the available_agents list.
    """

    async def select_next_agent(
        self,
        state: AgentWorkflowState,
        available_agents: List[str],
    ) -> Optional[str]:
        """Select next agent based on last message receiver.

        Args:
            state: Current workflow state
            available_agents: List of available agent names

        Returns:
            Agent name from message receiver, or None if workflow should end
        """
        # Check if last message indicates __next__
        if state.mailbox:
            last_msg = state.mailbox[-1]
            if last_msg.receiver in (ReservedReceiver.NEXT, "__next__"):
                # Find current agent's position and return next
                if state.step_history:
                    last_agent = state.step_history[-1]
                    try:
                        idx = available_agents.index(last_agent)
                        if idx + 1 < len(available_agents):
                            return available_agents[idx + 1]
                    except ValueError:
                        pass
                # If no history or agent not found, return first available
                return available_agents[0] if available_agents else None

        next_agent = state.next_agent()

        if next_agent and next_agent in available_agents:
            return next_agent

        return None


class WorkflowCoordinator(BaseCoordinator):
    """Stateful workflow coordinator with message-driven routing.

    Unlike FanOutCoordinator (parallel) or ChainCoordinator (sequential),
    WorkflowCoordinator uses AgentWorkflowState and message routing to
    determine execution order dynamically.

    Features:
    - Message-driven routing via AgentMessage.receiver
    - Step counting with max_steps limit
    - Per-agent retry with max_retries_per_agent
    - Optional checkpointing for recovery
    - Optional audit logging for observability

    Example:
        ```python
        config = CoordinationConfig(
            topology="workflow",
            max_steps=20,
            max_retries_per_agent=3,
        )

        coordinator = WorkflowCoordinator(config)
        coordinator.add_agents([planner, executor, validator])

        result = await coordinator.run(
            messages=[Message(role="user", content="Process document")],
            workflow_id="doc-123",
            goal="Extract and validate document data",
        )
        ```
    """

    def __init__(
        self,
        config: "CoordinationConfig",
        routing_policy: Optional[RoutingPolicy] = None,
        checkpoint_store: Optional["CheckpointStore"] = None,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        """Initialize workflow coordinator.

        Args:
            config: Coordination configuration
            routing_policy: Optional routing policy (defaults to MessageDrivenRoutingPolicy)
            checkpoint_store: Optional checkpoint store for state persistence
            audit_logger: Optional audit logger for execution tracking
        """
        super().__init__(config)
        self._routing_policy = routing_policy or MessageDrivenRoutingPolicy()
        self._checkpoint_store = checkpoint_store
        self._audit_logger = audit_logger
        self._state: Optional[AgentWorkflowState] = None
        self._agent_map: Dict[str, "BaseAgent"] = {}
        self._retry_counts: Dict[str, int] = {}

    @property
    def state(self) -> Optional[AgentWorkflowState]:
        """Current workflow state."""
        return self._state

    def delete_agent(self, agent_name: str) -> "WorkflowCoordinator":
        """Remove an agent from the workflow.

        Matches BeeAI Workflow.delete_step() API for compatibility.

        Args:
            agent_name: Name of agent to remove

        Returns:
            Self for method chaining

        Raises:
            ValueError: If agent not found
        """
        for i, agent in enumerate(self._agents):
            if agent.name == agent_name:
                del self._agents[i]
                if agent_name in self._agent_map:
                    del self._agent_map[agent_name]
                return self
        raise ValueError(f"Agent '{agent_name}' not found in workflow")

    def set_start_agent(self, agent_name: str) -> "WorkflowCoordinator":
        """Set the starting agent for the workflow.

        Args:
            agent_name: Name of agent to start with

        Returns:
            Self for method chaining

        Raises:
            ValueError: If agent not found
        """
        agent_names = [a.name for a in self._agents]
        if agent_name not in agent_names:
            raise ValueError(f"Agent '{agent_name}' not found in workflow")
        # Move agent to front of list
        for i, agent in enumerate(self._agents):
            if agent.name == agent_name:
                self._agents.insert(0, self._agents.pop(i))
                break
        return self

    async def run(
        self,
        messages: List["Message"],
        **kwargs: Any,
    ) -> CoordinationResult:
        """Execute workflow with message-driven routing.

        Args:
            messages: Input messages to process
            **kwargs: Additional arguments:
                - workflow_id: Optional workflow ID for checkpointing
                - goal: Workflow objective description
                - initial_agent: Optional first agent to execute
                - restore_checkpoint: Whether to restore from checkpoint

        Returns:
            CoordinationResult with workflow execution results
        """
        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        workflow_id = kwargs.get("workflow_id")
        goal = kwargs.get("goal", "")
        initial_agent = kwargs.get("initial_agent")
        restore_checkpoint = kwargs.get("restore_checkpoint", False)
        session_id = kwargs.get("session_id")
        user_id = kwargs.get("user_id")

        # OTel CHAIN span for workflow orchestration
        _tracer = trace_api.get_tracer("marie.agent.coordination")
        _span = start_span(
            _tracer,
            f"workflow:{workflow_id or 'anonymous'}",
            span_kind="chain",
        )
        _span.set_attribute("marie.workflow_id", workflow_id or "")
        _span.set_attribute("marie.agent_count", len(self._agents))
        if session_id:
            _span.set_attribute(SpanAttributes.SESSION_ID, session_id)
        if user_id:
            _span.set_attribute(SpanAttributes.USER_ID, user_id)
        _span_token = otel_context.attach(trace_api.set_span_in_context(_span))

        # Activate OI context so auto-instrumented LLM calls also get tagged
        from marie.instrumentation.context import using_session, using_user

        _oi_ctx = contextlib.ExitStack()
        if session_id:
            _oi_ctx.enter_context(using_session(session_id))
        if user_id:
            _oi_ctx.enter_context(using_user(user_id))

        try:
            # Build agent map
            self._agent_map = {agent.name: agent for agent in self._agents}
            available_agents = list(self._agent_map.keys())

            if not available_agents:
                _span.set_status(StatusCode.OK)
                return CoordinationResult(
                    results=[],
                    merged_output=None,
                    topology="workflow",
                    merge_strategy=self.config.merge_strategy,
                    total_duration_ms=0.0,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )

            # Initialize or restore state
            self._state = await self._initialize_state(
                workflow_id=workflow_id,
                goal=goal,
                initial_agent=initial_agent or available_agents[0],
                restore_checkpoint=restore_checkpoint,
            )
            self._retry_counts = {}

            # Create initial task message if no mailbox messages
            if not self._state.mailbox:
                content = self._extract_user_content(messages)
                initial_msg = create_task_message(
                    sender="coordinator",
                    receiver=initial_agent or available_agents[0],
                    content=content,
                )
                self._state.post_message(initial_msg)

            agent_results: List[AgentResult] = []

            # Main workflow loop
            step_count = 0
            max_steps = self.config.max_steps

            while step_count < max_steps:
                # Select next agent via routing policy
                next_agent = await self._routing_policy.select_next_agent(
                    self._state,
                    available_agents,
                )

                # Check for workflow completion
                if next_agent is None:
                    break

                # Check for broadcast (fan-out signal)
                if next_agent == "__broadcast__":
                    logger.info(
                        f"Workflow {self._state.workflow_id}: broadcast not implemented, ending"
                    )
                    break

                # Execute agent with retry
                result = await self._execute_with_retry(
                    agent_name=next_agent,
                    messages=messages,
                    step=step_count,
                    **kwargs,
                )
                agent_results.append(result)

                # Post result message to state
                await self._post_result_message(next_agent, result)

                # Checkpoint if enabled
                if self._checkpoint_store:
                    await self._checkpoint_store.save(
                        self._state.workflow_id, self._state
                    )

                # Check for terminal condition
                if self._state.is_terminal():
                    break

                step_count += 1

            # Mark workflow complete if not already terminal
            if not self._state.is_terminal():
                self._state.complete()

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            merged = self._merge_results(agent_results)

            # Audit log workflow completion
            if self._audit_logger:
                await self._audit_logger.log(
                    AuditEvent(
                        event_type=AuditEventType.WORKFLOW_COMPLETED,
                        workflow_id=self._state.workflow_id,
                        timestamp=datetime.now(timezone.utc),
                        details={
                            "status": self._state.status.value,
                            "total_steps": step_count,
                            "total_duration_ms": elapsed_ms,
                        },
                    )
                )

            _span.set_attribute("marie.workflow_steps", step_count)
            _span.set_status(StatusCode.OK)

            return CoordinationResult(
                results=agent_results,
                merged_output=merged,
                topology="workflow",
                merge_strategy=self.config.merge_strategy,
                total_duration_ms=elapsed_ms,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                workflow_state=self._state,
            )

        except Exception as exc:
            _span.set_status(StatusCode.ERROR, str(exc))
            _span.record_exception(exc)
            raise

        finally:
            _oi_ctx.close()
            otel_context.detach(_span_token)
            _span.end()

    async def _initialize_state(
        self,
        workflow_id: Optional[str],
        goal: str,
        initial_agent: str,
        restore_checkpoint: bool,
    ) -> AgentWorkflowState:
        """Initialize or restore workflow state.

        Args:
            workflow_id: Optional workflow ID
            goal: Workflow objective
            initial_agent: First agent to execute
            restore_checkpoint: Whether to attempt checkpoint restore

        Returns:
            Initialized or restored workflow state
        """
        # Try to restore from checkpoint
        if restore_checkpoint and self._checkpoint_store and workflow_id:
            restored = await self._checkpoint_store.load(workflow_id)
            if restored:
                logger.info(f"Restored workflow state: {workflow_id}")
                return restored

        # Create new state
        return create_workflow_state(
            goal=goal,
            workflow_id=workflow_id,
            initial_agent=None,  # We handle initial message separately
        )

    async def _execute_with_retry(
        self,
        agent_name: str,
        messages: List["Message"],
        step: int,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute agent with retry logic.

        Args:
            agent_name: Name of agent to execute
            messages: Input messages
            step: Current step number
            **kwargs: Additional arguments

        Returns:
            AgentResult from execution
        """
        agent = self._agent_map[agent_name]
        max_retries = self.config.max_retries_per_agent

        # Initialize retry count
        if agent_name not in self._retry_counts:
            self._retry_counts[agent_name] = 0

        # Record agent start
        self._state.record_agent_start(agent_name)

        # Audit log agent start
        if self._audit_logger:
            await self._audit_logger.log(
                AuditEvent(
                    event_type=AuditEventType.AGENT_STARTED,
                    workflow_id=self._state.workflow_id,
                    agent_name=agent_name,
                    timestamp=datetime.now(timezone.utc),
                    details={"step": step},
                )
            )

        # Build messages with workflow context
        agent_messages = self._state.build_messages_for_agent(
            agent_name,
            [
                (
                    msg.model_dump()
                    if hasattr(msg, "model_dump")
                    else {"role": msg.role, "content": msg.content}
                )
                for msg in messages
            ],
        )

        result = await execute_agent_with_timeout(
            agent=agent,
            messages=agent_messages,
            timeout=self.config.timeout,
            **kwargs,
        )

        # Handle retry on failure
        if not result.is_success:
            self._retry_counts[agent_name] += 1

            if self._retry_counts[agent_name] < max_retries:
                logger.warning(
                    f"Agent {agent_name} failed (attempt {self._retry_counts[agent_name]}/{max_retries}), retrying"
                )
                self._state.record_error(
                    f"Retry {self._retry_counts[agent_name]}: {result.error}"
                )

                # Recursive retry
                return await self._execute_with_retry(
                    agent_name=agent_name,
                    messages=messages,
                    step=step,
                    **kwargs,
                )
            else:
                logger.error(f"Agent {agent_name} exhausted retries ({max_retries})")
                self._state.fail(
                    f"Agent {agent_name} failed after {max_retries} retries: {result.error}"
                )

                # Audit log agent failure
                if self._audit_logger:
                    await self._audit_logger.log(
                        AuditEvent(
                            event_type=AuditEventType.AGENT_FAILED,
                            workflow_id=self._state.workflow_id,
                            agent_name=agent_name,
                            timestamp=datetime.now(timezone.utc),
                            details={
                                "step": step,
                                "duration_ms": result.duration_ms,
                                "error": result.error,
                                "retry_count": self._retry_counts[agent_name],
                            },
                        )
                    )
        else:
            # Record successful completion
            output_str = str(result.output) if result.output else ""
            self._state.record_agent_complete(agent_name, output_str)

            # Audit log agent completion
            if self._audit_logger:
                await self._audit_logger.log(
                    AuditEvent(
                        event_type=AuditEventType.AGENT_COMPLETED,
                        workflow_id=self._state.workflow_id,
                        agent_name=agent_name,
                        timestamp=datetime.now(timezone.utc),
                        details={
                            "step": step,
                            "duration_ms": result.duration_ms,
                            "status": result.status,
                        },
                    )
                )

        return result

    async def _post_result_message(
        self,
        agent_name: str,
        result: AgentResult,
    ) -> None:
        """Post result message to workflow state.

        Args:
            agent_name: Name of agent that produced result
            result: Agent execution result
        """
        # Determine next receiver from result
        next_receiver = self._extract_receiver_from_result(result)

        if result.is_success:
            msg = create_result_message(
                sender=agent_name,
                receiver=next_receiver,
                content=str(result.output) if result.output else "",
                duration_ms=result.duration_ms,
            )
        else:
            msg = AgentMessage(
                sender=agent_name,
                receiver=ReservedReceiver.END,
                msg_type=AgentMessageType.ERROR,
                content=result.error or "Unknown error",
            )

        self._state.post_message(msg)

    def _extract_receiver_from_result(self, result: AgentResult) -> str:
        """Extract next receiver from agent result.

        Looks for routing hints in result metadata or output.

        Args:
            result: Agent execution result

        Returns:
            Receiver string (agent name or reserved receiver)
        """
        # Check metadata for explicit routing
        if "next_agent" in result.metadata:
            return result.metadata["next_agent"]

        if "receiver" in result.metadata:
            return result.metadata["receiver"]

        # Check output for routing signal
        if isinstance(result.output, dict):
            if "next_agent" in result.output:
                return result.output["next_agent"]
            if "receiver" in result.output:
                return result.output["receiver"]

        # Default to coordinator decision (let routing policy decide)
        return ReservedReceiver.COORDINATOR

    def _extract_user_content(self, messages: List["Message"]) -> str:
        """Extract user content from messages.

        Args:
            messages: Input messages

        Returns:
            Concatenated user message content
        """
        content_parts = []
        for msg in messages:
            if hasattr(msg, "role") and msg.role == "user":
                if hasattr(msg, "content"):
                    content_parts.append(str(msg.content))

        return "\n".join(content_parts) if content_parts else ""
