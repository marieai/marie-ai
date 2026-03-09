"""Unit tests for WorkflowCoordinator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from marie.agent.config import CoordinationConfig
from marie.agent.coordination import (
    AgentWorkflowState,
    InMemoryAuditLogger,
    InMemoryCheckpointStore,
    MessageDrivenRoutingPolicy,
    ReservedReceiver,
    SequentialRoutingPolicy,
    WorkflowCoordinator,
)
from marie.agent.message import Message


@dataclass
class RoutingMockAgent:
    """Mock agent that returns routing instructions in metadata."""

    name: str = "mock_agent"
    next_receiver: str = "__end__"
    response: str = "mock response"
    delay: float = 0.0
    should_fail: bool = False
    call_count: int = field(default=0, init=False)

    async def arun(self, messages: List[Any], **kwargs) -> Dict[str, Any]:
        self.call_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError("Agent failed")
        return {
            "output": f"{self.name}: {self.response}",
            "messages": [],
            "metadata": {"next_agent": self.next_receiver},
        }


class TestWorkflowBasicExecution:
    """Tests for basic workflow execution."""

    @pytest.fixture
    def config(self):
        return CoordinationConfig(
            topology="workflow",
            max_steps=20,
            max_retries_per_agent=2,
            timeout=30.0,
        )

    @pytest.fixture
    def coordinator(self, config):
        return WorkflowCoordinator(config)

    @pytest.fixture
    def sample_messages(self):
        return [Message.user("Process this document")]

    @pytest.mark.asyncio
    async def test_run_single_agent_terminates(self, coordinator, sample_messages):
        """Test workflow with single agent that terminates."""
        agent = RoutingMockAgent(name="single", next_receiver="__end__")
        coordinator.add_agent(agent)

        result = await coordinator.run(sample_messages)

        assert result.topology == "workflow"
        assert len(result.results) == 1
        assert result.results[0].agent_name == "single"
        assert agent.call_count == 1

    @pytest.mark.asyncio
    async def test_run_empty_workflow(self, coordinator, sample_messages):
        """Test running workflow with no agents."""
        result = await coordinator.run(sample_messages)

        assert len(result.results) == 0
        assert result.merged_output is None

    @pytest.mark.asyncio
    async def test_run_sequential_agents(self, coordinator, sample_messages):
        """Test agents execute in sequence via routing."""
        agent1 = RoutingMockAgent(name="planner", next_receiver="executor")
        agent2 = RoutingMockAgent(name="executor", next_receiver="validator")
        agent3 = RoutingMockAgent(name="validator", next_receiver="__end__")

        coordinator.add_agent(agent1)
        coordinator.add_agent(agent2)
        coordinator.add_agent(agent3)

        result = await coordinator.run(
            sample_messages,
            initial_agent="planner",
        )

        assert len(result.results) == 3
        assert result.results[0].agent_name == "planner"
        assert result.results[1].agent_name == "executor"
        assert result.results[2].agent_name == "validator"

    @pytest.mark.asyncio
    async def test_max_steps_limit(self, sample_messages):
        """Test workflow stops at max_steps."""
        config = CoordinationConfig(
            topology="workflow",
            max_steps=3,
        )
        coordinator = WorkflowCoordinator(config)

        # Agent that always routes to itself (infinite loop)
        looping_agent = RoutingMockAgent(name="looper", next_receiver="looper")
        coordinator.add_agent(looping_agent)

        result = await coordinator.run(
            sample_messages,
            initial_agent="looper",
        )

        # Should stop after max_steps
        assert len(result.results) <= 3


class TestWorkflowRouting:
    """Tests for message-driven routing."""

    @pytest.fixture
    def coordinator(self):
        config = CoordinationConfig(topology="workflow", max_steps=10)
        return WorkflowCoordinator(config)

    @pytest.fixture
    def sample_messages(self):
        return [Message.user("Test")]

    @pytest.mark.asyncio
    async def test_routing_end_terminates(self, coordinator, sample_messages):
        """Test __end__ receiver terminates workflow."""
        agent = RoutingMockAgent(name="terminator", next_receiver="__end__")
        coordinator.add_agent(agent)

        result = await coordinator.run(sample_messages)

        assert coordinator.state is not None
        assert coordinator.state.is_terminal()

    @pytest.mark.asyncio
    async def test_routing_self_retries(self, coordinator, sample_messages):
        """Test __self__ receiver causes retry."""
        # Agent fails first time, succeeds second
        call_count = 0

        class RetryAgent:
            name = "retrier"

            async def arun(self, messages, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {
                        "output": "retry",
                        "messages": [],
                        "metadata": {"next_agent": "__self__"},
                    }
                return {
                    "output": "success",
                    "messages": [],
                    "metadata": {"next_agent": "__end__"},
                }

        coordinator.add_agent(RetryAgent())
        await coordinator.run(sample_messages)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_routing_direct_agent_name(self, coordinator, sample_messages):
        """Test direct agent name routing."""
        agent1 = RoutingMockAgent(name="first", next_receiver="second")
        agent2 = RoutingMockAgent(name="second", next_receiver="__end__")

        coordinator.add_agent(agent1)
        coordinator.add_agent(agent2)

        result = await coordinator.run(sample_messages, initial_agent="first")

        assert len(result.results) == 2
        assert result.results[1].agent_name == "second"


class TestWorkflowAgentManagement:
    """Tests for agent management."""

    @pytest.fixture
    def coordinator(self):
        config = CoordinationConfig(topology="workflow")
        return WorkflowCoordinator(config)

    def test_add_agent(self, coordinator):
        """Test adding agents."""
        agent = RoutingMockAgent(name="test")
        coordinator.add_agent(agent)
        assert len(coordinator.agents) == 1

    def test_add_agents(self, coordinator):
        """Test adding multiple agents."""
        agents = [RoutingMockAgent(name=f"agent_{i}") for i in range(3)]
        coordinator.add_agents(agents)
        assert len(coordinator.agents) == 3

    def test_delete_agent(self, coordinator):
        """Test deleting an agent."""
        agents = [RoutingMockAgent(name=f"agent_{i}") for i in range(3)]
        coordinator.add_agents(agents)

        coordinator.delete_agent("agent_1")

        assert len(coordinator.agents) == 2
        agent_names = [a.name for a in coordinator.agents]
        assert "agent_1" not in agent_names

    def test_delete_nonexistent_agent_raises(self, coordinator):
        """Test deleting nonexistent agent raises."""
        with pytest.raises(ValueError, match="not found"):
            coordinator.delete_agent("nonexistent")

    def test_set_start_agent(self, coordinator):
        """Test setting start agent."""
        agent1 = RoutingMockAgent(name="first")
        agent2 = RoutingMockAgent(name="second")
        coordinator.add_agent(agent1)
        coordinator.add_agent(agent2)

        coordinator.set_start_agent("second")

        assert coordinator.agents[0].name == "second"

    def test_set_start_nonexistent_raises(self, coordinator):
        """Test setting nonexistent start agent raises."""
        with pytest.raises(ValueError, match="not found"):
            coordinator.set_start_agent("nonexistent")


class TestWorkflowRetry:
    """Tests for retry logic."""

    @pytest.fixture
    def sample_messages(self):
        return [Message.user("Test")]

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, sample_messages):
        """Test agent is retried on failure."""
        config = CoordinationConfig(
            topology="workflow",
            max_retries_per_agent=3,
        )
        coordinator = WorkflowCoordinator(config)

        call_count = 0

        class FailThenSucceed:
            name = "flaky"

            async def arun(self, messages, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError("Temporary failure")
                return {
                    "output": "flaky: success",
                    "messages": [],
                    "metadata": {"next_agent": "__end__"},
                }

        coordinator.add_agent(FailThenSucceed())
        result = await coordinator.run(sample_messages)

        assert call_count == 3
        assert result.results[-1].output == "flaky: success"

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, sample_messages):
        """Test workflow fails after max retries exhausted."""
        config = CoordinationConfig(
            topology="workflow",
            max_retries_per_agent=2,
        )
        coordinator = WorkflowCoordinator(config)

        agent = RoutingMockAgent(name="always_fails", should_fail=True)
        coordinator.add_agent(agent)

        result = await coordinator.run(sample_messages)

        # Should have attempted 2 retries
        assert agent.call_count == 2
        assert coordinator.state.status.value == "failed"


class TestWorkflowCheckpointing:
    """Tests for checkpoint/restore functionality."""

    @pytest.fixture
    def sample_messages(self):
        return [Message.user("Test")]

    @pytest.mark.asyncio
    async def test_checkpoint_saves_state(self, sample_messages):
        """Test checkpoints are saved during execution."""
        config = CoordinationConfig(topology="workflow")
        store = InMemoryCheckpointStore()
        coordinator = WorkflowCoordinator(config, checkpoint_store=store)

        agent1 = RoutingMockAgent(name="first", next_receiver="second")
        agent2 = RoutingMockAgent(name="second", next_receiver="__end__")
        coordinator.add_agent(agent1)
        coordinator.add_agent(agent2)

        await coordinator.run(
            sample_messages,
            workflow_id="wf-checkpoint-test",
            initial_agent="first",
        )

        # Checkpoint should exist
        checkpoints = await store.list_checkpoints()
        assert "wf-checkpoint-test" in checkpoints

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, sample_messages):
        """Test workflow can be restored from checkpoint."""
        config = CoordinationConfig(topology="workflow")
        store = InMemoryCheckpointStore()

        # Create and save state
        from marie.agent.coordination.state import create_workflow_state

        state = create_workflow_state(
            goal="Restore test",
            workflow_id="wf-restore",
            initial_agent="second",
        )
        state.step_history = ["first"]  # Simulate first agent completed
        await store.save("wf-restore", state)

        # New coordinator with same store
        coordinator = WorkflowCoordinator(config, checkpoint_store=store)
        agent2 = RoutingMockAgent(name="second", next_receiver="__end__")
        coordinator.add_agent(agent2)

        result = await coordinator.run(
            sample_messages,
            workflow_id="wf-restore",
            restore_checkpoint=True,
        )

        # Should have restored and continued
        assert coordinator.state.workflow_id == "wf-restore"


class TestWorkflowAuditLogging:
    """Tests for audit logging."""

    @pytest.fixture
    def sample_messages(self):
        return [Message.user("Test")]

    @pytest.mark.asyncio
    async def test_audit_logs_agent_events(self, sample_messages):
        """Test audit logs agent start/complete events."""
        config = CoordinationConfig(topology="workflow")
        audit_logger = InMemoryAuditLogger()
        coordinator = WorkflowCoordinator(config, audit_logger=audit_logger)

        agent = RoutingMockAgent(name="audited", next_receiver="__end__")
        coordinator.add_agent(agent)

        await coordinator.run(sample_messages)

        events = await audit_logger.query()
        event_types = [e.event_type.value for e in events]

        assert "agent_started" in event_types
        assert "agent_completed" in event_types
        assert "workflow_completed" in event_types

    @pytest.mark.asyncio
    async def test_audit_logs_agent_failure(self, sample_messages):
        """Test audit logs agent failure events."""
        config = CoordinationConfig(
            topology="workflow",
            max_retries_per_agent=1,
        )
        audit_logger = InMemoryAuditLogger()
        coordinator = WorkflowCoordinator(config, audit_logger=audit_logger)

        agent = RoutingMockAgent(name="failing", should_fail=True)
        coordinator.add_agent(agent)

        await coordinator.run(sample_messages)

        events = await audit_logger.query()
        event_types = [e.event_type.value for e in events]

        assert "agent_failed" in event_types


class TestRoutingPolicies:
    """Tests for routing policy implementations."""

    def test_sequential_policy_follows_sequence(self):
        """Test SequentialRoutingPolicy follows defined order."""
        policy = SequentialRoutingPolicy(["a", "b", "c"])
        state = AgentWorkflowState()
        available = ["a", "b", "c"]

        # First should be "a"
        result = asyncio.run(policy.select_next_agent(state, available))
        assert result == "a"

        # After "a" completes
        state.step_history = ["a"]
        result = asyncio.run(policy.select_next_agent(state, available))
        assert result == "b"

        # After "b" completes
        state.step_history = ["a", "b"]
        result = asyncio.run(policy.select_next_agent(state, available))
        assert result == "c"

        # After all complete
        state.step_history = ["a", "b", "c"]
        result = asyncio.run(policy.select_next_agent(state, available))
        assert result is None

    def test_message_driven_policy_follows_receiver(self):
        """Test MessageDrivenRoutingPolicy follows message receiver."""
        from marie.agent.coordination.message import create_task_message

        policy = MessageDrivenRoutingPolicy()
        state = AgentWorkflowState()
        available = ["planner", "executor", "validator"]

        # Post message to executor
        state.post_message(create_task_message("planner", "executor", "task"))

        result = asyncio.run(policy.select_next_agent(state, available))
        assert result == "executor"

    def test_message_driven_policy_handles_end(self):
        """Test MessageDrivenRoutingPolicy returns None for __end__."""
        from marie.agent.coordination.message import AgentMessage

        policy = MessageDrivenRoutingPolicy()
        state = AgentWorkflowState()
        available = ["a", "b"]

        state.post_message(AgentMessage(
            sender="a",
            receiver="__end__",
        ))

        result = asyncio.run(policy.select_next_agent(state, available))
        assert result is None
