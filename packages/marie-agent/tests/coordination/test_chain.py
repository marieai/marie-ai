"""Unit tests for ChainCoordinator (sequential execution)."""

from __future__ import annotations

import asyncio

import pytest

from marie.agent.config import CoordinationConfig
from marie.agent.coordination.chain import ChainCoordinator

from .conftest import MockAgent, SlowMockAgent


class TestChainBasicExecution:
    """Tests for basic sequential execution."""

    @pytest.mark.asyncio
    async def test_run_single_agent(
        self, chain_coordinator, mock_agent, sample_messages
    ):
        """Test running a single agent in chain."""
        chain_coordinator.add_agent(mock_agent)
        result = await chain_coordinator.run(sample_messages)

        assert result.topology == "sequential"
        assert len(result.results) == 1
        assert result.results[0].output == "test response"
        assert result.results[0].is_success is True

    @pytest.mark.asyncio
    async def test_run_multiple_agents_sequentially(
        self, chain_coordinator, sample_messages
    ):
        """Test agents run in sequence."""
        execution_order = []

        async def track_order(name):
            async def tracked_run(messages, **kwargs):
                execution_order.append(name)
                await asyncio.sleep(0.01)
                return {"output": f"{name}_output", "messages": [], "metadata": {}}

            return tracked_run

        agents = []
        for i in range(3):
            agent = MockAgent(name=f"agent_{i}")
            agent.arun = await track_order(f"agent_{i}")
            agents.append(agent)

        chain_coordinator.add_agents(agents)
        await chain_coordinator.run(sample_messages)

        # Verify sequential order
        assert execution_order == ["agent_0", "agent_1", "agent_2"]

    @pytest.mark.asyncio
    async def test_run_empty_chain(self, chain_coordinator, sample_messages):
        """Test running with no agents."""
        result = await chain_coordinator.run(sample_messages)

        assert len(result.results) == 0
        assert result.merged_output is None


class TestChainContextPassing:
    """Tests for context passing between agents."""

    @pytest.mark.asyncio
    async def test_chain_context_passed_to_agents(
        self, chain_coordinator, sample_messages
    ):
        """Test that chain_context is passed to agents."""
        received_contexts = []

        async def capture_context(messages, **kwargs):
            received_contexts.append(kwargs.get("chain_context"))
            return {"output": "done", "messages": [], "metadata": {}}

        agents = []
        for i in range(3):
            agent = MockAgent(name=f"agent_{i}")
            agent.arun = capture_context
            agents.append(agent)

        chain_coordinator.add_agents(agents)
        await chain_coordinator.run(sample_messages)

        assert len(received_contexts) == 3

        # First agent gets position 0
        assert received_contexts[0]["chain_position"] == 0
        assert received_contexts[0]["previous_outputs"] == []

        # Second agent gets position 1 with first output
        assert received_contexts[1]["chain_position"] == 1
        assert len(received_contexts[1]["previous_outputs"]) == 1

        # Third agent gets position 2 with both outputs
        assert received_contexts[2]["chain_position"] == 2
        assert len(received_contexts[2]["previous_outputs"]) == 2

    @pytest.mark.asyncio
    async def test_previous_output_in_messages(
        self, chain_coordinator, sample_messages
    ):
        """Test that previous agent output is added to messages."""
        received_messages = []

        async def capture_messages(messages, **kwargs):
            received_messages.append([m.content for m in messages])
            return {"output": "processed", "messages": [], "metadata": {}}

        agent1 = MockAgent(name="first", response="first_response")
        agent2 = MockAgent(name="second")
        agent2.arun = capture_messages

        chain_coordinator.add_agent(agent1)
        chain_coordinator.add_agent(agent2)

        await chain_coordinator.run(sample_messages)

        # Second agent should receive original message + first agent's output
        assert len(received_messages) == 1
        assert len(received_messages[0]) == 2
        assert "first_response" in received_messages[0][1]


class TestChainErrorHandling:
    """Tests for error handling in sequential execution."""

    @pytest.mark.asyncio
    async def test_stop_on_failure_enabled(self, sample_messages):
        """Test chain stops when agent fails (default behavior)."""
        config = CoordinationConfig(topology="sequential")
        coordinator = ChainCoordinator(config)
        coordinator.set_stop_on_failure(True)

        agent1 = MockAgent(name="first", response="ok")
        agent2 = MockAgent(name="failing", should_fail=True)
        agent3 = MockAgent(name="third")

        coordinator.add_agent(agent1)
        coordinator.add_agent(agent2)
        coordinator.add_agent(agent3)

        result = await coordinator.run(sample_messages)

        # Should stop after second agent
        assert len(result.results) == 2
        assert result.results[0].is_success is True
        assert result.results[1].is_success is False
        assert agent3.call_count == 0  # Third agent never called

    @pytest.mark.asyncio
    async def test_stop_on_failure_disabled(self, sample_messages):
        """Test chain continues when stop_on_failure is False."""
        config = CoordinationConfig(topology="sequential")
        coordinator = ChainCoordinator(config)
        coordinator.set_stop_on_failure(False)

        agent1 = MockAgent(name="first")
        agent2 = MockAgent(name="failing", should_fail=True)
        agent3 = MockAgent(name="third")

        coordinator.add_agent(agent1)
        coordinator.add_agent(agent2)
        coordinator.add_agent(agent3)

        result = await coordinator.run(sample_messages)

        # All agents should be called
        assert len(result.results) == 3
        assert result.results[2].is_success is True
        assert agent3.call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_messages):
        """Test timeout handling in chain."""
        config = CoordinationConfig(
            topology="sequential",
            timeout=0.1,  # Very short timeout
        )
        coordinator = ChainCoordinator(config)

        slow_agent = SlowMockAgent(name="slow", delay=1.0)
        next_agent = MockAgent(name="next")

        coordinator.add_agent(slow_agent)
        coordinator.add_agent(next_agent)

        result = await coordinator.run(sample_messages)

        # Slow agent should timeout
        assert result.results[0].status == "timeout"
        # Chain should stop
        assert len(result.results) == 1


class TestChainMergeStrategies:
    """Tests for merge strategies with chain coordinator."""

    @pytest.mark.asyncio
    async def test_first_wins_returns_first_output(self, sample_messages):
        """Test first_wins strategy returns first successful output."""
        config = CoordinationConfig(
            topology="sequential",
            merge_strategy="first_wins",
        )
        coordinator = ChainCoordinator(config)

        agents = [
            MockAgent(name=f"agent_{i}", response=f"output_{i}") for i in range(3)
        ]
        coordinator.add_agents(agents)

        result = await coordinator.run(sample_messages)

        assert result.merged_output == "output_0"

    @pytest.mark.asyncio
    async def test_aggregate_collects_all_outputs(self, sample_messages):
        """Test aggregate strategy collects all outputs."""
        config = CoordinationConfig(
            topology="sequential",
            merge_strategy="aggregate",
        )
        coordinator = ChainCoordinator(config)

        agents = [
            MockAgent(name=f"agent_{i}", response=f"output_{i}") for i in range(3)
        ]
        coordinator.add_agents(agents)

        result = await coordinator.run(sample_messages)

        assert "outputs" in result.merged_output
        assert len(result.merged_output["outputs"]) == 3


class TestChainMetrics:
    """Tests for chain execution metrics."""

    @pytest.mark.asyncio
    async def test_total_duration_is_sum_of_agents(self, sample_messages):
        """Test that total duration accounts for all agents."""
        config = CoordinationConfig(topology="sequential")
        coordinator = ChainCoordinator(config)

        agents = [SlowMockAgent(name=f"agent_{i}", delay=0.05) for i in range(3)]
        coordinator.add_agents(agents)

        result = await coordinator.run(sample_messages)

        # Total should be at least 3 * 50ms
        assert result.total_duration_ms >= 150

    @pytest.mark.asyncio
    async def test_individual_durations_tracked(
        self, chain_coordinator, sample_messages
    ):
        """Test individual agent durations are tracked."""
        agent = SlowMockAgent(name="timed", delay=0.05)
        chain_coordinator.add_agent(agent)

        result = await chain_coordinator.run(sample_messages)

        assert result.results[0].duration_ms >= 50


class TestChainWithFailedPreviousAgent:
    """Tests for context handling when previous agent failed."""

    @pytest.mark.asyncio
    async def test_failed_output_not_in_previous_outputs(self, sample_messages):
        """Test failed agent outputs are excluded from previous_outputs."""
        config = CoordinationConfig(topology="sequential")
        coordinator = ChainCoordinator(config)
        coordinator.set_stop_on_failure(False)

        received_contexts = []

        async def capture_context(messages, **kwargs):
            received_contexts.append(kwargs.get("chain_context", {}))
            return {"output": "success", "messages": [], "metadata": {}}

        agent1 = MockAgent(name="success", response="ok")
        agent2 = MockAgent(name="failing", should_fail=True)
        agent3 = MockAgent(name="checker")
        agent3.arun = capture_context

        coordinator.add_agent(agent1)
        coordinator.add_agent(agent2)
        coordinator.add_agent(agent3)

        await coordinator.run(sample_messages)

        # Third agent should only see first agent's output (second failed)
        assert len(received_contexts) == 1
        context = received_contexts[0]
        assert context["chain_position"] == 2
        # Only successful outputs are included
        assert len(context["previous_outputs"]) == 1
        assert context["previous_outputs"][0] == "ok"
