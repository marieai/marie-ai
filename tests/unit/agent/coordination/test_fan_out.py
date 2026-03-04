"""Unit tests for FanOutCoordinator (parallel execution)."""

from __future__ import annotations

import asyncio
import time

import pytest

from marie.agent.config import CoordinationConfig
from marie.agent.coordination.fan_out import FanOutCoordinator
from marie.agent.message import Message

from .conftest import ConfidenceMockAgent, MockAgent, SlowMockAgent


class TestFanOutBasicExecution:
    """Tests for basic parallel execution."""

    @pytest.mark.asyncio
    async def test_run_single_agent(self, fan_out_coordinator, mock_agent, sample_messages):
        """Test running a single agent."""
        fan_out_coordinator.add_agent(mock_agent)
        result = await fan_out_coordinator.run(sample_messages)

        assert result.topology == "parallel"
        assert len(result.results) == 1
        assert result.results[0].agent_name == "test_agent"
        assert result.results[0].output == "test response"
        assert result.results[0].is_success is True

    @pytest.mark.asyncio
    async def test_run_multiple_agents(self, fan_out_coordinator, mock_agents, sample_messages):
        """Test running multiple agents in parallel."""
        fan_out_coordinator.add_agents(mock_agents)
        result = await fan_out_coordinator.run(sample_messages)

        assert len(result.results) == 3
        assert result.all_succeeded is True

        outputs = [r.output for r in result.results]
        assert "response_0" in outputs
        assert "response_1" in outputs
        assert "response_2" in outputs

    @pytest.mark.asyncio
    async def test_run_empty_agents(self, fan_out_coordinator, sample_messages):
        """Test running with no agents."""
        result = await fan_out_coordinator.run(sample_messages)

        assert len(result.results) == 0
        assert result.merged_output is None
        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_agents_receive_messages(self, fan_out_coordinator, sample_messages):
        """Test that agents receive the input messages."""
        agent = MockAgent(name="receiver")
        fan_out_coordinator.add_agent(agent)

        await fan_out_coordinator.run(sample_messages)

        assert len(agent.received_messages) == 1
        assert len(agent.received_messages[0]) == 1
        assert agent.received_messages[0][0].content == "Hello, world!"


class TestFanOutConcurrency:
    """Tests for concurrency control."""

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, sample_messages):
        """Test that max_concurrent limit is respected."""
        config = CoordinationConfig(
            topology="parallel",
            max_concurrent=2,
            timeout=10.0,
        )
        coordinator = FanOutCoordinator(config)

        concurrent_count = 0
        max_concurrent_seen = 0

        async def track_concurrency(messages, **kwargs):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.1)
            concurrent_count -= 1
            return {"output": "done", "messages": [], "metadata": {}}

        agents = []
        for i in range(5):
            agent = MockAgent(name=f"agent_{i}")
            agent.arun = track_concurrency
            agents.append(agent)

        coordinator.add_agents(agents)
        await coordinator.run(sample_messages)

        assert max_concurrent_seen <= 2

    @pytest.mark.asyncio
    async def test_parallel_execution_faster_than_sequential(self, sample_messages):
        """Test that parallel execution is faster than sequential."""
        config = CoordinationConfig(
            topology="parallel",
            max_concurrent=5,
            timeout=10.0,
        )
        coordinator = FanOutCoordinator(config)

        delay = 0.1
        agents = [
            SlowMockAgent(name=f"agent_{i}", delay=delay)
            for i in range(3)
        ]
        coordinator.add_agents(agents)

        start = time.perf_counter()
        await coordinator.run(sample_messages)
        elapsed = time.perf_counter() - start

        # Parallel should be faster than 3 * delay
        # Allow some overhead but should be less than sequential
        assert elapsed < 2 * delay


class TestFanOutErrorHandling:
    """Tests for error handling in parallel execution."""

    @pytest.mark.asyncio
    async def test_single_agent_failure(
        self, fan_out_coordinator, mock_agent, failing_agent, sample_messages
    ):
        """Test that one failing agent doesn't stop others."""
        fan_out_coordinator.add_agent(mock_agent)
        fan_out_coordinator.add_agent(failing_agent)

        result = await fan_out_coordinator.run(sample_messages)

        assert len(result.results) == 2
        assert result.success_count == 1
        assert result.failure_count == 1

        # Find the failed result
        failed = next(r for r in result.results if r.agent_name == "failing_agent")
        assert failed.status == "failed"
        assert "Intentional failure" in failed.error

    @pytest.mark.asyncio
    async def test_all_agents_fail(self, fan_out_coordinator, sample_messages):
        """Test handling when all agents fail."""
        failing_agents = [
            MockAgent(name=f"fail_{i}", should_fail=True, fail_message=f"Error {i}")
            for i in range(3)
        ]
        fan_out_coordinator.add_agents(failing_agents)

        result = await fan_out_coordinator.run(sample_messages)

        assert result.all_succeeded is False
        assert result.success_count == 0
        assert result.merged_output is None

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_messages):
        """Test that agent timeout is handled gracefully."""
        config = CoordinationConfig(
            topology="parallel",
            max_concurrent=5,
            timeout=0.1,  # Very short timeout
        )
        coordinator = FanOutCoordinator(config)

        slow_agent = SlowMockAgent(name="slow", delay=1.0)  # Will timeout
        fast_agent = MockAgent(name="fast")

        coordinator.add_agent(slow_agent)
        coordinator.add_agent(fast_agent)

        result = await coordinator.run(sample_messages)

        # Fast agent should succeed
        fast_result = next(r for r in result.results if r.agent_name == "fast")
        assert fast_result.is_success is True

        # Slow agent should timeout
        slow_result = next(r for r in result.results if r.agent_name == "slow")
        assert slow_result.status == "timeout"
        assert "Timeout" in slow_result.error


class TestFanOutMergeStrategies:
    """Tests for merge strategy integration with FanOut."""

    @pytest.mark.asyncio
    async def test_aggregate_merge(self, sample_messages):
        """Test aggregate merge strategy."""
        config = CoordinationConfig(
            topology="parallel",
            merge_strategy="aggregate",
        )
        coordinator = FanOutCoordinator(config)

        agents = [
            MockAgent(name=f"agent_{i}", response=f"output_{i}")
            for i in range(3)
        ]
        coordinator.add_agents(agents)

        result = await coordinator.run(sample_messages)

        assert result.merge_strategy == "aggregate"
        assert "outputs" in result.merged_output
        assert len(result.merged_output["outputs"]) == 3

    @pytest.mark.asyncio
    async def test_vote_merge(self, sample_messages):
        """Test vote merge strategy with majority."""
        config = CoordinationConfig(
            topology="parallel",
            merge_strategy="vote",
        )
        coordinator = FanOutCoordinator(config)

        agents = [
            MockAgent(name="voter_1", response="yes"),
            MockAgent(name="voter_2", response="yes"),
            MockAgent(name="voter_3", response="no"),
        ]
        coordinator.add_agents(agents)

        result = await coordinator.run(sample_messages)

        assert result.merged_output == "yes"

    @pytest.mark.asyncio
    async def test_best_score_merge(self, confidence_agents, sample_messages):
        """Test best_score merge strategy."""
        config = CoordinationConfig(
            topology="parallel",
            merge_strategy="best_score",
        )
        coordinator = FanOutCoordinator(config)
        coordinator.add_agents(confidence_agents)

        result = await coordinator.run(sample_messages)

        assert result.merged_output == "high confidence"


class TestFanOutMetrics:
    """Tests for execution metrics."""

    @pytest.mark.asyncio
    async def test_total_duration_tracked(
        self, fan_out_coordinator, mock_agents, sample_messages
    ):
        """Test that total duration is tracked."""
        fan_out_coordinator.add_agents(mock_agents)
        result = await fan_out_coordinator.run(sample_messages)

        assert result.total_duration_ms > 0
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at

    @pytest.mark.asyncio
    async def test_individual_agent_duration(
        self, fan_out_coordinator, sample_messages
    ):
        """Test that individual agent durations are tracked."""
        agent = SlowMockAgent(name="timed", delay=0.05)
        fan_out_coordinator.add_agent(agent)

        result = await fan_out_coordinator.run(sample_messages)

        assert result.results[0].duration_ms >= 50  # At least 50ms


class TestFanOutKwargsForwarding:
    """Tests for kwargs forwarding to agents."""

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_to_agents(
        self, fan_out_coordinator, sample_messages
    ):
        """Test that kwargs are forwarded to agent run methods."""
        agent = MockAgent(name="receiver")
        fan_out_coordinator.add_agent(agent)

        await fan_out_coordinator.run(
            sample_messages,
            custom_arg="custom_value",
            another_arg=42,
        )

        assert len(agent.received_kwargs) == 1
        assert agent.received_kwargs[0]["custom_arg"] == "custom_value"
        assert agent.received_kwargs[0]["another_arg"] == 42
