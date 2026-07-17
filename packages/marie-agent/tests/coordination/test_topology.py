"""Unit tests for coordination topology base classes."""

from __future__ import annotations

import pytest

from marie.agent.config import CoordinationConfig
from marie.agent.coordination.topology import (
    AgentResult,
    CoordinationResult,
    CoordinatorFactory,
)


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful agent result."""
        result = AgentResult(
            agent_name="test_agent",
            output="test output",
            status="completed",
        )
        assert result.agent_name == "test_agent"
        assert result.output == "test output"
        assert result.is_success is True
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed agent result."""
        result = AgentResult(
            agent_name="failed_agent",
            output=None,
            status="failed",
            error="Something went wrong",
        )
        assert result.is_success is False
        assert result.error == "Something went wrong"

    def test_timeout_result(self):
        """Test timeout result is not success."""
        result = AgentResult(
            agent_name="slow_agent",
            output=None,
            status="timeout",
            error="Timeout after 30s",
        )
        assert result.is_success is False
        assert "timeout" in result.status.lower()

    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = AgentResult(
            agent_name="meta_agent",
            output="result",
            metadata={"confidence": 0.95, "tokens": 100},
        )
        assert result.metadata["confidence"] == 0.95
        assert result.metadata["tokens"] == 100

    def test_result_with_duration(self):
        """Test result tracks duration."""
        result = AgentResult(
            agent_name="timed_agent",
            output="result",
            duration_ms=150.5,
        )
        assert result.duration_ms == 150.5


class TestCoordinationResult:
    """Tests for CoordinationResult dataclass."""

    def test_all_succeeded(self):
        """Test all_succeeded property with all successful results."""
        results = [
            AgentResult(agent_name=f"agent_{i}", output=f"output_{i}") for i in range(3)
        ]
        coord_result = CoordinationResult(
            results=results,
            merged_output={"outputs": ["output_0", "output_1", "output_2"]},
            topology="parallel",
            merge_strategy="aggregate",
            total_duration_ms=100.0,
        )
        assert coord_result.all_succeeded is True
        assert coord_result.success_count == 3
        assert coord_result.failure_count == 0

    def test_partial_failure(self):
        """Test with some failed results."""
        results = [
            AgentResult(agent_name="success", output="ok"),
            AgentResult(
                agent_name="failure", output=None, status="failed", error="err"
            ),
            AgentResult(
                agent_name="timeout", output=None, status="timeout", error="to"
            ),
        ]
        coord_result = CoordinationResult(
            results=results,
            merged_output=None,
            topology="parallel",
            merge_strategy="aggregate",
            total_duration_ms=200.0,
        )
        assert coord_result.all_succeeded is False
        assert coord_result.success_count == 1
        assert coord_result.failure_count == 2

    def test_empty_results(self):
        """Test with no results."""
        coord_result = CoordinationResult(
            results=[],
            merged_output=None,
            topology="parallel",
            merge_strategy="aggregate",
            total_duration_ms=0.0,
        )
        assert coord_result.all_succeeded is True
        assert coord_result.success_count == 0
        assert coord_result.failure_count == 0

    def test_timestamps(self):
        """Test timestamps are set."""
        coord_result = CoordinationResult(
            results=[],
            merged_output=None,
            topology="sequential",
            merge_strategy="first_wins",
            total_duration_ms=50.0,
        )
        assert coord_result.started_at is not None


class TestMergeStrategies:
    """Tests for merge strategy implementations."""

    @pytest.fixture
    def mock_coordinator(self, coordination_config):
        """Create a concrete coordinator for testing merges."""
        from marie.agent.coordination.fan_out import FanOutCoordinator

        return FanOutCoordinator(coordination_config)

    def test_merge_aggregate(self, mock_coordinator):
        """Test aggregate merge strategy."""
        results = [
            AgentResult(agent_name="a1", output="output_1"),
            AgentResult(agent_name="a2", output="output_2"),
        ]
        mock_coordinator.config.merge_strategy = "aggregate"
        merged = mock_coordinator._merge_results(results)

        assert "outputs" in merged
        assert "output_1" in merged["outputs"]
        assert "output_2" in merged["outputs"]
        assert "a1" in merged["agents"]
        assert "a2" in merged["agents"]

    def test_merge_first_wins(self, mock_coordinator):
        """Test first_wins merge strategy."""
        results = [
            AgentResult(agent_name="first", output="first_output"),
            AgentResult(agent_name="second", output="second_output"),
        ]
        mock_coordinator.config.merge_strategy = "first_wins"
        merged = mock_coordinator._merge_results(results)

        assert merged == "first_output"

    def test_merge_vote_majority(self, mock_coordinator):
        """Test vote merge strategy with clear majority."""
        results = [
            AgentResult(agent_name="a1", output="winner"),
            AgentResult(agent_name="a2", output="winner"),
            AgentResult(agent_name="a3", output="loser"),
        ]
        mock_coordinator.config.merge_strategy = "vote"
        merged = mock_coordinator._merge_results(results)

        assert merged == "winner"

    def test_merge_vote_tie(self, mock_coordinator):
        """Test vote merge strategy with tie (first most common wins)."""
        results = [
            AgentResult(agent_name="a1", output="option_a"),
            AgentResult(agent_name="a2", output="option_b"),
        ]
        mock_coordinator.config.merge_strategy = "vote"
        merged = mock_coordinator._merge_results(results)

        # Either is valid in a tie
        assert merged in ["option_a", "option_b"]

    def test_merge_best_score(self, mock_coordinator):
        """Test best_score merge strategy."""
        results = [
            AgentResult(
                agent_name="low",
                output="low_conf",
                metadata={"confidence": 0.3},
            ),
            AgentResult(
                agent_name="high",
                output="high_conf",
                metadata={"confidence": 0.95},
            ),
            AgentResult(
                agent_name="medium",
                output="med_conf",
                metadata={"confidence": 0.6},
            ),
        ]
        mock_coordinator.config.merge_strategy = "best_score"
        merged = mock_coordinator._merge_results(results)

        assert merged == "high_conf"

    def test_merge_empty_results(self, mock_coordinator):
        """Test merge with no results."""
        merged = mock_coordinator._merge_results([])
        assert merged is None

    def test_merge_only_failures(self, mock_coordinator):
        """Test merge when all results failed."""
        results = [
            AgentResult(agent_name="f1", output=None, status="failed", error="err1"),
            AgentResult(agent_name="f2", output=None, status="failed", error="err2"),
        ]
        merged = mock_coordinator._merge_results(results)
        assert merged is None


class TestCoordinatorFactory:
    """Tests for CoordinatorFactory."""

    def test_create_parallel_coordinator(self):
        """Test creating parallel coordinator."""
        config = CoordinationConfig(topology="parallel")
        coordinator = CoordinatorFactory.create(config)

        from marie.agent.coordination.fan_out import FanOutCoordinator

        assert isinstance(coordinator, FanOutCoordinator)

    def test_create_sequential_coordinator(self):
        """Test creating sequential coordinator."""
        config = CoordinationConfig(topology="sequential")
        coordinator = CoordinatorFactory.create(config)

        from marie.agent.coordination.chain import ChainCoordinator

        assert isinstance(coordinator, ChainCoordinator)

    def test_create_unknown_topology_raises(self):
        """Test unknown topology raises ValueError."""
        config = CoordinationConfig(topology="unknown")
        with pytest.raises(ValueError, match="Unknown topology"):
            CoordinatorFactory.create(config)

    def test_register_custom_coordinator(self):
        """Test registering a custom coordinator."""
        from marie.agent.coordination.fan_out import FanOutCoordinator

        CoordinatorFactory.register("custom", FanOutCoordinator)
        config = CoordinationConfig(topology="custom")
        coordinator = CoordinatorFactory.create(config)

        assert isinstance(coordinator, FanOutCoordinator)


class TestBaseCoordinator:
    """Tests for BaseCoordinator agent management."""

    @pytest.fixture
    def coordinator(self, coordination_config):
        """Create a coordinator for testing."""
        from marie.agent.coordination.fan_out import FanOutCoordinator

        return FanOutCoordinator(coordination_config)

    def test_add_agent(self, coordinator, mock_agent):
        """Test adding a single agent."""
        coordinator.add_agent(mock_agent)
        assert len(coordinator.agents) == 1
        assert coordinator.agents[0] is mock_agent

    def test_add_agents(self, coordinator, mock_agents):
        """Test adding multiple agents."""
        coordinator.add_agents(mock_agents)
        assert len(coordinator.agents) == 3

    def test_clear_agents(self, coordinator, mock_agents):
        """Test clearing all agents."""
        coordinator.add_agents(mock_agents)
        assert len(coordinator.agents) == 3

        coordinator.clear_agents()
        assert len(coordinator.agents) == 0
