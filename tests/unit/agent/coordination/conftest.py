"""Fixtures for coordination tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from marie.agent.config import CoordinationConfig
from marie.agent.coordination.chain import ChainCoordinator
from marie.agent.coordination.fan_out import FanOutCoordinator
from marie.agent.coordination.topology import AgentResult, BaseCoordinator
from marie.agent.message import Message


@dataclass
class MockAgent:
    """Mock agent for testing coordination."""

    name: str = "mock_agent"
    response: Any = "mock response"
    delay: float = 0.0
    should_fail: bool = False
    fail_message: str = "Agent error"
    call_count: int = field(default=0, init=False)
    received_messages: List[List[Message]] = field(default_factory=list, init=False)
    received_kwargs: List[Dict[str, Any]] = field(default_factory=list, init=False)

    async def arun(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        """Async run method."""
        self.call_count += 1
        self.received_messages.append(list(messages))
        self.received_kwargs.append(kwargs)

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise RuntimeError(self.fail_message)

        return {
            "output": self.response,
            "messages": [],
            "metadata": {"confidence": 0.8},
        }

    def run(self, messages: List[Message], **kwargs):
        """Sync run method that returns coroutine."""
        return self.arun(messages, **kwargs)


@dataclass
class SlowMockAgent(MockAgent):
    """Mock agent with configurable delay for timeout testing."""

    delay: float = 0.5


@dataclass
class ConfidenceMockAgent(MockAgent):
    """Mock agent that returns confidence score in metadata."""

    confidence: float = 0.8

    async def arun(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        result = await super().arun(messages, **kwargs)
        result["metadata"]["confidence"] = self.confidence
        return result


@pytest.fixture
def coordination_config():
    """Default coordination config for testing."""
    return CoordinationConfig(
        topology="parallel",
        merge_strategy="aggregate",
        max_concurrent=5,
        timeout=30.0,
    )


@pytest.fixture
def sequential_config():
    """Sequential coordination config."""
    return CoordinationConfig(
        topology="sequential",
        merge_strategy="first_wins",
        max_concurrent=1,
        timeout=30.0,
    )


@pytest.fixture
def mock_agent():
    """Single mock agent."""
    return MockAgent(name="test_agent", response="test response")


@pytest.fixture
def mock_agents():
    """Multiple mock agents for parallel testing."""
    return [
        MockAgent(name=f"agent_{i}", response=f"response_{i}")
        for i in range(3)
    ]


@pytest.fixture
def slow_agent():
    """Slow agent for timeout testing."""
    return SlowMockAgent(name="slow_agent", delay=2.0)


@pytest.fixture
def failing_agent():
    """Agent that always fails."""
    return MockAgent(
        name="failing_agent",
        should_fail=True,
        fail_message="Intentional failure",
    )


@pytest.fixture
def confidence_agents():
    """Agents with different confidence scores."""
    return [
        ConfidenceMockAgent(name="low", response="low confidence", confidence=0.3),
        ConfidenceMockAgent(name="high", response="high confidence", confidence=0.9),
        ConfidenceMockAgent(name="medium", response="medium confidence", confidence=0.6),
    ]


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [Message.user("Hello, world!")]


@pytest.fixture
def fan_out_coordinator(coordination_config):
    """FanOutCoordinator instance."""
    return FanOutCoordinator(coordination_config)


@pytest.fixture
def chain_coordinator(sequential_config):
    """ChainCoordinator instance."""
    return ChainCoordinator(sequential_config)
