"""Integration tests for A2A client-server interaction.

These tests require the test agents to be running. Use the fixtures
from conftest.py to start them automatically.
"""

import pytest
import pytest_asyncio

from marie.agent.a2a.client import A2AClient
from marie.agent.a2a.discovery import A2AAgentDiscovery, AgentRegistry
from marie.agent.a2a.types import Message, Role, TaskState, TextPart


@pytest.mark.asyncio
class TestA2AClientWithEchoAgent:
    """Integration tests with echo agent."""

    async def test_discover_echo_agent(self, echo_agent: str):
        """Test agent discovery."""
        discovery = A2AAgentDiscovery()
        try:
            card = await discovery.discover(echo_agent)
            assert card.name == "Echo Test Agent"
            assert len(card.skills) == 1
            assert card.skills[0].id == "echo"
        finally:
            await discovery.close()

    async def test_send_message_to_echo(self, echo_agent: str):
        """Test sending a message to echo agent."""
        async with await A2AClient.from_url(echo_agent) as client:
            result = await client.send_message("Hello A2A")

            # Result should be a Task or Message
            assert result is not None

            # Echo agent should return our message
            response_text = str(result)
            assert "Echo" in response_text or "Hello" in response_text

    async def test_send_message_object(self, echo_agent: str):
        """Test sending a Message object."""
        async with await A2AClient.from_url(echo_agent) as client:
            message = Message(
                role=Role.USER,
                parts=[TextPart(text="Test message")],
            )
            result = await client.send_message(message)
            assert result is not None


@pytest.mark.asyncio
class TestA2AClientWithCalculatorAgent:
    """Integration tests with calculator agent."""

    async def test_discover_calculator_agent(self, calculator_agent: str):
        """Test calculator agent discovery."""
        discovery = A2AAgentDiscovery()
        try:
            card = await discovery.discover(calculator_agent)
            assert card.name == "Calculator Agent"
            assert len(card.skills) >= 2
            skill_ids = [s.id for s in card.skills]
            assert "add" in skill_ids
            assert "multiply" in skill_ids
        finally:
            await discovery.close()

    async def test_calculator_add(self, calculator_agent: str):
        """Test addition operation."""
        async with await A2AClient.from_url(calculator_agent) as client:
            result = await client.send_message("add 5 3")
            response_text = str(result)
            assert "8" in response_text

    async def test_calculator_multiply(self, calculator_agent: str):
        """Test multiplication operation."""
        async with await A2AClient.from_url(calculator_agent) as client:
            result = await client.send_message("multiply 4 7")
            response_text = str(result)
            assert "28" in response_text

    async def test_calculator_json_input(self, calculator_agent: str):
        """Test JSON input format."""
        async with await A2AClient.from_url(calculator_agent) as client:
            result = await client.send_message('{"op": "add", "a": 10, "b": 5}')
            response_text = str(result)
            assert "15" in response_text


@pytest.mark.asyncio
class TestA2AClientWithStreamingAgent:
    """Integration tests with streaming agent."""

    async def test_discover_streaming_agent(self, streaming_agent: str):
        """Test streaming agent discovery."""
        discovery = A2AAgentDiscovery()
        try:
            card = await discovery.discover(streaming_agent)
            assert card.name == "Streaming Test Agent"
            assert card.capabilities.streaming is True
        finally:
            await discovery.close()

    async def test_streaming_count(self, streaming_agent: str):
        """Test streaming response."""
        async with await A2AClient.from_url(streaming_agent) as client:
            events = []
            async for event in client.stream_message("3"):
                events.append(event)

            # Should have at least 3 count events plus completion
            assert len(events) >= 3

            # Check for count updates
            event_texts = [str(e) for e in events]
            found_counts = sum(1 for t in event_texts if "Count" in t)
            assert found_counts >= 1


@pytest.mark.asyncio
class TestA2AAgentDiscovery:
    """Tests for agent discovery service."""

    async def test_cache_agent_card(self, echo_agent: str):
        """Test agent card caching."""
        discovery = A2AAgentDiscovery(cache_ttl=300)
        try:
            # First discovery
            card1 = await discovery.discover(echo_agent)

            # Second should use cache
            card2 = await discovery.discover(echo_agent)

            assert card1.name == card2.name
            assert len(discovery.list_agents()) == 1
        finally:
            await discovery.close()

    async def test_force_refresh(self, echo_agent: str):
        """Test forced cache refresh."""
        discovery = A2AAgentDiscovery()
        try:
            card1 = await discovery.discover(echo_agent)
            card2 = await discovery.discover(echo_agent, force_refresh=True)
            assert card1.name == card2.name
        finally:
            await discovery.close()

    async def test_discover_many(self, echo_agent: str, calculator_agent: str):
        """Test discovering multiple agents."""
        discovery = A2AAgentDiscovery()
        try:
            cards = await discovery.discover_many([echo_agent, calculator_agent])
            assert len(cards) == 2
            assert echo_agent in cards
            assert calculator_agent in cards
        finally:
            await discovery.close()

    async def test_invalidate_cache(self, echo_agent: str):
        """Test cache invalidation."""
        discovery = A2AAgentDiscovery()
        try:
            await discovery.discover(echo_agent)
            assert len(discovery.list_agents()) == 1

            discovery.invalidate(echo_agent)
            assert len(discovery.list_agents()) == 0
        finally:
            await discovery.close()


@pytest.mark.asyncio
class TestAgentRegistry:
    """Tests for agent registry."""

    async def test_register_and_get_client(self, echo_agent: str):
        """Test registering and getting client."""
        registry = AgentRegistry()
        registry.register("echo", echo_agent)

        try:
            client = await registry.get_client("echo")
            assert client.name == "Echo Test Agent"
            await client.close()
        finally:
            await registry.close()

    async def test_list_registered(self, echo_agent: str, calculator_agent: str):
        """Test listing registered agents."""
        registry = AgentRegistry()
        registry.register("echo", echo_agent)
        registry.register("calc", calculator_agent)

        try:
            agents = registry.list_registered()
            assert len(agents) == 2
            assert "echo" in agents
            assert "calc" in agents
        finally:
            await registry.close()

    async def test_unregister(self, echo_agent: str):
        """Test unregistering an agent."""
        registry = AgentRegistry()
        registry.register("echo", echo_agent)

        try:
            assert registry.unregister("echo") is True
            assert "echo" not in registry.list_registered()
        finally:
            await registry.close()


@pytest.mark.asyncio
class TestTaskManagement:
    """Tests for task lifecycle management."""

    async def test_get_task(self, echo_agent: str):
        """Test getting task status."""
        async with await A2AClient.from_url(echo_agent) as client:
            # Send a message to create a task
            result = await client.send_message("Hello")

            if hasattr(result, "id"):
                # Get the task
                task = await client.get_task(result.id)
                assert task.id == result.id

    async def test_task_states(self, async_task_agent: str):
        """Test task state transitions."""
        discovery = A2AAgentDiscovery()
        try:
            card = await discovery.discover(async_task_agent)
            assert card.capabilities.streaming is True
        finally:
            await discovery.close()


@pytest.mark.asyncio
class TestAllAgents:
    """Tests using all agents together."""

    async def test_all_agents_discoverable(self, all_test_agents: dict[str, str]):
        """Test that all agents are discoverable."""
        discovery = A2AAgentDiscovery()
        try:
            for name, url in all_test_agents.items():
                card = await discovery.discover(url)
                assert card.name is not None
                assert card.url is not None
        finally:
            await discovery.close()

    async def test_multi_agent_workflow(self, all_test_agents: dict[str, str]):
        """Test a simple multi-agent workflow."""
        # This demonstrates using multiple agents
        echo_url = all_test_agents["echo"]
        calc_url = all_test_agents["calculator"]

        async with await A2AClient.from_url(echo_url) as echo_client:
            async with await A2AClient.from_url(calc_url) as calc_client:
                # First, calculate something
                calc_result = await calc_client.send_message("add 10 20")

                # Then echo the result
                echo_result = await echo_client.send_message(
                    f"The result is: {calc_result}"
                )

                assert echo_result is not None
