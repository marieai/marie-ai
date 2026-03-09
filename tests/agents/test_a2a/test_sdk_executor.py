"""Tests for the SDK-based A2A executor.

These tests verify that MarieA2AExecutor correctly bridges
Marie agents to the A2A SDK interface.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from marie.agent.a2a import (
    A2AExecutor,
    AgentCapabilities,
    AgentCard,
    AgentCardBuilder,
    MarieA2AExecutor,
    Message,
    Role,
    TextPart,
)


class MockAgent:
    """Mock Marie agent for testing."""

    def __init__(self, name: str = "Test Agent", description: str = "A test agent"):
        self.name = name
        self.description = description
        self.function_map = {}

    def run_nonstream(self, messages):
        """Synchronous non-streaming response."""
        user_message = messages[-1]["content"] if messages else ""
        return [MagicMock(content=f"Response to: {user_message}")]

    def run(self, messages):
        """Synchronous streaming response (yields chunks)."""
        user_message = messages[-1]["content"] if messages else ""
        yield [MagicMock(content=f"Chunk 1: {user_message}")]
        yield [MagicMock(content=f"Chunk 2: {user_message}")]


class TestMarieA2AExecutor:
    """Tests for MarieA2AExecutor class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        return MockAgent()

    @pytest.fixture
    def agent_card(self):
        """Create a test agent card."""
        return AgentCard(
            name="Test Agent",
            url="http://localhost:8000",
            description="A test agent",
            version="1.0.0",
            skills=[],
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        )

    @pytest.fixture
    def executor(self, mock_agent, agent_card):
        """Create an executor with mock agent."""
        return MarieA2AExecutor(
            agent=mock_agent,
            agent_card=agent_card,
            streaming=True,
        )

    def test_executor_creation(self, executor, agent_card):
        """Test executor can be created."""
        assert executor.agent_card == agent_card
        assert executor._streaming is True

    def test_extract_text_from_message(self, executor):
        """Test text extraction from A2A message."""
        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[
                TextPart(text="Hello"),
                TextPart(text="World"),
            ],
        )
        text = executor._extract_text(message)
        assert text == "Hello\nWorld"

    def test_extract_text_single_part(self, executor):
        """Test text extraction with single part."""
        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[TextPart(text="Single message")],
        )
        text = executor._extract_text(message)
        assert text == "Single message"


class TestA2AExecutor:
    """Tests for the high-level A2AExecutor class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        return MockAgent()

    def test_executor_creation(self, mock_agent):
        """Test A2AExecutor can be created with an agent."""
        executor = A2AExecutor(
            agent=mock_agent,
            name="My Agent",
            url="http://localhost:8000",
            description="Test description",
            streaming=True,
        )

        assert executor.agent_card.name == "My Agent"
        assert executor.agent_card.url == "http://localhost:8000"
        assert executor.agent_card.capabilities.streaming is True

    def test_executor_uses_agent_name(self, mock_agent):
        """Test executor uses agent's name if not specified."""
        executor = A2AExecutor(
            agent=mock_agent,
            url="http://localhost:8000",
        )

        assert executor.agent_card.name == mock_agent.name

    def test_get_app(self, mock_agent):
        """Test getting Starlette application."""
        executor = A2AExecutor(
            agent=mock_agent,
            url="http://localhost:8000",
        )

        app = executor.get_app()
        assert app is not None
        # Calling again should return same instance
        assert executor.get_app() is app


class TestAgentCardBuilder:
    """Tests for AgentCardBuilder class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        return MockAgent(
            name="Builder Test Agent",
            description="Agent for testing builder",
        )

    def test_build_basic_card(self):
        """Test building a basic agent card."""
        card = (
            AgentCardBuilder()
            .with_name("Basic Agent")
            .with_url("http://localhost:9000")
            .build()
        )

        assert card.name == "Basic Agent"
        assert card.url == "http://localhost:9000"

    def test_build_with_capabilities(self):
        """Test building card with capabilities."""
        card = (
            AgentCardBuilder()
            .with_name("Capable Agent")
            .with_url("http://localhost:9000")
            .with_capabilities(streaming=True, push_notifications=False)
            .build()
        )

        assert card.capabilities.streaming is True
        assert card.capabilities.push_notifications is False

    def test_build_from_agent(self, mock_agent):
        """Test building card from agent."""
        card = (
            AgentCardBuilder()
            .with_url("http://localhost:9000")
            .from_agent(mock_agent)
            .build()
        )

        assert card.name == mock_agent.name
        assert card.description == mock_agent.description

    def test_build_with_skills(self):
        """Test building card with skills."""
        card = (
            AgentCardBuilder()
            .with_name("Skilled Agent")
            .with_url("http://localhost:9000")
            .with_skill(
                id="search",
                name="Search",
                description="Searches the web",
            )
            .with_skill(
                id="calculate",
                name="Calculate",
                description="Performs calculations",
            )
            .build()
        )

        assert len(card.skills) == 2
        assert card.skills[0].id == "search"
        assert card.skills[1].id == "calculate"

    def test_build_requires_name(self):
        """Test that name is required."""
        with pytest.raises(ValueError, match="name is required"):
            AgentCardBuilder().with_url("http://localhost:9000").build()

    def test_build_requires_url(self):
        """Test that URL is required."""
        with pytest.raises(ValueError, match="URL is required"):
            AgentCardBuilder().with_name("Agent").build()

    def test_build_with_provider(self):
        """Test building card with provider info."""
        card = (
            AgentCardBuilder()
            .with_name("Provider Agent")
            .with_url("http://localhost:9000")
            .with_provider(organization="Marie AI", url="https://marie-ai.com")
            .build()
        )

        assert card.provider.organization == "Marie AI"
        assert card.provider.url == "https://marie-ai.com"

    def test_build_with_modes(self):
        """Test building card with custom input/output modes."""
        card = (
            AgentCardBuilder()
            .with_name("Modes Agent")
            .with_url("http://localhost:9000")
            .with_input_modes(["text/plain", "application/json"])
            .with_output_modes(["text/plain", "image/png"])
            .build()
        )

        assert "text/plain" in card.default_input_modes
        assert "application/json" in card.default_input_modes
        assert "image/png" in card.default_output_modes
