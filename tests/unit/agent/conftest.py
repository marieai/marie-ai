"""Pytest fixtures for Marie agent framework tests."""

from typing import Any, Dict, List, Optional

import pytest

from marie.agent.backends.base import AgentBackend, AgentResult, BackendConfig
from marie.agent.message import Message
from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput


class MockTool(AgentTool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", return_value: str = "mock result"):
        self._name = name
        self._return_value = return_value
        self._metadata = ToolMetadata(
            name=name,
            description=f"Mock tool: {name}",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input value"},
                },
                "required": ["input"],
            },
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Mock tool: {self._name}"

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, **kwargs: Any) -> ToolOutput:
        return ToolOutput(
            content=self._return_value,
            tool_name=self._name,
            raw_output={"input": kwargs.get("input"), "result": self._return_value},
        )

    async def acall(self, **kwargs: Any) -> ToolOutput:
        return self.call(**kwargs)


class MockBackend(AgentBackend):
    """Mock backend for testing."""

    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        response: str = "Mock response",
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)
        self._response = response

    async def run(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        return AgentResult(
            output=Message.assistant(self._response),
            messages=[m.model_dump() for m in messages],
            iterations=1,
        )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        return []


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    return MockTool()


@pytest.fixture
def mock_backend():
    """Create a mock backend."""
    return MockBackend()


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        Message.system("You are a helpful assistant."),
        Message.user("Hello, how are you?"),
        Message.assistant("I'm doing well, thank you!"),
    ]


@pytest.fixture
def sample_user_message():
    """Create a sample user message."""
    return Message.user("What is 2 + 2?")


@pytest.fixture
def sample_tools():
    """Create sample tools dictionary."""
    return {
        "calculator": MockTool(name="calculator", return_value="4"),
        "search": MockTool(name="search", return_value="Search results"),
    }
