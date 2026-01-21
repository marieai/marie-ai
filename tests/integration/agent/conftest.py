"""Fixtures for agent integration tests.

Provides mocked LLM wrappers, tool fixtures, and agent configurations
for testing the agent framework without requiring actual LLM inference.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from unittest.mock import MagicMock

import pytest

from marie.agent import (
    TOOL_REGISTRY,
    AgentTool,
    AssistantAgent,
    BasicAgent,
    ChatAgent,
    FunctionTool,
    Message,
    PlanningAgent,
    ToolMetadata,
    ToolOutput,
    register_tool,
)
from marie.agent.llm_wrapper import BaseLLMWrapper

# =============================================================================
# Mock LLM Wrapper
# =============================================================================


class MockLLMWrapper(BaseLLMWrapper):
    """Mock LLM wrapper for testing without actual LLM inference.

    Supports configurable responses and function calling simulation.
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        function_call_responses: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ):
        """Initialize mock LLM wrapper.

        Args:
            responses: List of text responses to return in order
            function_call_responses: List of function call dicts to return
            stream: Whether to simulate streaming
        """
        self._responses = responses or ["This is a mock response."]
        self._function_call_responses = function_call_responses or []
        self._stream = stream
        self._call_count = 0
        self._chat_history: List[List[Message]] = []

    def chat(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = False,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Iterator[List[Message]]:
        """Mock chat method.

        Returns configured responses in order, cycling if exhausted.
        """
        self._chat_history.append(messages)
        idx = self._call_count % max(
            len(self._responses), len(self._function_call_responses) or 1
        )
        self._call_count += 1

        # Check if we should return a function call
        if self._function_call_responses and idx < len(self._function_call_responses):
            fc = self._function_call_responses[idx]
            from marie.agent.message import FunctionCall

            response = Message.assistant(
                content=fc.get("content", ""),
                function_call=FunctionCall(
                    name=fc["name"], arguments=fc.get("arguments", {})
                ),
            )
        else:
            # Return text response
            text_idx = idx % len(self._responses)
            response = Message.assistant(content=self._responses[text_idx])

        yield [response]

    async def achat(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Async mock chat method."""
        self._chat_history.append(messages)
        idx = self._call_count % len(self._responses)
        self._call_count += 1
        return Message.assistant(content=self._responses[idx])

    @property
    def call_count(self) -> int:
        """Number of times chat was called."""
        return self._call_count

    @property
    def last_messages(self) -> Optional[List[Message]]:
        """Messages from the last chat call."""
        return self._chat_history[-1] if self._chat_history else None

    def reset(self) -> None:
        """Reset call count and history."""
        self._call_count = 0
        self._chat_history = []


class SequenceMockLLMWrapper(MockLLMWrapper):
    """Mock LLM that returns different responses for each call in sequence.

    Useful for testing multi-turn conversations and tool calling loops.
    """

    def __init__(self, response_sequence: List[Union[str, Dict[str, Any]]]):
        """Initialize with a sequence of responses.

        Args:
            response_sequence: List of responses. Can be:
                - str: Text response
                - dict with "name" and "arguments": Function call
        """
        super().__init__()
        self._response_sequence = response_sequence

    def chat(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = False,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Iterator[List[Message]]:
        """Return next response in sequence."""
        self._chat_history.append(messages)

        if self._call_count >= len(self._response_sequence):
            # Exhausted sequence, return final text response
            response = Message.assistant(content="I have completed the task.")
        else:
            item = self._response_sequence[self._call_count]
            if isinstance(item, str):
                response = Message.assistant(content=item)
            elif isinstance(item, dict) and "name" in item:
                from marie.agent.message import FunctionCall

                response = Message.assistant(
                    content=item.get("content", ""),
                    function_call=FunctionCall(
                        name=item["name"], arguments=item.get("arguments", {})
                    ),
                )
            else:
                response = Message.assistant(content=str(item))

        self._call_count += 1
        yield [response]


# =============================================================================
# Test Tools
# =============================================================================


class MockSearchTool(AgentTool):
    """Mock search tool for testing."""

    def __init__(self, results: Optional[List[str]] = None):
        self._results = results or ["Result 1", "Result 2", "Result 3"]

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="mock_search",
            description="Search for information. Returns relevant results.",
            fn_schema=None,
        )

    @property
    def name(self) -> str:
        return "mock_search"

    def call(self, **kwargs) -> ToolOutput:
        """Execute search with keyword arguments."""
        query = kwargs.get("query", kwargs.get("input", ""))
        result = json.dumps({"query": query, "results": self._results})
        return ToolOutput(
            content=result,
            tool_name=self.name,
            raw_input=kwargs,
            raw_output=result,
            is_error=False,
        )


class MockCalculatorTool(AgentTool):
    """Mock calculator tool for testing."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="mock_calculator",
            description="Perform mathematical calculations.",
            fn_schema=None,
        )

    @property
    def name(self) -> str:
        return "mock_calculator"

    def call(self, **kwargs) -> ToolOutput:
        """Execute calculation with keyword arguments."""
        expression = kwargs.get("expression", kwargs.get("input", "0"))
        try:
            # Safe evaluation for testing
            result = eval(expression, {"__builtins__": {}}, {})
            content = json.dumps({"expression": expression, "result": result})
            is_error = False
        except Exception as e:
            content = json.dumps({"error": str(e)})
            is_error = True

        return ToolOutput(
            content=content,
            tool_name=self.name,
            raw_input=kwargs,
            raw_output=content,
            is_error=is_error,
        )


class FailingTool(AgentTool):
    """Tool that always fails, for error handling tests."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="failing_tool",
            description="A tool that always fails.",
            fn_schema=None,
        )

    @property
    def name(self) -> str:
        return "failing_tool"

    def call(self, **kwargs) -> ToolOutput:
        """This tool always raises an error."""
        raise RuntimeError("This tool always fails!")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Provide a basic mock LLM wrapper."""
    return MockLLMWrapper(responses=["Hello! I am a helpful assistant."])


@pytest.fixture
def mock_llm_factory():
    """Factory for creating mock LLM wrappers with custom responses."""

    def factory(
        responses: Optional[List[str]] = None,
        function_call_responses: Optional[List[Dict[str, Any]]] = None,
    ) -> MockLLMWrapper:
        return MockLLMWrapper(
            responses=responses or ["Default mock response."],
            function_call_responses=function_call_responses,
        )

    return factory


@pytest.fixture
def sequence_llm_factory():
    """Factory for creating sequence mock LLM wrappers."""

    def factory(response_sequence: List[Union[str, Dict[str, Any]]]) -> SequenceMockLLMWrapper:
        return SequenceMockLLMWrapper(response_sequence)

    return factory


@pytest.fixture
def mock_search_tool():
    """Provide a mock search tool."""
    return MockSearchTool()


@pytest.fixture
def mock_calculator_tool():
    """Provide a mock calculator tool."""
    return MockCalculatorTool()


@pytest.fixture
def failing_tool():
    """Provide a tool that always fails."""
    return FailingTool()


@pytest.fixture
def clean_tool_registry():
    """Provide a clean tool registry, restoring original state after test."""
    # Save original tools
    original_tools = dict(TOOL_REGISTRY._tools)

    yield TOOL_REGISTRY

    # Restore original tools
    TOOL_REGISTRY._tools = original_tools


@pytest.fixture
def basic_agent(mock_llm):
    """Provide a basic agent without tools."""
    return BasicAgent(
        llm=mock_llm,
        system_message="You are a helpful test assistant.",
    )


@pytest.fixture
def assistant_agent(mock_llm, mock_search_tool, mock_calculator_tool):
    """Provide an assistant agent with tools."""
    return AssistantAgent(
        llm=mock_llm,
        function_list=[mock_search_tool, mock_calculator_tool],
        system_message="You are a helpful assistant with search and calculator tools.",
        max_iterations=5,
    )


@pytest.fixture
def chat_agent(mock_llm):
    """Provide a chat agent."""
    return ChatAgent(
        llm=mock_llm,
        system_message="You are a friendly chat assistant.",
    )


@pytest.fixture
def sample_messages():
    """Provide sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def sample_messages_with_image():
    """Provide sample multimodal messages."""
    from marie.agent.message import ContentItem

    return [
        Message.user(
            [
                ContentItem(text="What do you see in this image?"),
                ContentItem(image="https://example.com/image.jpg"),
            ]
        ),
    ]


@pytest.fixture
def sample_conversation():
    """Provide a multi-turn conversation for testing."""
    return [
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "What's 2 + 2?"},
    ]


# =============================================================================
# Utility Functions
# =============================================================================


def run_agent_to_completion(
    agent, messages: List[Union[Dict, Message]]
) -> List[Message]:
    """Run an agent and collect all responses."""
    all_responses = []
    for responses in agent.run(messages):
        all_responses = responses
    return all_responses


def get_final_response(agent, messages: List[Union[Dict, Message]]) -> Message:
    """Run an agent and get the final response."""
    responses = run_agent_to_completion(agent, messages)
    return responses[-1] if responses else None
