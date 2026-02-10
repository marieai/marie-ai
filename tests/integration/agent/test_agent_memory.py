"""Integration tests for agent memory integration with Mem0.

Tests the memory hooks in ReactAgent, FunctionCallingAgent, ChatAgent,
and PlanAndExecuteAgent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from marie.agent import ChatAgent, FunctionCallingAgent, Message, ReactAgent
from marie.agent.agents.assistant import PlanAndExecuteAgent
from tests.integration.agent.conftest import (
    MockLLMWrapper,
    MockSearchTool,
    SequenceMockLLMWrapper,
)


def run_agent_to_completion(agent, messages, **kwargs):
    """Run an agent and collect all responses, passing kwargs."""
    all_responses = []
    for responses in agent.run(messages, **kwargs):
        all_responses = responses
    return all_responses


class MockMem0Memory:
    """Mock Mem0Memory for testing memory integration."""

    def __init__(self, enabled: bool = True, memories: Optional[List[Dict]] = None):
        self._enabled = enabled
        self._memories = memories or []
        self.search_calls: List[Dict] = []
        self.add_calls: List[Dict] = []

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def search(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict]:
        """Mock search that records calls and returns configured memories."""
        self.search_calls.append({
            "query": query,
            "user_id": user_id,
            "agent_id": agent_id,
            "limit": limit,
        })
        return self._memories

    def add(
        self,
        messages: List[Dict],
        user_id: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Mock add that records calls."""
        self.add_calls.append({
            "messages": messages,
            "user_id": user_id,
            "agent_id": agent_id,
            "metadata": metadata,
        })
        return {"id": "mock-memory-id"}


class TestReactAgentMemoryIntegration:
    """Test memory integration in ReactAgent."""

    def test_augment_with_memories_called_when_user_id_provided(self):
        """Test that _augment_with_memories is called when user_id is provided."""
        llm = MockLLMWrapper(responses=["Hello! I remember you."])
        agent = ReactAgent(llm=llm)

        mock_mem0 = MockMem0Memory(
            enabled=True,
            memories=[{"memory": "User prefers formal language"}],
        )
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Hello"}]
        run_agent_to_completion(agent, messages, user_id="test-user-123")

        assert len(mock_mem0.search_calls) == 1
        assert mock_mem0.search_calls[0]["user_id"] == "test-user-123"
        assert mock_mem0.search_calls[0]["query"] == "Hello"

    def test_store_interaction_called_after_response(self):
        """Test that _store_interaction is called after getting a response."""
        llm = MockLLMWrapper(responses=["Hello! How can I help you?"])
        agent = ReactAgent(llm=llm)

        mock_mem0 = MockMem0Memory(enabled=True)
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Hello"}]
        run_agent_to_completion(agent, messages, user_id="test-user-123")

        assert len(mock_mem0.add_calls) == 1
        assert mock_mem0.add_calls[0]["user_id"] == "test-user-123"
        # Check that the conversation was stored
        stored_messages = mock_mem0.add_calls[0]["messages"]
        assert any(m["role"] == "user" for m in stored_messages)
        assert any(m["role"] == "assistant" for m in stored_messages)

    def test_memory_not_used_when_user_id_not_provided(self):
        """Test that memory is not used when user_id is not provided."""
        llm = MockLLMWrapper(responses=["Hello!"])
        agent = ReactAgent(llm=llm)

        mock_mem0 = MockMem0Memory(enabled=True)
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Hello"}]
        run_agent_to_completion(agent, messages)  # No user_id

        assert len(mock_mem0.search_calls) == 0
        assert len(mock_mem0.add_calls) == 0

    def test_memory_not_used_when_disabled(self):
        """Test that memory is not used when disabled."""
        llm = MockLLMWrapper(responses=["Hello!"])
        agent = ReactAgent(llm=llm)

        mock_mem0 = MockMem0Memory(enabled=False)
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Hello"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        assert len(mock_mem0.search_calls) == 0
        assert len(mock_mem0.add_calls) == 0

    def test_agent_id_from_kwargs(self):
        """Test that agent_id from kwargs is used."""
        llm = MockLLMWrapper(responses=["Hello!"])
        agent = ReactAgent(llm=llm, name="default-agent")

        mock_mem0 = MockMem0Memory(enabled=True)
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Hello"}]
        run_agent_to_completion(
            agent, messages, user_id="test-user", agent_id="custom-agent"
        )

        assert mock_mem0.search_calls[0]["agent_id"] == "custom-agent"
        assert mock_mem0.add_calls[0]["agent_id"] == "custom-agent"

    def test_agent_id_falls_back_to_agent_name(self):
        """Test that agent_id falls back to agent name if not provided."""
        llm = MockLLMWrapper(responses=["Hello!"])
        agent = ReactAgent(llm=llm, name="my-react-agent")

        mock_mem0 = MockMem0Memory(enabled=True)
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Hello"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        assert mock_mem0.search_calls[0]["agent_id"] == "my-react-agent"
        assert mock_mem0.add_calls[0]["agent_id"] == "my-react-agent"

    def test_memory_storage_failure_does_not_crash_agent(self):
        """Test that memory storage failures are handled gracefully."""
        llm = MockLLMWrapper(responses=["Hello!"])
        agent = ReactAgent(llm=llm)

        mock_mem0 = MockMem0Memory(enabled=True)
        # Make add() raise an exception
        mock_mem0.add = MagicMock(side_effect=Exception("Storage failed"))
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Hello"}]
        # Should not raise
        responses = run_agent_to_completion(agent, messages, user_id="test-user")

        assert len(responses) > 0

    def test_memory_context_injected_into_messages(self):
        """Test that memory context is properly injected into messages."""
        llm = MockLLMWrapper(responses=["Hello John!"])
        agent = ReactAgent(llm=llm)

        mock_mem0 = MockMem0Memory(
            enabled=True,
            memories=[
                {"memory": "User's name is John"},
                {"memory": "User works as a software engineer"},
            ],
        )
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "What's my name?"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        # Check that the LLM received the memory context
        last_messages = llm.last_messages
        # Should have a system message with memories
        memory_found = False
        for msg in last_messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, str) and "User's name is John" in content:
                memory_found = True
                break
        assert memory_found, "Memory context should be injected into messages"


class TestFunctionCallingAgentMemoryIntegration:
    """Test memory integration in FunctionCallingAgent."""

    def test_augment_with_memories_called(self):
        """Test that memories are augmented in FunctionCallingAgent."""
        llm = MockLLMWrapper(responses=["Hello!"])
        agent = FunctionCallingAgent(llm=llm)

        mock_mem0 = MockMem0Memory(
            enabled=True,
            memories=[{"memory": "User likes Python"}],
        )
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Hello"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        assert len(mock_mem0.search_calls) == 1
        assert len(mock_mem0.add_calls) == 1

    def test_store_interaction_called(self):
        """Test that interactions are stored in FunctionCallingAgent."""
        llm = MockLLMWrapper(responses=["Here's your answer."])
        agent = FunctionCallingAgent(llm=llm)

        mock_mem0 = MockMem0Memory(enabled=True)
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Help me"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        assert len(mock_mem0.add_calls) == 1


class TestChatAgentMemoryIntegration:
    """Test memory integration in ChatAgent."""

    def test_augment_with_memories_called(self):
        """Test that memories are augmented in ChatAgent."""
        llm = MockLLMWrapper(responses=["Hello friend!"])
        agent = ChatAgent(llm=llm)

        mock_mem0 = MockMem0Memory(
            enabled=True,
            memories=[{"memory": "User prefers casual conversation"}],
        )
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Hi there"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        assert len(mock_mem0.search_calls) == 1

    def test_store_interaction_called(self):
        """Test that interactions are stored in ChatAgent."""
        llm = MockLLMWrapper(responses=["Nice to chat with you!"])
        agent = ChatAgent(llm=llm)

        mock_mem0 = MockMem0Memory(enabled=True)
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Let's chat"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        assert len(mock_mem0.add_calls) == 1


class TestPlanAndExecuteAgentMemoryIntegration:
    """Test memory integration in PlanAndExecuteAgent."""

    def test_augment_with_memories_called(self):
        """Test that memories are augmented in PlanAndExecuteAgent."""
        # Create a mock that returns FINAL ANSWER to complete quickly
        llm = MockLLMWrapper(responses=["FINAL ANSWER: Done!"])
        agent = PlanAndExecuteAgent(llm=llm, max_iterations=5)

        mock_mem0 = MockMem0Memory(
            enabled=True,
            memories=[{"memory": "User prefers detailed plans"}],
        )
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Plan something"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        assert len(mock_mem0.search_calls) == 1

    def test_store_interaction_called_on_final_answer(self):
        """Test that interactions are stored when FINAL ANSWER is reached."""
        llm = MockLLMWrapper(responses=["FINAL ANSWER: The task is complete."])
        agent = PlanAndExecuteAgent(llm=llm, max_iterations=5)

        mock_mem0 = MockMem0Memory(enabled=True)
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Complete this task"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        assert len(mock_mem0.add_calls) == 1


class TestMemoryWithToolCalling:
    """Test memory integration with tool calling workflows."""

    def test_memory_with_tool_calls(self):
        """Test memory works correctly with tool calling."""
        llm = SequenceMockLLMWrapper([
            {"name": "mock_search", "arguments": {"query": "AI"}, "content": "Searching..."},
            "Based on my search, here's what I found.",
        ])
        search_tool = MockSearchTool()
        agent = ReactAgent(
            llm=llm,
            function_list=[search_tool],
            max_iterations=5,
        )

        mock_mem0 = MockMem0Memory(
            enabled=True,
            memories=[{"memory": "User researching AI topics"}],
        )
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Search for AI"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        # Memory should be searched and stored
        assert len(mock_mem0.search_calls) == 1
        assert len(mock_mem0.add_calls) == 1

    def test_memory_stored_after_multiple_tool_calls(self):
        """Test that memory is stored after multiple tool calls complete."""
        llm = SequenceMockLLMWrapper([
            {"name": "mock_search", "arguments": {"query": "first"}, "content": ""},
            {"name": "mock_search", "arguments": {"query": "second"}, "content": ""},
            "Here are my findings from both searches.",
        ])
        search_tool = MockSearchTool()
        agent = ReactAgent(
            llm=llm,
            function_list=[search_tool],
            max_iterations=10,
        )

        mock_mem0 = MockMem0Memory(enabled=True)
        agent._mem0 = mock_mem0

        messages = [{"role": "user", "content": "Do multiple searches"}]
        run_agent_to_completion(agent, messages, user_id="test-user")

        # Should search once at start, store once at end
        assert len(mock_mem0.search_calls) == 1
        assert len(mock_mem0.add_calls) == 1
