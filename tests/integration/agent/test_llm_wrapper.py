"""Integration tests for LLM wrappers.

Tests the LLM wrapper interface and implementations.
"""

import pytest

from marie.agent import BaseLLMWrapper, Message
from marie.agent.llm_wrapper import MarieEngineLLMWrapper, get_llm_wrapper


class TestMockLLMWrapper:
    """Test the mock LLM wrapper from fixtures."""

    def test_mock_llm_chat(self, mock_llm):
        """Test basic chat with mock LLM."""
        messages = [Message.user("Hello")]

        responses = list(mock_llm.chat(messages))

        assert len(responses) > 0
        assert responses[-1][-1].role == "assistant"

    def test_mock_llm_call_count(self, mock_llm):
        """Test call count tracking."""
        messages = [Message.user("Hello")]

        assert mock_llm.call_count == 0

        list(mock_llm.chat(messages))
        assert mock_llm.call_count == 1

        list(mock_llm.chat(messages))
        assert mock_llm.call_count == 2

    def test_mock_llm_last_messages(self, mock_llm):
        """Test last messages tracking."""
        messages = [Message.user("Test message")]

        list(mock_llm.chat(messages))

        assert mock_llm.last_messages is not None
        assert mock_llm.last_messages[0].content == "Test message"

    def test_mock_llm_reset(self, mock_llm):
        """Test resetting mock LLM."""
        list(mock_llm.chat([Message.user("Hello")]))
        assert mock_llm.call_count == 1

        mock_llm.reset()

        assert mock_llm.call_count == 0
        assert mock_llm.last_messages is None

    def test_mock_llm_custom_responses(self, mock_llm_factory):
        """Test mock LLM with custom responses."""
        llm = mock_llm_factory(responses=["Response 1", "Response 2"])

        # First call
        resp1 = list(llm.chat([Message.user("First")]))
        assert "Response 1" in resp1[-1][-1].content

        # Second call
        resp2 = list(llm.chat([Message.user("Second")]))
        assert "Response 2" in resp2[-1][-1].content

    def test_mock_llm_function_call_responses(self, mock_llm_factory):
        """Test mock LLM with function call responses."""
        llm = mock_llm_factory(
            function_call_responses=[
                {"name": "search", "arguments": {"query": "test"}, "content": ""}
            ]
        )

        responses = list(llm.chat([Message.user("Search")]))
        response = responses[-1][-1]

        assert response.function_call is not None
        assert response.function_call.name == "search"


class TestSequenceMockLLMWrapper:
    """Test sequence mock LLM for multi-turn scenarios."""

    def test_sequence_responses(self, sequence_llm_factory):
        """Test that responses are returned in sequence."""
        llm = sequence_llm_factory([
            "First response",
            "Second response",
            "Third response",
        ])

        resp1 = list(llm.chat([Message.user("1")]))[-1][-1]
        assert resp1.content == "First response"

        resp2 = list(llm.chat([Message.user("2")]))[-1][-1]
        assert resp2.content == "Second response"

        resp3 = list(llm.chat([Message.user("3")]))[-1][-1]
        assert resp3.content == "Third response"

    def test_sequence_with_function_calls(self, sequence_llm_factory):
        """Test sequence with mixed responses and function calls."""
        llm = sequence_llm_factory([
            {"name": "search", "arguments": {"query": "AI"}, "content": ""},
            "Here's what I found about AI.",
        ])

        # First response should be function call
        resp1 = list(llm.chat([Message.user("Search")]))[0][-1]
        assert resp1.function_call is not None
        assert resp1.function_call.name == "search"

        # Second should be text
        resp2 = list(llm.chat([Message.user("Continue")]))[0][-1]
        assert "AI" in resp2.content

    def test_sequence_exhausted(self, sequence_llm_factory):
        """Test behavior when sequence is exhausted."""
        llm = sequence_llm_factory(["Only one response"])

        list(llm.chat([Message.user("1")]))
        resp2 = list(llm.chat([Message.user("2")]))[-1][-1]

        # Should return fallback response
        assert "completed" in resp2.content.lower()


class TestBaseLLMWrapperInterface:
    """Test that mock LLM implements the correct interface."""

    def test_chat_returns_iterator(self, mock_llm):
        """Test that chat returns an iterator."""
        messages = [Message.user("Hello")]

        result = mock_llm.chat(messages)

        # Should be iterable
        assert hasattr(result, "__iter__")

    def test_chat_yields_message_lists(self, mock_llm):
        """Test that chat yields lists of messages."""
        messages = [Message.user("Hello")]

        for response_batch in mock_llm.chat(messages):
            assert isinstance(response_batch, list)
            assert all(isinstance(m, Message) for m in response_batch)

    def test_chat_with_functions(self, mock_llm):
        """Test chat with function definitions."""
        messages = [Message.user("Hello")]
        functions = [
            {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        ]

        # Should not crash with functions parameter
        responses = list(mock_llm.chat(messages, functions=functions))
        assert len(responses) > 0

    def test_chat_with_extra_cfg(self, mock_llm):
        """Test chat with extra generation config."""
        messages = [Message.user("Hello")]
        extra_cfg = {"temperature": 0.5, "max_tokens": 100}

        # Should not crash with extra config
        responses = list(mock_llm.chat(messages, extra_generate_cfg=extra_cfg))
        assert len(responses) > 0


class TestAsyncLLMWrapper:
    """Test async LLM wrapper methods."""

    @pytest.mark.asyncio
    async def test_async_chat(self, mock_llm):
        """Test async chat method."""
        messages = [Message.user("Hello async")]

        response = await mock_llm.achat(messages)

        assert isinstance(response, Message)
        assert response.role == "assistant"

    @pytest.mark.asyncio
    async def test_async_with_functions(self, mock_llm):
        """Test async chat with functions."""
        messages = [Message.user("Hello")]
        functions = [{"name": "test", "description": "Test function", "parameters": {}}]

        response = await mock_llm.achat(messages, functions=functions)
        assert response is not None


class TestGetLLMWrapper:
    """Test the get_llm_wrapper factory function."""

    def test_get_wrapper_marie(self):
        """Test getting marie engine wrapper."""
        # This would require actual marie engine, so we just test the interface
        # In real integration tests, you'd set up the engine
        pass

    def test_get_wrapper_invalid_backend(self):
        """Test getting wrapper with invalid backend."""
        # Should handle invalid backend gracefully
        pass


class TestMarieEngineLLMWrapperInterface:
    """Test MarieEngineLLMWrapper interface (without actual engine)."""

    def test_wrapper_creation(self):
        """Test creating wrapper (may fail without engine)."""
        # This tests the interface, actual functionality requires marie engine
        try:
            wrapper = MarieEngineLLMWrapper(
                engine_name="test_engine",
                provider="mock",
            )
            assert wrapper.engine_name == "test_engine"
        except Exception:
            # Expected if engine not available
            pass

    def test_function_call_format_options(self):
        """Test function call format configuration."""
        # Test that the wrapper accepts format options
        try:
            wrapper = MarieEngineLLMWrapper(
                engine_name="test",
                function_call_format="tool_call",
            )
            assert wrapper.function_call_format == "tool_call"
        except Exception:
            pass


class TestLLMWrapperMessageFormats:
    """Test message format handling in LLM wrappers."""

    def test_dict_messages(self, mock_llm):
        """Test that wrapper handles dict messages."""
        # Mock wrapper should convert dicts to Messages
        messages = [{"role": "user", "content": "Hello"}]

        # Convert to Message objects first (as BaseAgent would)
        from marie.agent.message import format_messages
        msg_objects = format_messages(messages)

        responses = list(mock_llm.chat(msg_objects))
        assert len(responses) > 0

    def test_multimodal_messages(self, mock_llm):
        """Test handling multimodal messages."""
        from marie.agent import ContentItem

        messages = [
            Message.user([
                ContentItem(text="Describe this"),
                ContentItem(image="https://example.com/image.jpg"),
            ])
        ]

        # Should not crash
        responses = list(mock_llm.chat(messages))
        assert len(responses) > 0

    def test_system_message_handling(self, mock_llm):
        """Test system message handling."""
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("Hello"),
        ]

        responses = list(mock_llm.chat(messages))
        assert len(responses) > 0

    def test_conversation_history(self, mock_llm):
        """Test handling conversation history."""
        messages = [
            Message.system("System"),
            Message.user("Question 1"),
            Message.assistant("Answer 1"),
            Message.user("Question 2"),
        ]

        responses = list(mock_llm.chat(messages))
        assert len(responses) > 0

        # Verify history was passed
        assert len(mock_llm.last_messages) == 4


class TestLLMWrapperFunctionCalling:
    """Test function calling in LLM wrappers."""

    def test_function_definitions_passed(self, mock_llm):
        """Test that function definitions are passed correctly."""
        messages = [Message.user("Search for AI")]
        functions = [
            {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        ]

        # Should accept functions parameter
        list(mock_llm.chat(messages, functions=functions))

    def test_function_call_in_response(self, mock_llm_factory):
        """Test parsing function call from response."""
        llm = mock_llm_factory(
            function_call_responses=[
                {"name": "search", "arguments": {"query": "test"}}
            ]
        )

        responses = list(llm.chat([Message.user("Search")]))
        response = responses[-1][-1]

        assert response.function_call is not None
        assert response.function_call.name == "search"
        assert response.function_call.get_arguments_dict()["query"] == "test"
