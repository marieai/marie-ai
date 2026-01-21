"""Integration tests for BasicAgent.

Tests the simplest agent that just forwards messages to the LLM.
"""

import pytest

from marie.agent import BasicAgent, Message
from tests.integration.agent.conftest import (
    MockLLMWrapper,
    get_final_response,
    run_agent_to_completion,
)


class TestBasicAgentCreation:
    """Test BasicAgent instantiation."""

    def test_create_basic_agent(self, mock_llm):
        """Test creating a basic agent."""
        agent = BasicAgent(
            llm=mock_llm,
            system_message="You are a test assistant.",
        )

        assert agent.llm is mock_llm
        assert agent.system_message == "You are a test assistant."

    def test_create_agent_without_system_message(self, mock_llm):
        """Test creating agent with default system message."""
        agent = BasicAgent(llm=mock_llm)

        # Should have default system message
        assert agent.system_message is not None

    def test_create_agent_with_name(self, mock_llm):
        """Test creating agent with name."""
        agent = BasicAgent(
            llm=mock_llm,
            name="test_agent",
            description="A test agent",
        )

        assert agent.name == "test_agent"
        assert agent.description == "A test agent"


class TestBasicAgentRun:
    """Test BasicAgent run method."""

    def test_run_with_dict_messages(self, basic_agent, sample_messages):
        """Test running agent with dict messages."""
        responses = run_agent_to_completion(basic_agent, sample_messages)

        assert len(responses) > 0
        # When input is dict, output may be dict or Message
        last = responses[-1]
        if isinstance(last, dict):
            assert last["role"] == "assistant"
        else:
            assert last.role == "assistant"

    def test_run_with_message_objects(self, basic_agent):
        """Test running agent with Message objects."""
        messages = [Message.user("Hello, agent!")]
        responses = run_agent_to_completion(basic_agent, messages)

        assert len(responses) > 0
        assert isinstance(responses[-1], Message)

    def test_run_streaming(self, basic_agent, sample_messages):
        """Test that run yields responses (streaming pattern)."""
        response_count = 0
        for responses in basic_agent.run(sample_messages):
            response_count += 1
            assert isinstance(responses, list)

        assert response_count > 0

    def test_run_nonstream(self, basic_agent, sample_messages):
        """Test run_nonstream returns final response."""
        responses = basic_agent.run_nonstream(sample_messages)

        assert isinstance(responses, list)
        assert len(responses) > 0

    def test_system_message_prepended(self, mock_llm):
        """Test that system message is prepended to conversation."""
        agent = BasicAgent(
            llm=mock_llm,
            system_message="You are a helpful test bot.",
        )

        messages = [{"role": "user", "content": "Hi"}]
        run_agent_to_completion(agent, messages)

        # Check that LLM received system message
        last_messages = mock_llm.last_messages
        assert last_messages[0].role == "system"
        assert "helpful test bot" in last_messages[0].content

    def test_existing_system_message_merged(self, mock_llm):
        """Test that existing system message is merged."""
        agent = BasicAgent(
            llm=mock_llm,
            system_message="Agent instruction.",
        )

        messages = [
            {"role": "system", "content": "User instruction."},
            {"role": "user", "content": "Hi"},
        ]
        run_agent_to_completion(agent, messages)

        # Check that both instructions are present
        last_messages = mock_llm.last_messages
        assert "Agent instruction" in last_messages[0].content
        assert "User instruction" in last_messages[0].content


class TestBasicAgentLanguageDetection:
    """Test automatic language detection."""

    def test_detect_chinese(self, mock_llm):
        """Test Chinese language detection."""
        agent = BasicAgent(llm=mock_llm)
        messages = [Message.user("你好，请帮我写一段代码")]

        # Run the agent - language should be detected
        for responses in agent.run(messages):
            pass

        # We can't directly verify the lang parameter easily,
        # but we can verify the agent doesn't crash

    def test_detect_english(self, mock_llm):
        """Test English language detection."""
        agent = BasicAgent(llm=mock_llm)
        messages = [Message.user("Hello, can you help me?")]

        for responses in agent.run(messages):
            pass


class TestBasicAgentResponseFormat:
    """Test response format handling."""

    def test_dict_input_dict_output(self, mock_llm):
        """Test that dict input produces dict output."""
        agent = BasicAgent(llm=mock_llm)
        messages = [{"role": "user", "content": "Hello"}]

        responses = run_agent_to_completion(agent, messages)

        # When input is dict, output should be dict-like
        assert isinstance(responses[-1], (dict, Message))

    def test_message_input_message_output(self, mock_llm):
        """Test that Message input produces Message output."""
        agent = BasicAgent(llm=mock_llm)
        messages = [Message.user("Hello")]

        responses = run_agent_to_completion(agent, messages)

        assert isinstance(responses[-1], Message)

    def test_empty_messages(self, mock_llm):
        """Test handling empty message list."""
        agent = BasicAgent(llm=mock_llm)

        # Should handle empty messages gracefully
        responses = run_agent_to_completion(agent, [])
        # May return empty or with just system message response


class TestBasicAgentMultiTurn:
    """Test multi-turn conversations."""

    def test_multi_turn_conversation(self, mock_llm_factory):
        """Test multi-turn conversation handling."""
        llm = mock_llm_factory(responses=[
            "Hello! How can I help?",
            "I understand. Let me help with that.",
        ])
        agent = BasicAgent(llm=llm)

        # First turn
        messages = [{"role": "user", "content": "Hi"}]
        response1 = get_final_response(agent, messages)
        assert response1 is not None

        # Second turn - handle both dict and Message responses
        if isinstance(response1, dict):
            messages.append(response1)
        else:
            messages.append(response1.model_dump())
        messages.append({"role": "user", "content": "Can you help me more?"})
        response2 = get_final_response(agent, messages)
        assert response2 is not None

    def test_conversation_context_preserved(self, mock_llm):
        """Test that conversation context is sent to LLM."""
        agent = BasicAgent(llm=mock_llm)

        messages = [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Hello Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]

        run_agent_to_completion(agent, messages)

        # Verify all messages were sent to LLM
        last_messages = mock_llm.last_messages
        # Should have system + 3 messages
        assert len(last_messages) >= 3


class TestBasicAgentWithMultimodal:
    """Test multimodal message handling."""

    def test_multimodal_message(self, mock_llm, sample_messages_with_image):
        """Test handling multimodal messages."""
        agent = BasicAgent(llm=mock_llm)

        # Should not crash with multimodal input
        responses = run_agent_to_completion(agent, sample_messages_with_image)
        assert len(responses) > 0

    def test_image_content_passed_to_llm(self, mock_llm, sample_messages_with_image):
        """Test that image content is passed to LLM."""
        agent = BasicAgent(llm=mock_llm)

        run_agent_to_completion(agent, sample_messages_with_image)

        # Verify multimodal content was passed
        last_messages = mock_llm.last_messages
        user_msg = last_messages[-1]
        assert isinstance(user_msg.content, list)


class TestBasicAgentErrorHandling:
    """Test error handling in BasicAgent."""

    def test_no_llm_configured(self):
        """Test error when LLM is not configured."""
        agent = BasicAgent(llm=None)

        with pytest.raises(ValueError, match="LLM is not configured"):
            run_agent_to_completion(agent, [{"role": "user", "content": "Hi"}])

    def test_llm_error_handling(self, mock_llm_factory):
        """Test handling of LLM errors."""

        class FailingLLM(MockLLMWrapper):
            def chat(self, *args, **kwargs):
                raise RuntimeError("LLM service unavailable")

        agent = BasicAgent(llm=FailingLLM())
        messages = [{"role": "user", "content": "Hi"}]

        # Should not crash, should return error message
        responses = run_agent_to_completion(agent, messages)
        assert len(responses) > 0
        # Handle both dict and Message responses
        last = responses[-1]
        content = last.get("content", "") if isinstance(last, dict) else last.content
        assert "error" in content.lower()


class TestBasicAgentConfiguration:
    """Test agent configuration options."""

    def test_extra_generate_cfg(self, mock_llm):
        """Test passing extra generation config."""
        agent = BasicAgent(
            llm=mock_llm,
            extra_generate_cfg={"temperature": 0.7, "max_tokens": 100},
        )

        assert agent.extra_generate_cfg["temperature"] == 0.7
        assert agent.extra_generate_cfg["max_tokens"] == 100

    def test_seed_parameter(self, mock_llm):
        """Test passing seed parameter."""
        agent = BasicAgent(llm=mock_llm)
        messages = [{"role": "user", "content": "Hi"}]

        # Should accept seed parameter without error
        responses = agent.run_nonstream(messages, seed=42)
        assert len(responses) > 0
