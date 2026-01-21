"""Integration tests for AssistantAgent.

Tests the ReAct-style agent with tool calling capabilities.
"""

import json

import pytest

from marie.agent import AssistantAgent, FunctionCall, Message, register_tool
from tests.integration.agent.conftest import (
    FailingTool,
    MockCalculatorTool,
    MockLLMWrapper,
    MockSearchTool,
    SequenceMockLLMWrapper,
    get_final_response,
    run_agent_to_completion,
)


class TestAssistantAgentCreation:
    """Test AssistantAgent instantiation."""

    def test_create_assistant_agent(self, mock_llm, mock_search_tool):
        """Test creating an assistant agent."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
            system_message="You are a helpful assistant.",
        )

        assert agent.llm is mock_llm
        assert "mock_search" in agent.function_map

    def test_create_with_multiple_tools(self, mock_llm, mock_search_tool, mock_calculator_tool):
        """Test creating agent with multiple tools."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool, mock_calculator_tool],
        )

        assert len(agent.function_map) == 2
        assert "mock_search" in agent.function_map
        assert "mock_calculator" in agent.function_map

    def test_create_with_max_iterations(self, mock_llm):
        """Test creating agent with custom max iterations."""
        agent = AssistantAgent(
            llm=mock_llm,
            max_iterations=20,
        )

        assert agent.max_iterations == 20

    def test_default_react_prompt(self, mock_llm):
        """Test default ReAct prompt is used."""
        agent = AssistantAgent(llm=mock_llm)

        # Should have default ReAct prompt
        assert "tool" in agent.system_message.lower()

    def test_custom_system_message(self, mock_llm):
        """Test custom system message overrides default."""
        agent = AssistantAgent(
            llm=mock_llm,
            system_message="Custom message",
        )

        assert agent.system_message == "Custom message"


class TestAssistantAgentToolManagement:
    """Test dynamic tool management."""

    def test_add_tool(self, mock_llm, mock_search_tool, mock_calculator_tool):
        """Test adding a tool dynamically."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        assert "mock_calculator" not in agent.function_map

        agent.add_tool(mock_calculator_tool)
        assert "mock_calculator" in agent.function_map

    def test_remove_tool(self, mock_llm, mock_search_tool, mock_calculator_tool):
        """Test removing a tool."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool, mock_calculator_tool],
        )

        result = agent.remove_tool("mock_search")
        assert result is True
        assert "mock_search" not in agent.function_map

    def test_remove_nonexistent_tool(self, mock_llm, mock_search_tool):
        """Test removing a tool that doesn't exist."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        result = agent.remove_tool("nonexistent")
        assert result is False

    def test_get_tool_definitions(self, mock_llm, mock_search_tool):
        """Test getting tool definitions."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        definitions = agent._get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "mock_search"


class TestAssistantAgentWithoutTools:
    """Test AssistantAgent behavior without tools."""

    def test_run_without_tools(self, mock_llm):
        """Test running agent without any tools."""
        agent = AssistantAgent(llm=mock_llm)
        messages = [{"role": "user", "content": "Hello"}]

        responses = run_agent_to_completion(agent, messages)

        assert len(responses) > 0
        # Handle both dict and Message responses
        last = responses[-1]
        if isinstance(last, dict):
            assert last["role"] == "assistant"
        else:
            assert last.role == "assistant"

    def test_no_tool_calling_when_no_tools(self, mock_llm):
        """Test that tool calling is skipped when no tools available."""
        agent = AssistantAgent(llm=mock_llm)
        messages = [{"role": "user", "content": "Search for something"}]

        # Should complete without trying to call tools
        responses = run_agent_to_completion(agent, messages)
        assert len(responses) > 0


class TestAssistantAgentToolCalling:
    """Test tool calling workflow."""

    def test_detect_tool_call(self, mock_llm, mock_search_tool):
        """Test tool call detection from message."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        msg = Message.assistant(
            content="Let me search for that.",
            function_call=FunctionCall(name="mock_search", arguments={"query": "test"}),
        )

        has_call, name, args, text = agent._detect_tool_call(msg)

        assert has_call is True
        assert name == "mock_search"
        assert "query" in args

    def test_detect_no_tool_call(self, mock_llm, mock_search_tool):
        """Test detection when no tool call present."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        msg = Message.assistant(content="Just a regular response.")
        has_call, name, args, text = agent._detect_tool_call(msg)

        assert has_call is False
        assert text == "Just a regular response."

    def test_call_tool(self, mock_llm, mock_search_tool):
        """Test calling a tool directly."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        result = agent._call_tool("mock_search", '{"query": "test"}')

        assert isinstance(result, str)
        data = json.loads(result)
        assert data["query"] == "test"
        assert "results" in data

    def test_call_nonexistent_tool(self, mock_llm, mock_search_tool):
        """Test calling a tool that doesn't exist."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        result = agent._call_tool("nonexistent", "{}")

        assert "does not exist" in result


class TestAssistantAgentReActLoop:
    """Test the ReAct reasoning loop."""

    def test_single_tool_call_loop(self, sequence_llm_factory, mock_search_tool):
        """Test a single iteration tool call loop."""
        # Sequence: tool call -> final response
        llm = sequence_llm_factory([
            {"name": "mock_search", "arguments": {"query": "AI"}, "content": "Searching..."},
            "Based on my search, AI is artificial intelligence.",
        ])

        agent = AssistantAgent(
            llm=llm,
            function_list=[mock_search_tool],
            max_iterations=5,
        )

        messages = [{"role": "user", "content": "What is AI?"}]
        responses = run_agent_to_completion(agent, messages)

        # Should have called LLM twice (once for tool call, once for response)
        assert llm.call_count == 2

        # Final response should be text - handle both dict and Message
        final = responses[-1]
        content = final.get("content", "") if isinstance(final, dict) else final.content
        assert "artificial intelligence" in content.lower()

    def test_multiple_tool_calls(self, sequence_llm_factory, mock_search_tool, mock_calculator_tool):
        """Test multiple tool calls in sequence."""
        llm = sequence_llm_factory([
            {"name": "mock_search", "arguments": {"query": "math"}, "content": "Searching..."},
            {"name": "mock_calculator", "arguments": {"expression": "2+2"}, "content": "Calculating..."},
            "The answer is 4.",
        ])

        agent = AssistantAgent(
            llm=llm,
            function_list=[mock_search_tool, mock_calculator_tool],
            max_iterations=10,
        )

        messages = [{"role": "user", "content": "Search for math and calculate 2+2"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 3
        # Handle both dict and Message responses
        last = responses[-1]
        content = last.get("content", "") if isinstance(last, dict) else last.content
        assert "4" in content

    def test_max_iterations_limit(self, sequence_llm_factory, mock_search_tool):
        """Test that max iterations limits the loop."""
        # Create infinite tool call loop
        llm = sequence_llm_factory([
            {"name": "mock_search", "arguments": {"query": "1"}, "content": ""},
            {"name": "mock_search", "arguments": {"query": "2"}, "content": ""},
            {"name": "mock_search", "arguments": {"query": "3"}, "content": ""},
            {"name": "mock_search", "arguments": {"query": "4"}, "content": ""},
            {"name": "mock_search", "arguments": {"query": "5"}, "content": ""},
        ])

        agent = AssistantAgent(
            llm=llm,
            function_list=[mock_search_tool],
            max_iterations=3,
        )

        messages = [{"role": "user", "content": "Keep searching"}]
        run_agent_to_completion(agent, messages)

        # Should stop at max_iterations
        assert llm.call_count <= 3

    def test_no_tool_call_exits_loop(self, sequence_llm_factory, mock_search_tool):
        """Test that no tool call exits the loop immediately."""
        llm = sequence_llm_factory([
            "I don't need to use any tools for this.",
        ])

        agent = AssistantAgent(
            llm=llm,
            function_list=[mock_search_tool],
            max_iterations=10,
        )

        messages = [{"role": "user", "content": "Just say hello"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 1
        # Handle both dict and Message responses
        last = responses[-1]
        content = last.get("content", "") if isinstance(last, dict) else last.content
        assert "don't need" in content


class TestAssistantAgentToolErrorHandling:
    """Test error handling with tools."""

    def test_tool_failure_recovery(self, sequence_llm_factory, failing_tool):
        """Test that agent recovers from tool failures."""
        llm = sequence_llm_factory([
            {"name": "failing_tool", "arguments": {}, "content": "Let me try this tool"},
            "The tool failed, but I can still help.",
        ])

        agent = AssistantAgent(
            llm=llm,
            function_list=[failing_tool],
            max_iterations=5,
        )

        messages = [{"role": "user", "content": "Use the failing tool"}]
        responses = run_agent_to_completion(agent, messages)

        # Agent should continue after tool failure
        assert len(responses) > 0

    def test_invalid_tool_args(self, sequence_llm_factory, mock_calculator_tool):
        """Test handling of invalid tool arguments."""
        llm = sequence_llm_factory([
            {"name": "mock_calculator", "arguments": {"invalid": "args"}, "content": ""},
            "I couldn't calculate that.",
        ])

        agent = AssistantAgent(
            llm=llm,
            function_list=[mock_calculator_tool],
            max_iterations=5,
        )

        messages = [{"role": "user", "content": "Calculate something"}]
        responses = run_agent_to_completion(agent, messages)

        # Should not crash
        assert len(responses) > 0


class TestAssistantAgentConversationHistory:
    """Test conversation history handling."""

    def test_tool_results_in_conversation(self, sequence_llm_factory, mock_search_tool):
        """Test that tool results are added to conversation."""
        llm = sequence_llm_factory([
            {"name": "mock_search", "arguments": {"query": "test"}, "content": ""},
            "Based on the search results, I found...",
        ])

        agent = AssistantAgent(
            llm=llm,
            function_list=[mock_search_tool],
        )

        messages = [{"role": "user", "content": "Search for test"}]
        run_agent_to_completion(agent, messages)

        # The second LLM call should include the tool result
        last_messages = llm.last_messages
        roles = [m.role for m in last_messages]

        # Should have function result in history
        assert "function" in roles or "tool" in roles

    def test_multi_turn_with_tools(self, sequence_llm_factory, mock_search_tool):
        """Test multi-turn conversation with tools."""
        llm = sequence_llm_factory([
            {"name": "mock_search", "arguments": {"query": "first"}, "content": ""},
            "I found results for the first query.",
            {"name": "mock_search", "arguments": {"query": "second"}, "content": ""},
            "I found results for the second query.",
        ])

        agent = AssistantAgent(
            llm=llm,
            function_list=[mock_search_tool],
        )

        # First turn
        messages = [{"role": "user", "content": "Search for first"}]
        responses1 = run_agent_to_completion(agent, messages)

        # Second turn - handle both dict and Message responses
        for r in responses1:
            if isinstance(r, dict):
                messages.append(r)
            else:
                messages.append(r.model_dump())
        messages.append({"role": "user", "content": "Now search for second"})
        responses2 = run_agent_to_completion(agent, messages)

        assert len(responses2) > 0


class TestAssistantAgentReturnDirect:
    """Test return_direct tool behavior."""

    def test_return_direct_tool(self, sequence_llm_factory):
        """Test tool with return_direct=True."""
        from marie.agent import AgentTool, ToolMetadata, ToolOutput

        class DirectReturnTool(AgentTool):
            @property
            def metadata(self):
                return ToolMetadata(
                    name="direct_tool",
                    description="Returns directly",
                    return_direct=True,
                )

            @property
            def name(self):
                return "direct_tool"

            def call(self, **kwargs):
                return ToolOutput(
                    content="Direct result",
                    tool_name=self.name,
                    raw_input=kwargs,
                    raw_output="Direct result",
                    is_error=False,
                )

        llm = sequence_llm_factory([
            {"name": "direct_tool", "arguments": {}, "content": ""},
        ])

        agent = AssistantAgent(
            llm=llm,
            function_list=[DirectReturnTool()],
            return_direct_tool_results=True,
        )

        messages = [{"role": "user", "content": "Use the direct tool"}]
        responses = run_agent_to_completion(agent, messages)

        # Should return tool result directly - handle both dict and Message
        final = responses[-1]
        content = final.get("content", "") if isinstance(final, dict) else final.content
        assert "Direct result" in content


class TestAssistantAgentAsync:
    """Test async tool calling."""

    @pytest.mark.asyncio
    async def test_async_tool_call(self, mock_llm, mock_search_tool):
        """Test async tool call method."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        result = await agent._acall_tool("mock_search", '{"query": "test"}')

        assert isinstance(result, str)
        data = json.loads(result)
        assert "results" in data

    @pytest.mark.asyncio
    async def test_async_nonexistent_tool(self, mock_llm, mock_search_tool):
        """Test async call to nonexistent tool."""
        agent = AssistantAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        result = await agent._acall_tool("nonexistent", "{}")
        assert "does not exist" in result
