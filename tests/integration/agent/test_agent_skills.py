"""Integration tests for agent skills and tool usage.

Tests the interaction between agents, skills, and tools in realistic scenarios.
"""

from __future__ import annotations

import json

import pytest

from marie.agent import (
    AgentTool,
    AssistantAgent,
    FunctionTool,
    Message,
    ReactAgent,
    ToolMetadata,
    ToolOutput,
)
from marie.agent.skills.models import Skill, SkillInstructions, SkillMetadata
from marie.agent.skills.registry import SkillRegistry
from marie.agent.skills.router import SkillRouter
from tests.integration.agent.conftest import (
    MockCalculatorTool,
    MockLLMWrapper,
    MockSearchTool,
    SequenceMockLLMWrapper,
    run_agent_to_completion,
)


class TestAgentWithSkills:
    """Tests for agents using skills."""

    def test_skill_prompt_injection(self, mock_llm, mock_search_tool):
        """Test that skill instructions are injected into agent."""
        skill = Skill(
            metadata=SkillMetadata(
                name="test-skill",
                description="A test skill",
                allowed_tools=["mock_search"],
            ),
            _instructions=SkillInstructions(
                when_to_use="Use when testing",
                instructions="Follow test instructions",
            ),
        )

        prompt = skill.to_system_prompt_injection()

        assert "test-skill" in prompt
        assert "test instructions" in prompt.lower()
        assert "mock_search" in prompt

    def test_skill_routing_with_agent(self, mock_llm, mock_search_tool):
        """Test skill routing integrates with agent flow."""
        registry = SkillRegistry()
        registry.register_skill(Skill(
            metadata=SkillMetadata(
                name="search-skill",
                description="Search for documents",
                tags=["search", "document"],
            ),
            _instructions=SkillInstructions(
                when_to_use="When user asks to search",
                instructions="Use search tool to find documents",
            ),
        ))

        router = SkillRouter(registry=registry)
        result = router.match_skill("find documents about testing")

        assert result is not None
        skill, score = result
        assert skill.name == "search-skill"
        assert score > 0

    @pytest.mark.asyncio
    async def test_skill_context_in_routing(self, mock_search_tool):
        """Test that skill context is properly created."""
        registry = SkillRegistry()
        skill = Skill(
            metadata=SkillMetadata(
                name="doc-skill",
                description="Process documents",
                user_invokable=True,
            ),
        )
        registry.register_skill(skill)
        router = SkillRouter(registry=registry)

        context = await router.route("/doc-skill process invoice.pdf")

        assert context.has_skill
        assert context.skill.name == "doc-skill"
        assert context.explicit_invocation
        assert "invoice.pdf" in context.message


class TestAgentToolIntegration:
    """Tests for agent tool integration."""

    def test_agent_with_multiple_tools(
        self, mock_llm, mock_search_tool, mock_calculator_tool
    ):
        """Test agent with multiple tools."""
        agent = ReactAgent(
            llm=mock_llm,
            function_list=[mock_search_tool, mock_calculator_tool],
        )

        assert len(agent.function_map) == 2
        assert "mock_search" in agent.function_map
        assert "mock_calculator" in agent.function_map

    def test_agent_tool_definitions(self, mock_llm, mock_search_tool):
        """Test agent generates proper tool definitions."""
        agent = ReactAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        definitions = agent._get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "mock_search"
        assert "description" in definitions[0]

    def test_agent_tool_call_flow(self, sequence_llm_factory, mock_search_tool):
        """Test complete tool call flow."""
        llm = sequence_llm_factory([
            {
                "name": "mock_search",
                "arguments": {"query": "test query"},
                "content": "Searching...",
            },
            "Based on search results, I found the answer.",
        ])

        agent = ReactAgent(
            llm=llm,
            function_list=[mock_search_tool],
        )

        messages = [{"role": "user", "content": "Search for test"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 2
        final = responses[-1]
        content = final.get("content", "") if isinstance(final, dict) else final.content
        assert "found" in content.lower()


class TestCustomTools:
    """Tests for custom tool implementations."""

    def test_function_tool_from_function(self):
        """Test creating FunctionTool from a plain function."""
        def search_documents(query: str, limit: int = 10) -> str:
            """Search documents by query."""
            return json.dumps({"query": query, "limit": limit, "results": []})

        tool = FunctionTool.from_defaults(fn=search_documents)

        assert tool.name == "search_documents"
        assert "Search documents" in tool.metadata.description

    def test_function_tool_execution(self):
        """Test FunctionTool execution."""
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = FunctionTool.from_defaults(fn=add_numbers)
        result = tool.call(a=5, b=3)

        assert result.raw_output == 8

    def test_custom_agent_tool(self):
        """Test implementing custom AgentTool."""
        class DocumentTool(AgentTool):
            def __init__(self, documents: list):
                self.documents = documents

            @property
            def metadata(self):
                return ToolMetadata(
                    name="document_tool",
                    description="Access project documents",
                )

            @property
            def name(self):
                return "document_tool"

            def call(self, **kwargs):
                doc_id = kwargs.get("id", 0)
                if 0 <= doc_id < len(self.documents):
                    return ToolOutput(
                        content=self.documents[doc_id],
                        tool_name=self.name,
                        raw_input=kwargs,
                        raw_output=self.documents[doc_id],
                        is_error=False,
                    )
                return ToolOutput(
                    content="Document not found",
                    tool_name=self.name,
                    raw_input=kwargs,
                    raw_output="Document not found",
                    is_error=True,
                )

        tool = DocumentTool(documents=["Doc 1", "Doc 2", "Doc 3"])
        result = tool.call(id=1)

        assert result.content == "Doc 2"
        assert result.is_error is False


class TestToolErrorHandling:
    """Tests for tool error handling scenarios."""

    def test_tool_handles_invalid_args(self, mock_calculator_tool):
        """Test tool handles invalid arguments gracefully."""
        result = mock_calculator_tool.safe_call('{"invalid": "args"}')

        # Should not crash, returns error output
        assert isinstance(result, ToolOutput)

    def test_failing_tool_safe_call(self, failing_tool):
        """Test failing tool via safe_call."""
        result = failing_tool.safe_call({})

        assert result.is_error is True

    def test_agent_recovers_from_tool_error(self, sequence_llm_factory, failing_tool):
        """Test agent continues after tool failure."""
        llm = sequence_llm_factory([
            {"name": "failing_tool", "arguments": {}, "content": "Trying..."},
            "I couldn't complete that, but here's an alternative.",
        ])

        agent = ReactAgent(
            llm=llm,
            function_list=[failing_tool],
            max_iterations=5,
        )

        messages = [{"role": "user", "content": "Use the tool"}]
        responses = run_agent_to_completion(agent, messages)

        # Agent should have recovered
        assert len(responses) > 0


class TestMultiToolScenarios:
    """Tests for scenarios with multiple tools."""

    def test_agent_selects_correct_tool(
        self, sequence_llm_factory, mock_search_tool, mock_calculator_tool
    ):
        """Test agent selects appropriate tool for task."""
        llm = sequence_llm_factory([
            {"name": "mock_calculator", "arguments": {"expression": "5+5"}},
            "The result is 10.",
        ])

        agent = ReactAgent(
            llm=llm,
            function_list=[mock_search_tool, mock_calculator_tool],
        )

        messages = [{"role": "user", "content": "What is 5 + 5?"}]
        responses = run_agent_to_completion(agent, messages)

        # Calculator should have been called
        final = responses[-1]
        content = final.get("content", "") if isinstance(final, dict) else final.content
        assert "10" in content

    def test_chained_tool_calls(
        self, sequence_llm_factory, mock_search_tool, mock_calculator_tool
    ):
        """Test agent chains multiple tool calls."""
        llm = sequence_llm_factory([
            {"name": "mock_search", "arguments": {"query": "numbers"}},
            {"name": "mock_calculator", "arguments": {"expression": "1+2+3"}},
            "I searched and calculated. The sum is 6.",
        ])

        agent = ReactAgent(
            llm=llm,
            function_list=[mock_search_tool, mock_calculator_tool],
            max_iterations=10,
        )

        messages = [{"role": "user", "content": "Search for numbers and add them"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 3


class TestSkillToolCombination:
    """Tests combining skills and tools."""

    def test_skill_defines_allowed_tools(self):
        """Test that skills can define allowed tools."""
        skill = Skill(
            metadata=SkillMetadata(
                name="document-skill",
                description="Process documents",
                allowed_tools=["read_file", "write_file", "search"],
            ),
        )

        assert len(skill.metadata.allowed_tools) == 3
        assert "read_file" in skill.metadata.allowed_tools

    def test_skill_prompt_includes_tools(self):
        """Test skill system prompt includes tool information."""
        skill = Skill(
            metadata=SkillMetadata(
                name="analysis-skill",
                description="Data analysis skill",
                allowed_tools=["calculator", "search", "database"],
            ),
            _instructions=SkillInstructions(
                when_to_use="For data analysis",
                instructions="Analyze data using available tools",
            ),
        )

        prompt = skill.to_system_prompt_injection()

        assert "calculator" in prompt
        assert "search" in prompt
        assert "database" in prompt

    @pytest.mark.asyncio
    async def test_skill_routing_and_tool_execution(self, mock_search_tool):
        """Test integrated skill routing followed by tool execution."""
        # Set up skill
        registry = SkillRegistry()
        skill = Skill(
            metadata=SkillMetadata(
                name="research-skill",
                description="Research topics using search",
                allowed_tools=["mock_search"],
                user_invokable=True,
            ),
            _instructions=SkillInstructions(
                when_to_use="When researching topics",
                instructions="Use search tool to find information",
            ),
        )
        registry.register_skill(skill)

        # Route to skill
        router = SkillRouter(registry=registry)
        context = await router.route("/research-skill AI trends")

        assert context.has_skill
        assert context.skill.name == "research-skill"

        # Execute tool based on skill
        result = mock_search_tool.call(query="AI trends")
        data = json.loads(result.content)

        assert data["query"] == "AI trends"
        assert "results" in data


class TestAgentStateManagement:
    """Tests for agent state and conversation management."""

    def test_agent_maintains_conversation_history(self, sequence_llm_factory):
        """Test agent maintains conversation through tool calls."""
        llm = sequence_llm_factory([
            "First response",
        ])

        agent = ReactAgent(llm=llm)

        messages = [{"role": "user", "content": "Hello"}]
        run_agent_to_completion(agent, messages)

        # Check LLM received proper conversation
        assert len(llm._chat_history) == 1
        assert len(llm._chat_history[0]) >= 1

    def test_tool_results_added_to_conversation(
        self, sequence_llm_factory, mock_search_tool
    ):
        """Test tool results are added to conversation history."""
        llm = sequence_llm_factory([
            {"name": "mock_search", "arguments": {"query": "test"}},
            "Based on results...",
        ])

        agent = ReactAgent(
            llm=llm,
            function_list=[mock_search_tool],
        )

        messages = [{"role": "user", "content": "Search"}]
        run_agent_to_completion(agent, messages)

        # Second call should include tool result
        last_call = llm._chat_history[-1]
        roles = [m.role for m in last_call]
        assert "function" in roles or "tool" in roles
