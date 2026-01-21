"""Integration tests for PlanningAgent.

Tests the multi-step planning agent with tool orchestration capabilities.
"""

import json

import pytest

from marie.agent import Message, PlanningAgent, register_tool
from tests.integration.agent.conftest import (
    MockCalculatorTool,
    MockLLMWrapper,
    MockSearchTool,
    SequenceMockLLMWrapper,
    get_final_response,
    run_agent_to_completion,
)


class TestPlanningAgentCreation:
    """Test PlanningAgent instantiation."""

    def test_create_planning_agent(self, mock_llm):
        """Test creating a planning agent."""
        agent = PlanningAgent(
            llm=mock_llm,
            system_message="You are a planning assistant.",
        )

        assert agent.llm is mock_llm
        assert "planning" in agent.__class__.__name__.lower()

    def test_create_with_tools(self, mock_llm, mock_search_tool, mock_calculator_tool):
        """Test creating planning agent with tools."""
        agent = PlanningAgent(
            llm=mock_llm,
            function_list=[mock_search_tool, mock_calculator_tool],
        )

        assert len(agent.function_map) == 2
        assert "mock_search" in agent.function_map
        assert "mock_calculator" in agent.function_map

    def test_default_planning_prompt(self, mock_llm):
        """Test default planning prompt is used."""
        agent = PlanningAgent(llm=mock_llm)

        # Should have planning-related content in system message
        assert "plan" in agent.system_message.lower()
        assert "step" in agent.system_message.lower()

    def test_custom_max_iterations(self, mock_llm):
        """Test custom max iterations."""
        agent = PlanningAgent(
            llm=mock_llm,
            max_iterations=25,
        )

        assert agent.max_iterations == 25


class TestPlanningAgentPlanCreation:
    """Test plan creation functionality."""

    def test_plan_format_in_response(self, sequence_llm_factory, mock_search_tool):
        """Test that agent creates properly formatted plans."""
        llm = sequence_llm_factory([
            """PLAN:
1. Search for information about the topic
2. Analyze the results
3. Provide a summary

FINAL ANSWER:
Based on my analysis, here is the summary.""",
        ])

        agent = PlanningAgent(
            llm=llm,
            function_list=[mock_search_tool],
        )

        messages = [{"role": "user", "content": "Create a plan to research AI"}]
        responses = run_agent_to_completion(agent, messages)

        assert len(responses) > 0
        # Handle both dict and Message responses
        last = responses[-1]
        content = last.get("content", "") if isinstance(last, dict) else last.content
        assert "PLAN" in content or "plan" in content.lower()

    def test_numbered_steps_in_plan(self, sequence_llm_factory):
        """Test that plans have numbered steps."""
        llm = sequence_llm_factory([
            """PLAN:
1. First step
2. Second step
3. Third step

FINAL ANSWER:
Task completed.""",
        ])

        agent = PlanningAgent(llm=llm)
        messages = [{"role": "user", "content": "Plan a task"}]
        responses = run_agent_to_completion(agent, messages)

        last = responses[-1]
        content = last.get("content", "") if isinstance(last, dict) else last.content
        # Should have numbered items
        assert "1." in content
        assert "2." in content


class TestPlanningAgentExecution:
    """Test plan execution with tools."""

    def test_single_tool_in_plan(self, sequence_llm_factory, mock_search_tool):
        """Test executing a plan with a single tool call."""
        llm = sequence_llm_factory([
            # First response: plan with tool call
            {"name": "mock_search", "arguments": {"query": "AI research"}, "content": "STEP 1: Searching..."},
            # Second response: final answer
            "FINAL ANSWER:\nBased on the search results, AI research is progressing rapidly.",
        ])

        agent = PlanningAgent(
            llm=llm,
            function_list=[mock_search_tool],
            max_iterations=5,
        )

        messages = [{"role": "user", "content": "Research AI and summarize"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 2
        assert len(responses) > 0

    def test_multiple_tools_in_plan(self, sequence_llm_factory, mock_search_tool, mock_calculator_tool):
        """Test executing a plan with multiple tool calls."""
        llm = sequence_llm_factory([
            # Step 1: Search
            {"name": "mock_search", "arguments": {"query": "math problem"}, "content": "STEP 1: Searching..."},
            # Step 2: Calculate
            {"name": "mock_calculator", "arguments": {"expression": "10 * 5"}, "content": "STEP 2: Calculating..."},
            # Final answer
            "FINAL ANSWER:\nThe search found the problem and the calculation shows 50.",
        ])

        agent = PlanningAgent(
            llm=llm,
            function_list=[mock_search_tool, mock_calculator_tool],
            max_iterations=10,
        )

        messages = [{"role": "user", "content": "Find a math problem and solve it"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 3

    def test_tool_results_inform_next_step(self, sequence_llm_factory, mock_search_tool):
        """Test that tool results are used in subsequent planning."""
        llm = sequence_llm_factory([
            {"name": "mock_search", "arguments": {"query": "initial"}, "content": "Step 1"},
            {"name": "mock_search", "arguments": {"query": "followup"}, "content": "Step 2 based on results"},
            "FINAL ANSWER:\nCompleted both searches.",
        ])

        agent = PlanningAgent(
            llm=llm,
            function_list=[mock_search_tool],
        )

        messages = [{"role": "user", "content": "Do two related searches"}]
        run_agent_to_completion(agent, messages)

        # Verify second call received first tool's results
        assert llm.call_count == 3
        # The second LLM call should have function result in messages
        second_call_messages = llm._chat_history[1]
        roles = [m.role for m in second_call_messages]
        assert "function" in roles


class TestPlanningAgentFinalAnswer:
    """Test final answer detection and handling."""

    def test_final_answer_stops_execution(self, sequence_llm_factory):
        """Test that FINAL ANSWER stops the planning loop."""
        llm = sequence_llm_factory([
            "FINAL ANSWER:\nHere is my immediate response without tools.",
        ])

        agent = PlanningAgent(llm=llm)
        messages = [{"role": "user", "content": "Quick question"}]
        responses = run_agent_to_completion(agent, messages)

        # Should stop after first response with FINAL ANSWER
        assert llm.call_count == 1
        last = responses[-1]
        content = last.get("content", "") if isinstance(last, dict) else last.content
        assert "FINAL ANSWER" in content

    def test_final_answer_case_insensitive(self, sequence_llm_factory):
        """Test that final answer detection is case insensitive."""
        llm = sequence_llm_factory([
            "final answer:\nThe answer is 42.",
        ])

        agent = PlanningAgent(llm=llm)
        messages = [{"role": "user", "content": "What is the answer?"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 1


class TestPlanningAgentIterationLimits:
    """Test iteration limit handling."""

    def test_max_iterations_respected(self, sequence_llm_factory, mock_search_tool):
        """Test that max iterations limit is respected."""
        # Create infinite loop of tool calls (no FINAL ANSWER)
        llm = sequence_llm_factory([
            {"name": "mock_search", "arguments": {"query": "1"}, "content": "Step 1"},
            {"name": "mock_search", "arguments": {"query": "2"}, "content": "Step 2"},
            {"name": "mock_search", "arguments": {"query": "3"}, "content": "Step 3"},
            {"name": "mock_search", "arguments": {"query": "4"}, "content": "Step 4"},
            {"name": "mock_search", "arguments": {"query": "5"}, "content": "Step 5"},
        ])

        agent = PlanningAgent(
            llm=llm,
            function_list=[mock_search_tool],
            max_iterations=3,
        )

        messages = [{"role": "user", "content": "Keep searching forever"}]
        run_agent_to_completion(agent, messages)

        # Should stop at max_iterations
        assert llm.call_count <= 3

    def test_continues_without_final_answer_up_to_limit(self, sequence_llm_factory):
        """Test that planning continues without FINAL ANSWER until limit."""
        llm = sequence_llm_factory([
            "Step 1: Thinking...",
            "Step 2: Still thinking...",
            "Step 3: Almost there...",
            "FINAL ANSWER:\nDone thinking.",
        ])

        agent = PlanningAgent(
            llm=llm,
            max_iterations=10,
        )

        messages = [{"role": "user", "content": "Think step by step"}]
        run_agent_to_completion(agent, messages)

        # Should continue until FINAL ANSWER
        assert llm.call_count == 4


class TestPlanningAgentErrorHandling:
    """Test error handling in planning."""

    def test_tool_failure_continues_planning(self, sequence_llm_factory, failing_tool):
        """Test that tool failures don't stop planning."""
        llm = sequence_llm_factory([
            {"name": "failing_tool", "arguments": {}, "content": "Trying the tool..."},
            "FINAL ANSWER:\nThe tool failed but I can still provide an answer.",
        ])

        agent = PlanningAgent(
            llm=llm,
            function_list=[failing_tool],
            max_iterations=5,
        )

        messages = [{"role": "user", "content": "Use the failing tool"}]
        responses = run_agent_to_completion(agent, messages)

        # Should continue after tool failure
        assert len(responses) > 0

    def test_empty_plan_handled(self, mock_llm):
        """Test handling of empty or minimal responses."""
        agent = PlanningAgent(llm=mock_llm)
        messages = [{"role": "user", "content": "Hello"}]

        # Should not crash with minimal response
        responses = run_agent_to_completion(agent, messages)
        assert len(responses) >= 0


class TestPlanningAgentConversationHistory:
    """Test conversation history in planning."""

    def test_tool_results_in_history(self, sequence_llm_factory, mock_search_tool):
        """Test that tool results are properly added to history."""
        llm = sequence_llm_factory([
            {"name": "mock_search", "arguments": {"query": "test"}, "content": "Searching..."},
            "FINAL ANSWER:\nFound the results.",
        ])

        agent = PlanningAgent(
            llm=llm,
            function_list=[mock_search_tool],
        )

        messages = [{"role": "user", "content": "Search and report"}]
        run_agent_to_completion(agent, messages)

        # Second call should include tool result
        second_messages = llm._chat_history[1]
        roles = [m.role for m in second_messages]
        assert "function" in roles or "tool" in roles

    def test_multi_turn_planning(self, sequence_llm_factory, mock_search_tool):
        """Test multi-turn conversation with planning."""
        llm = sequence_llm_factory([
            "FINAL ANSWER:\nFirst task complete.",
            {"name": "mock_search", "arguments": {"query": "second"}, "content": "Second search..."},
            "FINAL ANSWER:\nSecond task complete.",
        ])

        agent = PlanningAgent(
            llm=llm,
            function_list=[mock_search_tool],
        )

        # First turn
        messages = [{"role": "user", "content": "First task"}]
        responses1 = run_agent_to_completion(agent, messages)

        # Second turn
        for r in responses1:
            if isinstance(r, dict):
                messages.append(r)
            else:
                messages.append(r.model_dump())
        messages.append({"role": "user", "content": "Second task"})
        responses2 = run_agent_to_completion(agent, messages)

        assert len(responses2) > 0


class TestPlanningAgentWithDocumentTools:
    """Test planning agent with document processing tools."""

    @pytest.fixture
    def document_tools(self):
        """Create mock document processing tools."""
        from marie.agent import AgentTool, ToolMetadata, ToolOutput

        class MockOCRTool(AgentTool):
            @property
            def name(self):
                return "ocr"

            @property
            def metadata(self):
                return ToolMetadata(
                    name="ocr",
                    description="Extract text from document images",
                )

            def call(self, **kwargs):
                return ToolOutput(
                    content=json.dumps({"text": "Extracted text content", "confidence": 0.95}),
                    tool_name=self.name,
                    raw_input=kwargs,
                    raw_output={"text": "Extracted text content"},
                    is_error=False,
                )

        class MockClassifierTool(AgentTool):
            @property
            def name(self):
                return "classifier"

            @property
            def metadata(self):
                return ToolMetadata(
                    name="classifier",
                    description="Classify document type",
                )

            def call(self, **kwargs):
                return ToolOutput(
                    content=json.dumps({"type": "invoice", "confidence": 0.92}),
                    tool_name=self.name,
                    raw_input=kwargs,
                    raw_output={"type": "invoice"},
                    is_error=False,
                )

        return [MockOCRTool(), MockClassifierTool()]

    def test_document_processing_plan(self, sequence_llm_factory, document_tools):
        """Test a document processing workflow plan."""
        llm = sequence_llm_factory([
            # Plan step 1: Classify
            {"name": "classifier", "arguments": {"document": "doc.pdf"}, "content": "STEP 1: Classifying..."},
            # Plan step 2: OCR
            {"name": "ocr", "arguments": {"document": "doc.pdf"}, "content": "STEP 2: Extracting text..."},
            # Final answer
            "FINAL ANSWER:\nDocument is an invoice. Extracted text: 'Extracted text content'",
        ])

        agent = PlanningAgent(
            llm=llm,
            function_list=document_tools,
            max_iterations=10,
        )

        messages = [{"role": "user", "content": "Process this document: classify it and extract text"}]
        responses = run_agent_to_completion(agent, messages)

        assert llm.call_count == 3
        last = responses[-1]
        content = last.get("content", "") if isinstance(last, dict) else last.content
        assert "invoice" in content.lower() or "FINAL ANSWER" in content


class TestPlanningAgentAsync:
    """Test async functionality of PlanningAgent."""

    @pytest.mark.asyncio
    async def test_async_tool_calls_in_plan(self, mock_llm, mock_search_tool):
        """Test async tool calls work in planning context."""
        agent = PlanningAgent(
            llm=mock_llm,
            function_list=[mock_search_tool],
        )

        result = await agent._acall_tool("mock_search", '{"query": "async test"}')

        assert isinstance(result, str)
        data = json.loads(result)
        assert "results" in data
