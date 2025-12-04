"""AssistantAgent - ReAct-style agent implementation.

This module provides the primary agent implementation following the
ReAct (Reason + Act) paradigm where the agent reasons about the task,
decides on actions (tool calls), and iterates until completion.
"""

from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from marie.agent.base import BaseAgent
from marie.agent.message import (
    ASSISTANT,
    FUNCTION,
    TOOL,
    FunctionCall,
    Message,
)
from marie.agent.tools.base import AgentTool
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.agent.llm_wrapper import BaseLLMWrapper

logger = MarieLogger("marie.agent.agents.assistant")


class AssistantAgent(BaseAgent):
    """ReAct-style assistant agent with tool calling capabilities.

    This agent follows the ReAct paradigm:
    1. Receive user input
    2. Reason about what to do (via LLM)
    3. Optionally call tools to gather information
    4. Observe tool results
    5. Continue reasoning until task is complete
    6. Provide final response

    The agent supports:
    - Multiple tool calls per turn
    - Streaming responses
    - Configurable iteration limits
    - Error recovery from tool failures

    Example:
        ```python
        from marie.agent.agents import AssistantAgent
        from marie.agent.llm_wrapper import MarieEngineLLMWrapper
        from marie.agent import register_tool


        @register_tool("search")
        def search(query: str) -> str:
            '''Search for information.'''
            return json.dumps({"results": [...]})


        agent = AssistantAgent(
            llm=MarieEngineLLMWrapper(engine_name="qwen2_5_vl_7b"),
            function_list=["search"],
            system_message="You are a helpful research assistant.",
            max_iterations=10,
        )

        for responses in agent.run([{"role": "user", "content": "Find info about AI"}]):
            print(responses[-1].content)
        ```
    """

    DEFAULT_REACT_PROMPT = """You are a helpful assistant that can use tools to accomplish tasks.

When you need to use a tool, respond with a tool call in this format:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

After receiving tool results, continue reasoning until you can provide a final answer.
Always explain your reasoning before making tool calls.
When you have enough information to answer, provide a clear and helpful response."""

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, AgentTool]]] = None,
        llm: Optional["BaseLLMWrapper"] = None,
        system_message: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_iterations: int = 10,
        return_direct_tool_results: bool = False,
        **kwargs: Any,
    ):
        """Initialize the AssistantAgent.

        Args:
            function_list: List of tools available to the agent
            llm: LLM wrapper for generating responses
            system_message: Custom system message (defaults to ReAct prompt)
            name: Agent name
            description: Agent description
            max_iterations: Maximum reasoning iterations before stopping
            return_direct_tool_results: If True, return tool results directly
                when tool.return_direct is True
            **kwargs: Additional configuration
        """
        # Use ReAct prompt if no system message provided
        if system_message is None:
            system_message = self.DEFAULT_REACT_PROMPT

        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
            **kwargs,
        )

        self.max_iterations = max_iterations
        self.return_direct_tool_results = return_direct_tool_results

    def _run(
        self,
        messages: List[Message],
        lang: str = "en",
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Execute the ReAct loop.

        Args:
            messages: Input messages with system message prepended
            lang: Language code
            **kwargs: Additional arguments

        Yields:
            Lists of response Messages (streaming partial results)
        """
        # Get function definitions if we have tools
        functions = self._get_tool_definitions() if self.function_map else None

        # Track conversation for this run
        conversation = list(messages)
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Call LLM
            extra_cfg = {"lang": lang}
            if kwargs.get("seed") is not None:
                extra_cfg["seed"] = kwargs["seed"]

            llm_responses = []
            for responses in self._call_llm(
                conversation,
                functions=functions,
                extra_generate_cfg=extra_cfg,
            ):
                llm_responses = responses
                yield responses

            if not llm_responses:
                break

            # Get the last response
            response = llm_responses[-1]
            conversation.append(response)

            # Check for tool calls
            has_call, tool_name, tool_args, text_content = self._detect_tool_call(
                response
            )

            if not has_call:
                # No tool call - this is the final response
                break

            # Execute tool
            logger.debug(f"Calling tool '{tool_name}' with args: {tool_args}")
            tool_result = self._call_tool(tool_name, tool_args, **kwargs)

            # Check if tool wants direct return
            if self.return_direct_tool_results:
                tool = self.function_map.get(tool_name)
                if tool and tool.metadata.return_direct:
                    final_msg = Message.assistant(content=str(tool_result))
                    yield [final_msg]
                    return

            # Add tool result to conversation
            tool_msg = Message.function_result(
                name=tool_name,
                content=str(tool_result),
            )
            conversation.append(tool_msg)

            # Yield intermediate state showing tool was called
            yield [response, tool_msg]

        # If we hit max iterations, yield a warning
        if iteration >= self.max_iterations:
            logger.warning(
                f"Agent reached max iterations ({self.max_iterations}). "
                "Consider increasing max_iterations or simplifying the task."
            )


class FunctionCallingAgent(BaseAgent):
    """Agent that uses OpenAI-style function calling.

    This agent is optimized for LLMs that support native function calling
    (like GPT-4, Claude, etc.) rather than the ReAct text-based approach.

    The agent handles:
    - Parallel tool calls (when supported by the LLM)
    - Tool call IDs for proper response threading
    - Automatic retry on tool errors
    """

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, AgentTool]]] = None,
        llm: Optional["BaseLLMWrapper"] = None,
        system_message: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ):
        """Initialize the FunctionCallingAgent."""
        if system_message is None:
            system_message = "You are a helpful assistant with access to tools."

        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
            **kwargs,
        )

        self.max_iterations = max_iterations

    def _run(
        self,
        messages: List[Message],
        lang: str = "en",
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Execute function calling loop.

        Args:
            messages: Input messages
            lang: Language code
            **kwargs: Additional arguments

        Yields:
            Response Messages
        """
        # Get OpenAI-style tool definitions
        tools = self._get_tool_definitions_openai() if self.function_map else None

        conversation = list(messages)
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Call LLM with tools
            extra_cfg = {"lang": lang}
            if kwargs.get("seed") is not None:
                extra_cfg["seed"] = kwargs["seed"]

            llm_responses = []
            for responses in self._call_llm(
                conversation,
                functions=tools,
                extra_generate_cfg=extra_cfg,
            ):
                llm_responses = responses
                yield responses

            if not llm_responses:
                break

            response = llm_responses[-1]
            conversation.append(response)

            # Check for tool_calls (OpenAI style)
            if not response.tool_calls:
                # No tool calls - final response
                break

            # Execute all tool calls
            tool_results = []
            for tool_call in response.tool_calls:
                func = tool_call.function
                logger.debug(f"Executing tool call: {func.name}")

                result = self._call_tool(
                    func.name,
                    func.get_arguments_str(),
                    **kwargs,
                )

                tool_msg = Message.tool_result(
                    tool_call_id=tool_call.id,
                    content=str(result),
                    name=func.name,
                )
                tool_results.append(tool_msg)
                conversation.append(tool_msg)

            # Yield intermediate state
            yield [response] + tool_results


class ChatAgent(BaseAgent):
    """Simple conversational agent without tool calling.

    A lightweight agent for pure conversation without tools.
    Useful for chatbots and simple Q&A systems.
    """

    def __init__(
        self,
        llm: Optional["BaseLLMWrapper"] = None,
        system_message: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize ChatAgent."""
        if system_message is None:
            system_message = "You are a helpful, friendly assistant."

        super().__init__(
            function_list=None,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
            **kwargs,
        )

    def _run(
        self,
        messages: List[Message],
        lang: str = "en",
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Generate a conversational response.

        Args:
            messages: Input messages
            lang: Language code
            **kwargs: Additional arguments

        Yields:
            Response Messages
        """
        extra_cfg = {"lang": lang}
        if kwargs.get("seed") is not None:
            extra_cfg["seed"] = kwargs["seed"]

        return self._call_llm(messages, extra_generate_cfg=extra_cfg)


class PlanningAgent(BaseAgent):
    """Agent that creates and executes multi-step plans.

    This agent first creates a plan for complex tasks, then executes
    each step systematically. Useful for tasks requiring multiple
    sequential operations.

    The planning approach:
    1. Analyze the task and create a step-by-step plan
    2. Execute each step, potentially using tools
    3. Verify results and adjust plan if needed
    4. Provide final synthesized response
    """

    PLANNING_PROMPT = """You are a planning assistant that breaks down complex tasks into steps.

When given a task:
1. First, create a numbered plan of steps needed to complete the task
2. Execute each step, using tools when necessary
3. After completing all steps, provide a final summary

Format your plan as:
PLAN:
1. [First step]
2. [Second step]
...

Then execute each step, showing your work.
When all steps are complete, provide a FINAL ANSWER."""

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, AgentTool]]] = None,
        llm: Optional["BaseLLMWrapper"] = None,
        system_message: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_iterations: int = 20,
        **kwargs: Any,
    ):
        """Initialize PlanningAgent."""
        if system_message is None:
            system_message = self.PLANNING_PROMPT

        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
            **kwargs,
        )

        self.max_iterations = max_iterations

    def _run(
        self,
        messages: List[Message],
        lang: str = "en",
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Execute planning and execution loop.

        This uses the same ReAct-style loop but with planning-focused prompts.
        """
        functions = self._get_tool_definitions() if self.function_map else None
        conversation = list(messages)
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            extra_cfg = {"lang": lang}
            if kwargs.get("seed") is not None:
                extra_cfg["seed"] = kwargs["seed"]

            llm_responses = []
            for responses in self._call_llm(
                conversation,
                functions=functions,
                extra_generate_cfg=extra_cfg,
            ):
                llm_responses = responses
                yield responses

            if not llm_responses:
                break

            response = llm_responses[-1]
            conversation.append(response)

            # Check for tool calls
            has_call, tool_name, tool_args, text_content = self._detect_tool_call(
                response
            )

            # Check if we've reached final answer
            if "FINAL ANSWER" in (text_content or "").upper():
                break

            if not has_call:
                # No tool call and no final answer - continue planning
                continue

            # Execute tool
            tool_result = self._call_tool(tool_name, tool_args, **kwargs)

            tool_msg = Message.function_result(
                name=tool_name,
                content=str(tool_result),
            )
            conversation.append(tool_msg)
            yield [response, tool_msg]
