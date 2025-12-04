"""Base agent class for Marie agent framework.

This module provides the abstract base class for all agents following
the Qwen-Agent template method pattern with marie.engine integration.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from marie.agent.message import (
    ASSISTANT,
    CONTENT,
    DEFAULT_SYSTEM_MESSAGE,
    ROLE,
    SYSTEM,
    ContentItem,
    FunctionCall,
    Message,
    format_messages,
    has_chinese_content,
)
from marie.agent.tools.base import AgentTool, ToolOutput
from marie.agent.tools.registry import resolve_tools
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.agent.llm_wrapper import BaseLLMWrapper

logger = MarieLogger("marie.agent.base")


class BaseAgent(ABC):
    """Abstract base class for Marie agents.

    Implements the template method pattern where `run()` handles normalization
    and `_run()` contains the core agent logic. This design follows Qwen-Agent
    architecture for consistency and extensibility.

    Subclasses must implement:
        - `_run()`: Core agent execution logic

    Example:
        ```python
        class MyAgent(BaseAgent):
            def _run(self, messages, lang="en", **kwargs):
                # Process messages and generate response
                for response in self._call_llm(messages):
                    yield [response]


        agent = MyAgent(
            llm=MarieEngineLLMWrapper(engine_name="qwen2_5_vl_7b"),
            function_list=["search", "calculator"],
            system_message="You are a helpful assistant.",
        )

        for responses in agent.run([{"role": "user", "content": "Hello"}]):
            print(responses[-1].content)
        ```
    """

    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, AgentTool, Callable]]] = None,
        llm: Optional["BaseLLMWrapper"] = None,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the agent.

        Args:
            function_list: List of tools available to the agent. Can be:
                - Tool name strings (looked up from registry)
                - Configuration dicts with 'name' key
                - AgentTool instances
                - Callable functions
            llm: LLM wrapper for generating responses
            system_message: System message prepended to conversations
            name: Agent name (used in multi-agent scenarios)
            description: Agent description (used for delegation decisions)
            **kwargs: Additional configuration
        """
        self.llm = llm
        self.system_message = system_message
        self.name = name
        self.description = description
        self.extra_generate_cfg: Dict[str, Any] = kwargs.get("extra_generate_cfg", {})

        # Initialize tools
        self.function_map: Dict[str, AgentTool] = {}
        if function_list:
            self._init_tools(function_list)

    def _init_tools(
        self,
        function_list: List[Union[str, Dict, AgentTool, Callable]],
    ) -> None:
        """Initialize tools from the function list.

        Args:
            function_list: List of tool specifications
        """
        resolved = resolve_tools(function_list)
        for name, tool in resolved.items():
            if name in self.function_map:
                logger.warning(
                    f"Repeatedly adding tool {name}, will use the newest tool"
                )
            self.function_map[name] = tool

    def add_tool(self, tool: Union[str, Dict, AgentTool, Callable]) -> None:
        """Add a tool to the agent.

        Args:
            tool: Tool specification (name, config, instance, or callable)
        """
        self._init_tools([tool])

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the agent.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name in self.function_map:
            del self.function_map[name]
            return True
        return False

    def run(
        self,
        messages: List[Union[Dict, Message]],
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Execute the agent with the given messages.

        This is the public entry point that normalizes input and delegates
        to `_run()` for core logic.

        Args:
            messages: Input messages (can be dicts or Message objects)
            **kwargs: Additional arguments passed to `_run()`

        Yields:
            Lists of response Messages (streaming, yields partial results)
        """
        # Deep copy to avoid mutation
        messages = copy.deepcopy(messages)

        # Track original message types for return format
        _return_message_type = "dict"
        new_messages: List[Message] = []

        if not messages:
            _return_message_type = "message"
        else:
            for msg in messages:
                if isinstance(msg, dict):
                    new_messages.append(Message(**msg))
                else:
                    new_messages.append(msg)
                    _return_message_type = "message"

        # Auto-detect language
        if "lang" not in kwargs:
            if has_chinese_content(new_messages):
                kwargs["lang"] = "zh"
            else:
                kwargs["lang"] = "en"

        # Prepend system message
        if self.system_message:
            if not new_messages or new_messages[0][ROLE] != SYSTEM:
                new_messages.insert(
                    0, Message(role=SYSTEM, content=self.system_message)
                )
            else:
                # Merge with existing system message
                existing_content = new_messages[0][CONTENT]
                if isinstance(existing_content, str):
                    new_messages[0][CONTENT] = (
                        self.system_message + "\n\n" + existing_content
                    )
                elif isinstance(existing_content, list):
                    new_messages[0][CONTENT] = [
                        ContentItem(text=self.system_message + "\n\n")
                    ] + existing_content

        # Execute core logic
        for responses in self._run(messages=new_messages, **kwargs):
            # Set agent name on responses
            for resp in responses:
                if not resp.name and self.name:
                    resp.name = self.name

            # Convert output format based on input format
            if _return_message_type == "message":
                yield [Message(**r) if isinstance(r, dict) else r for r in responses]
            else:
                yield [
                    r.model_dump() if isinstance(r, Message) else r for r in responses
                ]

    def run_nonstream(
        self,
        messages: List[Union[Dict, Message]],
        **kwargs: Any,
    ) -> List[Message]:
        """Execute the agent and return the final response.

        Same as `run()` but returns only the final result instead of streaming.

        Args:
            messages: Input messages
            **kwargs: Additional arguments

        Returns:
            Final list of response Messages
        """
        *_, last_responses = self.run(messages, **kwargs)
        return last_responses

    @abstractmethod
    def _run(
        self,
        messages: List[Message],
        lang: str = "en",
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Core agent execution logic.

        Subclasses must implement this method to define the agent's behavior.

        Args:
            messages: Normalized list of Messages with system message prepended
            lang: Language code ('en' or 'zh')
            **kwargs: Additional arguments

        Yields:
            Lists of response Messages
        """
        raise NotImplementedError

    def _call_llm(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = False,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Iterator[List[Message]]:
        """Call the LLM with the given messages.

        Args:
            messages: Messages to send to the LLM
            functions: Optional function definitions for function calling
            stream: Whether to stream the response (not yet implemented)
            extra_generate_cfg: Additional generation configuration

        Yields:
            LLM response Messages

        Raises:
            ValueError: If LLM is not configured

        Note:
            Streaming is not yet implemented. Responses are returned complete.
        """
        if self.llm is None:
            raise ValueError("LLM is not configured for this agent")

        # Merge generation configs
        merged_cfg = {**self.extra_generate_cfg}
        if extra_generate_cfg:
            merged_cfg.update(extra_generate_cfg)

        try:
            return self.llm.chat(
                messages=messages,
                functions=functions,
                stream=stream,
                extra_generate_cfg=merged_cfg,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {type(e).__name__}: {e}")
            # Yield an error message so the agent can handle it gracefully
            error_msg = Message.assistant(
                f"I encountered an error while processing: {type(e).__name__}. "
                "Please try again or rephrase your request."
            )

            # Return a generator that yields the error message
            def error_generator():
                yield [error_msg]

            return error_generator()

    def _call_tool(
        self,
        tool_name: str,
        tool_args: Union[str, Dict] = "{}",
        **kwargs: Any,
    ) -> Union[str, List[ContentItem]]:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool (string or dict)
            **kwargs: Additional arguments passed to the tool

        Returns:
            Tool output (string or list of ContentItems for multimodal)
        """
        if tool_name not in self.function_map:
            return f"Tool '{tool_name}' does not exist."

        tool = self.function_map[tool_name]
        result = tool.safe_call(tool_args, **kwargs)

        if result.is_error:
            logger.warning(f"Tool '{tool_name}' failed: {result.content}")

        return result.content

    async def _acall_tool(
        self,
        tool_name: str,
        tool_args: Union[str, Dict] = "{}",
        **kwargs: Any,
    ) -> Union[str, List[ContentItem]]:
        """Execute a tool asynchronously.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            **kwargs: Additional arguments

        Returns:
            Tool output
        """
        if tool_name not in self.function_map:
            return f"Tool '{tool_name}' does not exist."

        tool = self.function_map[tool_name]
        result = await tool.safe_acall(tool_args, **kwargs)

        if result.is_error:
            logger.warning(f"Tool '{tool_name}' failed: {result.content}")

        return result.content

    def _detect_tool_call(self, message: Message) -> Tuple[bool, str, str, str]:
        """Detect if a message contains a tool/function call.

        Args:
            message: Message to analyze

        Returns:
            Tuple of (has_call, tool_name, tool_args, text_content)
        """
        func_name: Optional[str] = None
        func_args: Optional[str] = None

        if message.function_call:
            func_name = message.function_call.name
            func_args = message.function_call.get_arguments_str()

        text = message.text_content or ""

        return (func_name is not None), func_name or "", func_args or "{}", text

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible function definitions for all tools.

        Returns:
            List of function definitions
        """
        return [tool.get_function_definition() for tool in self.function_map.values()]

    def _get_tool_definitions_openai(self) -> List[Dict[str, Any]]:
        """Get OpenAI tool format definitions.

        Returns:
            List of tool definitions in OpenAI format
        """
        return [tool.to_openai_tool() for tool in self.function_map.values()]


class BasicAgent(BaseAgent):
    """Simple agent that just calls the LLM without tools.

    The most basic form of an agent - passes messages directly to the LLM
    without any tool augmentation or complex workflows.

    Example:
        ```python
        agent = BasicAgent(
            llm=MarieEngineLLMWrapper(engine_name="qwen2_5_vl_7b"),
            system_message="You are a helpful assistant.",
        )

        for responses in agent.run([{"role": "user", "content": "Hello"}]):
            print(responses[-1].content)
        ```
    """

    def _run(
        self,
        messages: List[Message],
        lang: str = "en",
        **kwargs: Any,
    ) -> Iterator[List[Message]]:
        """Simply forward messages to the LLM.

        Args:
            messages: Input messages
            lang: Language code
            **kwargs: Additional arguments (seed, etc.)

        Yields:
            LLM responses
        """
        extra_generate_cfg = {"lang": lang}
        if kwargs.get("seed") is not None:
            extra_generate_cfg["seed"] = kwargs["seed"]

        return self._call_llm(messages, extra_generate_cfg=extra_generate_cfg)
