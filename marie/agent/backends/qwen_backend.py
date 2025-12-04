"""Qwen-style agent backend for Marie agent framework.

This module provides the native Qwen-style backend that uses marie.engine
for LLM inference with ReAct-style reasoning.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from marie.agent.agents.assistant import AssistantAgent
from marie.agent.backends.base import (
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendConfig,
    ToolCallRecord,
)
from marie.agent.llm_wrapper import BaseLLMWrapper, MarieEngineLLMWrapper
from marie.agent.message import Message
from marie.agent.tools.base import AgentTool
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.backends.qwen")


class QwenBackendConfig(BackendConfig):
    """Configuration specific to the Qwen backend."""

    engine_name: str = "qwen2_5_vl_7b"
    provider: str = "vllm"
    system_message: str = "You are a helpful assistant."
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class QwenAgentBackend(AgentBackend):
    """Native Qwen-style agent backend using marie.engine.

    This backend uses the AssistantAgent with MarieEngineLLMWrapper
    to provide ReAct-style reasoning with tool calling.

    Example:
        ```python
        backend = QwenAgentBackend(
            config=QwenBackendConfig(
                engine_name="qwen2_5_vl_7b",
                max_iterations=10,
            )
        )

        tools = {"search": search_tool, "calc": calc_tool}
        messages = [Message.user("What is 2+2?")]

        result = await backend.run(messages, tools)
        print(result.output_text)
        ```
    """

    def __init__(
        self,
        config: Optional[QwenBackendConfig] = None,
        llm: Optional[BaseLLMWrapper] = None,
        **kwargs: Any,
    ):
        """Initialize the Qwen backend.

        Args:
            config: Backend configuration
            llm: Optional pre-configured LLM wrapper
            **kwargs: Additional arguments
        """
        if config is None:
            config = QwenBackendConfig(**kwargs)
        elif not isinstance(config, QwenBackendConfig):
            # Convert base config to QwenBackendConfig
            config = QwenBackendConfig(**config.model_dump(), **kwargs)

        super().__init__(config=config)

        self._llm = llm
        self._agent: Optional[AssistantAgent] = None
        self._tools: Dict[str, AgentTool] = {}
        self._tool_call_history: List[ToolCallRecord] = []

    @property
    def qwen_config(self) -> QwenBackendConfig:
        """Get typed configuration."""
        return self.config  # type: ignore

    def _get_llm(self) -> BaseLLMWrapper:
        """Get or create LLM wrapper."""
        if self._llm is None:
            self._llm = MarieEngineLLMWrapper(
                engine_name=self.qwen_config.engine_name,
                provider=self.qwen_config.provider,
            )
        return self._llm

    def _create_agent(
        self,
        tools: Optional[Dict[str, AgentTool]] = None,
    ) -> AssistantAgent:
        """Create an AssistantAgent with current configuration.

        Args:
            tools: Tools to make available to the agent

        Returns:
            Configured AssistantAgent
        """
        function_list = list(tools.values()) if tools else None

        return AssistantAgent(
            llm=self._get_llm(),
            function_list=function_list,
            system_message=self.qwen_config.system_message,
            max_iterations=self.qwen_config.max_iterations,
        )

    async def run(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the agent with Qwen-style reasoning.

        Args:
            messages: Input messages
            tools: Available tools
            config: Optional override configuration
            **kwargs: Additional arguments

        Returns:
            AgentResult with output and metadata
        """
        # Apply config override if provided
        effective_config = config or self.config

        # Track execution
        self._tool_call_history = []
        start_time = time.time()
        iterations = 0

        try:
            # Create agent with tools
            agent = self._create_agent(tools)

            # Convert messages to dicts for agent.run()
            message_dicts = [msg.model_dump() for msg in messages]

            # Collect responses
            all_responses: List[Message] = []
            final_response: Optional[Message] = None

            for responses in agent.run(message_dicts, **kwargs):
                iterations += 1
                all_responses.extend(responses)

                # Track the latest response
                for resp in responses:
                    if isinstance(resp, dict):
                        resp = Message(**resp)

                    # Track tool calls
                    if resp.function_call:
                        self._tool_call_history.append(
                            ToolCallRecord(
                                tool_name=resp.function_call.name,
                                tool_args=resp.function_call.get_arguments_dict(),
                            )
                        )

                    # Update final response
                    if resp.role == "assistant" and resp.text_content:
                        final_response = resp

            # Build result
            duration_ms = (time.time() - start_time) * 1000

            return AgentResult(
                output=final_response or (all_responses[-1] if all_responses else ""),
                messages=[
                    msg.model_dump() if isinstance(msg, Message) else msg
                    for msg in all_responses
                ],
                tool_calls=self._tool_call_history,
                status=AgentStatus.COMPLETED,
                iterations=iterations,
                is_complete=True,
                metadata={
                    "duration_ms": duration_ms,
                    "engine": self.qwen_config.engine_name,
                    "provider": self.qwen_config.provider,
                },
            )

        except Exception as e:
            logger.error(f"Qwen backend execution failed: {e}")
            return AgentResult(
                output="",
                status=AgentStatus.FAILED,
                error=str(e),
                iterations=iterations,
                is_complete=False,
                tool_calls=self._tool_call_history,
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions."""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def set_tools(self, tools: Dict[str, AgentTool]) -> None:
        """Set available tools.

        Args:
            tools: Tool name to AgentTool mapping
        """
        self._tools = tools

    def set_llm(self, llm: BaseLLMWrapper) -> None:
        """Set the LLM wrapper.

        Args:
            llm: LLM wrapper instance
        """
        self._llm = llm


class SimpleQwenBackend(AgentBackend):
    """Simplified Qwen backend for single-turn interactions.

    This backend doesn't maintain agent state between calls,
    making it suitable for simple query-response patterns.
    """

    def __init__(
        self,
        engine_name: str = "qwen2_5_vl_7b",
        provider: str = "vllm",
        system_message: str = "You are a helpful assistant.",
        **kwargs: Any,
    ):
        """Initialize simple backend.

        Args:
            engine_name: Engine name
            provider: Provider (vllm, openai, etc.)
            system_message: System message
            **kwargs: Additional arguments
        """
        super().__init__(config=BackendConfig(**kwargs))
        self.engine_name = engine_name
        self.provider = provider
        self.system_message = system_message
        self._llm: Optional[MarieEngineLLMWrapper] = None

    def _get_llm(self) -> MarieEngineLLMWrapper:
        """Get or create LLM wrapper."""
        if self._llm is None:
            self._llm = MarieEngineLLMWrapper(
                engine_name=self.engine_name,
                provider=self.provider,
            )
        return self._llm

    async def run(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute a simple single-turn query.

        Args:
            messages: Input messages
            tools: Tools (not used in simple mode)
            config: Configuration
            **kwargs: Additional arguments

        Returns:
            AgentResult
        """
        try:
            llm = self._get_llm()

            # Add system message
            full_messages = [Message.system(self.system_message)] + messages

            # Get response
            for responses in llm.chat(full_messages, stream=False):
                if responses:
                    return AgentResult(
                        output=responses[-1],
                        messages=[r.model_dump() for r in responses],
                        status=AgentStatus.COMPLETED,
                        is_complete=True,
                    )

            return AgentResult(
                output="",
                status=AgentStatus.COMPLETED,
                is_complete=True,
            )

        except Exception as e:
            return AgentResult(
                output="",
                status=AgentStatus.FAILED,
                error=str(e),
                is_complete=False,
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Simple backend has no tools."""
        return []
