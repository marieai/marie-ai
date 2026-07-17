"""Agent backend for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from pydantic import ConfigDict, Field

from marie.agent.agents.assistant import ReactAgent
from marie.agent.backends.base import (
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendConfig,
    ToolCallRecord,
)
from marie.agent.cancellation import AbortSignal
from marie.agent.llm_wrapper import BaseLLMWrapper, OpenAICompatibleWrapper
from marie.agent.message import Message
from marie.agent.streaming import StreamChunk
from marie.agent.tools.base import AgentTool

logger = logging.getLogger("marie.agent.backends.openai")


class OpenAIBackendConfig(BackendConfig):
    """Configuration for an OpenAI-compatible agent backend."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Model identifier sent to the API")
    system_message: str = "You are a helpful assistant."
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    tool_call_format: str = "auto"
    max_retries: int = Field(default=0, ge=0)


class OpenAIAgentBackend(AgentBackend):
    """Run a ReAct agent through an OpenAI-compatible API."""

    config: OpenAIBackendConfig

    def __init__(
        self,
        config: Optional[OpenAIBackendConfig] = None,
        llm: Optional[BaseLLMWrapper] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = OpenAIBackendConfig(**kwargs)
        elif not isinstance(config, OpenAIBackendConfig):
            config = OpenAIBackendConfig(**config.model_dump(), **kwargs)

        super().__init__(config=config)
        self._llm = llm
        self._tools: Dict[str, AgentTool] = {}
        self._tool_call_history: List[ToolCallRecord] = []

    def _get_llm(self) -> BaseLLMWrapper:
        if self._llm is None:
            self._llm = OpenAICompatibleWrapper(
                api_key=self.config.api_key,
                model=self.config.model,
                base_url=self.config.base_url,
                tool_call_format=self.config.tool_call_format,
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
            )
        return self._llm

    def _create_agent(
        self,
        tools: Optional[Dict[str, AgentTool]] = None,
    ) -> ReactAgent:
        function_list = list(tools.values()) if tools else None
        generate_config = {
            name: value
            for name, value in {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }.items()
            if value is not None
        }

        return ReactAgent(
            llm=self._get_llm(),
            function_list=function_list,
            system_message=self.config.system_message,
            max_iterations=self.config.max_iterations,
            extra_generate_cfg=generate_config,
        )

    async def run(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the agent and return its final response."""
        return await asyncio.to_thread(
            self._run_sync,
            messages,
            tools,
            kwargs,
        )

    def _run_sync(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]],
        run_kwargs: Dict[str, Any],
    ) -> AgentResult:
        start_time = time.time()
        iterations = 0
        tool_call_history: List[ToolCallRecord] = []

        try:
            agent = self._create_agent(tools)
            message_dicts = [message.model_dump() for message in messages]
            all_responses: List[Message] = []
            final_response: Optional[Message] = None

            for responses in agent.run(message_dicts, **run_kwargs):
                iterations += 1

                for response in responses:
                    if isinstance(response, dict):
                        response = Message(**response)
                    all_responses.append(response)

                    if response.tool_calls:
                        for tool_call in response.tool_calls:
                            tool_call_history.append(
                                ToolCallRecord(
                                    tool_name=tool_call.function.name,
                                    tool_args=(tool_call.function.get_arguments_dict()),
                                )
                            )
                    elif response.function_call:
                        tool_call_history.append(
                            ToolCallRecord(
                                tool_name=response.function_call.name,
                                tool_args=response.function_call.get_arguments_dict(),
                            )
                        )

                    if (
                        response.role == "assistant"
                        and response.text_content
                        and not response.function_call
                        and not response.tool_calls
                    ):
                        final_response = response

            if final_response is None:
                raise RuntimeError("Agent returned no final assistant response")

            self._tool_call_history = tool_call_history
            return AgentResult(
                output=final_response,
                messages=[message.model_dump() for message in all_responses],
                tool_calls=tool_call_history,
                status=AgentStatus.COMPLETED,
                iterations=iterations,
                is_complete=True,
                metadata={
                    "duration_ms": (time.time() - start_time) * 1000,
                    "model": self.config.model,
                },
            )
        except Exception as exc:
            self._tool_call_history = tool_call_history
            logger.error(f"OpenAI backend execution failed: {exc}")
            return AgentResult(
                output="",
                status=AgentStatus.FAILED,
                error=str(exc),
                iterations=iterations,
                is_complete=False,
                tool_calls=tool_call_history,
            )

    async def run_stream(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        abort_signal: Optional[AbortSignal] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, AgentResult], None]:
        """Stream agent output followed by the final execution result."""
        start_time = time.time()
        self._tool_call_history = []

        try:
            agent = self._create_agent(tools)
            message_dicts = [message.model_dump() for message in messages]

            async for chunk in agent.arun_stream(
                message_dicts,
                abort_signal=abort_signal,
                **kwargs,
            ):
                yield chunk

            yield AgentResult(
                output="",
                messages=[],
                status=AgentStatus.COMPLETED,
                iterations=0,
                is_complete=True,
                metadata={
                    "duration_ms": (time.time() - start_time) * 1000,
                    "model": self.config.model,
                    "streamed": True,
                },
            )
        except Exception as exc:
            logger.error(f"OpenAI streaming execution failed: {exc}")
            yield AgentResult(
                output="",
                status=AgentStatus.FAILED,
                error=str(exc),
                is_complete=False,
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions in OpenAI function-tool format."""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def set_tools(self, tools: Dict[str, AgentTool]) -> None:
        """Replace the tools available to the backend."""
        self._tools = tools

    def set_llm(self, llm: BaseLLMWrapper) -> None:
        """Replace the LLM wrapper used by the backend."""
        self._llm = llm
