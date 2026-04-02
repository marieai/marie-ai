"""LLM wrapper for Marie agent framework.

This module provides LLM abstractions that bridge the agent framework
with marie.engine and other LLM backends.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from opentelemetry import trace as trace_api
from opentelemetry.trace import StatusCode

from marie.agent.cancellation import AbortSignal
from marie.agent.message import (
    ASSISTANT,
    FUNCTION,
    SYSTEM,
    TOOL,
    USER,
    ContentItem,
    FunctionCall,
    Message,
)
from marie.agent.streaming import StreamChunk, ToolCallAccumulator
from marie.agent.tool_call_parser import ToolCallTextParser
from marie.instrumentation import start_as_current_span as oi_start_as_current_span
from marie.instrumentation import start_span as oi_start_span
from marie.instrumentation.openinference import infer_llm_system
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.agent.emitter import Emitter
    from marie.engine.base import EngineLM

logger = MarieLogger("marie.agent.llm_wrapper")


class BaseLLMWrapper(ABC):
    """Abstract base class for LLM wrappers.

    Provides a unified interface for different LLM backends to be used
    with the Marie agent framework.
    """

    _emitter: Optional["Emitter"] = None

    @property
    def emitter(self) -> Optional["Emitter"]:
        """Get the LLM wrapper's event emitter.

        Set by the agent when executing LLM calls within a run context.
        """
        return self._emitter

    @emitter.setter
    def emitter(self, value: Optional["Emitter"]) -> None:
        self._emitter = value

    @property
    def supports_native_tool_calling(self) -> bool:
        """Whether this LLM backend supports native tool/function calling.

        When True, the LLM uses API-level tool definitions (like OpenAI's
        `tools` parameter) and returns structured `tool_calls` in responses.

        When False, tools are described in the system prompt and the model
        outputs tool calls as text (e.g., <tool_call>...</tool_call>).

        Returns:
            bool: True if native tool calling is supported
        """
        return False

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Iterator[List[Message]]:
        """Generate a chat response.

        Args:
            messages: List of conversation messages
            functions: Optional function/tool definitions for function calling
            stream: Whether to stream the response
            extra_generate_cfg: Additional generation configuration

        Yields:
            Lists of response Messages
        """
        pass

    @abstractmethod
    async def achat(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Generate a chat response asynchronously.

        Args:
            messages: List of conversation messages
            functions: Optional function/tool definitions
            extra_generate_cfg: Additional generation configuration

        Returns:
            Response Message
        """
        pass

    async def achat_stream(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        abort_signal: Optional[AbortSignal] = None,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream chat response as chunks.

        Default implementation falls back to ``achat()`` and yields a single
        chunk.  Subclasses may override for real token-level streaming.
        """
        if abort_signal:
            abort_signal.throw_if_aborted()

        message = await self.achat(messages, functions, extra_generate_cfg)

        tool_calls = message.tool_calls if message.tool_calls else None

        yield StreamChunk(
            content=message.text_content or None,
            finish_reason="stop",
            tool_calls=tool_calls,
        )


class MarieEngineLLMWrapper(BaseLLMWrapper):
    """LLM wrapper using marie.engine.

    Bridges the agent framework with Marie's EngineLM backends (VLLM, OpenAI, etc).
    Supports both text-only and multimodal (vision) inputs.

    Example:
        ```python
        wrapper = MarieEngineLLMWrapper(
            engine_name="qwen2_5_vl_7b",
            provider="vllm",
        )

        # Text-only
        messages = [Message.user("What is 2+2?")]
        for responses in wrapper.chat(messages):
            print(responses[-1].content)

        # Multimodal (vision)
        messages = [
            Message.user(
                [
                    ContentItem(image="/path/to/image.jpg"),
                    ContentItem(text="What do you see in this image?"),
                ]
            )
        ]
        for responses in wrapper.chat(messages):
            print(responses[-1].content)
        ```
    """

    def __init__(
        self,
        engine_name: str = "model_name",
        provider: str = "vllm",
        system_prompt: Optional[str] = None,
        function_call_format: str = "auto",
        **engine_kwargs: Any,
    ):
        """Initialize the Marie engine wrapper.

        Args:
            engine_name: Name of the engine (e.g., 'model_name')
            provider: Provider backend ('vllm', 'openai', etc.)
            system_prompt: Default system prompt (overridden by message system prompt)
            function_call_format: How to format function calls ('auto', 'tool_call', 'action')
            **engine_kwargs: Additional arguments for engine initialization
        """
        from marie.engine import get_engine

        self.engine: EngineLM = get_engine(
            engine_name, provider=provider, **engine_kwargs
        )
        self.engine_name = engine_name
        self.system_prompt = system_prompt
        self.function_call_format = function_call_format
        self._tool_call_parser = ToolCallTextParser(format=function_call_format)

    def _has_multimodal_content(self, messages: List[Message]) -> bool:
        """Check if any message contains multimodal content (images, etc).

        Args:
            messages: List of messages to check

        Returns:
            True if any message contains non-text content
        """
        for msg in messages:
            if msg.content is None:
                continue
            if isinstance(msg.content, str):
                continue
            # Content is a list - check for images/media
            for item in msg.content:
                if isinstance(item, ContentItem):
                    if item.image or item.audio or item.video or item.file:
                        return True
                elif isinstance(item, dict):
                    if any(item.get(k) for k in ("image", "audio", "video", "file")):
                        return True
        return False

    def _build_multimodal_content(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
    ) -> tuple[List[Union[str, Any]], str]:
        """Build multimodal content list for the engine.

        Extracts images and text from messages in the format expected by
        marie.engine (list of strings, bytes, or PIL Images).

        Args:
            messages: List of messages
            functions: Optional function definitions

        Returns:
            Tuple of (content_list, system_prompt) where content_list contains
            images (as paths/URLs) and text prompts for the engine.
        """
        from pathlib import Path

        from PIL import Image

        system_parts = []
        content_items: List[Union[str, Image.Image]] = []
        conversation_parts = []

        for msg in messages:
            role = msg.role

            if role == SYSTEM:
                system_parts.append(msg.text_content)
                continue

            if role == FUNCTION or role == TOOL:
                # Function/tool results - add as text context
                name = msg.name or "tool"
                conversation_parts.append(f"[{name} result]: {msg.text_content}")
                continue

            # Handle user/assistant messages with potential multimodal content
            role_prefix = "User" if role == USER else "Assistant"

            if msg.content is None:
                continue

            if isinstance(msg.content, str):
                conversation_parts.append(f"{role_prefix}: {msg.content}")
                continue

            # Multimodal content - extract images and text
            msg_text_parts = []
            for item in msg.content:
                if isinstance(item, ContentItem):
                    if item.image:
                        # Load image from path/URL
                        image_src = item.image
                        try:
                            if image_src.startswith(("http://", "https://", "data:")):
                                # URL or data URI - pass as string for engine to handle
                                content_items.append(image_src)
                            else:
                                # Local file path - load as PIL Image
                                img_path = Path(image_src)
                                if img_path.exists():
                                    img = Image.open(img_path).convert("RGB")
                                    content_items.append(img)
                                else:
                                    logger.warning(f"Image not found: {image_src}")
                                    msg_text_parts.append(
                                        f"[Image not found: {image_src}]"
                                    )
                        except Exception as e:
                            logger.error(f"Failed to load image {image_src}: {e}")
                            msg_text_parts.append(
                                f"[Failed to load image: {image_src}]"
                            )
                    elif item.text:
                        msg_text_parts.append(item.text)
                elif isinstance(item, dict):
                    if item.get("image"):
                        image_src = item["image"]
                        try:
                            if image_src.startswith(("http://", "https://", "data:")):
                                content_items.append(image_src)
                            else:
                                img_path = Path(image_src)
                                if img_path.exists():
                                    img = Image.open(img_path).convert("RGB")
                                    content_items.append(img)
                                else:
                                    logger.warning(f"Image not found: {image_src}")
                                    msg_text_parts.append(
                                        f"[Image not found: {image_src}]"
                                    )
                        except Exception as e:
                            logger.error(f"Failed to load image {image_src}: {e}")
                            msg_text_parts.append(
                                f"[Failed to load image: {image_src}]"
                            )
                    elif item.get("text"):
                        msg_text_parts.append(item["text"])

            if msg_text_parts:
                conversation_parts.append(f"{role_prefix}: {' '.join(msg_text_parts)}")

        # Add function definitions to system prompt if provided
        if functions:
            func_desc = self._format_functions(functions)
            system_parts.append(func_desc)

        system_prompt = (
            "\n\n".join(system_parts) if system_parts else self.system_prompt or ""
        )

        # Build final prompt text
        prompt_text = "\n".join(conversation_parts)
        if conversation_parts and not conversation_parts[-1].startswith("Assistant:"):
            prompt_text += "\nAssistant:"

        # Add prompt text to content items
        content_items.append(prompt_text)

        return content_items, system_prompt

    def chat(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = False,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Iterator[List[Message]]:
        """Generate a chat response using marie.engine.

        Supports both text-only and multimodal (vision) inputs. When messages
        contain images, they are automatically extracted and passed to the
        underlying engine for vision-language processing.

        Args:
            messages: List of conversation messages (can include images via ContentItem)
            functions: Optional function definitions for function calling
            stream: Whether to stream responses. NOTE: Streaming is not yet
                implemented - this parameter is accepted for API compatibility
                but responses are always returned as complete messages.
            extra_generate_cfg: Additional generation configuration

        Yields:
            Lists containing the response Message

        Note:
            Streaming support is planned for a future release. Currently,
            all responses are returned as complete messages regardless of
            the stream parameter value.
        """
        if stream:
            logger.debug(
                "Streaming requested but not yet implemented. "
                "Returning complete response."
            )

        # OTel LLM span — manual lifecycle because chat() is a generator
        _llm_tracer = trace_api.get_tracer("marie.agent.llm")
        _model_name = self.engine_name or "unknown"
        _llm_span = oi_start_span(
            _llm_tracer,
            f"llm:{_model_name}",
            span_kind="llm",
        )
        _llm_span.set_attribute("llm.model_name", _model_name)
        _llm_span.set_attribute("llm.system", infer_llm_system(_model_name))

        try:
            # Check for multimodal content
            is_multimodal = self._has_multimodal_content(messages)

            # Prepare generation kwargs
            gen_kwargs = {}
            if extra_generate_cfg:
                gen_kwargs.update(extra_generate_cfg)

            # Handle guided generation (JSON schema, etc.)
            guided_json = gen_kwargs.pop("guided_json", None)
            guided_regex = gen_kwargs.pop("guided_regex", None)

            if is_multimodal:
                # Build multimodal content (images + text)
                content, system_prompt = self._build_multimodal_content(
                    messages, functions
                )
                logger.debug(
                    f"Using multimodal path with {sum(1 for c in content if not isinstance(c, str))} images"
                )
            else:
                # Build text-only prompt
                content, system_prompt = self._build_prompt(messages, functions)

            # Generate response
            response = self.engine.generate(
                content=content,
                system_prompt=system_prompt,
                guided_json=guided_json,
                guided_regex=guided_regex,
                **gen_kwargs,
            )

            # Parse response and detect function calls
            message = self._parse_response(response, functions)

            _llm_span.set_status(StatusCode.OK)
            yield [message]

        except GeneratorExit:
            _llm_span.set_status(StatusCode.OK)

        except Exception as exc:
            _llm_span.set_status(StatusCode.ERROR, str(exc))
            _llm_span.record_exception(exc)
            raise

        finally:
            _llm_span.end()

    async def achat(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Generate response asynchronously.

        Currently runs sync generation in a thread pool.

        Args:
            messages: List of conversation messages
            functions: Optional function definitions
            extra_generate_cfg: Additional configuration

        Returns:
            Response Message
        """

        def _sync_chat():
            for responses in self.chat(messages, functions, False, extra_generate_cfg):
                return responses[-1]
            return Message.assistant("")

        return await asyncio.to_thread(_sync_chat)

    async def achat_stream(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        abort_signal: Optional[AbortSignal] = None,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Simulated streaming — yields a single chunk.

        Real engine-level streaming can be added when ``EngineLM`` supports it.
        """
        _llm_tracer = trace_api.get_tracer("marie.agent.llm")
        _model_name = self.engine_name or "unknown"
        _llm_span = oi_start_span(
            _llm_tracer,
            f"llm:{_model_name}",
            span_kind="llm",
        )
        _llm_span.set_attribute("llm.model_name", _model_name)
        _llm_span.set_attribute("llm.system", infer_llm_system(_model_name))

        try:
            if abort_signal:
                abort_signal.throw_if_aborted()

            message = await self.achat(messages, functions, extra_generate_cfg)

            tool_calls = message.tool_calls if message.tool_calls else None
            function_call = message.function_call

            # Convert legacy function_call to tool_calls for consistent API
            if function_call and not tool_calls:
                from marie.agent.message import ToolCall

                tool_calls = [
                    ToolCall(
                        id="call_0",
                        type="function",
                        function=function_call,
                    )
                ]

            _llm_span.set_status(StatusCode.OK)
            yield StreamChunk(
                content=message.text_content or None,
                finish_reason="stop",
                tool_calls=tool_calls,
            )

        except GeneratorExit:
            _llm_span.set_attribute("marie.stream_cancelled", True)
            _llm_span.set_status(StatusCode.OK)

        except Exception as exc:
            _llm_span.set_status(StatusCode.ERROR, str(exc))
            _llm_span.record_exception(exc)
            raise

        finally:
            _llm_span.end()

    def _build_prompt(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
    ) -> tuple[str, str]:
        """Build prompt string from messages.

        Args:
            messages: List of messages
            functions: Optional function definitions

        Returns:
            Tuple of (prompt_content, system_prompt)
        """
        system_parts = []
        conversation_parts = []

        for msg in messages:
            role = msg.role
            content = msg.text_content

            if role == SYSTEM:
                system_parts.append(content)
            elif role == USER:
                conversation_parts.append(f"User: {content}")
            elif role == ASSISTANT:
                conversation_parts.append(f"Assistant: {content}")
            elif role == FUNCTION or role == TOOL:
                # Function/tool results
                name = msg.name or "tool"
                conversation_parts.append(f"[{name} result]: {content}")

        # Add function definitions to system prompt if provided
        if functions:
            func_desc = self._format_functions(functions)
            system_parts.append(func_desc)

        system_prompt = (
            "\n\n".join(system_parts) if system_parts else self.system_prompt or ""
        )
        prompt = "\n".join(conversation_parts)

        # Add assistant prefix for continuation
        if conversation_parts and not conversation_parts[-1].startswith("Assistant:"):
            prompt += "\nAssistant:"

        return prompt, system_prompt

    def _format_functions(self, functions: List[Dict]) -> str:
        """Format function definitions for the prompt.

        Args:
            functions: List of function definitions

        Returns:
            Formatted string describing available functions
        """
        if not functions:
            return ""

        lines = ["You have access to the following tools:", ""]

        for func in functions:
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            lines.append(f"Tool: {name}")
            lines.append(f"Description: {description}")

            # Format parameters
            props = parameters.get("properties", {})
            required = set(parameters.get("required", []))

            if props:
                lines.append("Parameters:")
                for param_name, param_info in props.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_marker = " (required)" if param_name in required else ""
                    lines.append(
                        f"  - {param_name}: {param_type}{req_marker} - {param_desc}"
                    )

            lines.append("")

        lines.extend(
            [
                "To use a tool, respond with:",
                "<tool_call>",
                '{"name": "tool_name", "arguments": {"arg1": "value1"}}',
                "</tool_call>",
                "",
                "Alternatively, you can use the Action/Action Input format:",
                "Action: tool_name",
                "Action Input: arguments as JSON",
                "",
            ]
        )

        return "\n".join(lines)

    def _parse_response(
        self,
        response: str,
        functions: Optional[List[Dict]] = None,
    ) -> Message:
        """Parse model response and extract function calls.

        Args:
            response: Raw model response string
            functions: Available function definitions (for validation)

        Returns:
            Parsed Message with function_call if detected
        """
        if not response:
            return Message.assistant("")

        function_call = None
        content = response

        parsed = self._tool_call_parser.parse(response)
        if parsed and parsed.tool_calls:
            function_call = parsed.tool_calls[0].function
            content = parsed.clean_content

        # Validate function name if functions provided
        if function_call and functions:
            valid_names = {f.get("name") for f in functions}
            if function_call.name not in valid_names:
                logger.warning(
                    f"Model called unknown function '{function_call.name}'. "
                    f"Valid functions: {valid_names}"
                )

        return Message.assistant(
            content=content if content else None,
            function_call=function_call,
        )


class OpenAICompatibleWrapper(BaseLLMWrapper):
    """LLM wrapper for OpenAI-compatible APIs.

    Supports OpenAI, Azure OpenAI, and other compatible endpoints.

    Example:
        ```python
        wrapper = OpenAICompatibleWrapper(
            api_key="sk-...",
            model="gpt-4",
        )

        messages = [Message.user("Hello")]
        for responses in wrapper.chat(messages):
            print(responses[-1].content)
        ```
    """

    @property
    def supports_native_tool_calling(self) -> bool:
        """OpenAI API supports native tool calling via the tools parameter."""
        return True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        tool_call_format: str = "auto",
        **client_kwargs: Any,
    ):
        """Initialize OpenAI-compatible wrapper.

        Args:
            api_key: API key (uses OPENAI_API_KEY env var if not provided)
            model: Model name
            base_url: Custom API base URL
            tool_call_format: Fallback text parser format when the API returns
                tool calls as text instead of structured tool_calls.
                One of "auto", "hermes", "llama3_json", "action", "none".
            **client_kwargs: Additional client configuration
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            **client_kwargs,
        )
        self.model = model
        self._tool_call_parser: Optional[ToolCallTextParser] = (
            ToolCallTextParser(format=tool_call_format)
            if tool_call_format != "none"
            else None
        )

    def chat(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Iterator[List[Message]]:
        """Generate response using OpenAI API.

        Args:
            messages: Conversation messages
            functions: Function definitions
            stream: Whether to stream response
            extra_generate_cfg: Additional configuration

        Yields:
            Response Messages
        """
        # Convert messages to OpenAI format
        openai_messages = [self._message_to_openai(msg) for msg in messages]

        # Build API call kwargs
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }

        if functions:
            kwargs["tools"] = [{"type": "function", "function": f} for f in functions]

        if extra_generate_cfg:
            # Map common config keys
            if "temperature" in extra_generate_cfg:
                kwargs["temperature"] = extra_generate_cfg["temperature"]
            if "max_tokens" in extra_generate_cfg:
                kwargs["max_tokens"] = extra_generate_cfg["max_tokens"]

        # Make API call
        response = self.client.chat.completions.create(**kwargs)

        # Parse response
        choice = response.choices[0]
        message = self._openai_to_message(choice.message)

        yield [message]

    def _get_async_client(self) -> Any:
        """Get or create the async OpenAI client."""
        if not hasattr(self, "_async_client") or self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required")

            self._async_client = AsyncOpenAI(
                api_key=self.client.api_key,
                base_url=str(self.client.base_url) if self.client.base_url else None,
            )
        return self._async_client

    def _build_api_kwargs(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build kwargs dict for the OpenAI API call."""
        openai_messages = [self._message_to_openai(msg) for msg in messages]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }

        if functions:
            kwargs["tools"] = [{"type": "function", "function": f} for f in functions]

        if extra_generate_cfg:
            if "temperature" in extra_generate_cfg:
                kwargs["temperature"] = extra_generate_cfg["temperature"]
            if "max_tokens" in extra_generate_cfg:
                kwargs["max_tokens"] = extra_generate_cfg["max_tokens"]

        return kwargs

    async def achat(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Generate response asynchronously."""
        _llm_tracer = trace_api.get_tracer("marie.agent.llm")
        _model_name = self.model or "unknown"

        with oi_start_as_current_span(
            _llm_tracer,
            f"llm:{_model_name}",
            span_kind="llm",
        ) as _llm_span:
            _llm_span.set_attribute("llm.model_name", _model_name)
            _llm_span.set_attribute("llm.system", infer_llm_system(_model_name))

            try:
                async_client = self._get_async_client()
                kwargs = self._build_api_kwargs(messages, functions, extra_generate_cfg)

                response = await async_client.chat.completions.create(**kwargs)
                choice = response.choices[0]

                # Extract token usage from response
                if response.usage:
                    _llm_span.set_attribute(
                        "llm.token_count.prompt", response.usage.prompt_tokens or 0
                    )
                    _llm_span.set_attribute(
                        "llm.token_count.completion",
                        response.usage.completion_tokens or 0,
                    )

                _llm_span.set_status(StatusCode.OK)
                return self._openai_to_message(choice.message)

            except Exception as exc:
                _llm_span.set_status(StatusCode.ERROR, str(exc))
                _llm_span.record_exception(exc)
                raise

    async def achat_stream(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        abort_signal: Optional[AbortSignal] = None,
        extra_generate_cfg: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using OpenAI's streaming API.

        Properly handles:
        - Text content deltas
        - Indexed tool-call deltas (buffered until JSON is valid)
        - Abort signal checking per chunk
        - Both content AND tool_calls in the same delta (finding #5)
        - Event emission for observability (new_token, success, error)
        """
        start_time = time.perf_counter()

        if abort_signal:
            abort_signal.throw_if_aborted()

        # Emit start event
        if self._emitter:
            await self._emitter.emit(
                "start",
                {
                    "model_name": self.model,
                    "message_count": len(messages),
                    "has_functions": functions is not None,
                },
                source="llm",
            )

        # OTel LLM span — manual lifecycle for async generator
        _llm_tracer = trace_api.get_tracer("marie.agent.llm")
        _model_name = self.model or "unknown"
        _llm_span = oi_start_span(
            _llm_tracer,
            f"llm:{_model_name}",
            span_kind="llm",
        )
        _llm_span.set_attribute("llm.model_name", _model_name)
        _llm_span.set_attribute("llm.system", infer_llm_system(_model_name))

        async_client = self._get_async_client()
        kwargs = self._build_api_kwargs(messages, functions, extra_generate_cfg)
        kwargs["stream"] = True

        accumulated_content = ""
        _provider_usage = None

        try:
            response = await async_client.chat.completions.create(**kwargs)

            tool_accumulator = ToolCallAccumulator()
            has_tool_calls = False

            async for chunk in response:
                # Check abort signal each iteration
                if abort_signal and abort_signal.aborted:
                    break

                if not chunk.choices:
                    # Some providers send usage in a final chunk with no choices
                    if hasattr(chunk, "usage") and chunk.usage:
                        _provider_usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # Capture usage from the final chunk if provider sends it
                if hasattr(chunk, "usage") and chunk.usage:
                    _provider_usage = chunk.usage

                # Yield text content if present (finding #5: don't skip content
                # just because tool_calls are also present in this delta)
                content = getattr(delta, "content", None)
                if content:
                    accumulated_content += content
                    # Emit new_token event
                    if self._emitter:
                        await self._emitter.emit(
                            "new_token",
                            {
                                "token": content,
                                "accumulated_length": len(accumulated_content),
                            },
                            source="llm",
                        )
                    yield StreamChunk(content=content)

                # Accumulate tool-call deltas (finding #6: proper indexed protocol)
                delta_tool_calls = getattr(delta, "tool_calls", None)
                if delta_tool_calls:
                    has_tool_calls = True
                    tool_accumulator.feed(delta_tool_calls)

                # On finish, yield completed tool calls if any
                if finish_reason is not None:
                    completed_calls = (
                        tool_accumulator.get_complete_calls()
                        if has_tool_calls
                        else None
                    )

                    # Emit success event
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    if self._emitter:
                        await self._emitter.emit(
                            "success",
                            {
                                "model_name": self.model,
                                "has_tool_calls": completed_calls is not None,
                                "duration_ms": duration_ms,
                            },
                            source="llm",
                        )

                    yield StreamChunk(
                        content=None,
                        finish_reason=finish_reason,
                        tool_calls=completed_calls,
                        event_type="done",
                    )

            _llm_span.set_status(StatusCode.OK)

        except GeneratorExit:
            _llm_span.set_attribute("marie.stream_cancelled", True)
            _llm_span.set_status(StatusCode.OK)

        except Exception as e:
            _llm_span.set_status(StatusCode.ERROR, str(e))
            _llm_span.record_exception(e)
            # Emit error event
            if self._emitter:
                await self._emitter.emit(
                    "error",
                    {
                        "model_name": self.model,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    source="llm",
                )
            raise

        finally:
            # Token counting: prefer provider usage, fall back to tiktoken estimation
            if _provider_usage:
                _llm_span.set_attribute(
                    "llm.token_count.completion",
                    getattr(_provider_usage, "completion_tokens", 0) or 0,
                )
                _llm_span.set_attribute(
                    "llm.token_count.prompt",
                    getattr(_provider_usage, "prompt_tokens", 0) or 0,
                )
            elif accumulated_content:
                try:
                    from marie.instrumentation.token_counter import count_tokens_text

                    estimated = count_tokens_text(accumulated_content, self.model)
                    _llm_span.set_attribute("llm.token_count.completion", estimated)
                    _llm_span.set_attribute("marie.token_count_estimated", True)
                except Exception:
                    pass
            _llm_span.end()

    def _message_to_openai(self, msg: Message) -> Dict[str, Any]:
        """Convert Message to OpenAI format.

        Handles the nuances of OpenAI's message format:
        - 'name' is only valid for 'function' role (legacy), NOT for 'tool' role
        - 'tool_call_id' is only valid for 'tool' role
        - 'tool_calls' must be serialized for assistant messages
        - Assistant content can be null when making tool calls
        - Multimodal content (images) converted to OpenAI vision format
        """
        result: Dict[str, Any] = {"role": msg.role}

        # Handle content - may be text or multimodal
        if msg.content is not None:
            result["content"] = self._content_to_openai(msg.content)
            # Debug output with truncated base64 data
            if isinstance(result['content'], str):
                debug_content = result['content'][:200]
            elif isinstance(result['content'], list):
                # Truncate base64 data in image_url items
                debug_content = []
                for item in result['content']:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        url = item.get('image_url', {}).get('url', '')
                        if url.startswith('data:'):
                            debug_content.append(
                                {
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': f'{url[:50]}...[base64 truncated, {len(url)} chars]'
                                    },
                                }
                            )
                        else:
                            debug_content.append(item)
                    else:
                        debug_content.append(item)
            else:
                debug_content = result['content']
            logger.debug(
                f"[_message_to_openai] content type: {type(result['content'])}, value: {debug_content}"
            )
        elif msg.role == ASSISTANT:
            # OpenAI requires explicit null for assistant messages with tool_calls
            result["content"] = None

        # 'name' is only valid for 'function' role (legacy format), NOT for 'tool' role
        # OpenAI rejects messages with role='tool' that have 'name' field
        if msg.name and msg.role == FUNCTION:
            result["name"] = msg.name

        # 'tool_call_id' is only valid for 'tool' role responses
        if msg.tool_call_id and msg.role == TOOL:
            result["tool_call_id"] = msg.tool_call_id

        # Serialize tool_calls for assistant messages (current OpenAI format)
        if msg.tool_calls and msg.role == ASSISTANT:
            result["tool_calls"] = self._serialize_tool_calls(msg.tool_calls)

        # Legacy function_call (deprecated, keep for backward compatibility)
        if msg.function_call and msg.role == ASSISTANT:
            result["function_call"] = {
                "name": msg.function_call.name,
                "arguments": msg.function_call.get_arguments_str(),
            }

        return result

    def _content_to_openai(
        self, content: Union[str, List[Union[ContentItem, Dict[str, Any]]]]
    ) -> Union[str, List[Dict[str, Any]]]:
        """Convert content to OpenAI format, handling multimodal content.

        Args:
            content: String content or list of content items (text/image)

        Returns:
            String for text-only, or list of content blocks for multimodal
        """
        print(
            f"[DEBUG _content_to_openai] input type: {type(content)}, value: {content}"
        )

        # Simple string content
        if isinstance(content, str):
            return content

        # Check if content has any images
        has_images = False
        for item in content:
            if isinstance(item, ContentItem):
                if item.image:
                    has_images = True
                    break
            elif isinstance(item, dict):
                if item.get("image"):
                    has_images = True
                    break

        # Text-only content - return as simple string
        if not has_images:
            text_parts = []
            for item in content:
                if isinstance(item, ContentItem) and item.text:
                    text_parts.append(item.text)
                elif isinstance(item, dict) and item.get("text"):
                    text_parts.append(item["text"])
            return "\n".join(text_parts) if text_parts else ""

        # Multimodal content - convert to OpenAI vision format
        openai_content = []
        for item in content:
            if isinstance(item, ContentItem):
                if item.text:
                    openai_content.append({"type": "text", "text": item.text})
                elif item.image:
                    openai_content.append(self._image_to_openai(item.image))
            elif isinstance(item, dict):
                if item.get("text"):
                    openai_content.append({"type": "text", "text": item["text"]})
                elif item.get("image"):
                    openai_content.append(self._image_to_openai(item["image"]))

        return openai_content

    def _image_to_openai(self, image_path: str) -> Dict[str, Any]:
        """Convert image path to OpenAI vision format with base64 encoding.

        Args:
            image_path: Path to local image file or URL

        Returns:
            OpenAI image_url content block
        """
        import base64
        import mimetypes
        from pathlib import Path

        # Check if it's a URL (http/https or data URI)
        if image_path.startswith(("http://", "https://", "data:")):
            return {"type": "image_url", "image_url": {"url": image_path}}

        # Local file - read and base64 encode
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return {"type": "text", "text": f"[Image not found: {image_path}]"}

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            # Default to common image types based on extension
            ext = path.suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }
            mime_type = mime_map.get(ext, "image/png")

        # Read and encode
        try:
            with open(path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            }
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            return {"type": "text", "text": f"[Failed to read image: {image_path}]"}

    def _serialize_tool_calls(self, tool_calls: List) -> List[Dict[str, Any]]:
        """Serialize tool_calls to OpenAI format.

        Args:
            tool_calls: List of ToolCall objects or dicts

        Returns:
            List of tool call dicts in OpenAI format
        """
        result = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                # Already a dict, normalize to OpenAI format
                result.append(
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                        },
                    }
                )
            else:
                # ToolCall object
                arguments = tc.function.arguments
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                result.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": arguments,
                        },
                    }
                )
        return result

    def _openai_to_message(self, openai_msg: Any) -> Message:
        """Convert OpenAI message to Message.

        Includes fallback: when the API returns no structured tool_calls
        but the content contains tool call markup (e.g. from vLLM without
        a tool parser configured), extract them from the text.
        """
        function_call = None
        if hasattr(openai_msg, "function_call") and openai_msg.function_call:
            function_call = FunctionCall(
                name=openai_msg.function_call.name,
                arguments=openai_msg.function_call.arguments,
            )

        tool_calls = None
        if hasattr(openai_msg, "tool_calls") and openai_msg.tool_calls:
            from marie.agent.message import ToolCall

            tool_calls = []
            for tc in openai_msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function=FunctionCall(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                )

        content = openai_msg.content

        # Fallback: parse tool calls from content when API returned none
        if not tool_calls and not function_call and content and self._tool_call_parser:
            parsed = self._tool_call_parser.parse(content)
            if parsed and parsed.tool_calls:
                tool_calls = parsed.tool_calls
                content = parsed.clean_content or None

        return Message(
            role=openai_msg.role,
            content=content,
            function_call=function_call,
            tool_calls=tool_calls,
        )


def get_llm_wrapper(
    backend: str = "marie",
    **kwargs: Any,
) -> BaseLLMWrapper:
    """Factory function to create LLM wrappers.

    Args:
        backend: Wrapper backend ('marie', 'openai')
        **kwargs: Backend-specific configuration

    Returns:
        Configured LLM wrapper

    Example:
        ```python
        # Using marie.engine (local VLLM)
        wrapper = get_llm_wrapper("marie", engine_name="qwen2_5_vl_7b")

        # Using OpenAI direct
        wrapper = get_llm_wrapper("openai", model="gpt-4o")

        # Using Claude via LiteLLM proxy
        wrapper = get_llm_wrapper(
            "openai",
            model="claude/claude-sonnet-4-20250514",
            base_url="http://localhost:4000",  # LiteLLM server
        )

        # Using any provider via LiteLLM
        wrapper = get_llm_wrapper(
            "openai",
            model="anthropic/claude-3-opus",
            base_url="http://localhost:4000",
        )
        ```

    Note:
        For Claude and other providers, use the 'openai' backend with LiteLLM.
        LiteLLM provides an OpenAI-compatible endpoint that translates to
        various provider APIs. Set base_url to your LiteLLM server URL.
    """
    if backend == "marie":
        return MarieEngineLLMWrapper(**kwargs)
    elif backend == "openai":
        return OpenAICompatibleWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
