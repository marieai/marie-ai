"""LLM wrapper for Marie agent framework.

This module provides LLM abstractions that bridge the agent framework
with marie.engine and other LLM backends.
"""

from __future__ import annotations

import asyncio
import logging
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

from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.trace import StatusCode

from marie.agent.cancellation import AbortSignal
from marie.agent.llm_wrapper import BaseLLMWrapper, OpenAICompatibleWrapper
from marie.agent.message import (
    ASSISTANT,
    FUNCTION,
    SYSTEM,
    TOOL,
    USER,
    ContentItem,
    Message,
)
from marie.agent.streaming import StreamChunk
from marie.agent.tool_call_parser import ToolCallTextParser
from marie.instrumentation import set_llm_io, start_span
from marie.instrumentation.openinference import infer_llm_system

if TYPE_CHECKING:
    from marie.engine.base import EngineLM

logger = logging.getLogger(__name__)


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
        _llm_span = start_span(
            _llm_tracer,
            f"llm:{_model_name}",
            span_kind="llm",
        )
        _llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, _model_name)
        _llm_span.set_attribute(
            SpanAttributes.LLM_SYSTEM, infer_llm_system(_model_name)
        )
        set_llm_io(_llm_span, input_messages=messages)

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

            _resp_content = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            if isinstance(_resp_content, str):
                set_llm_io(_llm_span, output_messages=_resp_content)
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
        _llm_span = start_span(
            _llm_tracer,
            f"llm:{_model_name}",
            span_kind="llm",
        )
        _llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, _model_name)
        _llm_span.set_attribute(
            SpanAttributes.LLM_SYSTEM, infer_llm_system(_model_name)
        )

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


def get_llm_wrapper(
    backend: str = "marie",
    **kwargs: Any,
) -> BaseLLMWrapper:
    """Create a Marie-engine or OpenAI-compatible LLM wrapper."""
    if backend == "marie":
        return MarieEngineLLMWrapper(**kwargs)
    if backend == "openai":
        return OpenAICompatibleWrapper(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend}")
