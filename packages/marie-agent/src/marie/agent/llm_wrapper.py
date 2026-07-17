"""Portable LLM contracts and OpenAI-compatible implementation."""

from __future__ import annotations

import json
import logging
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

from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.trace import StatusCode

from marie.agent.cancellation import AbortSignal
from marie.agent.message import (
    ASSISTANT,
    FUNCTION,
    TOOL,
    ContentItem,
    FunctionCall,
    Message,
)
from marie.agent.streaming import StreamChunk, ToolCallAccumulator
from marie.agent.tool_call_parser import ToolCallTextParser
from marie.instrumentation import set_llm_io, start_as_current_span, start_span
from marie.instrumentation.openinference import infer_llm_system

if TYPE_CHECKING:
    from marie.agent.emitter import Emitter

logger = logging.getLogger("marie.agent.llm_wrapper")


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
            raise ImportError("openai package required. Install with: uv add openai")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            **client_kwargs,
        )
        self._request_timeout = self.client.timeout
        self._max_retries = self.client.max_retries
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

            import httpx

            http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=40,
                    max_keepalive_connections=20,
                ),
                timeout=self._request_timeout,
            )
            self._async_client = AsyncOpenAI(
                api_key=self.client.api_key,
                base_url=str(self.client.base_url) if self.client.base_url else None,
                http_client=http_client,
                max_retries=self._max_retries,
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

        with start_as_current_span(
            _llm_tracer,
            f"llm:{_model_name}",
            span_kind="llm",
        ) as _llm_span:
            _llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, _model_name)
            _llm_span.set_attribute(
                SpanAttributes.LLM_SYSTEM, infer_llm_system(_model_name)
            )
            set_llm_io(_llm_span, input_messages=messages)

            try:
                async_client = self._get_async_client()
                kwargs = self._build_api_kwargs(messages, functions, extra_generate_cfg)

                response = await async_client.chat.completions.create(**kwargs)
                choice = response.choices[0]

                # Extract token usage from response
                if response.usage:
                    prompt_tokens = response.usage.prompt_tokens or 0
                    completion_tokens = response.usage.completion_tokens or 0
                    _llm_span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                        prompt_tokens,
                    )
                    _llm_span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                        completion_tokens,
                    )
                    _llm_span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                        prompt_tokens + completion_tokens,
                    )

                result_msg = self._openai_to_message(choice.message)
                _resp = (
                    result_msg.get("content")
                    if isinstance(result_msg, dict)
                    else getattr(result_msg, "content", None)
                )
                if isinstance(_resp, str):
                    set_llm_io(_llm_span, output_messages=_resp)
                _llm_span.set_status(StatusCode.OK)
                return result_msg

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

        async_client = self._get_async_client()
        kwargs = self._build_api_kwargs(messages, functions, extra_generate_cfg)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

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

            if accumulated_content:
                set_llm_io(_llm_span, output_messages=accumulated_content)
            _llm_span.set_status(StatusCode.OK)

        except GeneratorExit:
            _llm_span.set_attribute("marie.stream_cancelled", True)
            if accumulated_content:
                set_llm_io(_llm_span, output_messages=accumulated_content)
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
                prompt_tokens = getattr(_provider_usage, "prompt_tokens", 0) or 0
                completion_tokens = (
                    getattr(_provider_usage, "completion_tokens", 0) or 0
                )
                _llm_span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                    prompt_tokens,
                )
                _llm_span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                    completion_tokens,
                )
                _llm_span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                    prompt_tokens + completion_tokens,
                )
            elif accumulated_content:
                try:
                    from marie.instrumentation.token_counter import count_tokens_text

                    estimated = count_tokens_text(accumulated_content, self.model)
                    _llm_span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, estimated
                    )
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
