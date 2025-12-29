"""
Token Counter - Hybrid token counting using API usage or tiktoken fallback.

Provides accurate token counting for LLM tracking, prioritizing API-reported
usage when available, falling back to tiktoken estimation.
"""

import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from marie.llm_tracking.config import get_settings
from marie.llm_tracking.types import Usage

logger = logging.getLogger(__name__)

# Model to tiktoken encoding mapping
MODEL_ENCODING_MAP: Dict[str, str] = {
    # OpenAI GPT-4 family
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    # OpenAI GPT-3.5 family
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    # OpenAI o1 family
    "o1": "o200k_base",
    "o1-preview": "o200k_base",
    "o1-mini": "o200k_base",
    # OpenAI embeddings
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # Anthropic Claude (approximation using cl100k_base)
    "claude-3-opus": "cl100k_base",
    "claude-3-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
    "claude-3-5-sonnet": "cl100k_base",
    "claude-3-5-haiku": "cl100k_base",
    # Default fallback
    "default": "cl100k_base",
}


@lru_cache(maxsize=8)
def get_encoding(model: str) -> Optional[Any]:
    """
    Get tiktoken encoding for a model (cached).

    Args:
        model: Model name or identifier

    Returns:
        tiktoken.Encoding or None if tiktoken not available
    """
    try:
        import tiktoken
    except ImportError:
        logger.warning("tiktoken not installed, token counting disabled")
        return None

    # Normalize model name
    model_lower = model.lower()

    # Find encoding name
    encoding_name = MODEL_ENCODING_MAP.get(model_lower)

    if encoding_name is None:
        # Try to find a partial match
        for key, enc in MODEL_ENCODING_MAP.items():
            if key in model_lower or model_lower in key:
                encoding_name = enc
                break

    if encoding_name is None:
        # Fall back to default
        encoding_name = MODEL_ENCODING_MAP["default"]

    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to get encoding {encoding_name} for model {model}: {e}")
        try:
            # Ultimate fallback
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens_text(text: str, model: str) -> int:
    """
    Count tokens in a text string.

    Args:
        text: Text to count tokens for
        model: Model name for encoding selection

    Returns:
        Token count
    """
    if not text:
        return 0

    encoding = get_encoding(model)
    if encoding is None:
        # Rough approximation: ~4 chars per token
        return len(text) // 4

    try:
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}")
        return len(text) // 4


def count_tokens_messages(
    messages: List[Dict[str, Any]],
    model: str,
) -> int:
    """
    Count tokens for a list of chat messages.

    Accounts for message formatting overhead (role, content delimiters).

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name for encoding selection

    Returns:
        Token count
    """
    if not messages:
        return 0

    encoding = get_encoding(model)
    if encoding is None:
        # Rough approximation
        total_chars = sum(
            len(str(m.get("content", ""))) + len(str(m.get("role", "")))
            for m in messages
        )
        return total_chars // 4

    # Token overhead per message (varies by model, this is approximate)
    tokens_per_message = 4  # <im_start>, role, <im_sep>, <im_end>
    tokens_per_name = -1  # Adjusts for 'name' field

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue
            if isinstance(value, str):
                try:
                    num_tokens += len(encoding.encode(value))
                except Exception:
                    num_tokens += len(value) // 4
            elif isinstance(value, list):
                # Handle multimodal content (text + images)
                for item in value:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            try:
                                num_tokens += len(encoding.encode(text))
                            except Exception:
                                num_tokens += len(text) // 4
                        elif item.get("type") == "image_url":
                            # Approximate token cost for images
                            # GPT-4V uses ~85 tokens for low-res, ~765 for high-res
                            detail = item.get("image_url", {}).get("detail", "auto")
                            if detail == "low":
                                num_tokens += 85
                            else:
                                num_tokens += 765
            if key == "name":
                num_tokens += tokens_per_name

    # Every reply is primed with <im_start>assistant
    num_tokens += 2

    return num_tokens


def count_tokens_json(data: Any, model: str) -> int:
    """
    Count tokens for arbitrary JSON-serializable data.

    Args:
        data: Data to serialize and count
        model: Model name for encoding selection

    Returns:
        Token count
    """
    if data is None:
        return 0

    try:
        text = json.dumps(data, ensure_ascii=False)
        return count_tokens_text(text, model)
    except (TypeError, ValueError):
        return 0


def extract_usage_from_response(
    response: Any,
    model: str,
) -> Usage:
    """
    Extract token usage from an LLM API response.

    Supports OpenAI, Anthropic, and similar response formats.

    Args:
        response: API response object or dict
        model: Model name (for fallback counting)

    Returns:
        Usage object with token counts
    """
    usage = Usage()

    # Try to extract from response object (OpenAI SDK style)
    if hasattr(response, "usage") and response.usage is not None:
        usage_obj = response.usage
        usage.input_tokens = getattr(usage_obj, "prompt_tokens", None)
        usage.output_tokens = getattr(usage_obj, "completion_tokens", None)
        usage.total_tokens = getattr(usage_obj, "total_tokens", None)

        # Handle detailed usage (OpenAI o1 models, etc.)
        if hasattr(usage_obj, "prompt_tokens_details"):
            details = usage_obj.prompt_tokens_details
            if details:
                if hasattr(details, "cached_tokens"):
                    usage.usage_details["cached_input_tokens"] = details.cached_tokens

        if hasattr(usage_obj, "completion_tokens_details"):
            details = usage_obj.completion_tokens_details
            if details:
                if hasattr(details, "reasoning_tokens"):
                    usage.usage_details["reasoning_tokens"] = details.reasoning_tokens

        return usage

    # Try dict access
    if isinstance(response, dict):
        usage_data = response.get("usage", {})
        if usage_data:
            # OpenAI format
            usage.input_tokens = usage_data.get("prompt_tokens")
            usage.output_tokens = usage_data.get("completion_tokens")
            usage.total_tokens = usage_data.get("total_tokens")

            # Handle input_tokens/output_tokens format (newer APIs)
            if usage.input_tokens is None:
                usage.input_tokens = usage_data.get("input_tokens")
            if usage.output_tokens is None:
                usage.output_tokens = usage_data.get("output_tokens")

            return usage

    return usage


def count_tokens_with_fallback(
    input_data: Optional[Any],
    output_data: Optional[Any],
    api_usage: Optional[Usage],
    model: str,
) -> Usage:
    """
    Get token counts with API usage preferred, tiktoken fallback.

    Args:
        input_data: Input messages or text
        output_data: Output text or structured data
        api_usage: Usage from API response (if available)
        model: Model name for tiktoken

    Returns:
        Usage object with best available counts
    """
    settings = get_settings()

    # If API usage is complete, use it
    if (
        api_usage
        and api_usage.input_tokens is not None
        and api_usage.output_tokens is not None
    ):
        # Ensure total is set
        if api_usage.total_tokens is None:
            api_usage.total_tokens = api_usage.input_tokens + api_usage.output_tokens
        return api_usage

    # Initialize with API usage if partial
    result = Usage()
    if api_usage:
        result.input_tokens = api_usage.input_tokens
        result.output_tokens = api_usage.output_tokens
        result.total_tokens = api_usage.total_tokens
        result.usage_details = api_usage.usage_details.copy()

    # Fall back to tiktoken if enabled
    if settings.TOKEN_COUNTING_ENABLED:
        effective_model = model or settings.DEFAULT_TOKENIZER_MODEL

        # Count input if not from API
        if result.input_tokens is None and input_data is not None:
            if isinstance(input_data, list) and len(input_data) > 0:
                # Assume chat messages format
                if isinstance(input_data[0], dict) and "role" in input_data[0]:
                    result.input_tokens = count_tokens_messages(
                        input_data, effective_model
                    )
                else:
                    result.input_tokens = count_tokens_json(input_data, effective_model)
            elif isinstance(input_data, str):
                result.input_tokens = count_tokens_text(input_data, effective_model)
            else:
                result.input_tokens = count_tokens_json(input_data, effective_model)

        # Count output if not from API
        if result.output_tokens is None and output_data is not None:
            if isinstance(output_data, str):
                result.output_tokens = count_tokens_text(output_data, effective_model)
            else:
                result.output_tokens = count_tokens_json(output_data, effective_model)

    # Calculate total
    if result.input_tokens is not None and result.output_tokens is not None:
        result.total_tokens = result.input_tokens + result.output_tokens
    elif result.total_tokens is None:
        if result.input_tokens is not None:
            result.total_tokens = result.input_tokens
        elif result.output_tokens is not None:
            result.total_tokens = result.output_tokens

    return result
