import logging
import os
import time
import traceback
from typing import Callable, Dict, List, Optional, Union

import diskcache as dc
from PIL import Image
from pydantic import BaseModel

from marie.engine.base import EngineLM
from marie.engine.batch_processor import BatchProcessor
from marie.engine.completion_contract import (
    CompletionCallParams,
    RequestContext,
    build_completion_call,
)
from marie.engine.engine_utils import (
    convert_openai_to_transformers_format,
    extract_text_info,
    is_batched_request,
    open_ai_like_formatting,
)
from marie.engine.openai_compat import build_async_openai_client

MISSING_API_KEY_ERROR_MESSAGE = """No API key found for LLM.
E.g. to use openai Please set the OPENAI_API_KEY environment variable or \
openai.api_key prior to initialization.
API keys can be found or created at \
https://platform.openai.com/account/api-keys
"""
INVALID_API_KEY_ERROR_MESSAGE = """Invalid LLM API key."""


# TODO: FIX DEFAULT_SYSTEM_PROMPT messages


def _check_repetition(
    text: str, min_repeats: int = 3, min_ngram_size: int = 1, max_ngram_size: int = 20
) -> bool:
    """
    Return True if any n-gram of size between min_ngram_size and max_ngram_size
    repeats at least min_repeats times consecutively at the very end of text.
    """
    tokens = text.split()
    L = len(tokens)
    # the largest n we could possibly repeat min_repeats times
    possible_max_n = L // min_repeats
    # clamp our n-gram window
    start_n = max(1, min_ngram_size)
    end_n = min(max_ngram_size, possible_max_n)

    if start_n > end_n:
        return False

    for n in range(start_n, end_n + 1):
        tail = tokens[-n:]
        repeats = 1
        # look back to see if the same tail appears min_repeats times
        for k in range(2, min_repeats + 1):
            start = -k * n
            end = -(k - 1) * n
            if tokens[start:end] == tail:
                repeats += 1
            else:
                break
        if repeats >= min_repeats:
            return True

    return False


class OpenAIEngine(EngineLM):
    """
    OpenAIEngine is a wrapper around the OpenAI API.
    Supports both multimodal and text-based models.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(
        self,
        model_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = True,
        cache: Union[dc.Cache, bool] = False,
        processor_kwargs: Dict = None,
        base_url: str = None,
        max_concurrency: Optional[int] = None,
        batch_timeout: Optional[float] = None,
        **kwargs,
    ):
        self.validate()
        super().__init__(
            model_string=model_name,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        processor_kwargs = processor_kwargs or {}
        api_key = os.getenv("OPENAI_API_KEY")

        self.client = build_async_openai_client(api_key=api_key, base_url=base_url)

        # Derive backend address for the circuit breaker key
        backend_address = base_url or "https://api.openai.com"

        self.client.models.list()
        self.model_string = model_name

        self.batch_processor = BatchProcessor(
            self.client,
            self.model_string,
            logger=self.logger,
            max_concurrency=max_concurrency,
            batch_timeout=batch_timeout,
            backend_address=backend_address,
            queue_enabled=kwargs.get("queue_enabled"),
            queue_client=kwargs.get("queue_client"),
            queue_pool_id=kwargs.get("queue_pool_id"),
            queue_producer_id=kwargs.get("queue_producer_id"),
            queue_valkey_url=kwargs.get("queue_valkey_url"),
        )

    def validate(self) -> None:
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError(
                "Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models."
            )

    def build_queue_dispatcher(self):
        return self.batch_processor.build_queue_dispatcher()

    def __del__(self):
        """Detach client to prevent cleanup errors during GC."""
        try:
            if hasattr(self, "batch_processor") and self.batch_processor is not None:
                close = getattr(self.batch_processor, "close", None)
                if callable(close):
                    close()
            if hasattr(self, "client") and self.client is not None:
                # Detach internal httpx client to prevent async cleanup issues
                if hasattr(self.client, "_client"):
                    self.client._client = None
                self.client = None
        except Exception:
            pass  # Never raise in __del__

    def _generate_from_single_prompt(
        self,
        content: Union[str, List[str]],
        system_prompt: str = None,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        **kwargs,
    ):
        return self.openai_generate(
            content,
            system_prompt,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_grammar=guided_grammar,
            guided_json_object=guided_json_object,
            guided_backend=guided_backend,
            guided_whitespace_pattern=guided_whitespace_pattern,
            on_result=on_result,
            **kwargs,
        )

    def _generate_from_multiple_input(
        self,
        content: Union[
            List[List[Union[str, bytes, Image.Image]]],
            List[Union[str, bytes, Image.Image]],
        ],
        system_prompt=None,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        **kwargs,
    ):
        return self.openai_generate(
            content,
            system_prompt,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_grammar=guided_grammar,
            guided_json_object=guided_json_object,
            guided_backend=guided_backend,
            guided_whitespace_pattern=guided_whitespace_pattern,
            on_result=on_result,
            **kwargs,
        )

    def __call__(
        self,
        content: Union[
            str,
            List[str],
            List[Union[Image.Image, bytes, str]],
            List[List[Union[Image.Image, bytes, str]]],
        ],
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        **kwargs,
    ):
        return self.generate(
            content,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_grammar=guided_grammar,
            guided_json_object=guided_json_object,
            guided_backend=guided_backend,
            guided_whitespace_pattern=guided_whitespace_pattern,
            **kwargs,
        )

    def _generate_prompt(self, messages, system_prompt: str):
        """
        Generates a formatted prompt based on user input and model requirements.

        :param messages: List of messages in OpenAI-like format.
        :param system_prompt: System-level prompt override.
        :return: Formatted prompt string.
        """
        if self.prompt:
            text_info = extract_text_info(messages)
            user_text = text_info[0]["text"] if text_info else ""
            formatted_prompt = self.prompt.replace(
                "SYSTEM_PROMPT_PLACEHOLDER", system_prompt or ""
            )
            formatted_prompt = formatted_prompt.replace(
                "QUESTION_PLACEHOLDER", user_text
            )
            return formatted_prompt

        if not self.is_multimodal:
            messages = convert_openai_to_transformers_format(messages)

        return self.tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=True
        )

    def batch_generate(
        self,
        batch_content: Union[List[str], List[List[Union[Image.Image, bytes, str]]]],
        system_prompt=None,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Performs batch inference for multiple inputs, supporting both text-only and multimodal content.

        This function processes input data, constructs formatted prompts, and executes batched inference
        with optimized error handling and logging.

        :param batch_content: A list of text prompts or multimodal inputs (image, text pairs).
        :param system_prompt: Optional system-level instructions for the model.
        :param on_result: Optional callback invoked when each task completes.
                         Signature: (task_id: str, response: Optional[str]) -> None
                         This enables incremental result processing (e.g., writing to disk as each completes).
        :param kwargs: Additional inference parameters. Recognized keys include
            reasoning_model (bool), mm_processor_kwargs (dict), completion_params
            (dict), metadata (dict), and request_contexts (list[RequestContext | None])
            aligned with batch_content.

        :return: A list of generated outputs corresponding to each input in batch_content.
        """
        calls = self._build_completion_calls(
            batch_content=batch_content,
            system_prompt=system_prompt,
            guided_json=guided_json,
            **kwargs,
        )

        self.logger.info(f"Initiating batch inference with {len(calls)} requests.")
        start_time = time.time()
        try:
            bp_kwargs = {}
            if "metadata" in kwargs:
                bp_kwargs["metadata"] = kwargs["metadata"]

            ordered_outputs = self.batch_processor.batch_generate_calls(
                calls=calls,
                on_result=on_result,
                **bp_kwargs,
            )
        except Exception:
            self.logger.error("Batch inference failed:\n%s", traceback.format_exc())
            raise  # Propagate to caller

        elapsed_time = time.time() - start_time
        self.logger.info(f"Batch inference completed in {elapsed_time:.2f} sec")

        return ordered_outputs

    def _build_completion_calls(
        self,
        *,
        batch_content: Union[List[str], List[List[Union[Image.Image, bytes, str]]]],
        system_prompt=None,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        **kwargs,
    ) -> List[CompletionCallParams]:
        effective_system_prompt = system_prompt or self.system_prompt
        effective_system_prompt = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        )

        if self.is_multimodal:
            default_options = {
                "min_pixels": 512 * 28 * 28,
                "max_pixels": 2560 * 28 * 28,
            }
            mm_kwargs = kwargs.get("mm_processor_kwargs", default_options)
            batch_content = [
                open_ai_like_formatting(content, True, **mm_kwargs)
                for content in batch_content
            ]

        reasoning_model = kwargs.get("reasoning_model", False)
        completion_params = kwargs.get("completion_params")
        request_contexts: Optional[List[RequestContext | None]] = kwargs.get(
            "request_contexts"
        )

        def transform_prompt_for_reasoning(reasoning_enabled: bool, prompt):
            if not reasoning_enabled:
                return prompt
            return f"""{prompt}

                ## Response Format

                Reply with JSON object ONLY."""

        messages_list = [
            [
                {"role": "system", "content": effective_system_prompt},
                {
                    "role": "user",
                    "content": transform_prompt_for_reasoning(reasoning_model, content),
                },
            ]
            for content in batch_content
        ]

        if request_contexts is not None and len(request_contexts) != len(messages_list):
            self.logger.warning(
                "Ignoring request_contexts length mismatch: expected=%s actual=%s",
                len(messages_list),
                len(request_contexts),
            )
            request_contexts = None

        return [
            build_completion_call(
                model=self.model_string,
                messages=messages,
                default_completion_params=self.batch_processor.default_completion_params,
                completion_params=completion_params,
                guided_json=guided_json,
                max_tokens=4096 * 4,
                stop=[],
                n=1,
                stream=False,
                context=(
                    request_contexts[index] if request_contexts is not None else None
                ),
            )
            for index, messages in enumerate(messages_list)
        ]

    def openai_generate(
        self,
        content,
        system_prompt=None,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        guided_regex: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_grammar: Optional[str] = None,
        guided_json_object: Optional[bool] = None,
        guided_backend: Optional[str] = None,
        guided_whitespace_pattern: Optional[str] = None,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate text using the model.

        Args:
            content: Input content (single or batch)
            system_prompt: Optional system prompt
            guided_json: Optional JSON schema for guided generation
            guided_regex: Optional regex for guided generation
            guided_choice: Optional choices for guided generation
            guided_grammar: Optional grammar for guided generation
            guided_json_object: Optional JSON object flag
            guided_backend: Optional guided backend
            guided_whitespace_pattern: Optional whitespace pattern
            on_result: Optional callback invoked when each task completes.
                      Signature: (task_id: str, response: Optional[str]) -> None
            **kwargs: Additional arguments

        Returns:
            Generated text (single string or list of strings)
        """
        batched = is_batched_request(content)
        if not batched:
            content = [content]
        results = self.batch_generate(
            content,
            system_prompt,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_grammar=guided_grammar,
            guided_json_object=guided_json_object,
            guided_backend=guided_backend,
            guided_whitespace_pattern=guided_whitespace_pattern,
            on_result=on_result,
            **kwargs,
        )
        if not batched:
            return results[0]
        return results
