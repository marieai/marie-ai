import re

from openai.types.chat import ChatCompletion

from marie.engine.batch_processor import BatchProcessor
from marie.engine.output_parser import parse_json_markdown
from marie.excepts import MaxTokensExceededError, RepetitionError
from marie.utils.utils import get_exception_traceback

try:
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        AsyncOpenAI,
        AuthenticationError,
        OpenAI,
        RateLimitError,
    )
except ImportError:
    raise ImportError(
        "If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables."
    )

import asyncio
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import diskcache as dc
from PIL import Image
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

from marie.engine.base import EngineLM
from marie.engine.engine_utils import (
    convert_openai_to_transformers_format,
    extract_text_info,
    is_batched_request,
    open_ai_like_formatting,
)
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger

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


def run_coroutine_in_current_loop(coroutine):
    """
    Runs `coroutine` to completion, even if we're inside a running loop.
    - Outside any loop: uses asyncio.run()
    - Inside a loop: spins up a fresh loop in a background thread,
      runs `coroutine`, shuts down async generators, then drains
      any *other* pending tasks before closing.
    """
    try:
        # If no loop is running here, just run normally.
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result_q = queue.Queue()

    def _thread_target():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        async def _runner():
            result = await coroutine
            # 2) clean up any async generators
            await new_loop.shutdown_asyncgens()

            # 3) drain *other* pending tasks, excluding this one
            current = asyncio.current_task()
            pending = [
                t for t in asyncio.all_tasks() if not t.done() and t is not current
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            return result

        try:
            res = new_loop.run_until_complete(_runner())
            result_q.put((True, res))
        except Exception as exc:
            result_q.put((False, exc))
        finally:
            new_loop.close()

    t = threading.Thread(target=_thread_target)
    t.start()
    t.join()

    ok, payload = result_q.get()
    if not ok:
        raise payload
    return payload


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
        **kwargs,
    ):
        self.validate()
        super().__init__(
            model_string=model_name,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache,
        )
        self.logger = MarieLogger(self.__class__.__name__).logger
        processor_kwargs = processor_kwargs or {}
        api_key = os.getenv("OPENAI_API_KEY")
        if not base_url:
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        models = self.client.models.list()
        self.model_string = model_name

        self.batch_processor = BatchProcessor(
            self.client, self.model_string, logger=self.logger
        )

    def validate(self) -> None:
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError(
                "Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models."
            )

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
            **kwargs,
        )

    # @cached
    # @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
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
        **kwargs,
    ) -> List[str]:
        """
        Performs batch inference for multiple inputs, supporting both text-only and multimodal content.

        This function processes input data, constructs formatted prompts, and executes batched inference
        with optimized error handling and logging.

        :param batch_content: A list of text prompts or multimodal inputs (image, text pairs).
        :param system_prompt: Optional system-level instructions for the model.
        :param kwargs: Additional inference parameters.
        :return: A list of generated outputs corresponding to each input in batch_content.
        """
        system_prompt = system_prompt or self.system_prompt
        system_prompt = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        )

        if self.is_multimodal:
            batch_content = [
                open_ai_like_formatting(content, True) for content in batch_content
            ]

        reasoning_model = kwargs.get("reasoning_model", False)

        # https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724
        # https://github.com/trustsight-io/deepseek-go/issues/2
        def transform_prompt_for_reasoning(reasoning_model: bool, prompt: str):
            if reasoning_model:
                return f"""{prompt}

                ## Response Format

                Reply with JSON object ONLY."""
            else:
                return prompt

        messages_list = [
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": transform_prompt_for_reasoning(reasoning_model, content),
                },
            ]
            for content in batch_content
        ]

        return_stats = kwargs.get("return_stats", False)

        self.logger.info(
            f"Initiating batch inference with {len(batch_content)} requests."
        )
        start_time = time.time()
        try:
            ordered_outputs = self.batch_processor.batch_generate(
                messages_list, guided_json=guided_json
            )
        except Exception as e:
            self.logger.error(f"Batch inference failed:\n" + get_exception_traceback())
            return ["ERROR: Inference failed"] * len(batch_content)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Batch inference completed in {elapsed_time:.2f} sec")

        return ordered_outputs

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
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate text using the  model."""
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
            **kwargs,
        )
        if not batched:
            return results[0]
        return results


if __name__ == "__main__":
    # https://github.com/novex-ai/parallel-parrot/blob/main/parallel_parrot/openai_api.py
    # https://github.com/openai/openai-cookbook/blob/main/examples/Structured_Outputs_Intro.ipynb
    # https://github.com/openai/openai-cookbook/blob/main/examples/Unit_test_writing_using_a_multi-step_prompt.ipynb
    # https://github.com/openai/openai-cookbook/blob/main/examples/Named_Entity_Recognition_to_enrich_text.ipynb
    # https://github.com/openai/openai-cookbook/blob/main/examples/Parse_PDF_docs_for_RAG.ipynb
    # https://docs.boundaryml.com/examples/prompt-engineering/chain-of-thought
    os.environ['OPENAI_API_KEY'] = 'EMPTY'

    text = "Need to fuzzy match. Data row at52. Totals? Not sure here. Wait, line52 is a data row but there's no totals line after. Wait, line52 is the only data row here. Wait, looking at line52: it's a data row, but after that, there's no totals line. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data row here. Wait, line52 is the only data"
    status = _check_repetition(text, 3, 20, 30)
    print("status : ", status)

    if True:
        engine = OpenAIEngine(
            model_name='deepseek_r1_32',
            system_prompt="You are a helpful assistant",
            is_multimodal=False,
            cache=False,
            processor_kwargs=None,
            # base_url="http://localhost:8090/v1",
            base_url="http://0.0.0.0:4000",
        )

        class Animal(BaseModel):
            name: str
            fact: str
            city: str

        prompts = []
        for i in range(1, 2):
            prompt = f"Tell me an animal fact, for animal from Poland  for random city # {i} : JSON Schema : {Animal.model_json_schema()}"
            prompts.append(prompt)

        results = engine.generate(prompts)
        print(results)

        for result in results:
            json_dict = parse_json_markdown(result)
            animal = Animal.model_validate(json_dict)
            print(animal)
