import re

from openai.types.chat import ChatCompletion

from marie.engine.async_helper import run_coroutine_in_current_loop
from marie.excepts import MaxTokensExceededError, RepetitionError

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
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
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


@dataclass
class BatchResult:
    task_id: str
    response: Optional[str]
    error: Optional[Exception]


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(RepetitionError)
            | retry_if_exception_type(MaxTokensExceededError)
            | retry_if_exception_type(APIError)
            | retry_if_exception_type(APIConnectionError)
            | retry_if_exception_type(APITimeoutError)
            | retry_if_exception_type(RateLimitError)
        ),
        # before_sleep=before_sleep_log(logger, logging.WARNING),
    )


class BatchProcessor:
    def __init__(self, client, model_string, logger: MarieLogger):
        self.client = client
        self.model_string = model_string
        self.logger = logger
        if not isinstance(self.client, AsyncOpenAI):
            raise ValueError(
                "Client must be an instance of OpenAI API client for async operations."
            )

    def extract_reasoning_content(
        self, model_output: str
    ) -> Tuple[Optional[str], Optional[str]]:
        think_start_token = "<think>"
        think_end_token = "</think>"
        reasoning_regex = re.compile(
            rf"{think_start_token}(.*?){think_end_token}", re.DOTALL
        )

        # DeepSeek R1 doesn't generate <think> now.
        # Thus we assume the reasoning content is always at the start.
        # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        if think_end_token not in model_output:
            return None, model_output
        else:
            # Add a start token if it's missing to keep compatibility.
            if think_start_token not in model_output:
                model_output = f"{think_start_token}{model_output}"
            # Use a regex to find the reasoning content
            reasoning_content = reasoning_regex.findall(model_output)[0]
            end_index = len(f"{think_start_token}{reasoning_content}{think_end_token}")
            final_output = model_output[end_index:]

            if len(final_output) == 0:
                return reasoning_content, None

            return reasoning_content, final_output

    def extract_text_from_response(
        self, completion: Union[ChatCompletion, str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract text from the OpenAI API response or a direct string response.

        Args:
            completion: The response object from OpenAI API or a string.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing (reasoning_content, extracted_text)
        """

        if isinstance(completion, str):
            extracted_text = completion.strip()
            self.logger.info(
                f"Input was string. Extracted text length: {len(extracted_text)} characters."
            )
            reasoning_content, extracted_text = self.extract_reasoning_content(
                extracted_text
            )
            return reasoning_content, extracted_text

        if (
            not completion.choices
            or not hasattr(completion.choices[0].message, "content")
            or not completion.choices[0].message.content
        ):
            self.logger.warning("No text extracted from Response.")
            raise ValueError(f"No text extracted from response. : {completion}")

        extracted_text = completion.choices[0].message.content.strip()
        self.logger.info(f"Extracted text length: {len(extracted_text)} characters.")

        try:
            reasoning_content = completion.choices[0].message.reasoning_content
        except Exception:
            reasoning_content, extracted_text = self.extract_reasoning_content(
                extracted_text
            )

        return reasoning_content, extracted_text

    async def completion_non_streaming(
        self,
        messages,
        task_id,
        request_id,
        guided_json: Optional[Union[Dict, BaseModel, str]],
    ):
        """
        Asynchronously performs inference for a single request,
        streaming and stopping as soon as finish_reason is set.
        """

        # FIXME: HANDLE CONNECTION ERRORS
        # we should retry on connection errors

        start = time.time()
        self.logger.info(f"Request {request_id} - Task {task_id} - Starting inference.")
        try:
            # estimate/max tokens
            estimated_tokens = 4096 * 4
            max_tokens = estimated_tokens
            stop: List[str] = (
                []
            )  # no extra stop tokens by default, we can add domain-specific ones here

            # persist the prompt for debugging
            if (
                debug_raw_messages := os.getenv("DEBUG_RAW_MESSAGES", "False").lower()
                == "true"
            ):
                temp_dir = "/tmp/openai_messages"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"{task_id}_messages.json")
                try:
                    import json

                    with open(temp_path, "w") as f:
                        json.dump(messages, f, indent=2)
                    self.logger.info(f"Messages saved: {temp_path}")
                except Exception as e:
                    self.logger.error(f"Could not save messages: {e}")

            # Handle authentication errors before making API call
            if not os.getenv("OPENAI_API_KEY"):
                raise AuthenticationError(MISSING_API_KEY_ERROR_MESSAGE)

            stream = await self.client.chat.completions.create(
                model=self.model_string,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop,
                max_tokens=max_tokens,
                n=1,
                stream=True,
            )

            full_response = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content
                    # FIXME: make this configurable
                    # if False:
                    #     if _check_repetition(full_response, 3, 20, 30):
                    #         self.logger.error(
                    #             f"Request {request_id} - Task {task_id} - Repetition detected in response."
                    #         )
                    #         # write the response to a file for debugging
                    #         await self.save_debug_msg(
                    #             full_response, task_id, "repetition"
                    #         )
                    #         raise RepetitionError()
                reason = chunk.choices[0].finish_reason
                if reason is not None:
                    # finish_reason == "length" → hit max_tokens
                    if reason == "length":
                        await self.save_debug_msg(full_response, task_id, "max_tokens")
                        raise MaxTokensExceededError()
                    break

            total_time = time.time() - start
            self.logger.info(
                f"Request {request_id} - Task {task_id} - Completed in {total_time:.2f}s"
            )
            reasoning_content, extracted_text = self.extract_text_from_response(
                full_response
            )

            return task_id, extracted_text
        except (RepetitionError, MaxTokensExceededError) as e:
            self.logger.error(
                f"Request {request_id} - Task {task_id} - Error in completion_non_streaming: {e}, retrying..."
            )
            raise e
        except Exception as e:  # swallow all other exceptions for now
            self.logger.error(
                f"Request {request_id} - Task {task_id} - Error in completion_non_streaming: {e}"
            )
            return task_id, None

    async def save_debug_msg(self, full_response: str, task_id: str, tag: str):
        os.makedirs("/tmp/marie/llm-engine", exist_ok=True)
        with open(f"/tmp/marie/llm-engine/{task_id}_{tag}.txt", "w") as f:
            f.write(full_response)

    async def acompletion_with_retry(
        self, max_retries: int, messages, task_id, request_id, guided_json
    ):
        try:
            """Use tenacity to retry the completion call."""
            retry_decorator = _create_retry_decorator(max_retries=max_retries)
            completion_with_retry = retry_decorator(self.completion_non_streaming)

            return await completion_with_retry(
                messages=messages,
                task_id=task_id,
                request_id=request_id,
                guided_json=guided_json,
            )
        except Exception as e:
            self.logger.error(f"Request {request_id} – Task {task_id} failed: {e!r}")
            return task_id, None

    async def load_batched_requestXXXX(self, messages_list, request_id, guided_json):
        """
        Processes the batch of requests, assigning a unique task_id to each request.
        """
        tasks = [
            self.acompletion_with_retry(
                3, msgs, f"{request_id}_task_{i}", request_id, guided_json
            )
            for i, msgs in enumerate(messages_list)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, asyncio.CancelledError):
                logger.warning(f"One of the tasks was cancelled : {r}")
            elif isinstance(r, Exception):
                logger.error("Task failed:", exc_info=r)
            else:
                logger.info(f"Task completed successfully with result: {r}")

        # results = await asyncio.gather(*tasks)  # never raises
        ordered_responses: List[Optional[str]] = [resp for (_tid, resp) in results]
        return ordered_responses, results

    async def load_batched_request(self, messages_list, request_id, guided_json):
        """
        Processes the batch of requests, returning exactly:
          ordered_responses: List[Optional[str]]
          raw_results: List[Tuple[task_id, Optional[str]]]
        """

        async def safe_call(i, msgs):
            tid = f"{request_id}_task_{i}"
            try:
                resp = await self.acompletion_with_retry(
                    max_retries=3,
                    messages=msgs,
                    task_id=tid,
                    request_id=request_id,
                    guided_json=guided_json,
                )
                return BatchResult(tid, resp, None)
            except asyncio.CancelledError:
                # allow cancellation to propagate
                raise
            except Exception as e:
                return BatchResult(tid, None, e)

        # build all coroutines, but do NOT wrap them in create_task()
        coros = [safe_call(i, msgs) for i, msgs in enumerate(messages_list)]

        # run them all together
        try:
            batch_results: List[BatchResult] = await asyncio.gather(*coros)
        except asyncio.CancelledError:
            # if the outer caller cancels this batch, it propagates here
            # no need to manually cancel anything — gather will do it for us
            raise

        # convert BatchResult → (task_id, response)
        raw_results: List[Tuple[str, Optional[str]]] = [
            (br.task_id, br.response) for br in batch_results
        ]

        # extract just the responses (or None) in the same order
        ordered_responses: List[Optional[str]] = [resp for (_tid, resp) in raw_results]

        return ordered_responses, raw_results

    def batch_generate(
        self,
        messages_list: Union[List[str], List[List[Union[Image.Image, bytes, str]]]],
        system_prompt=Optional[str],
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Performs batch inference for multiple inputs, supporting both text-only and multimodal content.
        """
        request_id = str(uuid.uuid4())
        system_prompt = system_prompt or "Default system prompt"
        system_prompt = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        )
        self.logger.info(
            f"Request {request_id} - Initiating batch inference with {len(messages_list)} requests."
        )
        start_time = time.time()
        # batch_outputs, task_results = asyncio.run(
        #     self.load_batched_request(messages_list, request_id, guided_json)
        # )

        batch_outputs, task_results = run_coroutine_in_current_loop(
            self.load_batched_request(messages_list, request_id, guided_json)
        )

        for task_id, response in task_results:
            if response:
                self.logger.info(
                    f"Request {request_id} - Task {task_id} - Response received."
                )
            else:
                self.logger.error(
                    f"Request {request_id} - Task {task_id} - Response failed."
                )
                self.logger.error(response)

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Request {request_id} - Batch inference completed in {elapsed_time:.2f} sec"
        )

        return batch_outputs
