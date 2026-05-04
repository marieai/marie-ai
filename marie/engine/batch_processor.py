import re

from openai.types.chat import ChatCompletion
from openinference.semconv.trace import SpanAttributes
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import StatusCode

from marie.engine.async_helper import run_coroutine_in_current_loop
from marie.excepts import (
    BatchExecutionError,
    CircuitOpenError,
    MaxTokensExceededError,
    RepetitionError,
)
from marie.instrumentation import (
    get_tracer,
    set_llm_io,
    start_as_current_span,
    start_span,
)
from marie.instrumentation.openinference import infer_llm_system

_tracer = get_tracer("marie.engine.batch_processor")


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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.queue_io import ListQueueClient, ValkeyListQueueClient
from marie.engine.llm_queue.result_types import BatchResult
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.serve.networking.balancer.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)

MISSING_API_KEY_ERROR_MESSAGE = """No API key found for LLM.
E.g. to use openai Please set the OPENAI_API_KEY environment variable or \
openai.api_key prior to initialization.
API keys can be found or created at \
https://platform.openai.com/account/api-keys
"""
INVALID_API_KEY_ERROR_MESSAGE = """Invalid LLM API key."""


def _is_pool_timeout(exc: BaseException) -> bool:
    """Check if an APITimeoutError was caused by httpx pool exhaustion."""
    try:
        import httpx

        cause = exc.__cause__
        return isinstance(cause, httpx.PoolTimeout)
    except ImportError:
        return False


def _should_retry(exc: BaseException) -> bool:
    """Return True if the exception is retryable.

    Excludes httpx.PoolTimeout (wrapped as APITimeoutError) because retrying
    when the connection pool is exhausted only makes things worse.
    """
    if isinstance(exc, APITimeoutError) and _is_pool_timeout(exc):
        logger.warning(
            "httpx.PoolTimeout detected — skipping retry to avoid pool exhaustion cascade"
        )
        return False
    return True


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
        )
        & retry_if_exception(_should_retry),
        # before_sleep=before_sleep_log(logger, logging.WARNING),
    )


class BatchProcessor:
    # Maximum number of concurrent requests sent to the LLM backend.
    # Keeps well under the httpx connection pool limit (40) so that retries
    # and other callers always have headroom.
    DEFAULT_MAX_CONCURRENCY = 20
    DEFAULT_BATCH_TIMEOUT = 600.0  # seconds

    def __init__(
        self,
        client,
        model_string,
        logger: MarieLogger,
        default_completion_params: Optional[Dict[str, Any]] = None,
        max_concurrency: Optional[int] = None,
        batch_timeout: Optional[float] = None,
        backend_address: Optional[str] = None,
        queue_enabled: Optional[bool] = None,
        queue_client: Optional[ListQueueClient] = None,
        queue_pool_id: Optional[str] = None,
        queue_producer_id: Optional[str] = None,
        queue_valkey_url: Optional[str] = None,
    ):
        self.client = client
        self.model_string = model_string
        self.logger = logger
        self.max_concurrency = max_concurrency or self.DEFAULT_MAX_CONCURRENCY
        self.batch_timeout = batch_timeout or self.DEFAULT_BATCH_TIMEOUT
        self.backend_address = backend_address or "unknown"
        if not isinstance(self.client, AsyncOpenAI):
            raise ValueError(
                "Client must be an instance of OpenAI API client for async operations."
            )
        _fallbacks = {
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "extra_body": None,
        }
        if default_completion_params:
            _fallbacks.update(default_completion_params)
        self.default_completion_params = _fallbacks
        self._shared_request_semaphore: Optional[asyncio.Semaphore] = None
        self._shared_request_semaphore_loop: Optional[asyncio.AbstractEventLoop] = None

        # Circuit breaker keyed by backend_address
        self._circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(),
            logger=MarieLogger("BatchProcessor.CircuitBreaker"),
        )
        # Async gate lock so HALF_OPEN slot reservation is serialized
        self._gate_lock: Optional[asyncio.Lock] = None
        self._gate_lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue_client = queue_client
        self._queued_executor = None
        self._queue_config = LlmQueueConfig.from_env(
            enabled=queue_enabled,
            valkey_url=queue_valkey_url,
            pool_id=queue_pool_id,
            producer_id=queue_producer_id,
        )

    def _queue_enabled(self) -> bool:
        config = getattr(self, "_queue_config", None)
        return bool(config and config.enabled)

    def _get_queue_client(self) -> ListQueueClient:
        if self._queue_client is not None:
            return self._queue_client
        if not self._queue_config.valkey_url:
            raise ValueError(
                "LLM queue is enabled but LLM_QUEUE_VALKEY_URL (or queue_valkey_url) is not configured."
            )
        self._queue_client = ValkeyListQueueClient(self._queue_config.valkey_url)
        return self._queue_client

    def _get_queued_executor(self):
        if self._queued_executor is None:
            from marie.engine.llm_queue.submitter import QueuedBatchExecutor

            self._queued_executor = QueuedBatchExecutor(
                queue_client=self._get_queue_client(),
                config=self._queue_config,
                model_string=self.model_string,
                logger=self.logger,
            )
        return self._queued_executor

    def build_queue_dispatcher(self):
        from marie.engine.llm_queue.adapters.litellm import LiteLlmExecutionAdapter
        from marie.engine.llm_queue.dispatcher import QueuedBatchDispatcher

        return QueuedBatchDispatcher(
            queue_client=self._get_queue_client(),
            execution_adapter=LiteLlmExecutionAdapter(self),
            config=self._queue_config,
            logger=self.logger,
        )

    def close(self) -> None:
        queued_executor = getattr(self, "_queued_executor", None)
        if queued_executor is not None:
            close = getattr(queued_executor, "close", None)
            if callable(close):
                close()
            self._queued_executor = None

        queue_client = getattr(self, "_queue_client", None)
        if queue_client is not None:
            close = getattr(queue_client, "close", None)
            if callable(close):
                close()
            self._queue_client = None

    def _get_request_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if (
            self._shared_request_semaphore is None
            or self._shared_request_semaphore_loop is not loop
        ):
            self._shared_request_semaphore = asyncio.Semaphore(self.max_concurrency)
            self._shared_request_semaphore_loop = loop
        return self._shared_request_semaphore

    def _get_gate_lock(self) -> asyncio.Lock:
        """Per-event-loop async lock for serializing HALF_OPEN slot reservation."""
        loop = asyncio.get_running_loop()
        if self._gate_lock is None or self._gate_lock_loop is not loop:
            self._gate_lock = asyncio.Lock()
            self._gate_lock_loop = loop
        return self._gate_lock

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
        completion_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """
        Asynchronously performs inference for a single request,
        streaming and stopping as soon as finish_reason is set.
        """
        start = time.time()
        self.logger.info(f"Request {request_id} - Task {task_id} - Starting inference.")

        with start_as_current_span(
            _tracer,
            "BatchProcessor.completion",
            span_kind="llm",
        ) as span:
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self.model_string)
            span.set_attribute(
                SpanAttributes.LLM_SYSTEM, infer_llm_system(self.model_string)
            )
            span.set_attribute("marie.task_id", task_id)
            span.set_attribute("marie.request_id", request_id)
            if metadata:
                for mk, mv in metadata.items():
                    span.set_attribute(f"marie.{mk}", str(mv))
            set_llm_io(span, input_messages=messages)

            try:
                # estimate/max tokens
                estimated_tokens = 4096 * 4
                max_tokens = estimated_tokens
                stop: List[str] = (
                    []
                )  # no extra stop tokens by default, we can add domain-specific ones here

                # persist the prompt for debugging
                if (
                    debug_raw_messages := os.getenv(
                        "DEBUG_RAW_MESSAGES", "False"
                    ).lower()
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

                effective = dict(self.default_completion_params)
                if completion_params:
                    effective.update(completion_params)

                extra_body = effective.pop("extra_body", None)

                create_kwargs = dict(
                    model=self.model_string,
                    messages=messages,
                    temperature=effective.get("temperature", 0.0),
                    top_p=effective.get("top_p", 1.0),
                    frequency_penalty=effective.get("frequency_penalty", 0.0),
                    presence_penalty=effective.get("presence_penalty", 0.0),
                    stop=stop,
                    max_tokens=max_tokens,
                    n=1,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                if extra_body is not None:
                    create_kwargs["extra_body"] = extra_body

                stream = await self.client.chat.completions.create(**create_kwargs)

                full_response = ""
                provider_usage = None
                finish_reason = None
                async for chunk in stream:
                    # Capture usage from any chunk (final chunk has usage only)
                    if hasattr(chunk, "usage") and chunk.usage:
                        provider_usage = chunk.usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_response += delta.content
                    reason = chunk.choices[0].finish_reason
                    if reason is not None:
                        finish_reason = reason
                        # finish_reason == "length" → hit max_tokens
                        if reason == "length":
                            await self.save_debug_msg(
                                full_response, task_id, "max_tokens"
                            )
                            raise MaxTokensExceededError()

                total_time = time.time() - start
                self.logger.info(
                    f"Request {request_id} - Task {task_id} - Completed in {total_time:.2f}s"
                )
                reasoning_content, extracted_text = self.extract_text_from_response(
                    full_response
                )

                set_llm_io(span, output_messages=extracted_text)
                span.set_attribute("marie.latency_seconds", total_time)
                span.set_attribute("marie.has_reasoning", reasoning_content is not None)

                # Set token usage from provider response
                if provider_usage:
                    prompt_tokens = getattr(provider_usage, "prompt_tokens", 0) or 0
                    completion_tokens = (
                        getattr(provider_usage, "completion_tokens", 0) or 0
                    )
                    span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
                    )
                    span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens
                    )
                    span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                        prompt_tokens + completion_tokens,
                    )

                span.set_status(StatusCode.OK)
                self._circuit_breaker.record_success(self.backend_address)

                return task_id, extracted_text
            except (RepetitionError, MaxTokensExceededError) as e:
                self.logger.error(
                    f"Request {request_id} - Task {task_id} - Error in completion_non_streaming: {e}, retrying..."
                )
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise e
            except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as e:
                self.logger.error(
                    f"Request {request_id} - Task {task_id} - API error in completion_non_streaming: {e}"
                )
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                self._circuit_breaker.record_failure(self.backend_address)
                raise  # Let tenacity retry these
            except Exception as e:
                self.logger.error(
                    f"Request {request_id} - Task {task_id} - Error in completion_non_streaming: {e}"
                )
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                # Propagate unexpected exceptions so the batch layer records the
                # task as failed instead of misclassifying it as a None response.
                raise

    async def save_debug_msg(self, full_response: str, task_id: str, tag: str):
        os.makedirs("/tmp/marie/llm-engine", exist_ok=True)
        with open(f"/tmp/marie/llm-engine/{task_id}_{tag}.txt", "w") as f:
            f.write(full_response)

    async def acompletion_with_retry(
        self,
        max_retries: int,
        messages,
        task_id,
        request_id,
        guided_json,
        completion_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
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
                completion_params=completion_params,
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(
                f"Request {request_id} – Task {task_id} failed after retries: {e!r}"
            )
            raise  # Propagate to caller

    async def load_batched_request(
        self,
        messages_list,
        request_id,
        guided_json,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        completion_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        metadata_list: Optional[List[Optional[Dict[str, str]]]] = None,
    ):
        """
        Processes the batch of requests, invoking on_result as each completes.

        Args:
            messages_list: List of message lists to process
            request_id: Unique identifier for this batch request
            guided_json: Optional JSON schema for guided generation
            on_result: Optional callback invoked when each task completes.
                       Signature: (task_id: str, response: Optional[str]) -> None

        Returns:
            Tuple of (ordered_responses, raw_results):
              ordered_responses: List[Optional[str]] - responses in original order
              raw_results: List[BatchResult] - full BatchResult objects

        Raises:
            asyncio.TimeoutError: If the batch exceeds batch_timeout seconds.
        """

        # Share one limiter across all overlapping batch calls that use this
        # engine/client instance. Per-call semaphores are not enough because
        # concurrent mini-batches can otherwise multiply in-flight requests.
        semaphore = self._get_request_semaphore()
        gate_lock = self._get_gate_lock()
        address = self.backend_address
        cb = self._circuit_breaker

        async def safe_call(i, msgs):
            tid = f"{request_id}_task_{i}"
            reserved_half_open = False
            try:
                # Check circuit breaker before acquiring semaphore
                async with gate_lock:
                    if not cb.is_available(address):
                        raise CircuitOpenError(address)
                    # Reserve HALF_OPEN probe slot if applicable
                    if cb.get_state(address) == CircuitState.HALF_OPEN:
                        cb.increment_half_open_calls(address)
                        reserved_half_open = True

                async with semaphore:
                    try:
                        resp = await self.acompletion_with_retry(
                            max_retries=3,
                            messages=msgs,
                            task_id=tid,
                            request_id=request_id,
                            guided_json=guided_json,
                            completion_params=completion_params,
                            metadata=(
                                metadata_list[i]
                                if metadata_list is not None
                                else metadata
                            ),
                        )
                        return BatchResult(tid, resp, None)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        return BatchResult(tid, None, e)
            except CircuitOpenError as e:
                # Intentional load shedding — do not count as backend failure
                self.logger.warning(
                    f"Task {tid} rejected: circuit breaker open for {address}"
                )
                return BatchResult(tid, None, e)
            except asyncio.CancelledError:
                raise
            finally:
                if reserved_half_open:
                    cb.decrement_half_open_calls(address)

        # Create tasks so we can use as_completed for incremental processing
        tasks = [
            asyncio.create_task(safe_call(i, msgs))
            for i, msgs in enumerate(messages_list)
        ]

        # Wrap with batch-level timeout
        results: List[Optional[BatchResult]] = [None] * len(messages_list)
        try:
            async with asyncio.timeout(self.batch_timeout):
                for completed_future in asyncio.as_completed(tasks):
                    try:
                        batch_result = await completed_future
                        idx = int(batch_result.task_id.rsplit("_", 1)[-1])
                        results[idx] = batch_result

                        # Invoke callback immediately when result is ready
                        if on_result and batch_result.error is None:
                            response_text = (
                                batch_result.response[1]
                                if isinstance(batch_result.response, tuple)
                                else batch_result.response
                            )
                            on_result(batch_result.task_id, response_text)
                    except asyncio.CancelledError:
                        raise
        except (asyncio.CancelledError, TimeoutError):
            # Cancel remaining tasks and drain them
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise asyncio.TimeoutError(
                f"Batch {request_id} timed out after {self.batch_timeout}s"
            )

        return results

    def batch_generate(
        self,
        messages_list: Union[List[str], List[List[Union[Image.Image, bytes, str]]]],
        system_prompt=Optional[str],
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Performs batch inference for multiple inputs, supporting both text-only and multimodal content.

        Args:
            messages_list: List of message lists to process
            system_prompt: Optional system prompt override
            guided_json: Optional JSON schema for guided generation
            on_result: Optional callback invoked when each task completes.
                       Signature: (task_id: str, response: Optional[str]) -> None
                       This enables incremental result processing.
            **kwargs: Additional arguments

        Returns:
            List of generated responses in original order

        Raises:
            BatchExecutionError: When one or more tasks failed after retries
                or were rejected by the circuit breaker.
            asyncio.TimeoutError: When the batch exceeds the configured timeout.
        """
        request_id = str(uuid.uuid4())
        system_prompt = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        )
        self.logger.info(
            f"Request {request_id} - Initiating batch inference with {len(messages_list)} requests."
        )
        start_time = time.time()

        # OTel span for the batch — manual lifecycle because batch_generate
        # is sync and runs an async event loop internally.
        batch_span = start_span(
            _tracer,
            "BatchProcessor.batch_generate",
            span_kind="chain",
        )
        batch_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self.model_string)
        batch_span.set_attribute("marie.request_id", request_id)
        batch_span.set_attribute("marie.batch_size", len(messages_list))
        if metadata:
            for mk, mv in metadata.items():
                batch_span.set_attribute(f"marie.{mk}", str(mv))
        batch_span.set_input(
            {
                "model": self.model_string,
                "batch_size": len(messages_list),
                "request_id": request_id,
            }
        )

        completion_params = kwargs.get("completion_params", None)

        # Attach batch span to OTel context so child spans (completion_non_streaming)
        # inherit it as their parent.
        ctx_token = otel_context.attach(otel_trace.set_span_in_context(batch_span))

        try:
            if self._queue_enabled():
                batch_results = self._get_queued_executor().execute(
                    messages_list=messages_list,
                    batch_request_id=request_id,
                    batch_timeout=self.batch_timeout,
                    on_result=on_result,
                    completion_params=completion_params,
                    metadata=metadata,
                )
            else:
                batch_results = run_coroutine_in_current_loop(
                    self.load_batched_request(
                        messages_list,
                        request_id,
                        guided_json,
                        on_result=on_result,
                        completion_params=completion_params,
                        metadata=metadata,
                    )
                )

            successful_count = 0
            failed_count = 0
            failed_results: List[BatchResult] = []
            for i, br in enumerate(batch_results):
                if br is None:
                    # Should not happen but treat as failure
                    failed_count += 1
                    failed_results.append(
                        BatchResult(f"{request_id}_task_{i}", None, None)
                    )
                elif br.error is not None:
                    self.logger.error(
                        f"Request {request_id} - Task {br.task_id} - Failed: {br.error!r}"
                    )
                    failed_count += 1
                    failed_results.append(br)
                else:
                    self.logger.info(
                        f"Request {request_id} - Task {br.task_id} - Response received."
                    )
                    successful_count += 1

            elapsed_time = time.time() - start_time

            batch_span.set_attribute("marie.latency_seconds", elapsed_time)
            batch_span.set_attribute("marie.successful_count", successful_count)
            batch_span.set_attribute("marie.failed_count", failed_count)
            batch_span.set_output(
                {
                    "successful": successful_count,
                    "failed": failed_count,
                    "elapsed_seconds": round(elapsed_time, 2),
                }
            )

            if failed_count > 0:
                raise BatchExecutionError(
                    request_id=request_id,
                    failed_results=failed_results,
                    total=len(messages_list),
                )

            self.logger.info(
                f"Request {request_id} - Batch inference completed in {elapsed_time:.2f} sec"
            )
            batch_span.set_status(StatusCode.OK)

            # Extract ordered responses from BatchResult objects
            ordered_responses: List[Optional[str]] = []
            for br in batch_results:
                if br and br.response is not None:
                    resp = (
                        br.response[1]
                        if isinstance(br.response, tuple)
                        else br.response
                    )
                    ordered_responses.append(resp)
                else:
                    ordered_responses.append(None)

            return ordered_responses
        except Exception as exc:
            batch_span.set_status(StatusCode.ERROR, str(exc))
            batch_span.record_exception(exc)
            raise
        finally:
            otel_context.detach(ctx_token)
            batch_span.end()
