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

from marie.engine.completion_contract import (
    CompletionCallParams,
    build_completion_call,
    completion_finish_reason,
    extract_completion_text,
)
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.queue_io import ListQueueClient, ValkeyListQueueClient
from marie.engine.llm_queue.result_types import BatchResult
from marie.engine.openai_compat import execute_completion_call
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


def _resolve_effective_queue_pool_id(
    fallback_pool_id: str, metadata: Optional[Dict[str, Any]]
) -> str:
    if isinstance(metadata, dict):
        value = metadata.get("pool_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback_pool_id


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
        self._queue_mode_logged = False

    def _queue_enabled(self) -> bool:
        config = getattr(self, "_queue_config", None)
        return bool(config and config.enabled)

    def _log_queue_mode_once(self) -> None:
        if getattr(self, "_queue_mode_logged", False):
            return
        self._queue_mode_logged = True
        config = getattr(self, "_queue_config", None)
        if config is None:
            return
        if config.enabled:
            self.logger.info(
                "LLM dispatch queue enabled: pool=%s valkey_configured=%s max_inline_payload_bytes=%s",
                config.pool_id,
                bool(config.valkey_url),
                config.max_inline_payload_bytes,
            )
        else:
            env_enabled = os.getenv("LLM_QUEUE_ENABLED")
            self.logger.info(
                "LLM dispatch queue disabled: env LLM_QUEUE_ENABLED=%r pool=%s",
                env_enabled,
                config.pool_id,
            )

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
                logger=self.logger,
            )
        return self._queued_executor

    def build_queue_dispatcher(self):
        from marie.engine.llm_queue.adapters.openai_compatible import (
            OpenAICompatibleExecutionAdapter,
        )
        from marie.engine.llm_queue.dispatcher import QueuedBatchDispatcher

        return QueuedBatchDispatcher(
            queue_client=self._get_queue_client(),
            execution_adapter=OpenAICompatibleExecutionAdapter(
                client=self.client,
                logger=self.logger,
                default_timeout=self.batch_timeout,
                backend_address=self.backend_address,
            ),
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

    def extract_text_from_response(
        self, completion: Union[ChatCompletion, str]
    ) -> Tuple[Optional[str], Optional[str]]:
        reasoning_content, extracted_text = extract_completion_text(completion)
        if extracted_text is None:
            self.logger.warning("No text extracted from Response.")
            raise ValueError(f"No text extracted from response. : {completion}")
        self.logger.info(f"Extracted text length: {len(extracted_text)} characters.")
        return reasoning_content, extracted_text

    async def completion_non_streaming_call(
        self,
        *,
        call: CompletionCallParams,
        task_id: str,
        request_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Asynchronously performs inference for one canonical completion call."""
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
            set_llm_io(span, input_messages=call.messages)

            try:
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
                            json.dump(call.messages, f, indent=2)
                        self.logger.info(f"Messages saved: {temp_path}")
                    except Exception as e:
                        self.logger.error(f"Could not save messages: {e}")

                # Handle authentication errors before making API call
                if not os.getenv("OPENAI_API_KEY"):
                    raise AuthenticationError(MISSING_API_KEY_ERROR_MESSAGE)
                completion = await execute_completion_call(self.client, call)
                finish_reason = completion_finish_reason(completion)
                if finish_reason == "length":
                    _, extracted_text = self.extract_text_from_response(completion)
                    await self.save_debug_msg(
                        extracted_text or "", task_id, "max_tokens"
                    )
                    raise MaxTokensExceededError()

                total_time = time.time() - start
                self.logger.info(
                    f"Request {request_id} - Task {task_id} - Completed in {total_time:.2f}s"
                )
                reasoning_content, extracted_text = self.extract_text_from_response(
                    completion
                )

                set_llm_io(span, output_messages=extracted_text)
                span.set_attribute("marie.latency_seconds", total_time)
                span.set_attribute("marie.has_reasoning", reasoning_content is not None)

                # Set token usage from provider response
                provider_usage = getattr(completion, "usage", None)
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

    async def acompletion_call_with_retry(
        self,
        *,
        max_retries: int,
        call: CompletionCallParams,
        task_id: str,
        request_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            """Use tenacity to retry the completion call."""
            retry_decorator = _create_retry_decorator(max_retries=max_retries)
            completion_with_retry = retry_decorator(self.completion_non_streaming_call)

            return await completion_with_retry(
                call=call,
                task_id=task_id,
                request_id=request_id,
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(
                f"Request {request_id} – Task {task_id} failed after retries: {e!r}"
            )
            raise  # Propagate to caller

    async def load_batched_completion_calls(
        self,
        *,
        calls: List[CompletionCallParams],
        request_id: str,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    ):

        # Share one limiter across all overlapping batch calls that use this
        # engine/client instance. Per-call semaphores are not enough because
        # concurrent mini-batches can otherwise multiply in-flight requests.
        semaphore = self._get_request_semaphore()
        gate_lock = self._get_gate_lock()
        address = self.backend_address
        cb = self._circuit_breaker

        async def safe_call(i, call):
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
                        resp = await self.acompletion_call_with_retry(
                            max_retries=3,
                            call=call,
                            task_id=tid,
                            request_id=request_id,
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
            asyncio.create_task(safe_call(i, call)) for i, call in enumerate(calls)
        ]

        # Wrap with batch-level timeout
        results: List[Optional[BatchResult]] = [None] * len(calls)
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
        system_prompt: Optional[str] = None,
        guided_json: Optional[Union[Dict, BaseModel, str]] = None,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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

        completion_params = kwargs.get("completion_params")
        calls = [
            build_completion_call(
                model=self.model_string,
                messages=messages,
                default_completion_params=self.default_completion_params,
                completion_params=completion_params,
                guided_json=guided_json,
                max_tokens=4096 * 4,
                stop=[],
                n=1,
                stream=False,
            )
            for messages in messages_list
        ]
        return self.batch_generate_calls(
            calls=calls,
            request_id=request_id,
            start_time=start_time,
            batch_span=batch_span,
            on_result=on_result,
            metadata=metadata,
        )

    def batch_generate_calls(
        self,
        *,
        calls: List[CompletionCallParams],
        request_id: Optional[str] = None,
        start_time: Optional[float] = None,
        batch_span=None,
        on_result: Optional[Callable[[str, Optional[str]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        request_id = request_id or str(uuid.uuid4())
        start_time = start_time or time.time()

        if batch_span is None:
            batch_span = start_span(
                _tracer,
                "BatchProcessor.batch_generate",
                span_kind="chain",
            )
            batch_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self.model_string)
            batch_span.set_attribute("marie.request_id", request_id)
            batch_span.set_attribute("marie.batch_size", len(calls))
            if metadata:
                for mk, mv in metadata.items():
                    batch_span.set_attribute(f"marie.{mk}", str(mv))
            batch_span.set_input(
                {
                    "model": self.model_string,
                    "batch_size": len(calls),
                    "request_id": request_id,
                }
            )

        ctx_token = otel_context.attach(otel_trace.set_span_in_context(batch_span))

        try:
            self._log_queue_mode_once()
            if self._queue_enabled():
                effective_pool_id = _resolve_effective_queue_pool_id(
                    self._queue_config.pool_id,
                    metadata,
                )
                self.logger.info(
                    "Submitting batch %s to LLM dispatch queue: pool=%s items=%s",
                    request_id,
                    effective_pool_id,
                    len(calls),
                )
                batch_results = self._get_queued_executor().execute(
                    calls=calls,
                    batch_request_id=request_id,
                    batch_timeout=self.batch_timeout,
                    on_result=on_result,
                    metadata=metadata,
                )
            else:
                self.logger.info(
                    "Executing batch %s directly against OpenAI-compatible endpoint: items=%s",
                    request_id,
                    len(calls),
                )
                batch_results = run_coroutine_in_current_loop(
                    self.load_batched_completion_calls(
                        calls=calls,
                        request_id=request_id,
                        on_result=on_result,
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
                    total=len(calls),
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
