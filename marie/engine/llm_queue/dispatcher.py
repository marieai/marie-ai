from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any, List, Optional

from openinference.semconv.trace import SpanAttributes
from opentelemetry.propagate import extract
from opentelemetry.trace import StatusCode

from marie.engine.async_helper import run_coroutine_in_current_loop
from marie.engine.completion_contract import (
    COMPLETION_QUEUE_CONTRACT_VERSION,
    CompletionReplyEnvelope,
    QueuedCompletionEnvelope,
    extract_completion_text,
    summarize_completion_call,
)
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.metrics import dispatch_metrics
from marie.engine.llm_queue.queue_io import ListQueueClient
from marie.engine.llm_queue.registry import register_dispatcher, unregister_dispatcher
from marie.engine.llm_queue.result_types import BatchResult
from marie.instrumentation import get_tracer, set_llm_io, start_as_current_span
from marie.instrumentation.openinference import infer_llm_system

_UNSET = object()
_tracer = get_tracer("marie.engine.llm_queue.dispatcher")


class QueuedBatchDispatcher:
    def __init__(
        self,
        *,
        queue_client: ListQueueClient,
        execution_adapter,
        config: LlmQueueConfig,
        logger,
    ):
        self.queue_client = queue_client
        self.execution_adapter = execution_adapter
        self.config = config
        self.logger = logger
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self.dispatcher_id = f"{self.config.pool_id}:{uuid.uuid4().hex[:12]}"
        self._last_error: Optional[str] = None
        self._processed_batches = 0
        self._processed_items = 0
        self._last_processed_at: Optional[float] = None
        self._last_batch_size = 0
        self._execution_failures = 0
        self._malformed_requests_dropped = 0
        self._offline_producer_requests_dropped = 0
        self._offline_producer_replies_dropped = 0
        self._inflight_requests: dict[str, dict[str, object]] = {}

    def run_once(self) -> int:
        first_request = self._pop_first_live_request()
        if first_request is None:
            return 0

        started_at = time.monotonic()
        batch = [first_request]
        batch.extend(self._fill_batch(first_request))
        results = self._execute_batch(batch)
        self._publish_replies(batch, results)
        dispatch_metrics.record_batch(
            pool_id=self.config.pool_id,
            dispatcher_id=self.dispatcher_id,
            batch_size=len(batch),
            duration_seconds=max(0.0, time.monotonic() - started_at),
        )
        with self._state_lock:
            self._processed_batches += 1
            self._processed_items += len(batch)
            self._last_processed_at = time.time()
            self._last_batch_size = len(batch)
            self._execution_failures += sum(
                1 for result in results if result.error is not None
            )
        return len(batch)

    def run_forever(self, stop_event=None) -> None:
        while stop_event is None or not stop_event.is_set():
            try:
                self.run_once()
                with self._state_lock:
                    self._last_error = None
            except Exception as exc:
                self.logger.error("Queued dispatcher loop failed: %r", exc)
                with self._state_lock:
                    self._last_error = str(exc)
                time.sleep(0.1)

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self.run_forever,
                kwargs={"stop_event": self._stop_event},
                name=f"llm-queue-dispatcher-{self.config.pool_id}",
                daemon=True,
            )
            register_dispatcher(self.dispatcher_id, self)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        unregister_dispatcher(self.dispatcher_id)

    def health(self) -> dict[str, object]:
        queue_depth = None
        queue_depth_error = None
        try:
            queue_depth = self.queue_client.request_queue_depth(self.config.pool_id)
        except Exception as exc:  # pragma: no cover - defensive
            queue_depth_error = str(exc)

        with self._state_lock:
            thread = self._thread
            return {
                "dispatcher_id": self.dispatcher_id,
                "enabled": self.config.enabled,
                "pool_id": self.config.pool_id,
                "fabric_group_id": self.config.fabric_group_id,
                "gateway_id": self.config.gateway_id,
                "valkey_configured": bool(self.config.valkey_url),
                "running": bool(
                    thread and thread.is_alive() and not self._stop_event.is_set()
                ),
                "last_error": self._last_error,
                "processed_batches": self._processed_batches,
                "processed_items": self._processed_items,
                "last_processed_at": self._last_processed_at,
                "last_batch_size": self._last_batch_size,
                "execution_failures": self._execution_failures,
                "malformed_requests_dropped": self._malformed_requests_dropped,
                "offline_producer_requests_dropped": self._offline_producer_requests_dropped,
                "offline_producer_replies_dropped": self._offline_producer_replies_dropped,
                "inflight_request_count": len(self._inflight_requests),
                "request_queue_depth": queue_depth,
                "request_queue_depth_error": queue_depth_error,
                "reply_queue_ttl_seconds": self.config.reply_queue_ttl_seconds,
                "dispatch_pop_timeout_seconds": self.config.dispatch_pop_timeout_seconds,
                "max_batch_items": self.config.max_batch_items,
                "max_batch_wait_ms": self.config.max_batch_wait_ms,
                "max_buffered_requests_per_pool": self.config.max_buffered_requests_per_pool,
            }

    def sample_pending_requests(self, limit: int) -> List[dict[str, object]]:
        now = time.time()
        requests = self.queue_client.sample_requests(self.config.pool_id, limit)
        return [
            _build_live_request_snapshot(
                request,
                lifecycle_stage="pending",
                state_source="valkey",
                dispatcher_id=None,
                now=now,
            )
            for request in requests
        ]

    def inflight_requests_snapshot(self) -> List[dict[str, object]]:
        now = time.time()
        with self._state_lock:
            snapshots = [dict(item) for item in self._inflight_requests.values()]
        for snapshot in snapshots:
            popped_at = snapshot.get("popped_at")
            if popped_at is not None:
                snapshot["inflight_age_seconds"] = max(0.0, now - float(popped_at))
        snapshots.sort(key=lambda item: float(item.get("submitted_at") or 0.0))
        return snapshots

    def _pop_first_live_request(self) -> Optional[QueuedCompletionEnvelope]:
        while True:
            try:
                request = self.queue_client.pop_request(
                    self.config.pool_id,
                    timeout=self.config.dispatch_pop_timeout_seconds,
                )
            except Exception as exc:
                self.logger.error("Dropping malformed queue request: %r", exc)
                with self._state_lock:
                    self._malformed_requests_dropped += 1
                dispatch_metrics.record_request_drop(
                    pool_id=self.config.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    reason="malformed_request",
                )
                continue
            if request is None:
                return None
            if not self.queue_client.is_producer_alive(request.producer_id):
                self.logger.info(
                    "Dropping request %s because producer %s is offline before dispatch",
                    request.request_id,
                    request.producer_id,
                )
                with self._state_lock:
                    self._offline_producer_requests_dropped += 1
                dispatch_metrics.record_request_drop(
                    pool_id=self.config.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    reason="offline_producer_before_dispatch",
                )
                continue
            self._mark_request_popped(request, lifecycle_stage="dispatching")
            return request

    def _fill_batch(
        self, first_request: QueuedCompletionEnvelope
    ) -> List[QueuedCompletionEnvelope]:
        collected: List[QueuedCompletionEnvelope] = []
        deadline = time.monotonic() + (self.config.max_batch_wait_ms / 1000.0)

        while len(collected) + 1 < self.config.max_batch_items:
            if len(collected) + 1 >= self.config.max_buffered_requests_per_pool:
                break

            try:
                candidate = self.queue_client.try_pop_request(self.config.pool_id)
            except Exception as exc:
                self.logger.error("Dropping malformed queue request: %r", exc)
                with self._state_lock:
                    self._malformed_requests_dropped += 1
                dispatch_metrics.record_request_drop(
                    pool_id=self.config.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    reason="malformed_request",
                )
                continue
            if candidate is None:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
                continue
            if not self.queue_client.is_producer_alive(candidate.producer_id):
                self.logger.warning(
                    "Dropping request %s because producer %s is offline before batching",
                    candidate.request_id,
                    candidate.producer_id,
                )
                with self._state_lock:
                    self._offline_producer_requests_dropped += 1
                dispatch_metrics.record_request_drop(
                    pool_id=self.config.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    reason="offline_producer_before_batching",
                )
                continue

            if candidate.batch_key != first_request.batch_key:
                self.queue_client.push_request_front(candidate)
                break

            self._mark_request_popped(candidate, lifecycle_stage="dispatching")
            collected.append(candidate)

        return collected

    def _execute_batch(
        self, requests: List[QueuedCompletionEnvelope]
    ) -> List[BatchResult]:
        try:
            results = run_coroutine_in_current_loop(self._execute_batch_async(requests))
            return _normalize_results(requests, results, self.logger)
        except Exception as exc:
            self.logger.error("Queued dispatch batch failed: %r", exc)
            return [BatchResult(request.request_id, None, exc) for request in requests]

    async def _execute_batch_async(
        self, requests: List[QueuedCompletionEnvelope]
    ) -> List[BatchResult]:
        tasks = [
            asyncio.create_task(self._execute_one(request)) for request in requests
        ]
        return await asyncio.gather(*tasks)

    async def _execute_one(self, request: QueuedCompletionEnvelope) -> BatchResult:
        started_monotonic = time.monotonic()
        started_wall = time.time()
        ok = False
        self._update_inflight_request(request, lifecycle_stage="executing")
        parent_context = _trace_context_from_request(request)
        backend_address = getattr(self.execution_adapter, "backend_address", None)
        span_kwargs = {"context": parent_context} if parent_context is not None else {}

        with start_as_current_span(
            _tracer,
            "LLMDispatch.completion",
            span_kind="llm",
            **span_kwargs,
        ) as span:
            _set_dispatch_span_base_attributes(
                span,
                request=request,
                config=self.config,
                dispatcher_id=self.dispatcher_id,
                backend_address=backend_address,
                started_wall=started_wall,
            )

            try:
                completion = await self.execution_adapter.execute(
                    request.call,
                    timeout_seconds=request.timeout_seconds,
                )
                response = (
                    completion.model_dump()
                    if hasattr(completion, "model_dump")
                    else completion
                )
                ok = True
                _set_dispatch_span_success_attributes(
                    span,
                    completion=response,
                    started_monotonic=started_monotonic,
                    request=request,
                )
                return BatchResult(
                    task_id=request.request_id,
                    response=response,
                    error=None,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error(
                    "Queued execution failed for %s: %r",
                    request.request_id,
                    exc,
                )
                self._update_inflight_request(
                    request,
                    current_error_summary=str(exc),
                )
                _set_dispatch_span_error_attributes(
                    span,
                    exc=exc,
                    started_monotonic=started_monotonic,
                    request=request,
                )
                return BatchResult(
                    task_id=request.request_id,
                    response=None,
                    error=exc,
                )
            finally:
                dispatch_metrics.record_request_execution(
                    pool_id=self.config.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    duration_seconds=max(0.0, time.monotonic() - started_monotonic),
                    ok=ok,
                )

    def _publish_replies(
        self, requests: List[QueuedCompletionEnvelope], results: List[BatchResult]
    ) -> None:
        for request, result in zip(requests, results):
            self._update_inflight_request(
                request,
                lifecycle_stage="replying",
                current_error_summary=(
                    None if result.error is None else str(result.error)
                ),
            )
            if not self.queue_client.is_producer_alive(request.producer_id):
                self.logger.info(
                    "Dropping reply for request %s because producer %s is offline",
                    request.request_id,
                    request.producer_id,
                )
                with self._state_lock:
                    self._offline_producer_replies_dropped += 1
                dispatch_metrics.record_reply_drop(
                    pool_id=self.config.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    reason="offline_producer_before_reply",
                )
                self._clear_inflight_request(request.request_id)
                continue

            reply = _build_reply(
                request,
                result,
                dispatcher_id=self.dispatcher_id,
                execution_backend_address=getattr(
                    self.execution_adapter,
                    "backend_address",
                    None,
                ),
            )
            self.queue_client.push_reply(
                reply,
                ttl_seconds=self.config.reply_queue_ttl_seconds,
            )
            self._clear_inflight_request(request.request_id)

    def _mark_request_popped(
        self, request: QueuedCompletionEnvelope, *, lifecycle_stage: str
    ) -> None:
        now = time.time()
        snapshot = _build_live_request_snapshot(
            request,
            lifecycle_stage=lifecycle_stage,
            state_source="dispatcher",
            dispatcher_id=self.dispatcher_id,
            now=now,
            popped_at=now,
        )
        with self._state_lock:
            self._inflight_requests[request.request_id] = snapshot

    def _update_inflight_request(
        self,
        request: QueuedCompletionEnvelope,
        *,
        lifecycle_stage: Optional[str] = None,
        current_error_summary: object = _UNSET,
    ) -> None:
        with self._state_lock:
            snapshot = self._inflight_requests.get(request.request_id)
            if snapshot is None:
                return

            now = time.time()
            popped_at = float(snapshot.get("popped_at") or now)
            if lifecycle_stage is not None:
                snapshot["lifecycle_stage"] = lifecycle_stage
            snapshot["state_updated_at"] = now
            snapshot["inflight_age_seconds"] = max(0.0, now - popped_at)
            if current_error_summary is not _UNSET:
                snapshot["current_error_summary"] = current_error_summary

    def _clear_inflight_request(self, request_id: str) -> None:
        with self._state_lock:
            self._inflight_requests.pop(request_id, None)


def _build_reply(
    request: QueuedCompletionEnvelope,
    result: BatchResult,
    *,
    dispatcher_id: str,
    execution_backend_address: Optional[str],
) -> CompletionReplyEnvelope:
    if result.error is None:
        return CompletionReplyEnvelope(
            request_id=request.request_id,
            producer_id=request.producer_id,
            pool_id=request.pool_id,
            status="ok",
            completion=result.response,
            completed_at=time.time(),
            dispatcher_id=dispatcher_id,
            execution_backend_address=execution_backend_address,
        )

    return CompletionReplyEnvelope(
        request_id=request.request_id,
        producer_id=request.producer_id,
        pool_id=request.pool_id,
        status="error",
        completed_at=time.time(),
        error_type=type(result.error).__name__,
        error_message=str(result.error),
        error_source="llm_dispatcher",
        dispatcher_id=dispatcher_id,
        execution_backend_address=execution_backend_address,
    )


def _trace_context_from_request(request: QueuedCompletionEnvelope):
    carrier = {}
    if request.traceparent:
        carrier["traceparent"] = request.traceparent
    if request.tracestate:
        carrier["tracestate"] = request.tracestate
    if not carrier:
        return None
    return extract(carrier)


def _set_dispatch_span_base_attributes(
    span,
    *,
    request: QueuedCompletionEnvelope,
    config: LlmQueueConfig,
    dispatcher_id: str,
    backend_address: Optional[str],
    started_wall: float,
) -> None:
    queue_wait_ms = max(0.0, (started_wall - request.submitted_at) * 1000.0)
    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, request.call.model)
    span.set_attribute(SpanAttributes.LLM_SYSTEM, infer_llm_system(request.call.model))
    span.set_attribute("marie.llm_dispatch.request_id", request.request_id)
    span.set_attribute("marie.llm_dispatch.producer_id", request.producer_id)
    span.set_attribute("marie.llm_dispatch.pool_id", request.pool_id)
    span.set_attribute(
        "marie.llm_dispatch.fabric_group_id", config.fabric_group_id or ""
    )
    span.set_attribute("marie.llm_dispatch.gateway_id", config.gateway_id or "")
    span.set_attribute("marie.llm_dispatch.dispatcher_id", dispatcher_id)
    span.set_attribute(
        "marie.llm_dispatch.dispatch_profile_key",
        request.dispatch_profile_key or request.call.model,
    )
    span.set_attribute("marie.llm_dispatch.backend_address", backend_address or "")
    span.set_attribute("marie.llm_dispatch.model", request.call.model)
    span.set_attribute("marie.llm_dispatch.queue_wait_ms", queue_wait_ms)
    span.set_attribute("marie.llm_dispatch.message_count", len(request.call.messages))
    span.set_attribute(
        "marie.llm_dispatch.contract_version",
        COMPLETION_QUEUE_CONTRACT_VERSION,
    )
    set_llm_io(span, input_messages=request.call.messages)


def _set_dispatch_span_success_attributes(
    span,
    *,
    completion: Any,
    started_monotonic: float,
    request: QueuedCompletionEnvelope,
) -> None:
    execution_ms = max(0.0, (time.monotonic() - started_monotonic) * 1000.0)
    total_latency_ms = max(0.0, (time.time() - request.submitted_at) * 1000.0)
    span.set_attribute("marie.llm_dispatch.status", "ok")
    span.set_attribute("marie.llm_dispatch.execution_ms", execution_ms)
    span.set_attribute("marie.llm_dispatch.total_latency_ms", total_latency_ms)
    _set_completion_output_attributes(span, completion)
    _set_usage_attributes(span, completion)
    span.set_status(StatusCode.OK)


def _set_dispatch_span_error_attributes(
    span,
    *,
    exc: Exception,
    started_monotonic: float,
    request: QueuedCompletionEnvelope,
) -> None:
    execution_ms = max(0.0, (time.monotonic() - started_monotonic) * 1000.0)
    total_latency_ms = max(0.0, (time.time() - request.submitted_at) * 1000.0)
    span.set_attribute("marie.llm_dispatch.status", "error")
    span.set_attribute("marie.llm_dispatch.error_type", type(exc).__name__)
    span.set_attribute("marie.llm_dispatch.error_message", str(exc))
    span.set_attribute("marie.llm_dispatch.execution_ms", execution_ms)
    span.set_attribute("marie.llm_dispatch.total_latency_ms", total_latency_ms)
    span.record_exception(exc)
    span.set_status(StatusCode.ERROR, str(exc))


def _set_completion_output_attributes(span, completion: Any) -> None:
    try:
        reasoning_content, extracted_text = extract_completion_text(completion)
    except Exception:
        return

    if extracted_text is not None:
        set_llm_io(span, output_messages=extracted_text)
    span.set_attribute("marie.has_reasoning", reasoning_content is not None)


def _set_usage_attributes(span, completion: Any) -> None:
    usage = _read_completion_usage(completion)
    if usage is None:
        return
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens)


def _read_completion_usage(completion: Any) -> Optional[dict[str, Any]]:
    if isinstance(completion, dict):
        usage = completion.get("usage")
    else:
        usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _normalize_results(
    requests: List[QueuedCompletionEnvelope], results: List[BatchResult], logger
) -> List[BatchResult]:
    if len(results) == len(requests):
        return results

    logger.error(
        "Execution adapter returned %s results for %s requests",
        len(results),
        len(requests),
    )
    normalized = list(results[: len(requests)])
    for request in requests[len(normalized) :]:
        normalized.append(
            BatchResult(
                task_id=request.request_id,
                response=None,
                error=RuntimeError("Execution adapter did not return a result"),
            )
        )
    return normalized


def _build_live_request_snapshot(
    request: QueuedCompletionEnvelope,
    *,
    lifecycle_stage: str,
    state_source: str,
    dispatcher_id: Optional[str],
    now: float,
    popped_at: Optional[float] = None,
    current_error_summary: Optional[str] = None,
) -> dict[str, object]:
    queue_wait_age_seconds = max(0.0, now - request.submitted_at)
    inflight_age_seconds = None
    if popped_at is not None:
        queue_wait_age_seconds = max(0.0, popped_at - request.submitted_at)
        inflight_age_seconds = max(0.0, now - popped_at)

    return {
        "request_id": request.request_id,
        "producer_id": request.producer_id,
        "pool_id": request.pool_id,
        "model": request.call.model,
        "lifecycle_stage": lifecycle_stage,
        "state_source": state_source,
        "submitted_at": request.submitted_at,
        "popped_at": popped_at,
        "state_updated_at": now,
        "queue_wait_age_seconds": queue_wait_age_seconds,
        "inflight_age_seconds": inflight_age_seconds,
        "dispatcher_id": dispatcher_id,
        "dispatch_profile_key": request.dispatch_profile_key,
        "timeout_seconds": request.timeout_seconds,
        "message_count": len(request.call.messages),
        "request_summary": summarize_completion_call(request.call),
        "current_error_summary": current_error_summary,
    }
