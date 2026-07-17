from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry import trace as otel_trace

from marie.engine.completion_contract import (
    CompletionCallParams,
    CompletionReplyEnvelope,
    QueuedCompletionEnvelope,
    build_dispatch_profile_key,
    completion_finish_reason,
    extract_completion_text,
)
from marie.engine.exceptions import MaxTokensExceededError
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.queue_io import ListQueueClient
from marie.engine.llm_queue.replies import ProducerSession, ReplyWaiter
from marie.engine.llm_queue.result_types import BatchResult


class QueuedBatchExecutor:
    def __init__(
        self,
        *,
        queue_client: ListQueueClient,
        config: LlmQueueConfig,
        logger,
    ):
        self.queue_client = queue_client
        self.config = config
        self.logger = logger
        self._producer_session = ProducerSession(
            queue_client=queue_client,
            producer_id=config.producer_id,
            alive_value=config.producer_id,
            producer_ttl_seconds=config.producer_ttl_seconds,
            refresh_interval_seconds=config.producer_refresh_interval_seconds,
            reply_pop_timeout_seconds=config.reply_pop_timeout_seconds,
            logger=logger,
        )

    def execute(
        self,
        *,
        calls: List[CompletionCallParams],
        batch_request_id: str,
        batch_timeout: float,
        on_result=None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[BatchResult]:
        traceparent, tracestate = _current_trace_headers()
        waiters: Dict[str, ReplyWaiter] = {}
        ordered_ids: List[str] = []
        positions: Dict[str, int] = {}
        requests: List[QueuedCompletionEnvelope] = []
        pool_id = _resolve_queue_pool_id(self.config.pool_id, metadata)

        try:
            for index, call in enumerate(calls):
                request_id = f"{batch_request_id}_task_{index}"
                waiter = self._producer_session.register_waiter(request_id)
                waiters[request_id] = waiter
                ordered_ids.append(request_id)
                positions[request_id] = index

                request = QueuedCompletionEnvelope(
                    request_id=request_id,
                    producer_id=self.config.producer_id,
                    pool_id=pool_id,
                    submitted_at=time.time(),
                    call=call,
                    metadata=metadata,
                    traceparent=traceparent,
                    tracestate=tracestate,
                    timeout_seconds=batch_timeout,
                    dispatch_profile_key=build_dispatch_profile_key(call),
                )
                payload = request.to_json()
                payload_size = len(payload.encode("utf-8"))
                if payload_size > self.config.max_inline_payload_bytes:
                    raise ValueError(
                        f"Queued request {request_id} is {payload_size} bytes; "
                        f"max inline payload is {self.config.max_inline_payload_bytes} bytes."
                    )
                requests.append(request)

            for request in requests:
                self.queue_client.push_request(request)
        except Exception:
            for request_id in ordered_ids:
                self._producer_session.remove_waiter(request_id)
            raise

        deadline = time.monotonic() + batch_timeout
        results: List[Optional[BatchResult]] = [None] * len(calls)
        pending = set(ordered_ids)

        try:
            while pending:
                ready_ids = self._wait_for_ready_ids(waiters, pending, deadline)
                if not ready_ids:
                    raise asyncio.TimeoutError(
                        f"Batch {batch_request_id} timed out after {batch_timeout}s"
                    )

                for request_id in ready_ids:
                    pending.remove(request_id)
                    waiter = waiters[request_id]
                    reply = waiter.reply
                    if reply is None:
                        continue

                    index = positions[request_id]
                    batch_result = _reply_to_batch_result(reply)
                    results[index] = batch_result
                    self._producer_session.remove_waiter(request_id)

                    if on_result and batch_result.error is None:
                        on_result(batch_result.task_id, batch_result.response)
        finally:
            for request_id in pending:
                self._producer_session.remove_waiter(request_id)

        return [result for result in results if result is not None]

    def close(self) -> None:
        self._producer_session.close()

    def _wait_for_ready_ids(
        self,
        waiters: Dict[str, ReplyWaiter],
        pending: set[str],
        deadline: float,
    ) -> List[str]:
        condition = self._producer_session.condition
        with condition:
            while True:
                ready_ids = [
                    request_id
                    for request_id in pending
                    if waiters[request_id].reply is not None
                ]
                if ready_ids:
                    return ready_ids

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []

                condition.wait(timeout=remaining)


def _current_trace_headers() -> Tuple[Optional[str], Optional[str]]:
    span = otel_trace.get_current_span()
    span_context = span.get_span_context()
    if not span_context or not span_context.is_valid:
        return None, None
    trace_id = f"{span_context.trace_id:032x}"
    span_id = f"{span_context.span_id:016x}"
    traceparent = f"00-{trace_id}-{span_id}-01"
    trace_state = None
    if span_context.trace_state:
        trace_state = str(span_context.trace_state)
    return traceparent, trace_state


def _resolve_queue_pool_id(
    fallback_pool_id: str,
    metadata: Optional[Dict[str, Any]],
) -> str:
    if not isinstance(metadata, dict):
        return fallback_pool_id

    return _non_empty_str(metadata.get("pool_id")) or fallback_pool_id


def _non_empty_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _reply_to_batch_result(reply: CompletionReplyEnvelope) -> BatchResult:
    if reply.status == "ok":
        try:
            if completion_finish_reason(reply.completion) == "length":
                raise MaxTokensExceededError()
            _, extracted_text = extract_completion_text(reply.completion)
            return BatchResult(
                task_id=reply.request_id,
                response=extracted_text,
                error=None,
            )
        except Exception as exc:
            return BatchResult(
                task_id=reply.request_id,
                response=None,
                error=exc,
            )

    error_type = reply.error_type or "RemoteQueueTaskError"
    error_message = reply.error_message or "queued execution failed"
    origin_parts = []
    if reply.error_source:
        origin_parts.append(reply.error_source)
    if reply.dispatcher_id:
        origin_parts.append(f"dispatcher={reply.dispatcher_id}")
    if reply.execution_backend_address:
        origin_parts.append(f"backend={reply.execution_backend_address}")
    origin = f" from {' '.join(origin_parts)}" if origin_parts else ""
    return BatchResult(
        task_id=reply.request_id,
        response=None,
        error=RuntimeError(f"{error_type}{origin}: {error_message}"),
    )
