from __future__ import annotations

import threading
import time
from typing import List, Optional

from opentelemetry import context as otel_context
from opentelemetry.propagate import extract

from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.models import QueueReply, QueueRequest
from marie.engine.llm_queue.queue_io import ListQueueClient
from marie.engine.llm_queue.result_types import BatchResult


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
        self._last_error: Optional[str] = None
        self._processed_batches = 0
        self._processed_items = 0

    def run_once(self) -> int:
        first_request = self._pop_first_live_request()
        if first_request is None:
            return 0

        batch = [first_request]
        batch.extend(self._fill_batch(first_request))
        results = self._execute_batch(batch)
        self._publish_replies(batch, results)
        with self._state_lock:
            self._processed_batches += 1
            self._processed_items += len(batch)
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
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def health(self) -> dict[str, object]:
        with self._state_lock:
            thread = self._thread
            return {
                "pool_id": self.config.pool_id,
                "running": bool(
                    thread and thread.is_alive() and not self._stop_event.is_set()
                ),
                "last_error": self._last_error,
                "processed_batches": self._processed_batches,
                "processed_items": self._processed_items,
            }

    def _pop_first_live_request(self) -> Optional[QueueRequest]:
        while True:
            payload = self.queue_client.pop_request(
                self.config.pool_id,
                timeout=self.config.dispatch_pop_timeout_seconds,
            )
            if payload is None:
                return None

            try:
                request = QueueRequest.from_json(payload)
            except Exception as exc:
                self.logger.error("Dropping malformed queue request: %r", exc)
                continue
            if not self.queue_client.is_producer_alive(request.producer_id):
                self.logger.info(
                    "Dropping request %s because producer %s is offline before dispatch",
                    request.request_id,
                    request.producer_id,
                )
                continue
            return request

    def _fill_batch(self, first_request: QueueRequest) -> List[QueueRequest]:
        collected: List[QueueRequest] = []
        deadline = time.monotonic() + (self.config.max_batch_wait_ms / 1000.0)

        while len(collected) + 1 < self.config.max_batch_items:
            if len(collected) + 1 >= self.config.max_buffered_requests_per_pool:
                break

            payload = self.queue_client.try_pop_request(self.config.pool_id)
            if payload is None:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
                continue

            try:
                candidate = QueueRequest.from_json(payload)
            except Exception as exc:
                self.logger.error("Dropping malformed queue request: %r", exc)
                continue
            if not self.queue_client.is_producer_alive(candidate.producer_id):
                self.logger.info(
                    "Dropping request %s because producer %s is offline before batching",
                    candidate.request_id,
                    candidate.producer_id,
                )
                continue

            if candidate.route_key != first_request.route_key:
                self.queue_client.push_request_front(self.config.pool_id, payload)
                break

            collected.append(candidate)

        return collected

    def _execute_batch(self, requests: List[QueueRequest]) -> List[BatchResult]:
        try:
            token = None
            trace_context = _trace_context_from_request(requests[0])
            if trace_context is not None:
                token = otel_context.attach(trace_context)
            try:
                results = self.execution_adapter.execute_requests(requests)
            finally:
                if token is not None:
                    otel_context.detach(token)
            return _normalize_results(requests, results, self.logger)
        except Exception as exc:
            self.logger.error("Queued dispatch batch failed: %r", exc)
            return [BatchResult(request.request_id, None, exc) for request in requests]

    def _publish_replies(
        self, requests: List[QueueRequest], results: List[BatchResult]
    ) -> None:
        for request, result in zip(requests, results):
            if not self.queue_client.is_producer_alive(request.producer_id):
                self.logger.info(
                    "Dropping reply for request %s because producer %s is offline",
                    request.request_id,
                    request.producer_id,
                )
                continue

            reply = _build_reply(request, result)
            self.queue_client.push_reply(
                request.producer_id,
                reply.to_json(),
                ttl_seconds=self.config.reply_queue_ttl_seconds,
            )


def _build_reply(request: QueueRequest, result: BatchResult) -> QueueReply:
    if result.error is None:
        response_text = result.response
        if isinstance(response_text, tuple):
            response_text = response_text[1]
        return QueueReply(
            request_id=request.request_id,
            producer_id=request.producer_id,
            pool_id=request.pool_id,
            route_key=request.route_key,
            status="ok",
            response=response_text,
            completed_at=time.time(),
        )

    return QueueReply(
        request_id=request.request_id,
        producer_id=request.producer_id,
        pool_id=request.pool_id,
        route_key=request.route_key,
        status="error",
        completed_at=time.time(),
        error_type=type(result.error).__name__,
        error_message=str(result.error),
    )


def _trace_context_from_request(request: QueueRequest):
    carrier = {}
    if request.traceparent:
        carrier["traceparent"] = request.traceparent
    if request.tracestate:
        carrier["tracestate"] = request.tracestate
    if not carrier:
        return None
    return extract(carrier)


def _normalize_results(
    requests: List[QueueRequest], results: List[BatchResult], logger
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
