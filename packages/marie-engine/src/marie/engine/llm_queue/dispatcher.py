from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import Counter
from typing import Any, List, Optional

from marie.engine.async_helper import run_coroutine_in_current_loop
from marie.engine.completion_contract import (
    COMPLETION_QUEUE_CONTRACT_VERSION,
    CompletionReplyEnvelope,
    QueuedCompletionEnvelope,
    require_terminal_completion,
)
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.metrics import dispatch_metrics
from marie.engine.llm_queue.queue_io import (
    ListQueueClient,
    MalformedQueueRequest,
    QueueUnavailable,
)
from marie.engine.llm_queue.registry import (
    SnapshotReadBudget,
    register_dispatcher,
    unregister_dispatcher,
)
from marie.engine.llm_queue.result_types import BatchResult
from marie.engine.llm_queue.scheduler import (
    DrrLaneConfig,
    DrrLaneScheduler,
    request_cost_units,
)
from marie.instrumentation import (
    MarieSpanAttributes,
    get_tracer,
    start_as_current_span,
)
from marie.instrumentation.openinference import infer_llm_system
from openinference.semconv.trace import SpanAttributes
from opentelemetry.propagate import extract
from opentelemetry.trace import StatusCode

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
        self._transport_failures = 0
        self._malformed_requests_dropped = 0
        self._held_requests: list[QueuedCompletionEnvelope] = []
        self._retained_until: dict[str, float] = {}
        self._pending_replies: list[tuple[QueuedCompletionEnvelope, BatchResult]] = []
        self._offline_producer_requests_dropped = 0
        self._offline_producer_replies_dropped = 0
        self._inflight_requests: dict[str, dict[str, object]] = {}

    def _execution_adapter_for(self, request: QueuedCompletionEnvelope):
        return self.execution_adapter

    def run_once(self) -> int:
        self._expire_retained()
        if self._pending_replies:
            self._publish_replies([], [])
            if self._pending_replies:
                return 0
        first_request = self._pop_first_live_request()
        if first_request is None:
            return 0

        return self._run_popped_batch([first_request], fill_compatible=True)

    def _run_popped_batch(
        self,
        first_requests: List[QueuedCompletionEnvelope],
        *,
        fill_compatible: bool,
    ) -> int:
        started_at = time.monotonic()
        batch = list(first_requests)
        if fill_compatible and len(batch) == 1:
            batch.extend(self._fill_batch(batch[0]))
        try:
            results = self._execute_batch(batch)
        except Exception:
            results = [
                BatchResult(request.request_id, None, RuntimeError('dispatch_error'))
                for request in batch
            ]
        self._publish_replies(batch, results)
        self._record_batches(
            batch,
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
                if self.run_once() == 0:
                    self._stop_event.wait(0.05)
            except Exception as exc:
                self.logger.error("Queued dispatcher loop failed")
                with self._state_lock:
                    self._last_error = (
                        'queue_unavailable'
                        if isinstance(exc, QueueUnavailable)
                        else 'dispatch_error'
                    )
                self._stop_event.wait(0.1)

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
        if thread is None or not thread.is_alive():
            self._expire_retained(closing=True)
        unregister_dispatcher(self.dispatcher_id)

    def health(
        self, *, read_budget: SnapshotReadBudget | None = None
    ) -> dict[str, object]:
        if read_budget is not None:
            read_budget.before_read()
        queue_depth = None
        queue_depth_error = None
        try:
            queue_depth = self.queue_client.request_queue_depth(self.config.pool_id)
        except Exception as exc:  # pragma: no cover - defensive
            queue_depth_error = 'store_unavailable'

        with self._state_lock:
            thread = self._thread
            return {
                "dispatcher_id": self.dispatcher_id,
                "enabled": self.config.enabled,
                "pool_id": self.config.pool_id,
                "fabric_group_id": self.config.fabric_group_id,
                "gateway_id": self.config.gateway_id,
                "queue_configured": bool(self.config.queue_url),
                "running": bool(
                    thread and thread.is_alive() and not self._stop_event.is_set()
                ),
                "last_error": self._last_error,
                "scheduler_policy": "fifo",
                "processed_batches": self._processed_batches,
                "processed_items": self._processed_items,
                "last_processed_at": self._last_processed_at,
                "last_batch_size": self._last_batch_size,
                "execution_failures": self._execution_failures,
                "transport_failures": self._transport_failures,
                "sampling_available": False,
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

    def sample_pending_requests(
        self, limit: int, *, read_budget: SnapshotReadBudget | None = None
    ) -> List[dict[str, object]]:
        return []

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
        while not self._stop_event.is_set():
            try:
                request = (
                    self._held_requests.pop(0)
                    if self._held_requests
                    else self.queue_client.pop_request(
                        self.config.pool_id,
                        timeout=self.config.dispatch_pop_timeout_seconds,
                    )
                )
            except MalformedQueueRequest:
                self.logger.error("Dropping malformed queue request")
                with self._state_lock:
                    self._malformed_requests_dropped += 1
                dispatch_metrics.record_request_drop(
                    pool_id=self.config.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    reason="malformed_request",
                )
                continue
            except QueueUnavailable:
                self._record_transport_failure()
                raise
            if request is None:
                return None
            try:
                live = self.queue_client.is_producer_alive(request.producer_id)
            except QueueUnavailable:
                self._hold_request(request)
                self._record_transport_failure()
                raise
            if not live:
                self.logger.info(
                    "Dropping request %s because producer %s is offline before dispatch",
                    request.request_id,
                    request.producer_id,
                )
                with self._state_lock:
                    self._offline_producer_requests_dropped += 1
                dispatch_metrics.record_request_drop(
                    pool_id=request.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    reason="offline_producer_before_dispatch",
                )
                continue
            self._mark_request_popped(request, lifecycle_stage="dispatching")
            return request

    def _hold_request(self, request: QueuedCompletionEnvelope) -> None:
        self._held_requests.append(request)
        self._retained_until.setdefault(
            request.request_id, time.monotonic() + self.config.reply_queue_ttl_seconds
        )

    def _expire_retained(self, *, closing: bool = False) -> None:
        now = time.monotonic()
        expired = {
            identity
            for identity, deadline in self._retained_until.items()
            if closing or now >= deadline
        }
        for request in self._held_requests:
            if request.request_id in expired:
                scheduler = getattr(self, 'scheduler', None)
                if scheduler is not None:
                    scheduler.release(request.pool_id)
                self._clear_inflight_request(request.request_id)
        self._held_requests = [
            request
            for request in self._held_requests
            if request.request_id not in expired
        ]
        for request, _ in self._pending_replies:
            if request.request_id in expired:
                self._clear_inflight_request(request.request_id)
        self._pending_replies = [
            (request, result)
            for request, result in self._pending_replies
            if request.request_id not in expired
        ]
        live = {request.request_id for request in self._held_requests} | {
            request.request_id for request, _ in self._pending_replies
        }
        self._retained_until = {
            identity: deadline
            for identity, deadline in self._retained_until.items()
            if identity in live
        }

    def _record_transport_failure(self) -> None:
        with self._state_lock:
            self._transport_failures += 1
            self._last_error = 'queue_unavailable'

    def _fill_batch(
        self, first_request: QueuedCompletionEnvelope
    ) -> List[QueuedCompletionEnvelope]:
        collected: List[QueuedCompletionEnvelope] = []
        deadline = time.monotonic() + (self.config.max_batch_wait_ms / 1000.0)

        while (
            len(collected) + 1 < self.config.max_batch_items
            and not self._stop_event.is_set()
            and time.monotonic() < deadline
        ):
            if len(collected) + 1 >= self.config.max_buffered_requests_per_pool:
                break

            try:
                candidate = self.queue_client.try_pop_request(first_request.pool_id)
            except MalformedQueueRequest:
                self.logger.error("Dropping malformed queue request")
                with self._state_lock:
                    self._malformed_requests_dropped += 1
                dispatch_metrics.record_request_drop(
                    pool_id=first_request.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    reason="malformed_request",
                )
                continue
            except QueueUnavailable:
                self._record_transport_failure()
                break
            if candidate is None:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
                continue
            try:
                live = self.queue_client.is_producer_alive(candidate.producer_id)
            except QueueUnavailable:
                self._hold_request(candidate)
                self._record_transport_failure()
                break
            if not live:
                self.logger.warning(
                    "Dropping request %s because producer %s is offline before batching",
                    candidate.request_id,
                    candidate.producer_id,
                )
                with self._state_lock:
                    self._offline_producer_requests_dropped += 1
                dispatch_metrics.record_request_drop(
                    pool_id=candidate.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    reason="offline_producer_before_batching",
                )
                continue

            if candidate.batch_key != first_request.batch_key:
                self._hold_request(candidate)
                break

            try:
                self._mark_request_popped(candidate, lifecycle_stage="dispatching")
            except Exception:
                self._hold_request(candidate)
                self._last_error = 'dispatch_error'
                break
            collected.append(candidate)

        return collected

    def _execute_batch(
        self, requests: List[QueuedCompletionEnvelope]
    ) -> List[BatchResult]:
        try:
            results = run_coroutine_in_current_loop(self._execute_batch_async(requests))
            return _normalize_results(requests, results, self.logger)
        except Exception as exc:
            self.logger.error("Queued dispatch batch failed")
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
        execution_adapter = self._execution_adapter_for(request)
        backend_address = getattr(execution_adapter, "backend_address", None)
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
                require_terminal_completion(request.call)
                completion = await execution_adapter.execute(
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
                    "Queued execution failed for %s",
                    request.request_id,
                )
                self._update_inflight_request(
                    request,
                    current_error_summary='execution_failed',
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
                    pool_id=request.pool_id,
                    dispatcher_id=self.dispatcher_id,
                    duration_seconds=max(0.0, time.monotonic() - started_monotonic),
                    ok=ok,
                )

    def _publish_replies(
        self, requests: List[QueuedCompletionEnvelope], results: List[BatchResult]
    ) -> None:
        self._pending_replies.extend(zip(requests, results))
        for request in requests:
            self._retained_until.setdefault(
                request.request_id,
                time.monotonic() + self.config.reply_queue_ttl_seconds,
            )
        while self._pending_replies:
            request, result = self._pending_replies[0]
            try:
                self._update_inflight_request(
                    request,
                    lifecycle_stage="replying",
                    current_error_summary=(
                        None if result.error is None else 'execution_failed'
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
                        pool_id=request.pool_id,
                        dispatcher_id=self.dispatcher_id,
                        reason="offline_producer_before_reply",
                    )
                    self._clear_inflight_request(request.request_id)
                    self._pending_replies.pop(0)
                    continue

                reply = _build_reply(
                    request,
                    result,
                    dispatcher_id=self.dispatcher_id,
                    execution_backend_address=getattr(
                        self._execution_adapter_for(request),
                        "backend_address",
                        None,
                    ),
                )
                self.queue_client.push_reply(
                    reply,
                    ttl_seconds=self.config.reply_queue_ttl_seconds,
                )
                self._clear_inflight_request(request.request_id)
                self._pending_replies.pop(0)
            except QueueUnavailable:
                self._record_transport_failure()
                return

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

    def _record_batches(
        self, batch: List[QueuedCompletionEnvelope], *, duration_seconds: float
    ) -> None:
        for pool_id, batch_size in Counter(
            request.pool_id for request in batch
        ).items():
            dispatch_metrics.record_batch(
                pool_id=pool_id,
                dispatcher_id=self.dispatcher_id,
                batch_size=batch_size,
                duration_seconds=duration_seconds,
            )


class DrrQueuedBatchDispatcher(QueuedBatchDispatcher):
    def __init__(
        self,
        *,
        queue_client: ListQueueClient,
        execution_adapter,
        config: LlmQueueConfig,
        logger,
        lanes: list[DrrLaneConfig],
        total_concurrent_dispatch: int,
        execution_adapters_by_pool: Optional[dict[str, object]] = None,
    ):
        super().__init__(
            queue_client=queue_client,
            execution_adapter=execution_adapter,
            config=config,
            logger=logger,
        )
        self.scheduler = DrrLaneScheduler(
            queue_client=queue_client,
            lanes=lanes,
            total_concurrent_dispatch=total_concurrent_dispatch,
        )
        self.dispatcher_id = f"{self.config.pool_id}:drr:{uuid.uuid4().hex[:12]}"
        self._lane_pool_ids = [lane.pool_id for lane in lanes if lane.enabled]
        self._execution_adapters_by_pool = execution_adapters_by_pool or {}

    def _execution_adapter_for(self, request: QueuedCompletionEnvelope):
        return self._execution_adapters_by_pool.get(
            request.pool_id,
            self.execution_adapter,
        )

    def run_once(self) -> int:
        self._expire_retained()
        if self._pending_replies:
            self._publish_replies([], [])
            if self._pending_replies:
                return 0
        batch: list[QueuedCompletionEnvelope] = []
        selected_pool_ids: list[str] = []
        try:
            while (
                self._held_requests
                or self.scheduler.inflight_count
                < self.scheduler.total_concurrent_dispatch
            ):
                try:
                    if self._held_requests:
                        request = self._held_requests.pop(0)
                        pool_id = request.pool_id
                    else:
                        dispatch = self.scheduler.select_next()
                        if dispatch is None:
                            break
                        request, pool_id = dispatch.request, dispatch.pool_id
                except Exception as exc:
                    if isinstance(exc, QueueUnavailable):
                        self._record_transport_failure()
                    if batch:
                        break
                    raise
                try:
                    live = self.queue_client.is_producer_alive(request.producer_id)
                except QueueUnavailable:
                    self._hold_request(request)
                    self._record_transport_failure()
                    break
                if not live:
                    self.scheduler.release(pool_id)
                    with self._state_lock:
                        self._offline_producer_requests_dropped += 1
                    continue
                try:
                    self._mark_request_popped(request, lifecycle_stage='dispatching')
                except Exception:
                    self._hold_request(request)
                    self._last_error = 'dispatch_error'
                    break
                batch.append(request)
                selected_pool_ids.append(pool_id)
            if not batch:
                return 0
            return self._run_popped_batch(batch, fill_compatible=False)
        finally:
            for pool_id in selected_pool_ids:
                self.scheduler.release(pool_id)

    def run_forever(self, stop_event=None) -> None:
        idle_sleep_seconds = min(
            0.05,
            max(0.001, self.config.dispatch_pop_timeout_seconds),
        )
        while stop_event is None or not stop_event.is_set():
            try:
                processed = self.run_once()
                if processed == 0:
                    time.sleep(idle_sleep_seconds)
            except Exception as exc:
                self.logger.error("Queued DRR dispatcher loop failed")
                with self._state_lock:
                    self._last_error = (
                        'queue_unavailable'
                        if isinstance(exc, QueueUnavailable)
                        else 'dispatch_error'
                    )
                self._stop_event.wait(0.1)

    def health(
        self, *, read_budget: SnapshotReadBudget | None = None
    ) -> dict[str, object]:
        health = super().health(read_budget=read_budget)
        lane_snapshots = [
            {
                "pool_id": lane.pool_id,
                "display_name": lane.display_name,
                "quantum": lane.quantum,
                "deficit": lane.deficit,
                "inflight": lane.inflight,
                "request_queue_depth": lane.queue_depth,
                "oldest_pending_age_seconds": lane.oldest_pending_age_seconds,
                "head_cost_units": lane.head_cost_units,
                "min_concurrent": lane.min_concurrent,
                "max_concurrent": lane.max_concurrent,
                "max_burst_per_visit": lane.max_burst_per_visit,
                "skip_counts": lane.skip_counts,
                "malformed_requests_dropped": lane.malformed_requests_dropped,
            }
            for lane in self.scheduler.lane_snapshots(
                before_read=read_budget.before_read if read_budget else None
            )
        ]
        total_depth = sum(
            int(lane["request_queue_depth"] or 0)
            for lane in lane_snapshots
            if lane["request_queue_depth"] is not None
        )
        health.update(
            {
                "scheduler_policy": "drr",
                "malformed_requests_dropped": self._malformed_requests_dropped
                + sum(
                    int(lane['malformed_requests_dropped']) for lane in lane_snapshots
                ),
                "request_queue_depth": total_depth,
                "total_concurrent_dispatch": self.scheduler.total_concurrent_dispatch,
                "lanes": lane_snapshots,
            }
        )
        return health

    def sample_pending_requests(
        self, limit: int, *, read_budget: SnapshotReadBudget | None = None
    ) -> List[dict[str, object]]:
        return []


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
    span.set_attribute(MarieSpanAttributes.LLM_DISPATCH_REQUEST_ID, request.request_id)
    span.set_attribute(MarieSpanAttributes.LLM_DISPATCH_POOL_ID, request.pool_id)
    span.set_attribute(
        MarieSpanAttributes.LLM_DISPATCH_FABRIC_GROUP_ID,
        config.fabric_group_id or "",
    )
    span.set_attribute(
        MarieSpanAttributes.LLM_DISPATCH_GATEWAY_ID,
        config.gateway_id or "",
    )
    span.set_attribute(MarieSpanAttributes.LLM_DISPATCH_DISPATCHER_ID, dispatcher_id)
    span.set_attribute(
        MarieSpanAttributes.LLM_DISPATCH_PROFILE_KEY,
        request.dispatch_profile_key or request.call.model,
    )
    span.set_attribute(MarieSpanAttributes.LLM_DISPATCH_MODEL, request.call.model)
    span.set_attribute(MarieSpanAttributes.LLM_DISPATCH_QUEUE_WAIT_MS, queue_wait_ms)
    span.set_attribute(
        MarieSpanAttributes.LLM_DISPATCH_MESSAGE_COUNT,
        len(request.call.messages),
    )
    span.set_attribute(
        MarieSpanAttributes.LLM_DISPATCH_CONTRACT_VERSION,
        COMPLETION_QUEUE_CONTRACT_VERSION,
    )


def _set_dispatch_span_success_attributes(
    span,
    *,
    completion: Any,
    started_monotonic: float,
    request: QueuedCompletionEnvelope,
) -> None:
    execution_ms = max(0.0, (time.monotonic() - started_monotonic) * 1000.0)
    total_latency_ms = max(0.0, (time.time() - request.submitted_at) * 1000.0)
    span.set_attribute(MarieSpanAttributes.LLM_DISPATCH_STATUS, "ok")
    span.set_attribute(MarieSpanAttributes.LLM_DISPATCH_EXECUTION_MS, execution_ms)
    span.set_attribute(
        MarieSpanAttributes.LLM_DISPATCH_TOTAL_LATENCY_MS,
        total_latency_ms,
    )
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
    span.set_attribute(MarieSpanAttributes.LLM_DISPATCH_STATUS, "error")
    span.set_attribute(MarieSpanAttributes.LLM_DISPATCH_ERROR_TYPE, type(exc).__name__)
    span.set_attribute(
        MarieSpanAttributes.LLM_DISPATCH_ERROR_MESSAGE, 'execution_failed'
    )
    span.set_attribute(MarieSpanAttributes.LLM_DISPATCH_EXECUTION_MS, execution_ms)
    span.set_attribute(
        MarieSpanAttributes.LLM_DISPATCH_TOTAL_LATENCY_MS,
        total_latency_ms,
    )
    span.set_status(StatusCode.ERROR, 'execution_failed')


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
        "estimated_cost_units": request_cost_units(request),
        "timeout_seconds": request.timeout_seconds,
        "message_count": len(request.call.messages),
        "current_error_summary": current_error_summary,
    }
