"""Original-process V3 admission and authoritative terminal result delivery."""

from __future__ import annotations

import atexit
import copy
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from uuid import uuid4

from marie.engine.completion_contract import (
    CompletionCallParams,
    QueuedCompletionEnvelopeV3,
    completion_finish_reason,
    extract_completion_text,
    require_terminal_completion,
)
from marie.engine.exceptions import MaxTokensExceededError
from marie.engine.llm_queue.config import LlmQueueConfig, resolve_fabric_id
from marie.engine.llm_queue.result_types import BatchResult
from marie.engine.llm_queue.scheduler import prepared_cost_units
from marie.engine.llm_queue.store import (
    ProducerDead,
    RequestStore,
    StoreUnavailable,
    TransitionRejected,
)
from marie.engine.llm_queue.submitter import _resolve_queue_pool_id


class QueueTaskError(RuntimeError):
    """Bounded task category; cancellation confirmation is distinct from a request."""

    def __init__(
        self, category: str, *, state: str = '', confirmed: bool = False
    ) -> None:
        self.category = category
        self.state = state
        self.confirmed = confirmed
        super().__init__(category)


class ProducerClosed(ProducerDead):
    """This process session has closed and cannot deliver any more results."""


@dataclass(slots=True)
class _Pending:
    task_id: str
    deadline: float
    result: BatchResult | None = None
    cancel: bool = False
    expire: bool = False
    category: str = 'deadline_exceeded'


@dataclass(frozen=True)
class PreparedCalls:
    """Prepare one original item at a time after validating shared completion options."""

    count: int
    prepare: Callable[[int], CompletionCallParams]
    validation_call: CompletionCallParams

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> CompletionCallParams:
        return self.prepare(index)


class V3Producer:
    def __init__(self, *, config: LlmQueueConfig) -> None:
        self.config = config
        self.fabric_id = resolve_fabric_id(config.fabric_group_id)
        if not config.queue_url:
            raise ValueError('V3 requires a configured queue URL')
        self._pid = os.getpid()
        self._condition = threading.Condition()
        self._start_lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._error: Exception | None = None
        self._store: RequestStore | None = None
        self.producer_id: str | None = None
        self._pending: dict[str, _Pending] = {}
        self._preparing = 0
        if (
            type(config.max_buffered_requests_per_pool) is not int
            or config.max_buffered_requests_per_pool < 1
        ):
            raise ValueError('Producer admission window must be positive')
        self._poll_ids: deque[str] = deque()
        self._threads: list[threading.Thread] = []
        self._lease_ms = int(config.producer_ttl_seconds * 1000)
        self._refresh = config.producer_refresh_interval_seconds
        if not 0 < self._refresh < config.producer_ttl_seconds / 2:
            raise ValueError(
                'Producer refresh must be positive and less than half its lease'
            )

    def _check(self) -> None:
        if os.getpid() != self._pid:
            raise ProducerClosed('An inherited producer cannot own child requests')
        if self._error is not None:
            raise self._error
        if self._stop.is_set():
            raise ProducerClosed('Producer session closed')

    def _start(self, deadline: float, cancellation: threading.Event) -> RequestStore:
        self._check()
        while True:
            self._check()
            if cancellation.is_set():
                raise QueueTaskError('cancellation_requested')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QueueTaskError('deadline_exceeded')
            if self._start_lock.acquire(timeout=min(0.05, remaining)):
                break
        try:
            while self.producer_id is None:
                self._check()
                if cancellation.is_set():
                    raise QueueTaskError('cancellation_requested')
                if time.monotonic() >= deadline:
                    raise QueueTaskError('store_unavailable')
                try:
                    if self._store is None:
                        self._store = RequestStore.for_producer(
                            self.config.queue_url,
                            fabric_id=self.fabric_id,
                            io_timeout_seconds=min(0.25, self._refresh / 2),
                        )
                    self._check()
                    if cancellation.is_set() or time.monotonic() >= deadline:
                        continue
                    self.producer_id = self._store.create_producer(
                        lease_ms=self._lease_ms
                    )
                    self._check()
                except (StoreUnavailable, TransitionRejected) as exc:
                    if (
                        isinstance(exc, TransitionRejected)
                        and str(exc) != 'backpressure'
                    ):
                        raise
                    cancellation.wait(min(0.05, max(0, deadline - time.monotonic())))
            if not self._threads:
                for target in (self._heartbeat, self._poll):
                    thread = threading.Thread(
                        target=self._supervise,
                        args=(target,),
                        daemon=True,
                        name=f'llm-v3-{target.__name__}',
                    )
                    self._threads.append(thread)
                    thread.start()
                atexit.register(self.close)
            return self._store
        finally:
            self._start_lock.release()

    def _supervise(self, target: Callable[[], None]) -> None:
        try:
            target()
        except Exception:
            # Never leave healthy-looking waiters after an unexpected worker failure.
            self._fail(ProducerClosed('Producer background worker stopped'))

    def _fail(self, error: Exception) -> None:
        with self._condition:
            self._error = error
            self._stop.set()
            self._condition.notify_all()

    def _heartbeat(self) -> None:
        while not self._stop.wait(self._refresh):
            try:
                if not self._store.renew_producer(
                    self.producer_id, lease_ms=self._lease_ms
                ):
                    self._fail(ProducerDead('Original producer lease expired'))
                    return
            except StoreUnavailable:
                # Unknown is not dead; a later renew must still validate the old lease.
                continue

    def _poll(self) -> None:
        while not self._stop.wait(0.02):
            with self._condition:
                ids = [
                    self._poll_ids.popleft()
                    for _ in range(min(64, len(self._poll_ids)))
                ]
            for attempt in ids:
                if self._stop.is_set():
                    return
                with self._condition:
                    pending = self._pending.get(attempt)
                if pending is None:
                    continue
                try:
                    if pending.cancel or pending.expire:
                        if time.monotonic() >= pending.deadline:
                            with self._condition:
                                self._pending.pop(attempt, None)
                            continue
                        reply = self._store.cancel_or_expire(
                            self.producer_id,
                            attempt,
                            expire=pending.expire and not pending.cancel,
                        )
                        if reply.disposition in {
                            'finished',
                            'existing',
                            'producer_dead',
                        }:
                            with self._condition:
                                self._pending.pop(attempt, None)
                            continue
                    else:
                        if self._stop.is_set():
                            return
                        metadata = self._store.metadata(attempt)
                        if metadata is not None and metadata.last_error:
                            pending.category = _category(metadata.last_error)
                        if metadata is not None and metadata.state in {
                            'succeeded',
                            'failed',
                            'cancelled',
                            'expired',
                        }:
                            if self._stop.is_set():
                                return
                            outcome = self._store.read_result(self.producer_id, attempt)
                            if outcome is not None:
                                result = _batch_result(
                                    pending.task_id, metadata.state, outcome
                                )
                                with self._condition:
                                    if not pending.cancel and not pending.expire:
                                        pending.result = result
                                    self._pending.pop(attempt, None)
                                    self._condition.notify_all()
                                continue
                except ProducerDead as exc:
                    self._fail(exc)
                    return
                except StoreUnavailable:
                    pass
                with self._condition:
                    if (
                        pending.cancel or pending.expire
                    ) and time.monotonic() >= pending.deadline:
                        self._pending.pop(attempt, None)
                    elif attempt in self._pending:
                        self._poll_ids.append(attempt)

    def execute(
        self,
        *,
        calls: Sequence[CompletionCallParams] | PreparedCalls,
        batch_request_id: str,
        batch_timeout: float,
        on_result: Callable | None = None,
        metadata: dict[str, Any] | None = None,
        queue_deadline: float | None = None,
        cancellation: threading.Event | None = None,
    ) -> list[BatchResult]:
        deadline = (
            queue_deadline
            if queue_deadline is not None
            else time.monotonic() + batch_timeout
        )
        if isinstance(calls, PreparedCalls):
            require_terminal_completion(calls.validation_call)
        else:
            for call in calls:
                require_terminal_completion(call)
        if not calls:
            return []
        cancellation = cancellation or threading.Event()
        results: list[BatchResult | None] = [None] * len(calls)
        owned: dict[str, tuple[int, _Pending]] = {}
        next_index = 0
        envelope = None
        preparing = False
        try:
            store = self._start(deadline, cancellation)
            pool = _resolve_queue_pool_id(self.config.pool_id, metadata)
            route = None
            expires = None
            while next_index < len(calls) or any(result is None for result in results):
                self._check()
                if cancellation.is_set():
                    raise QueueTaskError('cancellation_requested')
                for attempt, (index, pending) in list(owned.items()):
                    if pending.result is None or results[index] is not None:
                        continue
                    # Set delivery identity before entering arbitrary application code.
                    results[index] = pending.result
                    if on_result and pending.result.error is None:
                        try:
                            on_result(pending.task_id, pending.result.response)
                        except Exception:
                            results[index] = BatchResult(
                                pending.task_id, None, QueueTaskError('callback_failed')
                            )
                if cancellation.is_set():
                    raise QueueTaskError('cancellation_requested')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise QueueTaskError('deadline_exceeded')
                if all(result is not None for result in results):
                    break
                if next_index < len(calls):
                    try:
                        if route is None:
                            route = store.resolve_route(pool)
                            if route is None:
                                raise QueueTaskError('route_unavailable')
                        if time.monotonic() >= deadline or cancellation.is_set():
                            continue
                        if route is not None:
                            if expires is None:
                                # Timestamp taken before envelope serialization; subtract clock roundtrip.
                                server_now = store.server_time_ms()
                                expires = server_now + int(
                                    max(0, deadline - time.monotonic()) * 1000
                                )
                            if time.monotonic() >= deadline or cancellation.is_set():
                                continue
                            if envelope is None:
                                with self._condition:
                                    if (
                                        len(self._pending) + self._preparing
                                        >= self.config.max_buffered_requests_per_pool
                                    ):
                                        self._condition.wait(
                                            timeout=min(0.02, remaining)
                                        )
                                        continue
                                    self._preparing += 1
                                    preparing = True
                                attempt = uuid4().hex
                                task_id = f'{batch_request_id}_task_{next_index}'
                                prepared = calls[next_index]
                                if not isinstance(calls, PreparedCalls):
                                    prepared = copy.deepcopy(prepared)
                                require_terminal_completion(prepared)
                                envelope = QueuedCompletionEnvelopeV3(
                                    contract_version='v3',
                                    fabric_group_id=self.fabric_id,
                                    producer_id=self.producer_id,
                                    attempt_id=attempt,
                                    logical_batch_id=batch_request_id,
                                    logical_task_id=task_id,
                                    item_index=next_index,
                                    expires_at_ms=expires,
                                    call=prepared,
                                    estimated_cost_units=prepared_cost_units(
                                        prepared, metadata
                                    ),
                                    **route,
                                )
                                del prepared
                                pending = _Pending(task_id, deadline)
                                owned[attempt] = (next_index, pending)
                                with self._condition:
                                    self._preparing -= 1
                                    preparing = False
                                    self._pending[attempt] = pending
                                    self._poll_ids.append(attempt)
                            if time.monotonic() >= deadline or cancellation.is_set():
                                continue
                            reply = store.admit(envelope)
                            if reply.disposition in {'admitted', 'existing'}:
                                envelope = None
                                next_index += 1
                                continue
                            if reply.disposition == 'producer_dead':
                                raise ProducerDead('Original producer lease expired')
                            if reply.disposition != 'backpressure':
                                raise QueueTaskError(reply.disposition)
                    except StoreUnavailable:
                        pass
                    except (ValueError, TypeError, OverflowError):
                        if preparing:
                            with self._condition:
                                self._preparing -= 1
                                preparing = False
                        results[next_index] = BatchResult(
                            f'{batch_request_id}_task_{next_index}',
                            None,
                            QueueTaskError('invalid_request'),
                        )
                        if envelope is not None:
                            with self._condition:
                                self._pending.pop(envelope.attempt_id, None)
                        envelope = None
                        next_index += 1
                with self._condition:
                    self._condition.wait(
                        timeout=min(0.02, max(0, deadline - time.monotonic()))
                    )
        except QueueTaskError as error:
            for index, result in enumerate(results):
                if result is None:
                    cause = error
                    if error.category == 'deadline_exceeded':
                        pending = next(
                            (
                                entry
                                for original, entry in owned.values()
                                if original == index
                            ),
                            None,
                        )
                        if pending is not None:
                            cause = QueueTaskError(
                                pending.category, state='expiry_requested'
                            )
                    results[index] = BatchResult(
                        f'{batch_request_id}_task_{index}', None, cause
                    )
        finally:
            envelope = None
            with self._condition:
                if preparing:
                    self._preparing -= 1
                for attempt, (index, pending) in owned.items():
                    if attempt in self._pending:
                        pending.cancel = (
                            cancellation.is_set() or time.monotonic() < deadline
                        )
                        pending.expire = not pending.cancel
                self._condition.notify_all()
        return results

    def close(self) -> None:
        if os.getpid() != self._pid:
            return
        self._fail(ProducerClosed('Producer session closed'))
        # Startup owns any in-flight join/create result until it releases this lock.
        with self._start_lock:
            if self._closed:
                return
            self._closed = True
            atexit.unregister(self.close)
            for thread in self._threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=1)
            if self._store is not None:
                try:
                    if self.producer_id is not None:
                        self._store.close_producer(self.producer_id)
                        cleanup_until = time.monotonic() + 1
                        while time.monotonic() < cleanup_until:
                            cleaned = self._store.expire_producer(
                                self.producer_id,
                                limit=min(100, self._store.limits.cleanup_page_size),
                                should_stop=lambda: time.monotonic() >= cleanup_until,
                            )
                            if not cleaned:
                                break
                except StoreUnavailable:
                    pass  # The non-renewing lease is the final shutdown bound.
                finally:
                    self._store.close()
            with self._condition:
                self._pending.clear()
                self._poll_ids.clear()


def _batch_result(task_id: str, state: str, outcome: Any) -> BatchResult:
    if state != 'succeeded':
        category = _category(
            outcome.get('category') or outcome.get('error')
            if isinstance(outcome, dict)
            else None
        )
        return BatchResult(
            task_id, None, QueueTaskError(category, state=state, confirmed=True)
        )
    try:
        if not isinstance(outcome, dict):
            raise ValueError
        if completion_finish_reason(outcome) == 'length':
            return BatchResult(task_id, None, MaxTokensExceededError())
        _, text = extract_completion_text(outcome)
        return BatchResult(task_id, text, None)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        return BatchResult(
            task_id,
            None,
            QueueTaskError('invalid_completion', state=state, confirmed=True),
        )


def _category(value: Any) -> str:
    return (
        value
        if isinstance(value, str) and value.isidentifier() and len(value) <= 64
        else 'remote_failed'
    )
