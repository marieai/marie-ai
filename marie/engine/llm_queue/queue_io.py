from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Protocol

from marie.engine.completion_contract import (
    CompletionReplyEnvelope,
    QueuedCompletionEnvelope,
)
from marie.engine.llm_queue.valkey_keys import (
    producer_alive_key,
    reply_queue_key,
    request_queue_key,
)


class ListQueueClient(Protocol):
    def push_request(self, request: QueuedCompletionEnvelope) -> None: ...

    def pop_request(
        self, pool_id: str, timeout: float
    ) -> Optional[QueuedCompletionEnvelope]: ...

    def try_pop_request(self, pool_id: str) -> Optional[QueuedCompletionEnvelope]: ...

    def push_request_front(self, request: QueuedCompletionEnvelope) -> None: ...

    def push_reply(
        self, reply: CompletionReplyEnvelope, ttl_seconds: Optional[int] = None
    ) -> None: ...

    def pop_reply(
        self, producer_id: str, timeout: float
    ) -> Optional[CompletionReplyEnvelope]: ...

    def set_producer_alive(
        self, producer_id: str, value: str, ttl_seconds: int
    ) -> None: ...

    def clear_producer_alive(self, producer_id: str) -> None: ...

    def is_producer_alive(self, producer_id: str) -> bool: ...

    def request_queue_depth(self, pool_id: str) -> int: ...

    def sample_requests(
        self, pool_id: str, limit: int
    ) -> list[QueuedCompletionEnvelope]: ...

    def close(self) -> None: ...


class ValkeyListQueueClient:
    def __init__(self, url: str):
        self._client = _build_sync_client(url)

    def push_request(self, request: QueuedCompletionEnvelope) -> None:
        self._client.rpush(request_queue_key(request.pool_id), request.to_json())

    def pop_request(
        self, pool_id: str, timeout: float
    ) -> Optional[QueuedCompletionEnvelope]:
        result = self._client.blpop(
            request_queue_key(pool_id),
            timeout=max(1, math.ceil(timeout)),
        )
        if not result:
            return None
        return QueuedCompletionEnvelope.from_json(result[1])

    def try_pop_request(self, pool_id: str) -> Optional[QueuedCompletionEnvelope]:
        payload = self._client.lpop(request_queue_key(pool_id))
        if payload is None:
            return None
        return QueuedCompletionEnvelope.from_json(payload)

    def push_request_front(self, request: QueuedCompletionEnvelope) -> None:
        self._client.lpush(request_queue_key(request.pool_id), request.to_json())

    def push_reply(
        self, reply: CompletionReplyEnvelope, ttl_seconds: Optional[int] = None
    ) -> None:
        queue_key = reply_queue_key(reply.producer_id)
        pipe = self._client.pipeline()
        pipe.rpush(queue_key, reply.to_json())
        if ttl_seconds:
            pipe.expire(queue_key, ttl_seconds)
        pipe.execute()

    def pop_reply(
        self, producer_id: str, timeout: float
    ) -> Optional[CompletionReplyEnvelope]:
        result = self._client.blpop(
            reply_queue_key(producer_id),
            timeout=max(1, math.ceil(timeout)),
        )
        if not result:
            return None
        return CompletionReplyEnvelope.from_json(result[1])

    def set_producer_alive(
        self, producer_id: str, value: str, ttl_seconds: int
    ) -> None:
        self._client.set(producer_alive_key(producer_id), value, ex=ttl_seconds)

    def clear_producer_alive(self, producer_id: str) -> None:
        self._client.delete(producer_alive_key(producer_id))

    def is_producer_alive(self, producer_id: str) -> bool:
        return bool(self._client.exists(producer_alive_key(producer_id)))

    def request_queue_depth(self, pool_id: str) -> int:
        return int(self._client.llen(request_queue_key(pool_id)) or 0)

    def sample_requests(
        self, pool_id: str, limit: int
    ) -> list[QueuedCompletionEnvelope]:
        if limit <= 0:
            return []
        payloads = self._client.lrange(request_queue_key(pool_id), 0, limit - 1) or []
        return [QueuedCompletionEnvelope.from_json(payload) for payload in payloads]

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class InMemoryListQueueClient:
    def __init__(self):
        self._lists: Dict[str, Deque[str]] = defaultdict(deque)
        self._list_expiry: Dict[str, float] = {}
        self._alive: Dict[str, tuple[str, float]] = {}
        self._condition = threading.Condition()
        self._closed = False

    def push_request(self, request: QueuedCompletionEnvelope) -> None:
        with self._condition:
            self._cleanup_locked()
            self._lists[request_queue_key(request.pool_id)].append(request.to_json())
            self._condition.notify_all()

    def pop_request(
        self, pool_id: str, timeout: float
    ) -> Optional[QueuedCompletionEnvelope]:
        payload = self._blocking_pop(request_queue_key(pool_id), timeout)
        if payload is None:
            return None
        return QueuedCompletionEnvelope.from_json(payload)

    def try_pop_request(self, pool_id: str) -> Optional[QueuedCompletionEnvelope]:
        with self._condition:
            self._cleanup_locked()
            queue_key = request_queue_key(pool_id)
            if not self._lists[queue_key]:
                return None
            return QueuedCompletionEnvelope.from_json(self._lists[queue_key].popleft())

    def push_request_front(self, request: QueuedCompletionEnvelope) -> None:
        with self._condition:
            self._cleanup_locked()
            self._lists[request_queue_key(request.pool_id)].appendleft(
                request.to_json()
            )
            self._condition.notify_all()

    def push_reply(
        self, reply: CompletionReplyEnvelope, ttl_seconds: Optional[int] = None
    ) -> None:
        with self._condition:
            self._cleanup_locked()
            queue_key = reply_queue_key(reply.producer_id)
            self._lists[queue_key].append(reply.to_json())
            if ttl_seconds:
                self._list_expiry[queue_key] = time.monotonic() + ttl_seconds
            self._condition.notify_all()

    def pop_reply(
        self, producer_id: str, timeout: float
    ) -> Optional[CompletionReplyEnvelope]:
        payload = self._blocking_pop(reply_queue_key(producer_id), timeout)
        if payload is None:
            return None
        return CompletionReplyEnvelope.from_json(payload)

    def set_producer_alive(
        self, producer_id: str, value: str, ttl_seconds: int
    ) -> None:
        with self._condition:
            self._alive[producer_alive_key(producer_id)] = (
                value,
                time.monotonic() + ttl_seconds,
            )
            self._condition.notify_all()

    def clear_producer_alive(self, producer_id: str) -> None:
        with self._condition:
            self._alive.pop(producer_alive_key(producer_id), None)
            self._condition.notify_all()

    def is_producer_alive(self, producer_id: str) -> bool:
        with self._condition:
            self._cleanup_locked()
            return producer_alive_key(producer_id) in self._alive

    def request_queue_depth(self, pool_id: str) -> int:
        with self._condition:
            self._cleanup_locked()
            return len(self._lists.get(request_queue_key(pool_id), ()))

    def sample_requests(
        self, pool_id: str, limit: int
    ) -> list[QueuedCompletionEnvelope]:
        if limit <= 0:
            return []
        with self._condition:
            self._cleanup_locked()
            queue_key = request_queue_key(pool_id)
            payloads = list(self._lists.get(queue_key, ()))[:limit]
        return [QueuedCompletionEnvelope.from_json(payload) for payload in payloads]

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _blocking_pop(self, key: str, timeout: float) -> Optional[str]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._cleanup_locked()
                if self._closed:
                    return None
                if self._lists[key]:
                    return self._lists[key].popleft()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def _cleanup_locked(self) -> None:
        now = time.monotonic()
        expired_queues = [k for k, expiry in self._list_expiry.items() if expiry <= now]
        for key in expired_queues:
            self._list_expiry.pop(key, None)
            self._lists.pop(key, None)

        expired_alive = [k for k, (_, expiry) in self._alive.items() if expiry <= now]
        for key in expired_alive:
            self._alive.pop(key, None)


def _build_sync_client(url: str):
    try:
        from valkey import Valkey

        return Valkey.from_url(url, decode_responses=True)
    except ImportError:
        try:
            from redis import Redis

            return Redis.from_url(url, decode_responses=True)
        except ImportError as exc:
            raise ImportError(
                "LLM queue requires the `valkey` package (preferred) or `redis` package."
            ) from exc
