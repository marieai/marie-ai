from __future__ import annotations

import os
import socket
import uuid
from dataclasses import dataclass
from typing import Optional

from marie.utils.types import to_bool

DEFAULT_MAX_INLINE_PAYLOAD_BYTES = 16 * 1024 * 1024
DEFAULT_LLM_QUEUE_POOL_ID = "default"


@dataclass(frozen=True)
class LlmQueueConfig:
    enabled: bool
    valkey_url: Optional[str]
    pool_id: str
    producer_id: str
    producer_ttl_seconds: int
    producer_refresh_interval_seconds: float
    reply_queue_ttl_seconds: int
    reply_pop_timeout_seconds: float
    dispatch_pop_timeout_seconds: float
    max_batch_items: int
    max_batch_wait_ms: int
    max_buffered_requests_per_pool: int
    max_inline_payload_bytes: int
    fabric_group_id: Optional[str] = None
    gateway_id: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        *,
        enabled: Optional[bool] = None,
        valkey_url: Optional[str] = None,
        pool_id: Optional[str] = None,
        producer_id: Optional[str] = None,
    ) -> "LlmQueueConfig":
        alive_ttl = int(os.getenv("LLM_QUEUE_PRODUCER_TTL_SECONDS", "30"))
        refresh_interval = float(
            os.getenv(
                "LLM_QUEUE_PRODUCER_REFRESH_INTERVAL_SECONDS",
                str(max(1, alive_ttl // 3)),
            )
        )
        pool_id_value = pool_id or os.getenv(
            "LLM_QUEUE_POOL_ID",
            DEFAULT_LLM_QUEUE_POOL_ID,
        )
        max_batch_items = int(os.getenv("LLM_QUEUE_MAX_BATCH_ITEMS", "8"))

        return cls(
            enabled=(
                to_bool(os.getenv("LLM_QUEUE_ENABLED"), False)
                if enabled is None
                else enabled
            ),
            valkey_url=valkey_url or os.getenv("LLM_QUEUE_VALKEY_URL"),
            pool_id=pool_id_value,
            producer_id=producer_id
            or os.getenv("LLM_QUEUE_PRODUCER_ID")
            or _default_producer_id(),
            producer_ttl_seconds=alive_ttl,
            producer_refresh_interval_seconds=refresh_interval,
            reply_queue_ttl_seconds=int(
                os.getenv("LLM_QUEUE_REPLY_QUEUE_TTL_SECONDS", "300")
            ),
            reply_pop_timeout_seconds=float(
                os.getenv("LLM_QUEUE_REPLY_POP_TIMEOUT_SECONDS", "1.0")
            ),
            dispatch_pop_timeout_seconds=float(
                os.getenv("LLM_QUEUE_DISPATCH_POP_TIMEOUT_SECONDS", "1.0")
            ),
            max_batch_items=max_batch_items,
            max_batch_wait_ms=int(os.getenv("LLM_QUEUE_MAX_BATCH_WAIT_MS", "100")),
            max_buffered_requests_per_pool=int(
                os.getenv("LLM_QUEUE_MAX_BUFFERED_REQUESTS_PER_POOL", "32")
            ),
            max_inline_payload_bytes=int(
                os.getenv(
                    "LLM_QUEUE_MAX_INLINE_PAYLOAD_BYTES",
                    str(DEFAULT_MAX_INLINE_PAYLOAD_BYTES),
                )
            ),
            fabric_group_id=os.getenv("LLM_QUEUE_FABRIC_GROUP_ID") or None,
            gateway_id=os.getenv("LLM_QUEUE_GATEWAY_ID") or None,
        )


def _default_producer_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
