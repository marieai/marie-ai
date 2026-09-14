from __future__ import annotations

import os
import socket
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit

DEFAULT_MAX_INLINE_PAYLOAD_BYTES = 16 * 1024 * 1024
DEFAULT_LLM_QUEUE_POOL_ID = "default"


@dataclass(frozen=True, init=False)
class LlmQueueConfig:
    enabled: bool
    queue_url: Optional[str] = field(repr=False)
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
    queue_contract_version: str = 'v2'

    def __init__(
        self,
        enabled: bool,
        valkey_url: Optional[str] = None,
        pool_id: str = DEFAULT_LLM_QUEUE_POOL_ID,
        producer_id: str = "",
        producer_ttl_seconds: int = 30,
        producer_refresh_interval_seconds: float = 10.0,
        reply_queue_ttl_seconds: int = 300,
        reply_pop_timeout_seconds: float = 1.0,
        dispatch_pop_timeout_seconds: float = 1.0,
        max_batch_items: int = 8,
        max_batch_wait_ms: int = 100,
        max_buffered_requests_per_pool: int = 32,
        max_inline_payload_bytes: int = DEFAULT_MAX_INLINE_PAYLOAD_BYTES,
        fabric_group_id: Optional[str] = None,
        gateway_id: Optional[str] = None,
        *,
        queue_url: Optional[str] = None,
        queue_contract_version: str = 'v2',
    ) -> None:
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(
            self,
            "queue_url",
            _resolve_queue_url(
                queue_url,
                valkey_url,
                canonical_name="queue_url",
                legacy_name="valkey_url",
            ),
        )
        object.__setattr__(self, "pool_id", pool_id)
        object.__setattr__(self, "producer_id", producer_id)
        object.__setattr__(self, "producer_ttl_seconds", producer_ttl_seconds)
        object.__setattr__(
            self,
            "producer_refresh_interval_seconds",
            producer_refresh_interval_seconds,
        )
        object.__setattr__(self, "reply_queue_ttl_seconds", reply_queue_ttl_seconds)
        object.__setattr__(self, "reply_pop_timeout_seconds", reply_pop_timeout_seconds)
        object.__setattr__(
            self,
            "dispatch_pop_timeout_seconds",
            dispatch_pop_timeout_seconds,
        )
        object.__setattr__(self, "max_batch_items", max_batch_items)
        object.__setattr__(self, "max_batch_wait_ms", max_batch_wait_ms)
        object.__setattr__(
            self,
            "max_buffered_requests_per_pool",
            max_buffered_requests_per_pool,
        )
        object.__setattr__(
            self,
            "max_inline_payload_bytes",
            max_inline_payload_bytes,
        )
        object.__setattr__(self, "fabric_group_id", fabric_group_id)
        object.__setattr__(self, "gateway_id", gateway_id)
        object.__setattr__(
            self,
            'queue_contract_version',
            resolve_queue_contract_version(queue_contract_version),
        )

    @property
    def valkey_url(self) -> Optional[str]:
        return self.queue_url

    @classmethod
    def from_env(
        cls,
        *,
        enabled: Optional[bool] = None,
        queue_url: Optional[str] = None,
        valkey_url: Optional[str] = None,
        pool_id: Optional[str] = None,
        producer_id: Optional[str] = None,
        fabric_group_id: Optional[str] = None,
        queue_contract_version: Optional[str] = None,
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
            queue_contract_version=resolve_queue_contract_version(
                queue_contract_version, os.getenv('LLM_QUEUE_CONTRACT_VERSION')
            ),
            enabled=(
                _to_bool(os.getenv("LLM_QUEUE_ENABLED"), False)
                if enabled is None
                else enabled
            ),
            queue_url=_resolve_queue_url(
                queue_url,
                valkey_url,
                os.getenv("LLM_QUEUE_URL"),
                os.getenv("LLM_QUEUE_VALKEY_URL"),
                canonical_name="queue_url",
                legacy_name="valkey_url",
                canonical_env_name="LLM_QUEUE_URL",
                legacy_env_name="LLM_QUEUE_VALKEY_URL",
            ),
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
            fabric_group_id=fabric_group_id
            or os.getenv("LLM_QUEUE_FABRIC_GROUP_ID")
            or None,
            gateway_id=os.getenv("LLM_QUEUE_GATEWAY_ID") or None,
        )


def _default_producer_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def resolve_queue_contract_version(
    explicit: Optional[str] = None, environment: Optional[str] = None
) -> str:
    value = _nonempty(explicit) or _nonempty(environment) or 'v2'
    if value not in {'v2', 'v3'}:
        raise ValueError('queue_contract_version must be v2 or v3')
    return value


def resolve_fabric_id(*values: Optional[str]) -> str:
    from marie.engine.llm_queue.queue_keys import QueueKeys

    identities = {
        QueueKeys(value.strip()).fabric_id
        for value in values
        if value and value.strip()
    }
    if len(identities) != 1:
        raise ValueError('V3 requires one nonempty, consistent fabric identity')
    return identities.pop()


def _to_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_queue_url(
    queue_url: Optional[str],
    valkey_url: Optional[str],
    env_queue_url: Optional[str] = None,
    env_valkey_url: Optional[str] = None,
    *,
    canonical_name: str,
    legacy_name: str,
    canonical_env_name: Optional[str] = None,
    legacy_env_name: Optional[str] = None,
) -> Optional[str]:
    explicit_queue_url = _nonempty(queue_url)
    explicit_valkey_url = _nonempty(valkey_url)
    if explicit_queue_url:
        selected_name = canonical_name
        selected_url = explicit_queue_url
        if explicit_valkey_url:
            warnings.warn(
                f"Multiple queue URL settings configured; using {selected_name}.",
                DeprecationWarning,
                stacklevel=3,
            )
    elif explicit_valkey_url:
        selected_name = legacy_name
        selected_url = explicit_valkey_url
        warnings.warn(
            f"{legacy_name} is deprecated; use {canonical_name}.",
            DeprecationWarning,
            stacklevel=3,
        )
    else:
        selected_env_queue_url = _nonempty(env_queue_url)
        selected_env_valkey_url = _nonempty(env_valkey_url)
        if selected_env_queue_url:
            selected_name = canonical_env_name
            selected_url = selected_env_queue_url
            if selected_env_valkey_url:
                warnings.warn(
                    f"Multiple queue URL settings configured; using {selected_name}.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        elif selected_env_valkey_url:
            selected_name = legacy_env_name
            selected_url = selected_env_valkey_url
            warnings.warn(
                f"{legacy_env_name} is deprecated; use {canonical_env_name}.",
                DeprecationWarning,
                stacklevel=3,
            )
        else:
            return None

    if not _is_redis_url(selected_url):
        raise ValueError(
            f"Invalid {selected_name}: expected a redis:// or rediss:// URL."
        )
    return selected_url


def _nonempty(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _is_redis_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"redis", "rediss"} and bool(parsed.netloc)
