from __future__ import annotations

from typing import Any


def _build_sync_client(url: str, **connection_options: Any) -> Any:
    try:
        from valkey import Valkey

        return Valkey.from_url(url, decode_responses=True, **connection_options)
    except ImportError:
        try:
            from redis import Redis

            return Redis.from_url(url, decode_responses=True, **connection_options)
        except ImportError as exc:
            raise ImportError(
                "LLM queue requires the `valkey` package (preferred) or `redis` package."
            ) from exc
