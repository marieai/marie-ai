from __future__ import annotations

import os
from typing import Any

from marie.engine.completion_contract import CompletionCallParams


async def execute_completion_call(
    client: Any,
    call: CompletionCallParams,
):
    return await client.chat.completions.create(**call.to_create_kwargs())


def resolve_openai_base_url_from_env() -> str | None:
    return os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")


def build_async_openai_client(
    api_key: str,
    base_url: str | None = None,
):
    import httpx
    from openai import AsyncOpenAI

    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=40,
            max_keepalive_connections=20,
        ),
        timeout=httpx.Timeout(
            connect=10.0,
            read=300.0,
            write=10.0,
            pool=30.0,
        ),
    )

    if base_url:
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

    return AsyncOpenAI(
        api_key=api_key,
        http_client=http_client,
        max_retries=0,
    )
