from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit

from marie.engine.completion_contract import CompletionCallParams
from marie.engine.openai_compat import execute_completion_call


class OpenAICompatibleExecutionAdapter:
    def __init__(
        self,
        *,
        client: Any = None,
        client_factory: Optional[Callable[[], Any]] = None,
        logger,
        default_timeout: Optional[float] = None,
        backend_address: Optional[str] = None,
    ):
        self.client = client
        self.client_factory = client_factory
        self.logger = logger
        self.default_timeout = default_timeout
        self.backend_address = _safe_backend_address(backend_address)

    async def execute(
        self,
        call: CompletionCallParams,
        *,
        timeout_seconds: Optional[float] = None,
    ):
        timeout_budget = timeout_seconds or self.default_timeout
        if self.client is not None:
            return await self._execute_with_client(self.client, call, timeout_budget)

        if self.client_factory is None:
            raise RuntimeError("OpenAI-compatible execution adapter has no client")

        client = self.client_factory()
        try:
            return await self._execute_with_client(client, call, timeout_budget)
        finally:
            await _close_client(client)

    async def _execute_with_client(
        self,
        client: Any,
        call: CompletionCallParams,
        timeout_budget: Optional[float],
    ):
        try:
            completion_coro = execute_completion_call(client, call)
            if timeout_budget is None:
                return await completion_coro
            return await asyncio.wait_for(completion_coro, timeout=timeout_budget)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI-compatible dispatch failed against {self.backend_address}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def _safe_backend_address(backend_address: Optional[str]) -> str:
    if not backend_address:
        return "configured OpenAI-compatible endpoint"
    parsed = urlsplit(backend_address)
    if not parsed.username and not parsed.password:
        return backend_address
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


async def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        result = close()
        if hasattr(result, "__await__"):
            await result
