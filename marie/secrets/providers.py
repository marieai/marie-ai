"""Secret backends for marie-ai.

A `SecretProvider` resolves a secret *name* to its value. Backends are pluggable
so the same secret references resolve regardless of where the value actually
lives (environment, an in-memory mapping/mock, and — later — a database or a
dedicated secret manager). This lives at `marie.secrets` so any component
(executors, scheduler, connectors, agent tools, …) can reuse it.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class SecretProvider(Protocol):
    """Resolves a secret name to its value (or None if absent)."""

    def get_secret(self, key: str) -> str | None: ...


class EnvSecretProvider:
    """Resolves secrets from the process environment."""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self._env: dict[str, str] = dict(os.environ) if env is None else dict(env)

    def get_secret(self, key: str) -> str | None:
        return self._env.get(key)


class MappingSecretProvider:
    """Resolves secrets from an in-memory mapping (mock, or database-loaded)."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values: dict[str, str] = dict(values or {})

    def get_secret(self, key: str) -> str | None:
        return self._values.get(key)
