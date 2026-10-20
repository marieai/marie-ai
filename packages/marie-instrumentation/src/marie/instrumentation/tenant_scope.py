"""Per-item scope after server-side ownership resolution, never from OTLP claims.

These helpers propagate already-authorized identity inside a trusted process.
They do not authenticate external exporters, baggage, queue messages or payloads.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID


def _uuid(value: str) -> str:
    parsed = UUID(value)
    if not parsed.int:
        raise ValueError("Telemetry scope cannot use nil UUIDs")
    return str(parsed)


@dataclass(frozen=True)
class TelemetryScope:
    organization_id: str
    workspace_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "organization_id", _uuid(self.organization_id))
        if self.workspace_id != "":
            object.__setattr__(self, "workspace_id", _uuid(self.workspace_id))


_scope: ContextVar[TelemetryScope | None] = ContextVar(
    "marie_telemetry_scope", default=None
)
_RESERVED = frozenset(
    {
        "marie.organization_id",
        "marie.workspace_id",
        "marie.project_id",
        "marie_organization_id",
        "marie_workspace_id",
        "marie_project_id",
        "marie.identity_state",
        "marie.identity_version",
    }
)


def current_telemetry_scope() -> TelemetryScope | None:
    """Capture immutable identity before an item leaves the request/task context."""
    return _scope.get()


@contextmanager
def using_telemetry_scope(scope: TelemetryScope | None) -> Iterator[None]:
    """Activate resolved scope for one item; restore it on errors and nested calls."""
    if scope is not None and not isinstance(scope, TelemetryScope):
        raise TypeError("Expected a resolved TelemetryScope")
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


def scoped_attributes(attributes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot authoritative attributes, replacing every reserved ownership claim."""
    result = {
        key: value for key, value in (attributes or {}).items() if key not in _RESERVED
    }
    scope = current_telemetry_scope()
    result.update(
        {
            "marie.organization_id": scope.organization_id if scope else "",
            "marie.workspace_id": scope.workspace_id if scope else "",
            "marie.project_id": scope.workspace_id if scope else "",
            "marie.identity_state": "verified" if scope else "unattributed",
            "marie.identity_version": 1 if scope else 0,
        }
    )
    return result
