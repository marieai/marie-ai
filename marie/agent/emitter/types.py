"""Type definitions for the emitter system.

Provides EventTrace, EventMeta, EmitterOptions, and type aliases for
the event-driven middleware system inspired by BeeAI Framework.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

T = TypeVar("T")

# Type aliases for event handling
EventData = Dict[str, Any]
EventHandler = Callable[[EventData], Union[None, Awaitable[None]]]
EventFilter = Callable[[EventData], bool]


@dataclass(frozen=True)
class EventTrace:
    """Trace context for event propagation.

    Enables distributed tracing by carrying trace IDs through event chains.
    Child traces inherit run_id while generating new span_ids.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None

    def child(self) -> "EventTrace":
        """Create a child trace with a new span_id."""
        return EventTrace(
            run_id=self.run_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=self.span_id,
        )

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "run_id": self.run_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }


@dataclass
class EventMeta:
    """Metadata attached to every emitted event.

    Provides context about when and where an event was emitted,
    plus trace information for distributed observability.
    """

    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace: EventTrace = field(default_factory=EventTrace)
    source: Optional[str] = None
    group_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "trace": self.trace.to_dict(),
            "source": self.source,
            "group_id": self.group_id,
        }


@dataclass
class ListenerOptions:
    """Options for event listener registration.

    Attributes:
        priority: Higher priority listeners execute first (default 0)
        is_blocking: If True, listener must complete before event handling continues
        once: If True, listener is removed after first invocation
    """

    priority: int = 0
    is_blocking: bool = False
    once: bool = False


@dataclass
class EmitterOptions:
    """Configuration options for Emitter instances.

    Attributes:
        namespace: Event namespace prefix (e.g., "agent", "tool")
        trace: Initial trace context
        group_id: Group ID for related events (e.g., workflow ID)
    """

    namespace: Optional[str] = None
    trace: Optional[EventTrace] = None
    group_id: Optional[str] = None


@dataclass
class EmittedEvent:
    """A fully-formed event ready for emission.

    Combines the event name, data payload, and metadata.
    """

    name: str
    data: EventData
    meta: EventMeta

    @property
    def full_name(self) -> str:
        """Get the fully qualified event name including namespace."""
        return self.meta.name if self.meta else self.name


@dataclass
class ListenerEntry:
    """Internal representation of a registered listener."""

    pattern: str
    handler: EventHandler
    options: ListenerOptions
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def matches(self, event_name: str) -> bool:
        """Check if this listener matches the given event name.

        Supports:
        - Exact match: "agent.start" matches "agent.start"
        - Wildcard suffix: "agent.*" matches "agent.start", "agent.error"
        - Wildcard all: "*" matches everything
        """
        if self.pattern == "*":
            return True
        if self.pattern.endswith(".*"):
            prefix = self.pattern[:-2]
            return event_name == prefix or event_name.startswith(prefix + ".")
        return self.pattern == event_name
