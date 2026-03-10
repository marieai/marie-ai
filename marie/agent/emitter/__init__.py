"""Event emitter system for agent middleware.

Provides event-driven observability for agents, tools, and LLM calls.
"""

from marie.agent.emitter.emitter import Emitter
from marie.agent.emitter.sync import emit_sync
from marie.agent.emitter.types import (
    EmittedEvent,
    EmitterOptions,
    EventData,
    EventHandler,
    EventMeta,
    EventTrace,
    ListenerEntry,
    ListenerOptions,
)

__all__ = [
    "Emitter",
    "EmittedEvent",
    "EmitterOptions",
    "emit_sync",
    "EventData",
    "EventHandler",
    "EventMeta",
    "EventTrace",
    "ListenerEntry",
    "ListenerOptions",
]
