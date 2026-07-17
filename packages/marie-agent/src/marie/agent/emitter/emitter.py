"""Core Emitter class for event-driven middleware system.

Provides an async-first event emitter with namespace support, pattern matching,
priority-based listener ordering, and parent-child event bubbling.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Callable, List, Optional, Set, TypeVar, Union

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

logger = logging.getLogger("marie.agent.emitter")

T = TypeVar("T")


class Emitter:
    """Async event emitter with namespace, pattern matching, and priority support.

    Key features:
    - Namespaced events: Events are prefixed with emitter's namespace
    - Pattern matching: Listeners can use wildcards (*, agent.*)
    - Priority ordering: Higher priority listeners execute first
    - Blocking listeners: Can pause event propagation until complete
    - Parent piping: Child emitters can pipe events to parents

    Example:
        ```python
        emitter = Emitter(EmitterOptions(namespace="agent"))


        @emitter.on("start")
        async def on_start(data):
            print(f"Agent started: {data}")


        await emitter.emit("start", {"name": "MyAgent"})
        # Emits "agent.start" event
        ```
    """

    def __init__(self, options: Optional[EmitterOptions] = None) -> None:
        self._options = options or EmitterOptions()
        self._listeners: List[ListenerEntry] = []
        self._parents: List["Emitter"] = []
        self._trace = self._options.trace or EventTrace()
        self._destroyed = False

    @property
    def namespace(self) -> Optional[str]:
        return self._options.namespace

    @property
    def trace(self) -> EventTrace:
        return self._trace

    @property
    def group_id(self) -> Optional[str]:
        return self._options.group_id

    def _qualify_name(self, name: str) -> str:
        """Qualify event name with namespace."""
        if self.namespace and not name.startswith(self.namespace + "."):
            return f"{self.namespace}.{name}"
        return name

    def on(
        self,
        pattern: str,
        handler: Optional[EventHandler] = None,
        *,
        priority: int = 0,
        is_blocking: bool = False,
        once: bool = False,
    ) -> Union[Callable[[EventHandler], EventHandler], str]:
        """Register an event listener.

        Can be used as a decorator or called directly:

            @emitter.on("start")
            async def handler(data): ...

            # Or directly:
            listener_id = emitter.on("start", my_handler)

        Args:
            pattern: Event pattern to match (supports wildcards)
            handler: Event handler function
            priority: Higher priority executes first
            is_blocking: If True, waits for handler before continuing
            once: If True, removes listener after first match

        Returns:
            If handler provided: listener ID string
            If no handler: decorator function
        """
        if self._destroyed:
            raise RuntimeError("Cannot add listener to destroyed emitter")

        qualified_pattern = self._qualify_name(pattern)
        options = ListenerOptions(priority=priority, is_blocking=is_blocking, once=once)

        if handler is not None:
            entry = ListenerEntry(
                pattern=qualified_pattern,
                handler=handler,
                options=options,
            )
            self._listeners.append(entry)
            self._sort_listeners()
            return entry.id

        def decorator(fn: EventHandler) -> EventHandler:
            entry = ListenerEntry(
                pattern=qualified_pattern,
                handler=fn,
                options=options,
            )
            self._listeners.append(entry)
            self._sort_listeners()
            return fn

        return decorator

    def once(
        self,
        pattern: str,
        handler: Optional[EventHandler] = None,
        *,
        priority: int = 0,
        is_blocking: bool = False,
    ) -> Union[Callable[[EventHandler], EventHandler], str]:
        """Register a one-time event listener.

        Convenience wrapper for on(..., once=True).
        """
        return self.on(
            pattern,
            handler,
            priority=priority,
            is_blocking=is_blocking,
            once=True,
        )

    def off(self, listener_id: str) -> bool:
        """Remove a listener by ID.

        Args:
            listener_id: The ID returned from on()

        Returns:
            True if listener was found and removed
        """
        for i, entry in enumerate(self._listeners):
            if entry.id == listener_id:
                del self._listeners[i]
                return True
        return False

    def off_pattern(self, pattern: str) -> int:
        """Remove all listeners matching a pattern.

        Args:
            pattern: Pattern to match (exact match on registered pattern)

        Returns:
            Number of listeners removed
        """
        qualified_pattern = self._qualify_name(pattern)
        original_count = len(self._listeners)
        self._listeners = [e for e in self._listeners if e.pattern != qualified_pattern]
        return original_count - len(self._listeners)

    def _sort_listeners(self) -> None:
        """Sort listeners by priority (descending)."""
        self._listeners.sort(key=lambda e: -e.options.priority)

    async def emit(
        self,
        name: str,
        data: Optional[EventData] = None,
        *,
        source: Optional[str] = None,
    ) -> EmittedEvent:
        """Emit an event to all matching listeners.

        Args:
            name: Event name (will be namespaced)
            data: Event payload
            source: Source identifier for tracing

        Returns:
            The emitted event object
        """
        if self._destroyed:
            raise RuntimeError("Cannot emit on destroyed emitter")

        qualified_name = self._qualify_name(name)
        data = data or {}

        meta = EventMeta(
            name=qualified_name,
            trace=self._trace,
            source=source,
            group_id=self.group_id,
        )

        event = EmittedEvent(name=qualified_name, data=data, meta=meta)

        await self._dispatch(event)

        # Pipe to parents
        for parent in self._parents:
            if not parent._destroyed:
                await parent._dispatch(event)

        return event

    async def _dispatch(self, event: EmittedEvent) -> None:
        """Dispatch event to matching listeners."""
        to_remove: Set[str] = set()
        blocking_tasks: List[asyncio.Task] = []
        non_blocking_tasks: List[asyncio.Task] = []

        for entry in self._listeners:
            if not entry.matches(event.name):
                continue

            if entry.options.once:
                to_remove.add(entry.id)

            task = asyncio.create_task(
                self._invoke_handler(entry.handler, event.data, event.meta)
            )

            if entry.options.is_blocking:
                blocking_tasks.append(task)
            else:
                non_blocking_tasks.append(task)

        # Wait for blocking listeners first
        if blocking_tasks:
            await asyncio.gather(*blocking_tasks, return_exceptions=True)

        # Fire-and-forget non-blocking listeners
        if non_blocking_tasks:
            asyncio.gather(*non_blocking_tasks, return_exceptions=True)

        # Remove once listeners
        if to_remove:
            self._listeners = [e for e in self._listeners if e.id not in to_remove]

    async def _invoke_handler(
        self,
        handler: EventHandler,
        data: EventData,
        meta: EventMeta,
    ) -> None:
        """Invoke a handler, supporting both sync and async functions."""
        try:
            result = handler(data)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.warning(f"Event handler error for {meta.name}: {e}")

    def pipe(self, parent: "Emitter") -> None:
        """Pipe events from this emitter to a parent.

        All events emitted by this emitter will also be dispatched
        to the parent emitter's listeners.

        Args:
            parent: Parent emitter to pipe to
        """
        if parent not in self._parents:
            self._parents.append(parent)

    def unpipe(self, parent: "Emitter") -> bool:
        """Remove a parent from the pipe chain.

        Args:
            parent: Parent emitter to remove

        Returns:
            True if parent was found and removed
        """
        if parent in self._parents:
            self._parents.remove(parent)
            return True
        return False

    def child(
        self,
        namespace: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> "Emitter":
        """Create a child emitter that pipes events to this one.

        The child inherits:
        - A child trace (same run_id, new span_id)
        - Group ID (if not overridden)

        Args:
            namespace: Optional namespace override for child
            group_id: Optional group_id override

        Returns:
            New child Emitter piped to this one
        """
        child_trace = self._trace.child()
        child_options = EmitterOptions(
            namespace=namespace or self.namespace,
            trace=child_trace,
            group_id=group_id or self.group_id,
        )
        child_emitter = Emitter(child_options)
        child_emitter.pipe(self)
        return child_emitter

    def destroy(self) -> None:
        """Destroy the emitter, preventing further use.

        Removes all listeners and pipes.
        """
        self._destroyed = True
        self._listeners.clear()
        self._parents.clear()

    def listener_count(self, pattern: Optional[str] = None) -> int:
        """Count registered listeners.

        Args:
            pattern: If provided, count only matching this pattern

        Returns:
            Number of listeners
        """
        if pattern is None:
            return len(self._listeners)
        qualified = self._qualify_name(pattern)
        return sum(1 for e in self._listeners if e.pattern == qualified)

    def event_names(self) -> List[str]:
        """Get list of unique event patterns with registered listeners."""
        return list(set(e.pattern for e in self._listeners))

    def __repr__(self) -> str:
        return (
            f"Emitter(namespace={self.namespace!r}, "
            f"listeners={len(self._listeners)}, "
            f"trace={self._trace.span_id[:8]})"
        )
