"""Synchronous emission helper for sync code contexts.

Provides emit_sync() for calling the async Emitter.emit() from
synchronous code like BaseAgent.run() which is a sync generator.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from marie.agent.emitter.emitter import Emitter
    from marie.agent.emitter.types import EmittedEvent, EventData


def emit_sync(
    emitter: Optional["Emitter"],
    name: str,
    data: Optional["EventData"] = None,
    *,
    source: Optional[str] = None,
) -> Optional["EmittedEvent"]:
    """Emit an event synchronously from sync code.

    Handles the async-to-sync bridge:
    - If an event loop is already running, schedules the emit
      as a task and returns immediately (fire-and-forget)
    - If no event loop exists, creates one to run the emit

    Args:
        emitter: The emitter instance (if None, does nothing)
        name: Event name
        data: Event payload
        source: Source identifier

    Returns:
        EmittedEvent if emit completed synchronously, None otherwise

    Example:
        ```python
        def run(self, messages):
            emit_sync(self.emitter, "start", {"messages": messages})
            for chunk in self._run(messages):
                yield chunk
            emit_sync(self.emitter, "success", {"result": chunk})
        ```
    """
    if emitter is None:
        return None

    try:
        loop = asyncio.get_running_loop()
        # Already in async context - schedule as task
        loop.create_task(emitter.emit(name, data, source=source))
        return None
    except RuntimeError:
        # No running loop - create one
        return asyncio.run(emitter.emit(name, data, source=source))
