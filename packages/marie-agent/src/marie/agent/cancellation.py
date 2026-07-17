"""Cancellation primitives for async streaming operations.

Provides AbortController/AbortSignal pattern (inspired by BeeAI framework)
for cooperative cancellation of streaming LLM calls and agent loops.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional


class AbortError(Exception):
    """Raised when an operation is aborted via AbortSignal."""

    def __init__(self, reason: str = "Operation aborted"):
        self.reason = reason
        super().__init__(reason)


class AbortSignal:
    """Observable cancellation signal.

    Check ``aborted`` to see if cancellation was requested, or call
    ``throw_if_aborted()`` to raise ``AbortError`` when aborted.

    Example::

        signal = AbortSignal()
        async for chunk in stream:
            signal.throw_if_aborted()
            yield chunk
    """

    def __init__(self) -> None:
        self._aborted = False
        self._reason: Optional[str] = None
        self._listeners: List[Callable[[str], Any]] = []

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def _abort(self, reason: str = "Operation aborted") -> None:
        """Called by AbortController — not for external use."""
        if self._aborted:
            return
        self._aborted = True
        self._reason = reason
        for listener in self._listeners:
            listener(reason)

    def throw_if_aborted(self) -> None:
        """Raise AbortError if the signal has been aborted."""
        if self._aborted:
            raise AbortError(self._reason or "Operation aborted")

    def on_abort(self, callback: Callable[[str], Any]) -> None:
        """Register a callback invoked when abort is triggered."""
        self._listeners.append(callback)
        if self._aborted:
            callback(self._reason or "Operation aborted")

    @classmethod
    def timeout(cls, seconds: float) -> "AbortSignal":
        """Create a signal that auto-aborts after *seconds*.

        The timer starts immediately. Requires a running asyncio event loop.
        """
        controller = AbortController()

        async def _timer() -> None:
            await asyncio.sleep(seconds)
            controller.abort(f"Timeout after {seconds}s")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_timer())
        except RuntimeError:
            pass  # No event loop — caller must manage timeout themselves

        return controller.signal


class AbortController:
    """Controls an AbortSignal.

    Example::

        controller = AbortController()
        task = run_streaming(signal=controller.signal)
        # Later...
        controller.abort("User cancelled")
    """

    def __init__(self) -> None:
        self._signal = AbortSignal()

    @property
    def signal(self) -> AbortSignal:
        return self._signal

    def abort(self, reason: str = "Operation aborted") -> None:
        """Abort the associated signal."""
        self._signal._abort(reason)
