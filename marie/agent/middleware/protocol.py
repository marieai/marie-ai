"""Middleware protocol and type definitions.

Defines the RunMiddlewareProtocol that middleware implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from marie.agent.emitter import Emitter


@runtime_checkable
class RunMiddlewareProtocol(Protocol):
    """Protocol for agent run middleware.

    Middleware can observe and modify agent execution by registering
    event listeners on the emitter.
    """

    @property
    def name(self) -> str:
        """Middleware name for identification."""
        ...

    @property
    def priority(self) -> int:
        """Priority for middleware ordering. Higher = executed first."""
        ...

    def bind(self, emitter: "Emitter") -> None:
        """Bind middleware to an emitter.

        Called when an agent run starts. Middleware should register
        its event listeners here.

        Args:
            emitter: The emitter for this agent run
        """
        ...


class BaseMiddleware(ABC):
    """Base class for middleware implementations.

    Provides common functionality and enforces the middleware interface.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        priority: int = 0,
    ) -> None:
        self._name = name or self.__class__.__name__
        self._priority = priority
        self._listener_ids: List[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @abstractmethod
    def bind(self, emitter: "Emitter") -> None:
        """Bind middleware to an emitter.

        Subclasses must implement this to register event listeners.
        Store listener IDs in self._listener_ids for cleanup.
        """
        pass

    def unbind(self, emitter: "Emitter") -> None:
        """Unbind middleware from an emitter.

        Removes all listeners registered by this middleware.
        """
        for listener_id in self._listener_ids:
            emitter.off(listener_id)
        self._listener_ids.clear()


# Type alias for middleware list
MiddlewareList = List[RunMiddlewareProtocol]
