"""A2A Executor for exposing Marie agents via A2A protocol.

This module provides the A2AExecutor class that wraps Marie agents
and exposes them through the A2A protocol interface using the official SDK.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.types import AgentCard

from marie.agent.a2a.agent_card import AgentCardBuilder
from marie.executor.a2a.sdk_executor import MarieA2AExecutor

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent

logger = logging.getLogger(__name__)


class A2AExecutor:
    """Executes Marie agents via A2A protocol using the official SDK.

    Wraps a Marie agent and provides A2A protocol handlers for
    message processing, task management, and streaming responses.

    Example:
        executor = A2AExecutor(
            agent=my_agent,
            name="My Agent",
            url="http://localhost:8000",
        )

        # Get the agent card
        card = executor.agent_card

        # Get FastAPI/Starlette app for serving
        app = executor.get_app()
    """

    def __init__(
        self,
        agent: "BaseAgent",
        name: Optional[str] = None,
        url: str = "http://localhost:8000",
        version: str = "1.0.0",
        description: Optional[str] = None,
        streaming: bool = True,
        push_notifications: bool = False,
        task_store: Optional[TaskStore] = None,
    ):
        """Initialize the A2A executor.

        Args:
            agent: The Marie agent to expose.
            name: Agent name (defaults to agent.name).
            url: The agent's A2A endpoint URL.
            version: Agent version string.
            description: Agent description.
            streaming: Whether to support streaming responses.
            push_notifications: Whether to support push notifications.
            task_store: Optional custom task store (defaults to InMemoryTaskStore).
        """
        self.agent = agent
        self._url = url
        self._streaming = streaming
        self._push_notifications = push_notifications

        # Build agent card
        self._agent_card = (
            AgentCardBuilder()
            .with_name(name or agent.name or "Marie Agent")
            .with_url(url)
            .with_version(version)
            .with_description(description or agent.description or "")
            .with_capabilities(
                streaming=streaming,
                push_notifications=push_notifications,
            )
            .from_agent(agent)
            .build()
        )

        # Create SDK executor
        self._sdk_executor = MarieA2AExecutor(
            agent=agent,
            agent_card=self._agent_card,
            streaming=streaming,
        )

        # Create task store
        self._task_store = task_store or InMemoryTaskStore()

        # Create request handler
        self._request_handler = DefaultRequestHandler(
            agent_executor=self._sdk_executor,
            task_store=self._task_store,
        )

        # Create Starlette app (lazy initialization)
        self._app: Optional[A2AStarletteApplication] = None

    @property
    def agent_card(self) -> AgentCard:
        """Get the agent card."""
        return self._agent_card

    @property
    def request_handler(self) -> DefaultRequestHandler:
        """Get the SDK request handler."""
        return self._request_handler

    @property
    def task_store(self) -> TaskStore:
        """Get the task store."""
        return self._task_store

    def get_app(self) -> A2AStarletteApplication:
        """Get the Starlette application for serving.

        Returns:
            Configured A2AStarletteApplication instance.
        """
        if self._app is None:
            self._app = A2AStarletteApplication(
                agent_card=self._agent_card,
                http_handler=self._request_handler,
            )
        return self._app

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a JSON-RPC request directly.

        Args:
            request: The JSON-RPC request dict.

        Returns:
            JSON-RPC response dict.
        """
        return await self._request_handler.handle(request)
