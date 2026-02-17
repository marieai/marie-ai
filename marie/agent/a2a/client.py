"""A2A Client for calling external agents.

This module provides the A2AClient class for making requests to
external A2A-compatible agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional, Union
from uuid import uuid4

import httpx

from marie.agent.a2a.constants import (
    AGENT_CARD_PATH,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_STREAM_TIMEOUT,
    JSONRPC_VERSION,
    A2AMethod,
)
from marie.agent.a2a.errors import (
    A2AClientError,
    AgentDiscoveryError,
    TaskNotFoundError,
)
from marie.agent.a2a.types import (
    AgentCard,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskIdParams,
    TaskQueryParams,
    TextPart,
)

logger = logging.getLogger(__name__)


class A2AClient:
    """Client for communicating with external A2A agents.

    Provides methods for sending messages, managing tasks, and
    streaming responses from A2A-compatible agents.

    Example:
        # Discover and connect to an agent
        client = await A2AClient.from_url("http://agent.example.com")

        # Send a message
        result = await client.send_message("Hello, agent!")

        # Stream a response
        async for event in client.stream_message("Process this"):
            print(event)
    """

    def __init__(
        self,
        agent_card: AgentCard,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        stream_timeout: float = DEFAULT_STREAM_TIMEOUT,
    ):
        """Initialize the client.

        Args:
            agent_card: The agent card describing the remote agent.
            http_client: Optional pre-configured HTTP client.
            timeout: Request timeout in seconds.
            stream_timeout: Streaming timeout in seconds.
        """
        self.agent_card = agent_card
        self._base_url = agent_card.url.rstrip("/")
        self._timeout = timeout
        self._stream_timeout = stream_timeout
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    @classmethod
    async def from_url(
        cls,
        url: str,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> "A2AClient":
        """Create a client by discovering an agent at the given URL.

        Args:
            url: Base URL of the A2A agent.
            http_client: Optional pre-configured HTTP client.
            timeout: Request timeout in seconds.

        Returns:
            Configured A2AClient instance.

        Raises:
            AgentDiscoveryError: If agent discovery fails.
        """
        client = http_client or httpx.AsyncClient(timeout=timeout)
        owns_client = http_client is None

        try:
            base_url = url.rstrip("/")
            card_url = f"{base_url}{AGENT_CARD_PATH}"

            response = await client.get(card_url)
            response.raise_for_status()

            card_data = response.json()
            agent_card = AgentCard(**card_data)

            return cls(
                agent_card=agent_card,
                http_client=client if not owns_client else None,
                timeout=timeout,
            )

        except httpx.HTTPError as e:
            if owns_client:
                await client.aclose()
            raise AgentDiscoveryError(url, f"HTTP error: {e}")
        except Exception as e:
            if owns_client:
                await client.aclose()
            raise AgentDiscoveryError(url, str(e))

    async def close(self) -> None:
        """Close the HTTP client if owned by this instance."""
        if self._owns_client and self._client:
            await self._client.aclose()

    async def __aenter__(self) -> "A2AClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def send_message(
        self,
        content: Union[str, Message],
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        blocking: bool = True,
    ) -> Union[Message, Task]:
        """Send a message to the agent.

        Args:
            content: Text content or Message object.
            context_id: Optional context ID for conversation continuity.
            task_id: Optional task ID for continuing an existing task.
            blocking: Whether to wait for completion.

        Returns:
            Response Message or Task from the agent.
        """
        if isinstance(content, str):
            message = Message(
                role=Role.USER,
                parts=[TextPart(text=content)],
                message_id=str(uuid4()),
                context_id=context_id,
                task_id=task_id,
            )
        else:
            message = content

        params = MessageSendParams(
            message=message,
            configuration=MessageSendConfiguration(blocking=blocking),
        )

        request_body = {
            "jsonrpc": JSONRPC_VERSION,
            "id": str(uuid4()),
            "method": A2AMethod.SEND_MESSAGE.value,
            "params": params.model_dump(by_alias=True),
        }

        response = await self._client.post(
            self._base_url,
            json=request_body,
            timeout=self._timeout,
        )
        response.raise_for_status()

        result = response.json()

        if "error" in result:
            raise A2AClientError(
                result["error"].get("message", "Unknown error"),
                agent_url=self._base_url,
            )

        return self._parse_result(result.get("result", {}))

    async def stream_message(
        self,
        content: Union[str, Message],
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        """Send a message and stream the response.

        Args:
            content: Text content or Message object.
            context_id: Optional context ID.
            task_id: Optional task ID.

        Yields:
            Response events from the agent.
        """
        if isinstance(content, str):
            message = Message(
                role=Role.USER,
                parts=[TextPart(text=content)],
                message_id=str(uuid4()),
                context_id=context_id,
                task_id=task_id,
            )
        else:
            message = content

        params = MessageSendParams(message=message)

        request_body = {
            "jsonrpc": JSONRPC_VERSION,
            "id": str(uuid4()),
            "method": A2AMethod.SEND_MESSAGE_STREAM.value,
            "params": params.model_dump(by_alias=True),
        }

        async with self._client.stream(
            "POST",
            self._base_url,
            json=request_body,
            timeout=self._stream_timeout,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or line.startswith(":"):
                    continue

                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data:
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in SSE: {data}")

    async def get_task(
        self,
        task_id: str,
        history_length: Optional[int] = None,
    ) -> Task:
        """Get a task by ID.

        Args:
            task_id: The task ID.
            history_length: Optional limit on history length.

        Returns:
            The Task object.

        Raises:
            TaskNotFoundError: If task not found.
        """
        params = TaskQueryParams(id=task_id, history_length=history_length)

        request_body = {
            "jsonrpc": JSONRPC_VERSION,
            "id": str(uuid4()),
            "method": A2AMethod.GET_TASK.value,
            "params": params.model_dump(by_alias=True),
        }

        response = await self._client.post(
            self._base_url,
            json=request_body,
            timeout=self._timeout,
        )
        response.raise_for_status()

        result = response.json()

        if "error" in result:
            error = result["error"]
            if error.get("code") == -32001:
                raise TaskNotFoundError(task_id)
            raise A2AClientError(
                error.get("message", "Unknown error"),
                agent_url=self._base_url,
            )

        return Task(**result.get("result", {}))

    async def cancel_task(self, task_id: str) -> Task:
        """Cancel a task.

        Args:
            task_id: The task ID.

        Returns:
            The canceled Task object.
        """
        params = TaskIdParams(id=task_id)

        request_body = {
            "jsonrpc": JSONRPC_VERSION,
            "id": str(uuid4()),
            "method": A2AMethod.CANCEL_TASK.value,
            "params": params.model_dump(by_alias=True),
        }

        response = await self._client.post(
            self._base_url,
            json=request_body,
            timeout=self._timeout,
        )
        response.raise_for_status()

        result = response.json()

        if "error" in result:
            raise A2AClientError(
                result["error"].get("message", "Unknown error"),
                agent_url=self._base_url,
            )

        return Task(**result.get("result", {}))

    def _parse_result(self, result: dict[str, Any]) -> Union[Message, Task]:
        """Parse a result as Message or Task."""
        if result.get("kind") == "task" or "status" in result:
            return Task(**result)
        return Message(**result)

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self.agent_card.name

    @property
    def description(self) -> Optional[str]:
        """Get the agent description."""
        return self.agent_card.description

    @property
    def skills(self) -> list:
        """Get the agent skills."""
        return self.agent_card.skills or []

    @property
    def supports_streaming(self) -> bool:
        """Check if agent supports streaming."""
        if self.agent_card.capabilities:
            return self.agent_card.capabilities.streaming
        return False

    @property
    def supports_push_notifications(self) -> bool:
        """Check if agent supports push notifications."""
        if self.agent_card.capabilities:
            return self.agent_card.capabilities.push_notifications
        return False
