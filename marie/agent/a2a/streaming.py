"""SSE stream manager for A2A protocol.

This module provides utilities for managing Server-Sent Events (SSE)
streams for A2A protocol endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional, Union

from marie.agent.a2a.types import (
    Artifact,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

logger = logging.getLogger(__name__)


class SSEEvent:
    """Server-Sent Event wrapper."""

    def __init__(
        self,
        data: str,
        event: Optional[str] = None,
        id: Optional[str] = None,
        retry: Optional[int] = None,
    ):
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry

    def encode(self) -> str:
        """Encode as SSE format string."""
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")
        if self.event:
            lines.append(f"event: {self.event}")
        if self.retry:
            lines.append(f"retry: {self.retry}")

        # Data can be multi-line
        for line in self.data.split("\n"):
            lines.append(f"data: {line}")

        return "\n".join(lines) + "\n\n"


class StreamManager:
    """Manages SSE event streams for A2A responses.

    Handles buffering, serialization, and delivery of A2A events
    to connected clients via Server-Sent Events.
    """

    def __init__(
        self,
        task_id: str,
        context_id: str,
        max_queue_size: int = 1024,
    ):
        self.task_id = task_id
        self.context_id = context_id
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._closed = False
        self._event_counter = 0

    async def send_status_update(
        self,
        state: TaskState,
        message: Optional[Message] = None,
        final: bool = False,
    ) -> None:
        """Send a task status update event.

        Args:
            state: New task state.
            message: Optional status message.
            final: Whether this is the final event.
        """
        if self._closed:
            logger.warning("Attempted to send to closed stream")
            return

        event = TaskStatusUpdateEvent(
            status=TaskStatus(state=state, message=message),
            task_id=self.task_id,
            context_id=self.context_id,
            final=final,
        )

        await self._enqueue_event(event)

    async def send_artifact_update(
        self,
        artifact: Artifact,
        append: bool = False,
        last_chunk: bool = False,
    ) -> None:
        """Send a task artifact update event.

        Args:
            artifact: The artifact to send.
            append: Whether to append to existing artifact.
            last_chunk: Whether this is the last chunk.
        """
        if self._closed:
            logger.warning("Attempted to send to closed stream")
            return

        event = TaskArtifactUpdateEvent(
            artifact=artifact,
            task_id=self.task_id,
            context_id=self.context_id,
            append=append,
            last_chunk=last_chunk,
        )

        await self._enqueue_event(event)

    async def send_message(self, message: Message) -> None:
        """Send a message event.

        Args:
            message: The message to send.
        """
        if self._closed:
            logger.warning("Attempted to send to closed stream")
            return

        await self._enqueue_event(message)

    async def send_task(self, task: Task) -> None:
        """Send a complete task object.

        Args:
            task: The task to send.
        """
        if self._closed:
            logger.warning("Attempted to send to closed stream")
            return

        await self._enqueue_event(task)

    async def _enqueue_event(
        self,
        event: Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent, Message, Task],
    ) -> None:
        """Enqueue an event for streaming."""
        self._event_counter += 1
        await self._queue.put(event)

    async def close(self) -> None:
        """Close the stream."""
        self._closed = True
        # Send sentinel to signal end
        await self._queue.put(None)

    async def events(self) -> AsyncGenerator[SSEEvent, None]:
        """Yield SSE events from the stream.

        Yields:
            SSE events until the stream is closed.
        """
        event_id = 0
        while True:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=30.0,
                )

                if event is None:  # Sentinel
                    break

                event_id += 1
                data = self._serialize_event(event)

                yield SSEEvent(
                    data=data,
                    event="message",
                    id=str(event_id),
                )

            except asyncio.TimeoutError:
                # Send keep-alive
                yield SSEEvent(data="", event="ping")

    def _serialize_event(self, event: Any) -> str:
        """Serialize an event to JSON string."""
        if hasattr(event, "model_dump"):
            return json.dumps(event.model_dump(by_alias=True))
        return json.dumps(event)


class StreamingResponseBuilder:
    """Builds streaming responses for A2A endpoints.

    Provides a fluent interface for constructing streaming
    responses with proper SSE formatting.
    """

    def __init__(
        self,
        task_id: str,
        context_id: str,
    ):
        self.task_id = task_id
        self.context_id = context_id
        self._events: list[Any] = []

    def add_status_update(
        self,
        state: TaskState,
        message: Optional[Message] = None,
        final: bool = False,
    ) -> "StreamingResponseBuilder":
        """Add a status update event."""
        self._events.append(
            TaskStatusUpdateEvent(
                status=TaskStatus(state=state, message=message),
                task_id=self.task_id,
                context_id=self.context_id,
                final=final,
            )
        )
        return self

    def add_artifact(
        self,
        artifact: Artifact,
        append: bool = False,
        last_chunk: bool = False,
    ) -> "StreamingResponseBuilder":
        """Add an artifact update event."""
        self._events.append(
            TaskArtifactUpdateEvent(
                artifact=artifact,
                task_id=self.task_id,
                context_id=self.context_id,
                append=append,
                last_chunk=last_chunk,
            )
        )
        return self

    def add_message(self, message: Message) -> "StreamingResponseBuilder":
        """Add a message event."""
        self._events.append(message)
        return self

    def add_task(self, task: Task) -> "StreamingResponseBuilder":
        """Add a complete task."""
        self._events.append(task)
        return self

    async def stream(self) -> AsyncGenerator[SSEEvent, None]:
        """Stream all events."""
        for i, event in enumerate(self._events, 1):
            data = (
                json.dumps(event.model_dump(by_alias=True))
                if hasattr(event, "model_dump")
                else json.dumps(event)
            )
            yield SSEEvent(data=data, event="message", id=str(i))


async def stream_from_generator(
    task_id: str,
    context_id: str,
    generator: AsyncGenerator[Any, None],
) -> AsyncGenerator[SSEEvent, None]:
    """Stream events from an async generator.

    Wraps an async generator in SSE event formatting.

    Args:
        task_id: Task ID for context.
        context_id: Context ID for context.
        generator: Async generator yielding events.

    Yields:
        SSE formatted events.
    """
    event_id = 0
    async for event in generator:
        event_id += 1

        if hasattr(event, "model_dump"):
            data = json.dumps(event.model_dump(by_alias=True))
        elif isinstance(event, dict):
            data = json.dumps(event)
        else:
            data = str(event)

        yield SSEEvent(data=data, event="message", id=str(event_id))
