"""SDK-based A2A Executor for Marie agents.

This module provides MarieA2AExecutor that wraps Marie agents
using the official a2a-sdk package.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCard,
    Artifact,
    Message,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils import new_agent_text_message

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent

logger = logging.getLogger(__name__)


class MarieA2AExecutor(AgentExecutor):
    """SDK-based executor that wraps Marie agents for A2A protocol.

    Bridges Marie's synchronous agent API to the async A2A SDK interface
    using asyncio.to_thread() for non-blocking execution.

    Example:
        from marie.agent.a2a import AgentCardBuilder

        # Build agent card
        card = AgentCardBuilder().from_agent(my_agent).with_url(url).build()

        # Create executor
        executor = MarieA2AExecutor(my_agent, card)

        # Use with SDK's DefaultRequestHandler
        handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )
    """

    def __init__(
        self,
        agent: "BaseAgent",
        agent_card: AgentCard,
        streaming: bool = True,
    ):
        """Initialize the Marie A2A executor.

        Args:
            agent: The Marie agent to wrap.
            agent_card: The A2A agent card describing this agent.
            streaming: Whether to use streaming responses.
        """
        self._agent = agent
        self._agent_card = agent_card
        self._streaming = streaming
        self._cancel_events: dict[str, asyncio.Event] = {}

    @property
    def agent_card(self) -> AgentCard:
        """Get the agent card."""
        return self._agent_card

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent for a given request.

        Args:
            context: Request context with message and IDs.
            event_queue: Queue for sending response events.
        """
        task_id = context.task_id
        context_id = context.context_id

        # Set up cancellation event
        cancel_event = asyncio.Event()
        self._cancel_events[task_id] = cancel_event

        try:
            # Extract text from message
            input_text = self._extract_text(context.message)
            marie_messages = [{"role": "user", "content": input_text}]

            if self._streaming:
                await self._execute_streaming(
                    marie_messages, context_id, task_id, event_queue, cancel_event
                )
            else:
                await self._execute_nonstream(
                    marie_messages, context_id, task_id, event_queue, cancel_event
                )

        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")
            raise
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            # Send failure status
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(
                            text=f"Error: {e}",
                            context_id=context_id,
                            task_id=task_id,
                        ),
                    ),
                    task_id=task_id,
                    context_id=context_id,
                    final=True,
                )
            )
        finally:
            self._cancel_events.pop(task_id, None)

    async def _execute_nonstream(
        self,
        messages: list[dict[str, Any]],
        context_id: str,
        task_id: str,
        event_queue: EventQueue,
        cancel_event: asyncio.Event,
    ) -> None:
        """Execute agent without streaming."""
        # Run sync agent in thread pool
        responses = await asyncio.to_thread(self._agent.run_nonstream, messages)

        if cancel_event.is_set():
            return

        # Extract response text
        response_text = ""
        if responses:
            last_response = responses[-1]
            if hasattr(last_response, "content"):
                response_text = str(last_response.content)
            elif isinstance(last_response, dict):
                response_text = str(last_response.get("content", ""))

        # Send response
        await event_queue.enqueue_event(
            new_agent_text_message(
                text=response_text,
                context_id=context_id,
                task_id=task_id,
            )
        )

    async def _execute_streaming(
        self,
        messages: list[dict[str, Any]],
        context_id: str,
        task_id: str,
        event_queue: EventQueue,
        cancel_event: asyncio.Event,
    ) -> None:
        """Execute agent with streaming responses."""
        # Stream chunks from the agent
        async for chunk in self._stream_agent(messages):
            if cancel_event.is_set():
                return

            # Send chunk as artifact update
            artifact = Artifact(
                artifact_id=str(uuid.uuid4()),
                parts=[TextPart(text=chunk)],
                name="response",
            )
            await event_queue.enqueue_event(artifact)

    async def _stream_agent(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream responses from the sync agent iterator.

        Bridges the sync agent.run() iterator to an async iterator.
        """
        loop = asyncio.get_event_loop()

        # Run the iterator in a thread and yield chunks
        def run_iterator():
            return list(self._agent.run(messages))

        results = await loop.run_in_executor(None, run_iterator)

        for responses in results:
            if responses:
                last_response = responses[-1]
                if hasattr(last_response, "content"):
                    yield str(last_response.content)
                elif isinstance(last_response, dict):
                    yield str(last_response.get("content", ""))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel an executing task.

        Args:
            context: Request context with task ID.
            event_queue: Queue for sending cancellation event.
        """
        task_id = context.task_id

        # Signal cancellation
        if task_id in self._cancel_events:
            self._cancel_events[task_id].set()
            logger.info(f"Requested cancellation of task {task_id}")

        # Send cancelled status
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(state=TaskState.canceled),
                task_id=task_id,
                context_id=context.context_id,
                final=True,
            )
        )

    def _extract_text(self, message: Message) -> str:
        """Extract text content from an A2A message."""
        texts = []
        for part in message.parts:
            # SDK wraps parts in Part(root=...) RootModel
            actual_part = part.root if hasattr(part, "root") else part
            if isinstance(actual_part, TextPart):
                texts.append(actual_part.text)
            elif hasattr(actual_part, "text"):
                texts.append(actual_part.text)
        return "\n".join(texts)
