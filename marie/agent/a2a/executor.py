"""A2A Executor for exposing Marie agents via A2A protocol.

This module provides the A2AExecutor class that wraps Marie agents
and exposes them through the A2A protocol interface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Union
from uuid import uuid4

from marie.agent.a2a.agent_card import AgentCardBuilder
from marie.agent.a2a.constants import TERMINAL_STATES, A2AMethod, TaskState
from marie.agent.a2a.errors import (
    A2AServerError,
    InvalidParamsError,
    TaskNotCancelableError,
    TaskNotFoundError,
)
from marie.agent.a2a.jsonrpc import JSONRPCDispatcher
from marie.agent.a2a.streaming import SSEEvent, StreamManager
from marie.agent.a2a.task import TaskManager, create_text_message
from marie.agent.a2a.types import (
    AgentCard,
    Artifact,
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskIdParams,
    TaskQueryParams,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

if TYPE_CHECKING:
    from marie.agent.base import BaseAgent

logger = logging.getLogger(__name__)


class A2AExecutor:
    """Executes Marie agents via A2A protocol.

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

        # Process a message
        result = await executor.send_message(message_params)
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

        # Task management
        self._task_manager = TaskManager()
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._stream_managers: dict[str, StreamManager] = {}
        self._lock = asyncio.Lock()

        # JSON-RPC dispatcher
        self._dispatcher = JSONRPCDispatcher()
        self._register_handlers()

    @property
    def agent_card(self) -> AgentCard:
        """Get the agent card."""
        return self._agent_card

    def _register_handlers(self) -> None:
        """Register JSON-RPC handlers."""
        self._dispatcher.register(
            A2AMethod.SEND_MESSAGE,
            self._handle_send_message,
        )
        self._dispatcher.register(
            A2AMethod.SEND_MESSAGE_STREAM,
            self._handle_send_message_stream,
            streaming=True,
        )
        self._dispatcher.register(
            A2AMethod.GET_TASK,
            self._handle_get_task,
        )
        self._dispatcher.register(
            A2AMethod.CANCEL_TASK,
            self._handle_cancel_task,
        )

    async def dispatch(
        self,
        request: Union[str, bytes, dict[str, Any]],
    ) -> Any:
        """Dispatch a JSON-RPC request.

        Args:
            request: The JSON-RPC request.

        Returns:
            JSON-RPC response.
        """
        return await self._dispatcher.dispatch(request)

    async def dispatch_streaming(
        self,
        request: Union[str, bytes, dict[str, Any]],
    ) -> AsyncGenerator[SSEEvent, None]:
        """Dispatch a streaming JSON-RPC request.

        Args:
            request: The JSON-RPC request.

        Yields:
            SSE events.
        """
        async for event in self._dispatcher.dispatch_streaming(request):
            if hasattr(event, "encode"):
                yield event
            else:
                # Wrap non-SSE events
                import json

                data = (
                    json.dumps(event.model_dump(by_alias=True))
                    if hasattr(event, "model_dump")
                    else json.dumps(event)
                )
                yield SSEEvent(data=data, event="message")

    async def send_message(
        self,
        params: MessageSendParams,
    ) -> Union[Message, Task]:
        """Send a message to the agent.

        Args:
            params: Message send parameters.

        Returns:
            Response Message or Task.
        """
        return await self._handle_send_message(params.model_dump())

    async def send_message_stream(
        self,
        params: MessageSendParams,
    ) -> AsyncGenerator[Any, None]:
        """Send a message and stream the response.

        Args:
            params: Message send parameters.

        Yields:
            Response events.
        """
        async for event in self._handle_send_message_stream(params.model_dump()):
            yield event

    async def _handle_send_message(
        self,
        params: dict[str, Any],
    ) -> Union[Message, Task]:
        """Handle message/send request."""
        try:
            message_params = MessageSendParams(**params)
        except Exception as e:
            raise InvalidParamsError(f"Invalid message params: {e}")

        message = message_params.message

        # Get or create task
        task_id = message.task_id
        context_id = message.context_id or str(uuid4())

        if task_id:
            task = self._task_manager.get_task(task_id)
            if not task:
                raise TaskNotFoundError(task_id)
            if task.status.state in TERMINAL_STATES:
                raise InvalidParamsError(
                    f"Task {task_id} is in terminal state: {task.status.state}"
                )
            self._task_manager.add_message(task_id, message)
        else:
            task = self._task_manager.create_task(
                message=message,
                context_id=context_id,
            )
            task_id = task.id

        # Update to working state
        self._task_manager.update_status(task_id, TaskState.WORKING)

        try:
            # Execute agent
            result = await self._execute_agent(message)

            # Complete task with result
            self._task_manager.complete_task(task_id, result)

            return self._task_manager.get_task(task_id)

        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            self._task_manager.fail_task(task_id, str(e))
            return self._task_manager.get_task(task_id)

    async def _handle_send_message_stream(
        self,
        params: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        """Handle message/stream request."""
        try:
            message_params = MessageSendParams(**params)
        except Exception as e:
            raise InvalidParamsError(f"Invalid message params: {e}")

        message = message_params.message

        # Get or create task
        task_id = message.task_id or str(uuid4())
        context_id = message.context_id or str(uuid4())

        if message.task_id:
            task = self._task_manager.get_task(task_id)
            if not task:
                raise TaskNotFoundError(task_id)
            self._task_manager.add_message(task_id, message)
        else:
            task = self._task_manager.create_task(
                message=message,
                context_id=context_id,
                task_id=task_id,
            )

        # Create stream manager
        stream_manager = StreamManager(task_id, context_id)
        self._stream_managers[task_id] = stream_manager

        # Send working status
        yield TaskStatusUpdateEvent(
            status=TaskStatus(state=TaskState.WORKING),
            task_id=task_id,
            context_id=context_id,
            final=False,
        )

        try:
            # Execute agent with streaming
            async for chunk in self._execute_agent_streaming(message):
                # Send chunk as artifact update
                artifact = Artifact(
                    artifact_id=str(uuid4()),
                    parts=[TextPart(text=chunk)],
                )
                yield artifact

            # Send completion
            yield TaskStatusUpdateEvent(
                status=TaskStatus(state=TaskState.COMPLETED),
                task_id=task_id,
                context_id=context_id,
                final=True,
            )

            self._task_manager.complete_task(task_id)

        except Exception as e:
            logger.exception(f"Streaming execution failed: {e}")
            yield TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=TaskState.FAILED,
                    message=create_text_message(str(e)),
                ),
                task_id=task_id,
                context_id=context_id,
                final=True,
            )
            self._task_manager.fail_task(task_id, str(e))

        finally:
            await stream_manager.close()
            del self._stream_managers[task_id]

    async def _handle_get_task(
        self,
        params: dict[str, Any],
    ) -> Task:
        """Handle tasks/get request."""
        try:
            query_params = TaskQueryParams(**params)
        except Exception as e:
            raise InvalidParamsError(f"Invalid task params: {e}")

        task = self._task_manager.get_task(query_params.id)
        if not task:
            raise TaskNotFoundError(query_params.id)

        # Apply history length limit if specified
        if query_params.history_length is not None and task.history:
            task.history = task.history[-query_params.history_length :]

        return task

    async def _handle_cancel_task(
        self,
        params: dict[str, Any],
    ) -> Task:
        """Handle tasks/cancel request."""
        try:
            id_params = TaskIdParams(**params)
        except Exception as e:
            raise InvalidParamsError(f"Invalid params: {e}")

        task = self._task_manager.get_task(id_params.id)
        if not task:
            raise TaskNotFoundError(id_params.id)

        if task.status.state in TERMINAL_STATES:
            raise TaskNotCancelableError(
                id_params.id,
                task.status.state.value,
            )

        # Cancel running task if exists
        async with self._lock:
            if id_params.id in self._running_tasks:
                self._running_tasks[id_params.id].cancel()
                del self._running_tasks[id_params.id]

        self._task_manager.cancel_task(id_params.id)
        return self._task_manager.get_task(id_params.id)

    async def _execute_agent(self, message: Message) -> str:
        """Execute the agent with a message.

        Args:
            message: Input message.

        Returns:
            Agent response text.
        """
        # Extract text from message parts
        input_text = self._extract_text(message)

        # Convert to Marie message format
        marie_messages = [{"role": "user", "content": input_text}]

        # Run agent
        responses = self.agent.run_nonstream(marie_messages)

        # Extract response text
        if responses:
            last_response = responses[-1]
            if hasattr(last_response, "content"):
                return str(last_response.content)
            elif isinstance(last_response, dict):
                return str(last_response.get("content", ""))

        return ""

    async def _execute_agent_streaming(
        self,
        message: Message,
    ) -> AsyncGenerator[str, None]:
        """Execute the agent with streaming.

        Args:
            message: Input message.

        Yields:
            Response text chunks.
        """
        # Extract text from message parts
        input_text = self._extract_text(message)

        # Convert to Marie message format
        marie_messages = [{"role": "user", "content": input_text}]

        # Run agent with streaming
        for responses in self.agent.run(marie_messages):
            if responses:
                last_response = responses[-1]
                if hasattr(last_response, "content"):
                    yield str(last_response.content)
                elif isinstance(last_response, dict):
                    yield str(last_response.get("content", ""))

    def _extract_text(self, message: Message) -> str:
        """Extract text content from a message."""
        texts = []
        for part in message.parts:
            if isinstance(part, TextPart):
                texts.append(part.text)
            elif hasattr(part, "text"):
                texts.append(part.text)
        return "\n".join(texts)
