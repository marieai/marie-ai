"""Async Task Agent - Long-running task A2A test agent.

A2A agent that simulates a long-running task with multiple stages
and status updates. Tests the full task lifecycle.
"""

import asyncio

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps.jsonrpc import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.utils.message import get_message_text, new_agent_text_message

AGENT_NAME = "Async Task Agent"
AGENT_VERSION = "1.0.0"
DEFAULT_PORT = 9004

async_skill = AgentSkill(
    id="process",
    name="Async Processor",
    description="Simulates a long-running task with status updates",
    examples=["process data", "run task"],
)

async_agent_card = AgentCard(
    name=AGENT_NAME,
    description="Tests long-running task lifecycle",
    url=f"http://localhost:{DEFAULT_PORT}/",
    version=AGENT_VERSION,
    default_input_modes=["text", "text/plain"],
    default_output_modes=["text", "text/plain"],
    capabilities=AgentCapabilities(streaming=True, push_notifications=True),
    skills=[async_skill],
)


class AsyncTaskExecutor(AgentExecutor):
    """Agent executor that simulates long-running tasks."""

    def __init__(self, stage_delay: float = 2.0):
        self.stage_delay = stage_delay
        self._cancelled = False

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        self._cancelled = False
        input_text = get_message_text(context.message) if context.message else ""

        stages = ["Initializing", "Processing", "Validating", "Finalizing"]

        await event_queue.enqueue_event(
            new_agent_text_message(f"Starting task with input: {input_text}")
        )

        for i, stage in enumerate(stages, 1):
            if self._cancelled:
                await event_queue.enqueue_event(
                    new_agent_text_message(f"Task cancelled during {stage}")
                )
                return

            await event_queue.enqueue_event(
                new_agent_text_message(f"Status: {stage} ({i}/{len(stages)})")
            )
            await asyncio.sleep(self.stage_delay)

        await event_queue.enqueue_event(
            new_agent_text_message("Task completed successfully")
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        self._cancelled = True
        await event_queue.enqueue_event(
            new_agent_text_message("Task cancelled by user")
        )


def create_app() -> A2AStarletteApplication:
    """Create the A2A Starlette application."""
    handler = DefaultRequestHandler(
        agent_executor=AsyncTaskExecutor(),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(
        agent_card=async_agent_card,
        http_handler=handler,
    )


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app.build(), host="0.0.0.0", port=DEFAULT_PORT)
