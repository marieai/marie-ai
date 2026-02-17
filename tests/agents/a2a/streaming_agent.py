"""Streaming Agent - SSE streaming A2A test agent.

A2A agent that demonstrates streaming responses via Server-Sent Events.
Counts from 1 to N with incremental updates.
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

AGENT_NAME = "Streaming Test Agent"
AGENT_VERSION = "1.0.0"
DEFAULT_PORT = 9003

streaming_skill = AgentSkill(
    id="count",
    name="Streaming Counter",
    description="Counts from 1 to N with streaming updates",
    examples=["5", "10", "count to 3"],
)

streaming_agent_card = AgentCard(
    name=AGENT_NAME,
    description="Streams incremental results via SSE",
    url=f"http://localhost:{DEFAULT_PORT}/",
    version=AGENT_VERSION,
    default_input_modes=["text", "text/plain"],
    default_output_modes=["text", "text/plain"],
    capabilities=AgentCapabilities(streaming=True, push_notifications=False),
    skills=[streaming_skill],
)


class StreamingExecutor(AgentExecutor):
    """Agent executor that streams counting updates."""

    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self._cancelled = False

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        self._cancelled = False
        text = get_message_text(context.message) if context.message else "5"

        # Parse the count target
        n = self._parse_count(text)

        for i in range(1, n + 1):
            if self._cancelled:
                await event_queue.enqueue_event(
                    new_agent_text_message(f"Counting cancelled at {i}")
                )
                return

            await asyncio.sleep(self.delay)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Count: {i}")
            )

        await event_queue.enqueue_event(
            new_agent_text_message(f"Completed counting to {n}")
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        self._cancelled = True
        await event_queue.enqueue_event(
            new_agent_text_message("Streaming cancelled")
        )

    def _parse_count(self, text: str) -> int:
        """Parse count target from text."""
        text = text.strip().lower()

        # Handle "count to N" format
        if text.startswith("count to "):
            text = text[9:]

        try:
            n = int(text)
            return max(1, min(n, 100))  # Limit between 1 and 100
        except ValueError:
            return 5  # Default


def create_app() -> A2AStarletteApplication:
    """Create the A2A Starlette application."""
    handler = DefaultRequestHandler(
        agent_executor=StreamingExecutor(),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(
        agent_card=streaming_agent_card,
        http_handler=handler,
    )


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app.build(), host="0.0.0.0", port=DEFAULT_PORT)
