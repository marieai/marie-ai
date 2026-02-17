"""Echo Agent - Minimal A2A test agent.

A simple A2A agent that echoes back the input message. Used for basic
protocol compliance testing.
"""

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

AGENT_NAME = "Echo Test Agent"
AGENT_VERSION = "1.0.0"
DEFAULT_PORT = 9001

echo_skill = AgentSkill(
    id="echo",
    name="Echo Message",
    description="Echoes back the input message",
    tags=["test", "echo"],
    examples=["echo hello", "repeat this"],
)

echo_agent_card = AgentCard(
    name=AGENT_NAME,
    description="Simple A2A test agent that echoes messages",
    url=f"http://localhost:{DEFAULT_PORT}/",
    version=AGENT_VERSION,
    default_input_modes=["text", "text/plain"],
    default_output_modes=["text", "text/plain"],
    capabilities=AgentCapabilities(streaming=False, push_notifications=False),
    skills=[echo_skill],
)


class EchoAgentExecutor(AgentExecutor):
    """Agent executor that echoes back input messages."""

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        input_text = get_message_text(context.message) if context.message else "No input"
        result = f"Echo: {input_text}"
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        await event_queue.enqueue_event(
            new_agent_text_message("Echo cancelled")
        )


def create_app() -> A2AStarletteApplication:
    """Create the A2A Starlette application."""
    handler = DefaultRequestHandler(
        agent_executor=EchoAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(
        agent_card=echo_agent_card,
        http_handler=handler,
    )


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app.build(), host="0.0.0.0", port=DEFAULT_PORT)
