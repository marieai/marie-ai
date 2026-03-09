"""Calculator Agent - Multi-skill A2A test agent.

A2A agent that performs basic arithmetic operations. Demonstrates
multi-skill agents and JSON input handling.
"""

import json

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

AGENT_NAME = "Calculator Agent"
AGENT_VERSION = "1.0.0"
DEFAULT_PORT = 9002

calc_skills = [
    AgentSkill(
        id="add",
        name="Add Numbers",
        description="Add two numbers together",
        examples=["add 5 3", '{"op": "add", "a": 5, "b": 3}'],
    ),
    AgentSkill(
        id="multiply",
        name="Multiply Numbers",
        description="Multiply two numbers",
        examples=["multiply 5 3", '{"op": "multiply", "a": 5, "b": 3}'],
    ),
    AgentSkill(
        id="subtract",
        name="Subtract Numbers",
        description="Subtract two numbers",
        examples=["subtract 5 3", '{"op": "subtract", "a": 5, "b": 3}'],
    ),
    AgentSkill(
        id="divide",
        name="Divide Numbers",
        description="Divide two numbers",
        examples=["divide 10 2", '{"op": "divide", "a": 10, "b": 2}'],
    ),
]

calc_agent_card = AgentCard(
    name=AGENT_NAME,
    description="Performs basic arithmetic operations",
    url=f"http://localhost:{DEFAULT_PORT}/",
    version=AGENT_VERSION,
    default_input_modes=["text", "text/plain", "application/json"],
    default_output_modes=["text", "text/plain", "application/json"],
    capabilities=AgentCapabilities(streaming=False, push_notifications=False),
    skills=calc_skills,
)


class CalculatorExecutor(AgentExecutor):
    """Agent executor that performs arithmetic operations."""

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        text = get_message_text(context.message) if context.message else ""

        try:
            op, a, b = self._parse_input(text)
            result = self._calculate(op, a, b)
            await event_queue.enqueue_event(new_agent_text_message(str(result)))
        except ValueError as e:
            await event_queue.enqueue_event(new_agent_text_message(f"Error: {e}"))
        except ZeroDivisionError:
            await event_queue.enqueue_event(new_agent_text_message("Error: Division by zero"))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        await event_queue.enqueue_event(new_agent_text_message("Calculation cancelled"))

    def _parse_input(self, text: str) -> tuple[str, float, float]:
        """Parse input from text or JSON format."""
        text = text.strip()

        if text.startswith("{"):
            data = json.loads(text)
            return data["op"], float(data["a"]), float(data["b"])

        parts = text.split()
        if len(parts) != 3:
            raise ValueError(
                "Expected format: 'operation num1 num2' or JSON object"
            )
        return parts[0].lower(), float(parts[1]), float(parts[2])

    def _calculate(self, op: str, a: float, b: float) -> float:
        """Perform the calculation."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y,
        }

        if op not in operations:
            raise ValueError(f"Unknown operation: {op}")

        return operations[op](a, b)


def create_app() -> A2AStarletteApplication:
    """Create the A2A Starlette application."""
    handler = DefaultRequestHandler(
        agent_executor=CalculatorExecutor(),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(
        agent_card=calc_agent_card,
        http_handler=handler,
    )


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app.build(), host="0.0.0.0", port=DEFAULT_PORT)
