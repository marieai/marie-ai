"""Agent utils."""

from marie.core.agent.types import TaskStep
from marie.core.base.llms.types import ChatMessage, MessageRole
from marie.core.memory import BaseMemory


def add_user_step_to_memory(
    step: TaskStep, memory: BaseMemory, verbose: bool = False
) -> None:
    """Add user step to memory."""
    user_message = ChatMessage(content=step.input, role=MessageRole.USER)
    memory.put(user_message)
    if verbose:
        print(f"Added user message to memory: {step.input}")
