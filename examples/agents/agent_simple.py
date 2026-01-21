"""Simple Agent Example - Starter Template.

This is the simplest possible agent example. Use it as a starting point
for building your own agent tools.

Shows:
- Function-based tools using @register_tool (easiest approach)
- Class-based tools using AgentTool (more control)
- Minimal boilerplate to get started

Usage:
    python agent_simple.py --task "Add 5 and 3"
    python agent_simple.py --task "What time is it?"
    python agent_simple.py --tui
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from marie.agent import (
    AgentTool,
    AssistantAgent,
    MarieEngineLLMWrapper,
    OpenAICompatibleWrapper,
    ToolMetadata,
    ToolOutput,
    register_tool,
)

# =============================================================================
# FUNCTION-BASED TOOLS (using @register_tool)
#
# This is the simplest way to create tools:
# 1. Write a function with type hints
# 2. Add a docstring (used as tool description)
# 3. Return a JSON string
# 4. Decorate with @register_tool("name")
# =============================================================================


@register_tool("add")
def add(a: float, b: float) -> str:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        JSON with the sum.
    """
    result = a + b
    return json.dumps({"a": a, "b": b, "result": result})


@register_tool("multiply")
def multiply(a: float, b: float) -> str:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        JSON with the product.
    """
    result = a * b
    return json.dumps({"a": a, "b": b, "result": result})


@register_tool("get_time")
def get_time() -> str:
    """Get the current date and time.

    Returns:
        JSON with current time info.
    """
    now = datetime.now()
    return json.dumps(
        {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
        }
    )


@register_tool("list_files")
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files in a directory.

    Args:
        directory: Directory path (default: current)
        pattern: Glob pattern like "*.py" or "*.txt"

    Returns:
        JSON with file listing.
    """
    path = Path(directory)
    if not path.exists():
        return json.dumps({"error": f"Not found: {directory}"})

    files = [
        {"name": f.name, "size": f.stat().st_size}
        for f in path.glob(pattern)
        if f.is_file()
    ]

    return json.dumps(
        {
            "directory": str(path),
            "pattern": pattern,
            "files": files[:20],
            "count": len(files),
        }
    )


@register_tool("read_file")
def read_file(path: str) -> str:
    """Read a text file.

    Args:
        path: File path to read

    Returns:
        JSON with file contents.
    """
    file_path = Path(path)
    if not file_path.exists():
        return json.dumps({"error": f"File not found: {path}"})

    try:
        content = file_path.read_text()[:5000]  # Limit size
        return json.dumps({"path": path, "content": content})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# CLASS-BASED TOOL (using AgentTool)
#
# Use this when you need:
# - Configuration at initialization
# - State that persists between calls
# - Custom parameter schemas
# =============================================================================


class CounterTool(AgentTool):
    """A counter that maintains state between calls."""

    def __init__(self, start: int = 0):
        self.count = start

    @property
    def name(self) -> str:
        return "counter"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="A counter: increment, decrement, reset, or get value.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["increment", "decrement", "reset", "get"],
                        "description": "Action to perform",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Amount (default: 1)",
                    },
                },
                "required": ["action"],
            },
        )

    def call(self, **kwargs) -> ToolOutput:
        action = kwargs.get("action", "get")
        amount = kwargs.get("amount", 1)

        if action == "increment":
            self.count += amount
        elif action == "decrement":
            self.count -= amount
        elif action == "reset":
            self.count = 0

        result = {"action": action, "count": self.count}

        return ToolOutput(
            content=json.dumps(result),
            tool_name=self.name,
            raw_input=kwargs,
            raw_output=result,
            is_error=False,
        )


# =============================================================================
# AGENT SETUP
# =============================================================================


def create_agent(backend: str = "marie", model: Optional[str] = None) -> AssistantAgent:
    """Create an agent with our tools."""

    # Setup LLM
    if backend == "marie":
        llm = MarieEngineLLMWrapper(engine_name=model or "qwen2_5_vl_7b")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")
        llm = OpenAICompatibleWrapper(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            api_base="https://api.openai.com/v1",
        )

    # List tools - function names (strings) and class instances
    tools = [
        "add",
        "multiply",
        "get_time",
        "list_files",
        "read_file",
        CounterTool(start=0),
    ]

    return AssistantAgent(
        llm=llm,
        function_list=tools,
        name="Simple Agent",
        description="A simple agent with basic tools.",
        system_message="""You are a helpful assistant with these tools:
- add: Add two numbers
- multiply: Multiply two numbers
- get_time: Get current date/time
- list_files: List files in a directory
- read_file: Read a file
- counter: Increment/decrement a counter

Use the appropriate tool to help the user.""",
    )


# =============================================================================
# RUNNING THE AGENT
# =============================================================================


def run_task(task: str, backend: str = "marie"):
    """Run a single task."""
    print(f"Task: {task}")
    print("-" * 40)

    agent = create_agent(backend=backend)
    messages = [{"role": "user", "content": task}]

    for responses in agent.run(messages=messages):
        if responses:
            last = responses[-1]
            content = (
                last.get("content", "") if isinstance(last, dict) else last.content
            )
            if content:
                print(f"Agent: {content}")


def run_interactive():
    """Interactive mode."""
    print("Simple Agent - Interactive Mode")
    print("Type 'quit' to exit\n")

    agent = create_agent()
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})

        response_list = []
        for response_list in agent.run(messages=messages):
            if response_list:
                last = response_list[-1]
                content = (
                    last.get("content", "") if isinstance(last, dict) else last.content
                )
                if content:
                    print(f"Agent: {content}\n")

        if response_list:
            for r in response_list:
                messages.append(r if isinstance(r, dict) else r.model_dump())

    print("Goodbye!")


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Agent Example")
    parser.add_argument("--task", "-t", help="Task to run")
    parser.add_argument("--tui", action="store_true", help="Interactive mode")
    parser.add_argument("--backend", default="marie", choices=["marie", "openai"])

    args = parser.parse_args()

    if args.tui:
        run_interactive()
    elif args.task:
        run_task(args.task, backend=args.backend)
    else:
        print("Simple Agent - Starter Template")
        print("=" * 40)
        print()
        print("This is the simplest agent example.")
        print("Use it as a starting point for your own agents.")
        print()
        print("TWO WAYS TO CREATE TOOLS:")
        print()
        print("1. @register_tool decorator (easiest)")
        print("   - Just write a function that returns JSON")
        print("   - See: add(), multiply(), get_time()")
        print()
        print("2. AgentTool class (more control)")
        print("   - For stateful tools or custom schemas")
        print("   - See: CounterTool")
        print()
        print("USAGE:")
        print("  python agent_simple.py --task 'Add 5 and 3'")
        print("  python agent_simple.py --task 'What time is it?'")
        print("  python agent_simple.py --task 'List Python files'")
        print("  python agent_simple.py --tui")
