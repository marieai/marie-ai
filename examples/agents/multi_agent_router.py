"""Multi-Agent Router Example.

Demonstrates how to create a router that delegates tasks to specialized agents.

The Router class acts as an orchestrator that:
1. Receives user requests
2. Decides whether to answer directly or delegate
3. Routes to the appropriate specialized agent
4. Returns the agent's response with attribution

Shows:
- Creating specialized agents with focused tool sets
- Building a router that orchestrates multiple agents
- Agent selection based on capabilities
- Response attribution (which agent handled the request)

Available Agents:
    - time_assistant: Time queries (uses get_time tool)
    - math_assistant: Calculations (uses calculator tool)
    - file_assistant: File listing (uses list_files tool)

Available Tools (by agent):
    time_assistant:
        - get_time: Get current time in any IANA timezone

    math_assistant:
        - calculator: Math expressions, percentages, scientific functions

    file_assistant:
        - list_files: List files with glob patterns

Usage Examples:
    # Time queries -> routed to time_assistant
    python multi_agent_router.py --task "What time is it?"
    python multi_agent_router.py --task "What time is it in Tokyo?"
    python multi_agent_router.py --task "What day of the week is it?"
    python multi_agent_router.py --task "Current time in America/New_York?"

    # Math queries -> routed to math_assistant
    python multi_agent_router.py --task "Calculate 25 * 4"
    python multi_agent_router.py --task "What is 15% of 200?"
    python multi_agent_router.py --task "Calculate 15% tip on $85"
    python multi_agent_router.py --task "What is sqrt(144)?"

    # File queries -> routed to file_assistant
    python multi_agent_router.py --task "List files in current directory"
    python multi_agent_router.py --task "Show me the Python files here"
    python multi_agent_router.py --task "What files are in this folder?"

    # General questions -> answered directly by router
    python multi_agent_router.py --task "Hello, how are you?"
    python multi_agent_router.py --task "What can you help me with?"

    # Interactive mode
    python multi_agent_router.py --tui

    # With OpenAI backend
    python multi_agent_router.py --backend openai --task "What is 25 * 4?"

    # Debug mode
    python multi_agent_router.py --task "What time is it?" --debug
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from marie.agent import (
    ReactAgent,
    register_tool,
)
from marie.agent.agents.router import Router

# Load environment variables from .env file
load_dotenv()


@register_tool("get_time")
def get_time(timezone_name: str = "UTC") -> str:
    """Get current date and time in a specified timezone.

    Args:
        timezone_name: IANA timezone name (e.g., "America/New_York", "UTC")

    Returns:
        JSON string with current time information.
    """
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(timezone_name)
        now = datetime.now(tz)

        return json.dumps(
            {
                "timezone": timezone_name,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day": now.strftime("%A"),
                "iso_format": now.isoformat(),
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "error": f"Invalid timezone: {timezone_name}",
                "message": str(e),
                "hint": "Use IANA timezone names like 'America/New_York', 'UTC', 'Asia/Tokyo'",
            }
        )


@register_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, abs, round, min, max

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        JSON string with calculation result.
    """
    import math
    import re

    # Handle percentage expressions like "15% of 85"
    percent_match = re.match(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", expression, re.IGNORECASE
    )
    if percent_match:
        percent = float(percent_match.group(1))
        value = float(percent_match.group(2))
        result = (percent / 100) * value
        return json.dumps(
            {
                "expression": expression,
                "result": result,
                "formatted": f"{percent}% of {value} = {result}",
            }
        )

    # Safe evaluation with limited operations
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Remove any dangerous characters
        sanitized = re.sub(r'[^0-9+\-*/().,%\s\w]', '', expression)
        result = eval(sanitized, {"__builtins__": {}}, allowed_names)

        return json.dumps(
            {
                "expression": expression,
                "result": result,
                "type": type(result).__name__,
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "error": "Calculation failed",
                "expression": expression,
                "message": str(e),
            }
        )


@register_tool("list_files")
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files in a directory.

    Args:
        directory: Directory path to list (default: current directory)
        pattern: Glob pattern to filter files (default: all files)

    Returns:
        JSON string with file listing.
    """
    try:
        path = Path(directory).resolve()
        if not path.exists():
            return json.dumps(
                {
                    "error": f"Directory not found: {directory}",
                }
            )

        if not path.is_dir():
            return json.dumps(
                {
                    "error": f"Not a directory: {directory}",
                }
            )

        files = []
        for f in path.glob(pattern):
            try:
                stat = f.stat()
                files.append(
                    {
                        "name": f.name,
                        "type": "directory" if f.is_dir() else "file",
                        "size": stat.st_size if f.is_file() else None,
                    }
                )
            except (OSError, PermissionError):
                files.append(
                    {
                        "name": f.name,
                        "type": "unknown",
                        "error": "permission denied",
                    }
                )

        # Sort: directories first, then files, both alphabetically
        files.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))

        return json.dumps(
            {
                "directory": str(path),
                "pattern": pattern,
                "count": len(files),
                "files": files[:50],  # Limit to 50 entries
                "truncated": len(files) > 50,
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "error": "Failed to list files",
                "directory": directory,
                "message": str(e),
            }
        )


# =============================================================================
# Agent Setup
# =============================================================================


def create_router(backend: str = "marie", model: Optional[str] = None) -> Router:
    """Create a multi-agent router with specialized sub-agents.

    Args:
        backend: LLM backend ("marie" or "openai")
        model: Model name to use

    Returns:
        Configured Router instance with sub-agents.
    """
    from utils import create_llm

    llm = create_llm(backend=backend, model=model)

    # Create specialized agents
    time_agent = ReactAgent(
        llm=llm,
        function_list=["get_time"],
        name="time_assistant",
        description="Gets current date, time, and day of week in any timezone.",
        max_iterations=3,
    )

    math_agent = ReactAgent(
        llm=llm,
        function_list=["calculator"],
        name="math_assistant",
        description="Performs mathematical calculations, percentages, and arithmetic.",
        max_iterations=3,
    )

    file_agent = ReactAgent(
        llm=llm,
        function_list=["list_files"],
        name="file_assistant",
        description="Lists and explores files and directories on the filesystem.",
        max_iterations=3,
    )

    # Create router with all agents
    return Router(
        agents=[time_agent, math_agent, file_agent],
        llm=llm,
        name="Router",
        description="Routes requests to specialized assistants.",
    )


from utils import print_debug_response

# =============================================================================
# Running Modes
# =============================================================================


def run_task(
    task: str, backend: str = "marie", model: Optional[str] = None, debug: bool = False
):
    """Run a single task through the router.

    Args:
        task: The task/query to process
        backend: LLM backend to use
        model: Model name
        debug: Show raw LLM responses
    """
    print(f"Task: {task}")
    if debug:
        print(f"Backend: {backend}")
        print("Debug: ON")
    print("-" * 60)

    router = create_router(backend=backend, model=model)
    messages = [{"role": "user", "content": task}]

    iteration = 0
    for responses in router.run(messages=messages):
        if responses:
            iteration += 1
            if debug:
                for resp in responses:
                    print_debug_response(resp, iteration)

            last = responses[-1]
            content = (
                last.get("content", "") if isinstance(last, dict) else last.content
            )
            name = (
                last.get("name")
                if isinstance(last, dict)
                else getattr(last, 'name', None)
            )
            if content:
                prefix = f"[{name}] " if name else ""
                print(f"\n{prefix}{content}")


def run_interactive(
    backend: str = "marie", model: Optional[str] = None, debug: bool = False
):
    """Run in interactive mode.

    Args:
        backend: LLM backend to use
        model: Model name
        debug: Show raw LLM responses
    """
    print("=" * 60)
    print("Multi-Agent Router - Interactive Mode")
    print("=" * 60)
    if debug:
        print("Debug: ON")
    print()
    print("Available agents:")
    print("  - time_assistant: Gets date/time in any timezone")
    print("  - math_assistant: Performs calculations")
    print("  - file_assistant: Lists files and directories")
    print()
    print("Commands: 'quit', 'exit', 'clear'")
    print()

    router = create_router(backend=backend, model=model)
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            messages = []
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        iteration = 0
        response_list = []
        for response_list in router.run(messages=messages):
            if response_list:
                iteration += 1
                if debug:
                    for resp in response_list:
                        print_debug_response(resp, iteration)

                last = response_list[-1]
                content = (
                    last.get("content", "") if isinstance(last, dict) else last.content
                )
                name = (
                    last.get("name")
                    if isinstance(last, dict)
                    else getattr(last, 'name', None)
                )
                if content:
                    prefix = f"[{name}] " if name else ""
                    print(f"Router: {prefix}{content}")
                    print()

        if response_list:
            for r in response_list:
                messages.append(r if isinstance(r, dict) else r.model_dump())


def show_help():
    """Display usage information."""
    print("Multi-Agent Router - Task Delegation")
    print("=" * 60)
    print()
    print("AVAILABLE AGENTS:")
    print("  time_assistant  - Time queries (get_time tool)")
    print("  math_assistant  - Calculations (calculator tool)")
    print("  file_assistant  - File listing (list_files tool)")
    print()
    print("HOW IT WORKS:")
    print("  The router analyzes requests and delegates to specialists:")
    print("  - Time questions -> time_assistant")
    print("  - Math questions -> math_assistant")
    print("  - File questions -> file_assistant")
    print("  - General questions -> answered directly")
    print()
    print("USAGE EXAMPLES:")
    print()
    print("  # Time queries")
    print("  python multi_agent_router.py --task 'What time is it?'")
    print("  python multi_agent_router.py --task 'Time in Tokyo?'")
    print()
    print("  # Math queries")
    print("  python multi_agent_router.py --task 'Calculate 25 * 4'")
    print("  python multi_agent_router.py --task 'What is 15% of 200?'")
    print()
    print("  # File queries")
    print("  python multi_agent_router.py --task 'List Python files'")
    print("  python multi_agent_router.py --task 'Show files here'")
    print()
    print("  # Interactive / OpenAI / Debug")
    print("  python multi_agent_router.py --tui")
    print("  python multi_agent_router.py --backend openai --task '...'")
    print("  python multi_agent_router.py --task '...' --debug")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Router Example")
    parser.add_argument("--task", "-t", type=str, help="Single task to run")
    parser.add_argument("--tui", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--backend",
        default="marie",
        choices=["marie", "openai"],
        help="LLM backend to use",
    )
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Show raw LLM responses"
    )

    args = parser.parse_args()

    if args.tui:
        run_interactive(backend=args.backend, model=args.model, debug=args.debug)
    elif args.task:
        run_task(args.task, backend=args.backend, model=args.model, debug=args.debug)
    else:
        show_help()
