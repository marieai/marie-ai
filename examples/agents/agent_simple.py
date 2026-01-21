"""Simple Agent Example - Tutorial / Starter Template.

PURPOSE: Learn how to build agents. This is a minimal, well-commented example
designed for learning. Copy this file as a starting point for your own agents.

For a production-ready example with full features, see: agent_assistant.py

What You'll Learn:
- Function-based tools using @register_tool (easiest approach)
- Class-based tools using AgentTool (for stateful tools)
- How to create and run an agent
- Multimodal messages (text + images)
- Minimal boilerplate to get started

Available Tools:
    - add: Add two numbers
    - multiply: Multiply two numbers
    - get_time: Get current date and time
    - list_files: List files in a directory with glob patterns
    - read_file: Read contents of a text file
    - counter: Stateful counter (increment, decrement, reset, get)
    - image_info: Get image metadata (size, format, etc.)

Usage Examples:
    # Math operations
    python agent_simple.py --task "Add 5 and 3"
    python agent_simple.py --task "Multiply 7 by 8"
    python agent_simple.py --task "What is 12 plus 45?"

    # Time queries
    python agent_simple.py --task "What time is it?"
    python agent_simple.py --task "What day is today?"

    # File operations
    python agent_simple.py --task "List Python files in current directory"
    python agent_simple.py --task "List all .py files"
    python agent_simple.py --task "Read the file agent_simple.py"

    # Counter (stateful)
    python agent_simple.py --task "Increment the counter 3 times"
    python agent_simple.py --task "What is the counter value?"

    # Image analysis (multimodal)
    python agent_simple.py --image photo.jpg --task "Describe this image"
    python agent_simple.py --image document.png --task "What text is in this image?"
    python agent_simple.py --image chart.png --task "Get info about this image"

    # Interactive mode
    python agent_simple.py --tui

    # With OpenAI backend
    python agent_simple.py --backend openai --task "Add 5 and 3"

    # Debug mode - show raw LLM responses
    python agent_simple.py --task "Add 5 and 3" --debug
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from marie.agent import (
    AgentTool,
    ReactAgent,
    ToolMetadata,
    ToolOutput,
    register_tool,
)
from utils import create_llm, print_debug_response

# Load environment variables from .env file
load_dotenv()


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


@register_tool("image_info")
def image_info(image_path: str) -> str:
    """Get metadata about an image file.

    Args:
        image_path: Path to the image file

    Returns:
        JSON with image information (size, format, dimensions).
    """
    try:
        from PIL import Image

        path = Path(image_path)
        if not path.exists():
            return json.dumps({"error": f"File not found: {image_path}"})

        with Image.open(path) as img:
            info = {
                "file_name": path.name,
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "file_size_kb": round(path.stat().st_size / 1024, 2),
            }
        return json.dumps(info)

    except ImportError:
        return json.dumps(
            {"error": "PIL/Pillow not installed. Run: pip install Pillow"}
        )
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


def create_agent(backend: str = "marie", model: Optional[str] = None) -> ReactAgent:
    """Create an agent with our tools."""
    llm = create_llm(backend=backend, model=model)

    # List tools - function names (strings) and class instances
    tools = [
        "add",
        "multiply",
        "get_time",
        "list_files",
        "read_file",
        "image_info",
        CounterTool(start=0),
    ]

    return ReactAgent(
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
- image_info: Get image metadata (size, format, dimensions)
- counter: Increment/decrement a counter

You can see and analyze images directly. When an image is provided,
describe what you see. Use image_info to get technical details.

Use the appropriate tool to help the user.""",
    )


def run_task(task: str, backend: str = "marie", debug: bool = False):
    """Run a single task."""
    print(f"Task: {task}")
    if debug:
        print(f"Backend: {backend}")
        print(f"Debug: ON")
    print("-" * 40)

    agent = create_agent(backend=backend)
    messages = [{"role": "user", "content": task}]

    iteration = 0
    for responses in agent.run(messages=messages):
        if responses:
            iteration += 1
            for resp in responses:
                if debug:
                    print_debug_response(resp, iteration)

            last = responses[-1]
            content = (
                last.get("content", "") if isinstance(last, dict) else last.content
            )
            if content:
                print(f"\nAgent: {content}")


def run_image_task(
    image_path: str, task: str, backend: str = "marie", debug: bool = False
):
    """Run a task with an image (multimodal).

    Args:
        image_path: Path to the image file
        task: The query/task about the image
        backend: LLM backend to use
        debug: Show raw LLM responses
    """
    print(f"Image: {image_path}")
    print(f"Task: {task}")
    if debug:
        print(f"Backend: {backend}")
        print(f"Debug: ON")
    print("-" * 40)

    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return

    agent = create_agent(backend=backend)

    # Create multimodal message with image + text
    # The format uses {"image": path} and {"text": query} content items
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": task},
            ],
        }
    ]

    iteration = 0
    for responses in agent.run(messages=messages):
        if responses:
            iteration += 1
            for resp in responses:
                if debug:
                    print_debug_response(resp, iteration)

            last = responses[-1]
            content = (
                last.get("content", "") if isinstance(last, dict) else last.content
            )
            if content:
                print(f"\nAgent: {content}")


def run_interactive(backend: str = "marie", debug: bool = False):
    """Interactive mode."""
    print("Simple Agent - Interactive Mode")
    if debug:
        print("Debug: ON")
    print("Type 'quit' to exit\n")

    agent = create_agent(backend=backend)
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

        iteration = 0
        response_list = []
        for response_list in agent.run(messages=messages):
            if response_list:
                iteration += 1
                if debug:
                    for resp in response_list:
                        print_debug_response(resp, iteration)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Agent Example")
    parser.add_argument("--task", "-t", help="Task to run")
    parser.add_argument("--image", "-i", help="Image file for multimodal analysis")
    parser.add_argument("--tui", action="store_true", help="Interactive mode")
    parser.add_argument("--backend", default="marie", choices=["marie", "openai"])
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Show raw LLM responses"
    )

    args = parser.parse_args()

    if args.tui:
        run_interactive(backend=args.backend, debug=args.debug)
    elif args.image and args.task:
        run_image_task(args.image, args.task, backend=args.backend, debug=args.debug)
    elif args.image:
        # Image provided without task - use default description task
        run_image_task(
            args.image,
            "Describe this image in detail.",
            backend=args.backend,
            debug=args.debug,
        )
    elif args.task:
        run_task(args.task, backend=args.backend, debug=args.debug)
    else:
        print("Simple Agent - Tutorial / Starter Template")
        print("=" * 60)
        print()
        print("This is a LEARNING example. Copy this file to start your own agent.")
        print("For production features, see: agent_assistant.py")
        print()
        print("TOOLS (simple implementations):")
        print("  add        - Add two numbers")
        print("  multiply   - Multiply two numbers")
        print("  get_time   - Get current date and time")
        print("  list_files - List files in a directory")
        print("  read_file  - Read a text file")
        print("  image_info - Get image metadata (size, format, etc.)")
        print("  counter    - Stateful counter (shows AgentTool class)")
        print()
        print("EXAMPLES:")
        print("  # Text tasks")
        print("  python agent_simple.py --task 'Add 5 and 3'")
        print("  python agent_simple.py --task 'What time is it?'")
        print("  python agent_simple.py --task 'List Python files'")
        print("  python agent_simple.py --task 'Increment counter 3 times'")
        print()
        print("  # Image analysis (multimodal)")
        print("  python agent_simple.py --image photo.jpg")
        print("  python agent_simple.py --image doc.png --task 'What text is here?'")
        print()
        print("  # Other options")
        print("  python agent_simple.py --tui")
        print("  python agent_simple.py --backend openai --task 'Add 5 and 3'")
        print("  python agent_simple.py --task 'Add 5 and 3' --debug")
