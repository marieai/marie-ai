"""Assistant Agent Example - Production Ready.

PURPOSE: Production-quality agent with real-world tools. Use this as a reference
for building robust, full-featured agents with proper error handling.

For a simpler tutorial example, see: agent_simple.py

Features:
- Real tools that perform actual work (not mock implementations)
- Comprehensive error handling and input validation
- Security considerations (safe shell commands, path restrictions)
- Timezone support, math expressions, web fetching
- Production patterns you can adapt for your use case

Available Tools:
    - get_current_time: Get time in any timezone (IANA names)
    - calculator: Math expressions, percentages, scientific functions
    - run_shell_command: Safe shell commands (ls, pwd, cat, etc.)
    - read_file: Read file contents
    - write_file: Write to files (relative paths or /tmp)
    - web_fetch: Fetch and extract text from URLs
    - system_info: Get system/environment information

Usage Examples:
    # Time queries (supports IANA timezone names)
    python agent_assistant.py --task "What time is it in Tokyo?"
    python agent_assistant.py --task "What time is it in America/New_York?"
    python agent_assistant.py --task "What time is it in UTC?"

    # Calculator (supports percentages and math functions)
    python agent_assistant.py --task "Calculate 15% tip on $85"
    python agent_assistant.py --task "What is 20% of 150?"
    python agent_assistant.py --task "Calculate sqrt(144) + 25"
    python agent_assistant.py --task "What is sin(pi/2)?"

    # Shell commands (safe subset only)
    python agent_assistant.py --task "List files in current directory"
    python agent_assistant.py --task "Show current working directory"
    python agent_assistant.py --task "Run 'ls -la' command"

    # File operations
    python agent_assistant.py --task "Read the file agent_assistant.py"
    python agent_assistant.py --task "Write 'hello' to /tmp/test.txt"

    # System info
    python agent_assistant.py --task "What system am I running on?"
    python agent_assistant.py --task "What Python version is installed?"

    # Multiple tools in one query
    python agent_assistant.py --task "List Python files and tell me the time"

    # Interactive mode
    python agent_assistant.py --tui

    # With OpenAI backend
    python agent_assistant.py --backend openai --task "Calculate 15% tip on $85"

    # Debug mode - show raw LLM responses
    python agent_assistant.py --task "Calculate 15% tip on $85" --debug
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

from marie.agent import (
    AgentTool,
    ReactAgent,
    ToolMetadata,
    ToolOutput,
    register_tool,
)
from utils import print_debug_response

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Function-Based Tools (using @register_tool decorator)
# =============================================================================


@register_tool("get_current_time")
def get_current_time(timezone_name: str = "UTC") -> str:
    """Get the current time in a specified timezone.

    Args:
        timezone_name: IANA timezone name (e.g., "America/New_York", "Europe/London", "Asia/Tokyo")

    Returns:
        JSON string with current time information.
    """
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(timezone_name)
        now = datetime.now(tz)
        utc_now = datetime.now(timezone.utc)

        return json.dumps(
            {
                "timezone": timezone_name,
                "local_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "utc_time": utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "utc_offset": now.strftime("%z"),
                "day_of_week": now.strftime("%A"),
                "iso_format": now.isoformat(),
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "error": f"Invalid timezone: {timezone_name}",
                "message": str(e),
                "hint": "Use IANA timezone names like 'America/New_York', 'Europe/London', 'Asia/Tokyo'",
            }
        )


@register_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, abs, round, min, max

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)", "15% of 85")

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


@register_tool("run_shell_command")
def run_shell_command(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return the output.

    Only allows safe, read-only commands for security.

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds

    Returns:
        JSON string with command output.
    """
    # Whitelist of safe commands
    safe_commands = [
        "ls",
        "pwd",
        "whoami",
        "date",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "find",
        "echo",
        "which",
        "uname",
    ]
    base_cmd = command.split()[0] if command.split() else ""

    if base_cmd not in safe_commands:
        return json.dumps(
            {
                "error": "Command not allowed",
                "command": command,
                "allowed_commands": safe_commands,
            }
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return json.dumps(
            {
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0,
            }
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"error": "Command timed out", "command": command, "timeout": timeout}
        )
    except Exception as e:
        return json.dumps(
            {"error": "Command execution failed", "command": command, "message": str(e)}
        )


@register_tool("read_file")
def read_file(file_path: str, max_lines: int = 100) -> str:
    """Read contents of a file.

    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (default 100)

    Returns:
        JSON string with file contents.
    """
    try:
        # Security check - prevent reading sensitive files
        sensitive_patterns = [
            ".env",
            "password",
            "secret",
            "credential",
            ".ssh",
            ".gnupg",
        ]
        if any(pattern in file_path.lower() for pattern in sensitive_patterns):
            return json.dumps(
                {
                    "error": "Access denied",
                    "message": "Cannot read potentially sensitive files",
                }
            )

        with open(file_path, 'r') as f:
            lines = f.readlines()

        total_lines = len(lines)
        truncated = total_lines > max_lines
        content = ''.join(lines[:max_lines])

        return json.dumps(
            {
                "file_path": file_path,
                "content": content,
                "total_lines": total_lines,
                "lines_returned": min(total_lines, max_lines),
                "truncated": truncated,
            }
        )
    except FileNotFoundError:
        return json.dumps({"error": "File not found", "file_path": file_path})
    except PermissionError:
        return json.dumps({"error": "Permission denied", "file_path": file_path})
    except Exception as e:
        return json.dumps(
            {"error": "Failed to read file", "file_path": file_path, "message": str(e)}
        )


@register_tool("write_file")
def write_file(file_path: str, content: str, append: bool = False) -> str:
    """Write content to a file.

    Args:
        file_path: Path to the file to write
        content: Content to write
        append: If True, append to file instead of overwriting

    Returns:
        JSON string with write status.
    """
    try:
        # Security check - only allow relative paths or /tmp
        if file_path.startswith("/") and not file_path.startswith("/tmp"):
            return json.dumps(
                {
                    "error": "Security restriction",
                    "message": "Can only write to relative paths or /tmp",
                }
            )

        mode = 'a' if append else 'w'
        with open(file_path, mode) as f:
            f.write(content)

        return json.dumps(
            {
                "success": True,
                "file_path": file_path,
                "bytes_written": len(content),
                "mode": "append" if append else "overwrite",
            }
        )
    except Exception as e:
        return json.dumps(
            {"error": "Failed to write file", "file_path": file_path, "message": str(e)}
        )


# =============================================================================
# Class-Based Tool (extending AgentTool)
# =============================================================================


class WebFetchTool(AgentTool):
    """Tool for fetching content from URLs.

    Demonstrates a class-based tool with configuration.
    """

    def __init__(self, timeout: int = 10, max_size: int = 50000):
        self.timeout = timeout
        self.max_size = max_size

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Fetch content from a URL. Returns the text content of web pages.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        )

    def call(self, **kwargs) -> ToolOutput:
        """Fetch content from a URL."""
        url = kwargs.get("url", "")

        if not url:
            return ToolOutput(
                content=json.dumps({"error": "URL is required"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "URL is required"},
                is_error=True,
            )

        try:
            import urllib.request
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip_tags = {'script', 'style', 'head', 'meta'}
                    self.current_tag = None

                def handle_starttag(self, tag, attrs):
                    self.current_tag = tag

                def handle_data(self, data):
                    if self.current_tag not in self.skip_tags:
                        text = data.strip()
                        if text:
                            self.text.append(text)

                def get_text(self):
                    return ' '.join(self.text)

            req = urllib.request.Request(url, headers={'User-Agent': 'Marie-AI/1.0'})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read(self.max_size).decode('utf-8', errors='ignore')
                content_type = response.headers.get('Content-Type', '')

            if 'html' in content_type:
                parser = TextExtractor()
                parser.feed(content)
                text = parser.get_text()[:10000]
            else:
                text = content[:10000]

            result = {"url": url, "content": text, "content_type": content_type}
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
                is_error=False,
            )

        except Exception as e:
            result = {"error": str(e), "url": url}
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
                is_error=True,
            )


class SystemInfoTool(AgentTool):
    """Tool for getting system/environment information."""

    @property
    def name(self) -> str:
        return "system_info"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Get system and environment information (Python version, OS, etc.)",
        )

    def call(self, **kwargs) -> ToolOutput:
        import platform
        import sys

        info = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "cwd": os.getcwd(),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        }

        return ToolOutput(
            content=json.dumps(info, indent=2),
            tool_name=self.name,
            raw_input=kwargs,
            raw_output=info,
            is_error=False,
        )


# =============================================================================
# Agent Initialization
# =============================================================================


def create_assistant(backend: str = "marie", model: Optional[str] = None) -> ReactAgent:
    """Create an assistant agent with real tools.

    Args:
        backend: LLM backend ("marie" or "openai")
        model: Model name to use

    Returns:
        Configured ReactAgent instance.
    """
    from utils import create_llm

    llm = create_llm(backend=backend, model=model)

    tools = [
        "get_current_time",
        "calculator",
        "run_shell_command",
        "read_file",
        "write_file",
        WebFetchTool(timeout=10),
        SystemInfoTool(),
    ]

    return ReactAgent(
        llm=llm,
        function_list=tools,
        name="Basic Assistant",
        description="A helpful assistant with time, calculator, file, shell, and web tools.",
        system_message="""You are a helpful assistant with access to tools:

1. **get_current_time**: Get time in any timezone (use IANA names like "America/New_York")
2. **calculator**: Evaluate math expressions, percentages (e.g., "15% of 85")
3. **run_shell_command**: Run safe shell commands (ls, pwd, cat, etc.)
4. **read_file**: Read file contents
5. **write_file**: Write to files
6. **web_fetch**: Fetch content from URLs
7. **system_info**: Get system information

Use the appropriate tool for each task. Handle errors gracefully.""",
        max_iterations=10,
    )


def run_single_query(
    query: str, backend: str = "marie", model: Optional[str] = None, debug: bool = False
):
    """Run a single query."""
    print(f"Query: {query}")
    if debug:
        print(f"Backend: {backend}")
        print(f"Debug: ON")
    print("-" * 60)

    agent = create_assistant(backend=backend, model=model)
    messages = [{"role": "user", "content": query}]

    iteration = 0
    for responses in agent.run(messages=messages):
        if responses:
            iteration += 1
            if debug:
                for resp in responses:
                    print_debug_response(resp, iteration)

            last = responses[-1]
            content = (
                last.get("content", "") if isinstance(last, dict) else last.content
            )
            if content:
                print(f"\n{content}")


def run_interactive(backend: str = "marie", debug: bool = False):
    """Run in interactive mode."""
    print("=" * 60)
    print("Basic Assistant - Interactive Mode")
    print("=" * 60)
    if debug:
        print("Debug: ON")
    print("Commands: 'quit', 'exit', 'clear'")
    print()

    agent = create_assistant(backend=backend)
    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "clear":
            messages = []
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        print("\nAssistant: ", end="", flush=True)

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
                    print(content)

        if response_list:
            for r in response_list:
                messages.append(r if isinstance(r, dict) else r.model_dump())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic Assistant Agent")
    parser.add_argument("--task", "-t", type=str, help="Task to run")
    parser.add_argument("--tui", action="store_true", help="Interactive mode")
    parser.add_argument("--backend", default="marie", choices=["marie", "openai"])
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Show raw LLM responses"
    )

    args = parser.parse_args()

    if args.tui:
        run_interactive(backend=args.backend, debug=args.debug)
    elif args.task:
        run_single_query(
            args.task, backend=args.backend, model=args.model, debug=args.debug
        )
    else:
        print("Assistant Agent - Production Ready")
        print("=" * 60)
        print()
        print("Production-quality agent with robust tools and error handling.")
        print("For a simpler tutorial, see: agent_simple.py")
        print()
        print("TOOLS (production implementations):")
        print("  get_current_time  - Timezone-aware time (IANA names)")
        print("  calculator        - Math, percentages, scientific functions")
        print("  run_shell_command - Safe shell commands with whitelist")
        print("  read_file         - Read with size limits and error handling")
        print("  write_file        - Write with path restrictions")
        print("  web_fetch         - HTTP fetch with timeout and error handling")
        print("  system_info       - System/environment info")
        print()
        print("EXAMPLES:")
        print("  python agent_assistant.py --task 'What time is it in Tokyo?'")
        print("  python agent_assistant.py --task 'Calculate 15% tip on $85'")
        print("  python agent_assistant.py --task 'List files here'")
        print("  python agent_assistant.py --task 'List files and show time'")
        print("  python agent_assistant.py --tui")
        print("  python agent_assistant.py --backend openai --task '...'")
        print("  python agent_assistant.py --task '...' --debug")
