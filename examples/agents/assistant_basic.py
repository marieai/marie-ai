"""Basic Assistant Agent Example.

This example demonstrates how to create an assistant agent with tools
that perform actual work - not mock implementations.

Shows:
- Function-based tools with @register_tool decorator
- Class-based tools extending AgentTool
- Real operations (time, calculations, file I/O, shell commands)
- Proper error handling in tools
- Configuration via environment variables

Usage:
    # Interactive mode
    python assistant_basic.py --tui

    # Single query
    python assistant_basic.py --query "What time is it in Tokyo?"

    # With specific backend
    python assistant_basic.py --backend openai --query "Calculate 15% tip on $85"
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
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


def create_assistant(
    backend: str = "marie", model: Optional[str] = None
) -> AssistantAgent:
    """Create an assistant agent with real tools.

    Args:
        backend: LLM backend ("marie" or "openai")
        model: Model name to use

    Returns:
        Configured AssistantAgent instance.
    """
    if backend == "marie":
        llm = MarieEngineLLMWrapper(engine_name=model or "qwen2_5_vl_7b")
    elif backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        llm = OpenAICompatibleWrapper(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            api_base="https://api.openai.com/v1",
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    tools = [
        "get_current_time",
        "calculator",
        "run_shell_command",
        "read_file",
        "write_file",
        WebFetchTool(timeout=10),
        SystemInfoTool(),
    ]

    return AssistantAgent(
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


# =============================================================================
# Running Modes
# =============================================================================


def run_single_query(query: str, backend: str = "marie", model: Optional[str] = None):
    """Run a single query."""
    print(f"Query: {query}")
    print("-" * 60)

    agent = create_assistant(backend=backend, model=model)
    messages = [{"role": "user", "content": query}]

    for responses in agent.run(messages=messages):
        if responses:
            last = responses[-1]
            content = (
                last.get("content", "") if isinstance(last, dict) else last.content
            )
            if content:
                print(content)


def run_interactive():
    """Run in interactive mode."""
    print("=" * 60)
    print("Basic Assistant - Interactive Mode")
    print("=" * 60)
    print("Commands: 'quit', 'exit', 'clear'")
    print()

    agent = create_assistant()
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

        response_list = []
        for response_list in agent.run(messages=messages):
            if response_list:
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
    parser.add_argument("--query", "-q", type=str, help="Single query to run")
    parser.add_argument("--tui", action="store_true", help="Interactive mode")
    parser.add_argument("--backend", default="marie", choices=["marie", "openai"])
    parser.add_argument("--model", type=str, help="Model name")

    args = parser.parse_args()

    if args.tui:
        run_interactive()
    elif args.query:
        run_single_query(args.query, backend=args.backend, model=args.model)
    else:
        print("Examples:")
        print("  python assistant_basic.py --query 'What time is it in Tokyo?'")
        print("  python assistant_basic.py --query 'Calculate 15% tip on $85.50'")
        print("  python assistant_basic.py --query 'List files in current directory'")
        print("  python assistant_basic.py --tui")
