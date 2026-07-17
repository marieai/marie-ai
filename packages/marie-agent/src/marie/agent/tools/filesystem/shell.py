"""Shell command tool for agent framework."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput


class ShellInput(BaseModel):
    """Input schema for ShellTool."""

    command: str = Field(..., description="Shell command to execute")
    timeout: int = Field(30, description="Timeout in seconds")


class ShellTool(AgentTool):
    """Execute safe shell commands.

    Uses a fixed whitelist of allowed commands for security.
    Commands not in the whitelist are rejected.
    """

    ALLOWED_COMMANDS = frozenset(
        {
            "ls",
            "pwd",
            "cat",
            "head",
            "tail",
            "grep",
            "find",
            "wc",
            "echo",
            "date",
            "whoami",
            "hostname",
            "uname",
            "df",
            "du",
            "sort",
            "uniq",
            "cut",
            "tr",
            "sed",
            "awk",
            "diff",
            "file",
            "which",
        }
    )
    MAX_OUTPUT_SIZE = 100_000  # 100KB

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="shell",
            description=(
                "Execute safe shell commands. Only allows: "
                + ", ".join(sorted(self.ALLOWED_COMMANDS))
            ),
            fn_schema=ShellInput,
        )

    def call(self, command: str, timeout: int = 30, **kwargs: Any) -> ToolOutput:
        """Execute a shell command.

        Args:
            command: Shell command to execute
            timeout: Maximum execution time in seconds

        Returns:
            ToolOutput with command output or error
        """
        raw_input = {"command": command, "timeout": timeout}

        # Extract base command
        parts = command.split()
        if not parts:
            result = {"error": "Empty command", "command": command}
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )

        base_cmd = parts[0]

        # Security check - only allow whitelisted commands
        if base_cmd not in self.ALLOWED_COMMANDS:
            result = {
                "error": "Command not allowed",
                "command": command,
                "base_command": base_cmd,
                "allowed_commands": sorted(self.ALLOWED_COMMANDS),
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )

        try:
            result_proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stdout = result_proc.stdout
            stderr = result_proc.stderr

            # Truncate output if too large
            if len(stdout) > self.MAX_OUTPUT_SIZE:
                stdout = stdout[: self.MAX_OUTPUT_SIZE] + "\n... (output truncated)"
            if len(stderr) > self.MAX_OUTPUT_SIZE:
                stderr = stderr[: self.MAX_OUTPUT_SIZE] + "\n... (output truncated)"

            result = {
                "command": command,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": result_proc.returncode,
                "success": result_proc.returncode == 0,
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=result_proc.returncode != 0,
            )

        except subprocess.TimeoutExpired:
            result = {
                "error": "Command timed out",
                "command": command,
                "timeout": timeout,
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )
        except Exception as e:
            result = {"error": str(e), "command": command}
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )
