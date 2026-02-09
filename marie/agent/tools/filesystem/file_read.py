"""File read tool for agent framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput


class FileReadInput(BaseModel):
    """Input schema for FileReadTool."""

    path: str = Field(..., description="Path to file to read")
    max_lines: int = Field(200, description="Maximum lines to read")


class FileReadTool(AgentTool):
    """Read file contents with security restrictions.

    Blocks access to sensitive files (.env, .ssh, credentials, etc.)
    and enforces size/line limits.
    """

    BLOCKED_PATTERNS = [
        ".env",
        ".ssh",
        "credentials",
        "secret",
        "password",
        ".pem",
        ".key",
        ".gnupg",
    ]
    MAX_SIZE = 1_000_000  # 1MB

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_read",
            description="Read contents of a file. Blocks sensitive files for security.",
            fn_schema=FileReadInput,
        )

    def call(self, path: str, max_lines: int = 200, **kwargs: Any) -> ToolOutput:
        """Read file contents.

        Args:
            path: Path to the file to read
            max_lines: Maximum number of lines to read

        Returns:
            ToolOutput with file contents or error
        """
        raw_input = {"path": path, "max_lines": max_lines}

        # Security check - block sensitive files
        path_lower = path.lower()
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in path_lower:
                result = {
                    "error": "Access denied",
                    "message": f"Cannot read files matching pattern: {pattern}",
                    "path": path,
                }
                return ToolOutput(
                    content=json.dumps(result),
                    tool_name=self.name,
                    raw_input=raw_input,
                    raw_output=result,
                    is_error=True,
                )

        try:
            file_path = Path(path)
            if not file_path.exists():
                result = {"error": "File not found", "path": path}
                return ToolOutput(
                    content=json.dumps(result),
                    tool_name=self.name,
                    raw_input=raw_input,
                    raw_output=result,
                    is_error=True,
                )

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.MAX_SIZE:
                result = {
                    "error": "File too large",
                    "size_bytes": file_size,
                    "max_size": self.MAX_SIZE,
                    "path": path,
                }
                return ToolOutput(
                    content=json.dumps(result),
                    tool_name=self.name,
                    raw_input=raw_input,
                    raw_output=result,
                    is_error=True,
                )

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            total_lines = len(lines)
            truncated = total_lines > max_lines
            content = "".join(lines[:max_lines])

            result = {
                "path": str(file_path.absolute()),
                "content": content,
                "total_lines": total_lines,
                "lines_returned": min(total_lines, max_lines),
                "truncated": truncated,
                "size_bytes": file_size,
            }
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=False,
            )

        except PermissionError:
            result = {"error": "Permission denied", "path": path}
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )
        except Exception as e:
            result = {"error": str(e), "path": path}
            return ToolOutput(
                content=json.dumps(result),
                tool_name=self.name,
                raw_input=raw_input,
                raw_output=result,
                is_error=True,
            )
