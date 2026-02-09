"""File write tool for agent framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput


class FileWriteInput(BaseModel):
    """Input schema for FileWriteTool."""

    path: str = Field(..., description="Path to write (relative or /tmp/*)")
    content: str = Field(..., description="Content to write")
    append: bool = Field(False, description="Append instead of overwrite")


class FileWriteTool(AgentTool):
    """Write to files with path restrictions.

    Only allows writing to relative paths or /tmp/* for security.
    Creates parent directories if needed.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_write",
            description="Write content to a file. Only allows relative paths or /tmp/* for security.",
            fn_schema=FileWriteInput,
        )

    def call(
        self, path: str, content: str, append: bool = False, **kwargs: Any
    ) -> ToolOutput:
        """Write content to a file.

        Args:
            path: Path to the file to write
            content: Content to write
            append: If True, append instead of overwrite

        Returns:
            ToolOutput with write status or error
        """
        raw_input = {"path": path, "content": content, "append": append}

        # Security check - only allow relative paths or /tmp
        if path.startswith("/") and not path.startswith("/tmp"):
            result = {
                "error": "Security restriction",
                "message": "Can only write to relative paths or /tmp/*",
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

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)

            result = {
                "success": True,
                "path": str(file_path.absolute()),
                "bytes_written": len(content.encode("utf-8")),
                "mode": "append" if append else "write",
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
