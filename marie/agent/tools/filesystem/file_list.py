"""File list tool for agent framework."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput


class FileListInput(BaseModel):
    """Input schema for FileListTool."""

    path: str = Field(".", description="Directory to list")
    pattern: str = Field("*", description="Glob pattern")
    recursive: bool = Field(False, description="Include subdirectories")
    max_files: int = Field(100, description="Maximum files to return")


class FileListTool(AgentTool):
    """List files with glob patterns.

    Returns file information including name, size, and modification time.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_list",
            description="List files in a directory with optional pattern matching and recursive search.",
            fn_schema=FileListInput,
        )

    def call(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False,
        max_files: int = 100,
        **kwargs: Any,
    ) -> ToolOutput:
        """List files in a directory.

        Args:
            path: Directory to list
            pattern: Glob pattern to match
            recursive: If True, search recursively
            max_files: Maximum number of files to return

        Returns:
            ToolOutput with file listing or error
        """
        raw_input = {
            "path": path,
            "pattern": pattern,
            "recursive": recursive,
            "max_files": max_files,
        }

        try:
            dir_path = Path(path)
            if not dir_path.exists():
                result = {"error": "Directory not found", "path": path}
                return ToolOutput(
                    content=json.dumps(result),
                    tool_name=self.name,
                    raw_input=raw_input,
                    raw_output=result,
                    is_error=True,
                )

            if not dir_path.is_dir():
                result = {"error": "Path is not a directory", "path": path}
                return ToolOutput(
                    content=json.dumps(result),
                    tool_name=self.name,
                    raw_input=raw_input,
                    raw_output=result,
                    is_error=True,
                )

            if recursive:
                files = list(dir_path.rglob(pattern))
            else:
                files = list(dir_path.glob(pattern))

            file_info = []
            for f in files[:max_files]:
                if f.is_file():
                    try:
                        stat = f.stat()
                        file_info.append(
                            {
                                "path": str(f),
                                "name": f.name,
                                "size_bytes": stat.st_size,
                                "modified": datetime.fromtimestamp(
                                    stat.st_mtime
                                ).isoformat(),
                            }
                        )
                    except (OSError, PermissionError):
                        # Skip files we can't stat
                        continue

            result = {
                "directory": str(dir_path.absolute()),
                "pattern": pattern,
                "recursive": recursive,
                "total_matched": len(files),
                "files_returned": len(file_info),
                "truncated": len(files) > max_files,
                "files": file_info,
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
