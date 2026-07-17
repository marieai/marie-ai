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

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root_dir = Path(root_dir).expanduser().resolve() if root_dir else None

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
            if any(part == ".." for part in Path(pattern).parts):
                raise PermissionError(pattern)
            dir_path = Path(path).expanduser()
            if self._root_dir is not None:
                if not dir_path.is_absolute():
                    dir_path = self._root_dir / dir_path
                dir_path = dir_path.resolve()
                if not dir_path.is_relative_to(self._root_dir):
                    raise PermissionError(path)
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
            for f in files:
                if self._root_dir is not None:
                    resolved = f.resolve()
                    if not resolved.is_relative_to(self._root_dir):
                        continue
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
                if len(file_info) == max_files:
                    break

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
