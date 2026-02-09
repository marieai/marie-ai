"""Tests for FileReadTool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marie.agent.tools.filesystem import FileReadTool


class TestFileReadTool:
    """Tests for FileReadTool."""

    def test_metadata(self):
        """Test tool metadata is correct."""
        tool = FileReadTool()
        assert tool.name == "file_read"
        assert "Read" in tool.description
        assert tool.metadata.fn_schema is not None

    def test_read_existing_file(self, temp_file):
        """Test reading an existing file."""
        tool = FileReadTool()
        result = tool.call(path=str(temp_file))

        assert result.is_error is False
        data = json.loads(result.content)
        assert "content" in data
        assert "Line 1" in data["content"]
        assert data["total_lines"] == 3

    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        tool = FileReadTool()
        result = tool.call(path="/nonexistent/file.txt")

        assert result.is_error is True
        data = json.loads(result.content)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_blocks_sensitive_files(self):
        """Test that sensitive files are blocked."""
        tool = FileReadTool()

        sensitive_paths = [
            "/home/user/.env",
            "/etc/password",
            "config/secret.txt",
            "~/.ssh/id_rsa",
            "credentials.json",
        ]

        for path in sensitive_paths:
            result = tool.call(path=path)
            assert result.is_error is True
            data = json.loads(result.content)
            assert "denied" in data["error"].lower() or "access" in data["error"].lower()

    def test_max_lines_limit(self, temp_dir):
        """Test max_lines parameter limits output."""
        # Create file with many lines
        file_path = temp_dir / "many_lines.txt"
        file_path.write_text("\n".join([f"Line {i}" for i in range(100)]))

        tool = FileReadTool()
        result = tool.call(path=str(file_path), max_lines=10)

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["lines_returned"] == 10
        assert data["total_lines"] == 100
        assert data["truncated"] is True

    def test_large_file_blocked(self, temp_dir):
        """Test that files exceeding MAX_SIZE are blocked."""
        tool = FileReadTool()
        tool.MAX_SIZE = 100  # Set small limit for testing

        file_path = temp_dir / "large_file.txt"
        file_path.write_text("x" * 200)

        result = tool.call(path=str(file_path))

        assert result.is_error is True
        data = json.loads(result.content)
        assert "too large" in data["error"].lower()

    def test_returns_absolute_path(self, temp_file):
        """Test that result contains absolute path."""
        tool = FileReadTool()
        result = tool.call(path=str(temp_file))

        assert result.is_error is False
        data = json.loads(result.content)
        assert Path(data["path"]).is_absolute()

    def test_tool_name_in_output(self, temp_file):
        """Test that tool_name is correctly set in output."""
        tool = FileReadTool()
        result = tool.call(path=str(temp_file))
        assert result.tool_name == "file_read"

    def test_raw_input_captured(self, temp_file):
        """Test that raw_input is captured in output."""
        tool = FileReadTool()
        result = tool.call(path=str(temp_file), max_lines=50)

        assert result.raw_input["path"] == str(temp_file)
        assert result.raw_input["max_lines"] == 50
