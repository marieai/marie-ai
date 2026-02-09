"""Tests for FileWriteTool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marie.agent.tools.filesystem import FileWriteTool


class TestFileWriteTool:
    """Tests for FileWriteTool."""

    def test_metadata(self):
        """Test tool metadata is correct."""
        tool = FileWriteTool()
        assert tool.name == "file_write"
        assert "Write" in tool.description
        assert tool.metadata.fn_schema is not None

    def test_write_new_file(self, temp_dir):
        """Test writing to a new file."""
        tool = FileWriteTool()
        file_path = temp_dir / "new_file.txt"
        content = "Hello, World!"

        result = tool.call(path=str(file_path), content=content)

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["success"] is True
        assert file_path.read_text() == content

    def test_overwrite_existing_file(self, temp_file):
        """Test overwriting an existing file."""
        tool = FileWriteTool()
        new_content = "New content"

        result = tool.call(path=str(temp_file), content=new_content)

        assert result.is_error is False
        assert temp_file.read_text() == new_content

    def test_append_mode(self, temp_file):
        """Test appending to an existing file."""
        tool = FileWriteTool()
        original = temp_file.read_text()
        additional = "Additional content"

        result = tool.call(path=str(temp_file), content=additional, append=True)

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["mode"] == "append"
        assert temp_file.read_text() == original + additional

    def test_blocks_absolute_paths_outside_tmp(self):
        """Test that absolute paths outside /tmp are blocked."""
        tool = FileWriteTool()

        forbidden_paths = [
            "/etc/passwd",
            "/home/user/file.txt",
            "/root/test.txt",
            "/var/log/app.log",
        ]

        for path in forbidden_paths:
            result = tool.call(path=path, content="test")
            assert result.is_error is True
            data = json.loads(result.content)
            assert "security" in data["error"].lower()

    def test_allows_tmp_paths(self, temp_dir):
        """Test that /tmp paths are allowed."""
        tool = FileWriteTool()
        # Use actual temp dir which should be under /tmp
        file_path = temp_dir / "allowed.txt"

        result = tool.call(path=str(file_path), content="allowed")

        assert result.is_error is False

    def test_allows_relative_paths(self, temp_dir, monkeypatch):
        """Test that relative paths are allowed."""
        monkeypatch.chdir(temp_dir)
        tool = FileWriteTool()

        result = tool.call(path="relative_file.txt", content="relative")

        assert result.is_error is False
        assert (temp_dir / "relative_file.txt").exists()

    def test_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created if needed."""
        tool = FileWriteTool()
        file_path = temp_dir / "nested" / "deeply" / "file.txt"

        result = tool.call(path=str(file_path), content="nested content")

        assert result.is_error is False
        assert file_path.exists()
        assert file_path.read_text() == "nested content"

    def test_bytes_written_count(self, temp_dir):
        """Test that bytes_written count is correct."""
        tool = FileWriteTool()
        file_path = temp_dir / "bytes_test.txt"
        content = "Hello 世界"  # Mixed ASCII and Unicode

        result = tool.call(path=str(file_path), content=content)

        data = json.loads(result.content)
        assert data["bytes_written"] == len(content.encode("utf-8"))

    def test_tool_name_in_output(self, temp_dir):
        """Test that tool_name is correctly set in output."""
        tool = FileWriteTool()
        result = tool.call(path=str(temp_dir / "test.txt"), content="test")
        assert result.tool_name == "file_write"
