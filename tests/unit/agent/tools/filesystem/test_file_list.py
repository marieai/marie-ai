"""Tests for FileListTool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marie.agent.tools.filesystem import FileListTool


class TestFileListTool:
    """Tests for FileListTool."""

    def test_metadata(self):
        """Test tool metadata is correct."""
        tool = FileListTool()
        assert tool.name == "file_list"
        assert "List" in tool.description
        assert tool.metadata.fn_schema is not None

    def test_list_files_in_directory(self, temp_files, temp_dir):
        """Test listing files in a directory."""
        tool = FileListTool()
        result = tool.call(path=str(temp_dir))

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["total_matched"] >= 3  # At least our test files
        assert len(data["files"]) >= 3

    def test_list_with_pattern(self, temp_files, temp_dir):
        """Test listing files with glob pattern."""
        tool = FileListTool()
        result = tool.call(path=str(temp_dir), pattern="*.txt")

        assert result.is_error is False
        data = json.loads(result.content)
        # Should find file1.txt and file2.txt but not script.py
        file_names = [f["name"] for f in data["files"]]
        assert "file1.txt" in file_names
        assert "file2.txt" in file_names
        assert "script.py" not in file_names

    def test_recursive_listing(self, nested_temp_dir):
        """Test recursive file listing."""
        tool = FileListTool()
        result = tool.call(path=str(nested_temp_dir), pattern="*.txt", recursive=True)

        assert result.is_error is False
        data = json.loads(result.content)
        file_names = [f["name"] for f in data["files"]]
        assert "root.txt" in file_names
        assert "nested.txt" in file_names

    def test_non_recursive_excludes_subdirs(self, nested_temp_dir):
        """Test that non-recursive listing excludes subdirectories."""
        tool = FileListTool()
        result = tool.call(path=str(nested_temp_dir), pattern="*.txt", recursive=False)

        assert result.is_error is False
        data = json.loads(result.content)
        file_names = [f["name"] for f in data["files"]]
        assert "root.txt" in file_names
        assert "nested.txt" not in file_names

    def test_max_files_limit(self, temp_dir):
        """Test max_files parameter limits results."""
        # Create many files
        for i in range(20):
            (temp_dir / f"file_{i}.txt").write_text(f"Content {i}")

        tool = FileListTool()
        result = tool.call(path=str(temp_dir), max_files=5)

        assert result.is_error is False
        data = json.loads(result.content)
        assert len(data["files"]) <= 5
        assert data["truncated"] is True

    def test_nonexistent_directory(self):
        """Test listing a nonexistent directory."""
        tool = FileListTool()
        result = tool.call(path="/nonexistent/directory")

        assert result.is_error is True
        data = json.loads(result.content)
        assert "not found" in data["error"].lower()

    def test_file_path_instead_of_directory(self, temp_file):
        """Test error when path is a file instead of directory."""
        tool = FileListTool()
        result = tool.call(path=str(temp_file))

        assert result.is_error is True
        data = json.loads(result.content)
        assert "not a directory" in data["error"].lower()

    def test_file_info_contains_required_fields(self, temp_files, temp_dir):
        """Test that file info contains all required fields."""
        tool = FileListTool()
        result = tool.call(path=str(temp_dir))

        data = json.loads(result.content)
        for file_info in data["files"]:
            assert "path" in file_info
            assert "name" in file_info
            assert "size_bytes" in file_info
            assert "modified" in file_info

    def test_returns_absolute_directory(self, temp_dir):
        """Test that result contains absolute directory path."""
        tool = FileListTool()
        result = tool.call(path=str(temp_dir))

        data = json.loads(result.content)
        assert Path(data["directory"]).is_absolute()

    def test_default_path_is_current_directory(self, temp_dir, monkeypatch):
        """Test that default path is current directory."""
        monkeypatch.chdir(temp_dir)
        tool = FileListTool()
        result = tool.call()

        assert result.is_error is False
        data = json.loads(result.content)
        assert data["directory"] == str(temp_dir.absolute())
