"""Fixtures for filesystem tool unit tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file with sample content."""
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3\n")
    return file_path


@pytest.fixture
def temp_files(temp_dir):
    """Create multiple temporary files for testing."""
    files = {}
    for name in ["file1.txt", "file2.txt", "script.py"]:
        file_path = temp_dir / name
        file_path.write_text(f"Content of {name}")
        files[name] = file_path
    return files


@pytest.fixture
def nested_temp_dir(temp_dir):
    """Create a nested directory structure."""
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("Nested content")
    (temp_dir / "root.txt").write_text("Root content")
    return temp_dir
