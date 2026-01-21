"""
Tests for FileSystemStateBackend.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from marie_kernel.backends.filesystem import FileSystemStateBackend
from marie_kernel.ref import TaskInstanceRef


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def backend(temp_dir):
    """Create a FileSystemStateBackend with a temporary directory."""
    return FileSystemStateBackend(base_path=temp_dir)


@pytest.fixture
def task_ref():
    """Create a TaskInstanceRef for testing."""
    return TaskInstanceRef(
        tenant_id="test_tenant",
        dag_name="test_dag",
        dag_id="test_run_001",
        task_id="remarks",
        try_number=1,
    )


class TestFileSystemStateBackend:
    """Tests for FileSystemStateBackend."""

    def test_init(self, temp_dir):
        """Test backend initialization."""
        backend = FileSystemStateBackend(base_path=temp_dir)
        assert backend._base_path == temp_dir

    def test_init_with_tilde(self):
        """Test backend initialization with tilde path expansion."""
        backend = FileSystemStateBackend(base_path="~/test")
        assert backend._base_path == os.path.expanduser("~/test")

    def test_push_pull_memory(self, backend, task_ref):
        """Test push and pull from memory store."""
        backend.push(task_ref, "TEST_KEY", {"data": "test_value"})
        result = backend.pull(task_ref, "TEST_KEY")
        assert result == {"data": "test_value"}

    def test_push_with_metadata(self, backend, task_ref):
        """Test push with metadata."""
        backend.push(
            task_ref,
            "TEST_KEY",
            {"data": "value"},
            metadata={"version": "1.0"},
        )
        result = backend.pull(task_ref, "TEST_KEY")
        assert result == {"data": "value"}

    def test_pull_default(self, backend, task_ref):
        """Test pull returns default for missing key."""
        result = backend.pull(task_ref, "MISSING_KEY", default="default_value")
        assert result == "default_value"

    def test_pull_from_tasks(self, backend, task_ref):
        """Test pull from specific task IDs."""
        # Push to different task
        upstream_ti = TaskInstanceRef(
            tenant_id="test_tenant",
            dag_name="test_dag",
            dag_id="test_run_001",
            task_id="tables",
            try_number=1,
        )
        backend.push(upstream_ti, "DATA", {"tables": ["table1", "table2"]})

        # Pull from remarks task, specifying tables as from_tasks
        result = backend.pull(task_ref, "DATA", from_tasks=["tables"])
        assert result == {"tables": ["table1", "table2"]}

    def test_clear_for_task(self, backend, task_ref):
        """Test clearing state for a task."""
        backend.push(task_ref, "KEY1", "value1")
        backend.push(task_ref, "KEY2", "value2")

        backend.clear_for_task(task_ref)

        assert backend.pull(task_ref, "KEY1") is None
        assert backend.pull(task_ref, "KEY2") is None

    def test_clear(self, backend, task_ref):
        """Test clearing all state."""
        backend.push(task_ref, "KEY1", "value1")
        backend.clear()
        assert len(backend) == 0

    def test_len(self, backend, task_ref):
        """Test len returns number of entries."""
        assert len(backend) == 0
        backend.push(task_ref, "KEY1", "value1")
        assert len(backend) == 1
        backend.push(task_ref, "KEY2", "value2")
        assert len(backend) == 2

    def test_get_all_for_task(self, backend, task_ref):
        """Test getting all state for a task."""
        backend.push(task_ref, "KEY1", "value1")
        backend.push(task_ref, "KEY2", "value2")

        result = backend.get_all_for_task(task_ref)
        assert result["KEY1"] == "value1"
        assert result["KEY2"] == "value2"


class TestFileSystemStateBackendFilesystem:
    """Tests for FileSystemStateBackend reading from filesystem."""

    def test_read_task_output_json(self, temp_dir, task_ref):
        """Test reading JSON output from filesystem."""
        # Create output directory structure
        output_dir = os.path.join(temp_dir, "agent-output", "tables")
        os.makedirs(output_dir, exist_ok=True)

        # Create test JSON files
        test_data = {"regions": [{"role": "CODE", "data": "test"}]}
        with open(os.path.join(output_dir, "frame_0001.json"), "w") as f:
            json.dump(test_data, f)

        backend = FileSystemStateBackend(base_path=temp_dir)
        result = backend.pull(task_ref, "ANNOTATOR_RESULTS", from_tasks=["tables"])

        assert result is not None
        assert result["task_id"] == "tables"
        assert 1 in result["pages"]
        assert result["pages"][1]["type"] == "json"
        assert result["pages"][1]["data"] == test_data

    def test_read_task_output_markdown(self, temp_dir, task_ref):
        """Test reading markdown output from filesystem."""
        output_dir = os.path.join(temp_dir, "agent-output", "claims")
        os.makedirs(output_dir, exist_ok=True)

        # Create test markdown file
        md_content = "# Test Claim\n\nThis is a test."
        with open(os.path.join(output_dir, "frame_0002.md"), "w") as f:
            f.write(md_content)

        backend = FileSystemStateBackend(base_path=temp_dir)
        result = backend.pull(task_ref, "ANNOTATOR_RESULTS", from_tasks=["claims"])

        assert result is not None
        assert result["task_id"] == "claims"
        assert 2 in result["pages"]
        assert result["pages"][2]["type"] == "markdown"
        assert result["pages"][2]["data"] == md_content

    def test_read_task_output_skips_prompt_files(self, temp_dir, task_ref):
        """Test that prompt files are skipped when reading output."""
        output_dir = os.path.join(temp_dir, "agent-output", "tables")
        os.makedirs(output_dir, exist_ok=True)

        # Create JSON and prompt files
        with open(os.path.join(output_dir, "frame_0001.json"), "w") as f:
            json.dump({"data": "test"}, f)
        with open(os.path.join(output_dir, "frame_0001.png_prompt.txt"), "w") as f:
            f.write("prompt content")

        backend = FileSystemStateBackend(base_path=temp_dir)
        result = backend.pull(task_ref, "ANNOTATOR_RESULTS", from_tasks=["tables"])

        # Should have one page, not include prompt file
        assert len(result["raw_files"]) == 1
        assert result["raw_files"][0]["type"] == "json"

    def test_read_task_output_nonexistent(self, temp_dir, task_ref):
        """Test reading from nonexistent task directory returns None."""
        backend = FileSystemStateBackend(base_path=temp_dir)
        result = backend.pull(task_ref, "ANNOTATOR_RESULTS", from_tasks=["nonexistent"])
        assert result is None

    def test_extract_page_number(self, temp_dir):
        """Test page number extraction from filenames."""
        backend = FileSystemStateBackend(base_path=temp_dir)

        assert backend._extract_page_number("frame_0001.json") == 1
        assert backend._extract_page_number("frame_0001.png.json") == 1
        assert backend._extract_page_number("page_42.md") == 42
        assert backend._extract_page_number("0003.json") == 3
        assert backend._extract_page_number("invalid.json") is None


class TestFileSystemStateBackendMultiPage:
    """Tests for multi-page document handling."""

    def test_read_multiple_pages(self, temp_dir, task_ref):
        """Test reading output from multiple pages."""
        output_dir = os.path.join(temp_dir, "agent-output", "tables")
        os.makedirs(output_dir, exist_ok=True)

        # Create multiple page files
        for i in range(1, 4):
            data = {"page": i, "regions": [{"role": "DATA"}]}
            with open(os.path.join(output_dir, f"frame_{i:04d}.json"), "w") as f:
                json.dump(data, f)

        backend = FileSystemStateBackend(base_path=temp_dir)
        result = backend.pull(task_ref, "ANNOTATOR_RESULTS", from_tasks=["tables"])

        assert len(result["pages"]) == 3
        assert 1 in result["pages"]
        assert 2 in result["pages"]
        assert 3 in result["pages"]

    def test_mixed_output_types(self, temp_dir, task_ref):
        """Test reading mixed JSON and markdown output."""
        output_dir = os.path.join(temp_dir, "agent-output", "mixed")
        os.makedirs(output_dir, exist_ok=True)

        # Create JSON file
        with open(os.path.join(output_dir, "frame_0001.json"), "w") as f:
            json.dump({"type": "json"}, f)

        # Create markdown file
        with open(os.path.join(output_dir, "frame_0002.md"), "w") as f:
            f.write("# Markdown")

        backend = FileSystemStateBackend(base_path=temp_dir)
        result = backend.pull(task_ref, "ANNOTATOR_RESULTS", from_tasks=["mixed"])

        assert result["pages"][1]["type"] == "json"
        assert result["pages"][2]["type"] == "markdown"
