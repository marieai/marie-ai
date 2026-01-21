"""
Tests for RunContext API.
"""

import json
import os
import tempfile

import pytest
from marie_kernel import RunContext, TaskInstanceRef
from marie_kernel.backends.memory import InMemoryStateBackend


class TestRunContextBasic:
    """Tests for basic RunContext functionality."""

    def test_create_context(self, task_ref, backend):
        """Test creating a RunContext."""
        ctx = RunContext(task_ref, backend)
        assert ctx.ti == task_ref

    def test_ti_property(self, run_context, task_ref):
        """Test that ti property returns the task instance ref."""
        assert run_context.ti == task_ref
        assert run_context.ti.task_id == "task_1"

    def test_repr(self, run_context):
        """Test __repr__ produces useful output."""
        repr_str = repr(run_context)
        assert "RunContext" in repr_str
        assert "task_1" in repr_str


class TestRunContextSetGet:
    """Tests for the primary set/get API."""

    def test_set_and_get_basic(self, run_context):
        """Test basic set and get round-trip."""
        run_context.set("MY_KEY", "my_value")
        result = run_context.get("MY_KEY")
        assert result == "my_value"

    def test_set_and_get_dict(self, run_context):
        """Test storing and retrieving dict values."""
        data = {"rows": [1, 2, 3], "columns": ["a", "b"]}
        run_context.set("TABLE_DATA", data)
        result = run_context.get("TABLE_DATA")
        assert result == data

    def test_set_and_get_list(self, run_context):
        """Test storing and retrieving list values."""
        data = [1, 2, 3, "four", {"five": 5}]
        run_context.set("MY_LIST", data)
        result = run_context.get("MY_LIST")
        assert result == data

    def test_set_and_get_none(self, run_context):
        """Test storing and retrieving None."""
        run_context.set("NULLABLE", None)
        result = run_context.get("NULLABLE")
        assert result is None

    def test_get_missing_key_returns_none(self, run_context):
        """Test that missing key returns None by default."""
        result = run_context.get("NONEXISTENT")
        assert result is None

    def test_get_missing_key_with_default(self, run_context):
        """Test that missing key returns specified default."""
        result = run_context.get("NONEXISTENT", default="fallback")
        assert result == "fallback"

    def test_get_existing_ignores_default(self, run_context):
        """Test that existing key ignores default value."""
        run_context.set("EXISTS", "real_value")
        result = run_context.get("EXISTS", default="fallback")
        assert result == "real_value"

    def test_set_overwrites_existing(self, run_context):
        """Test that set overwrites existing values."""
        run_context.set("KEY", "first")
        run_context.set("KEY", "second")
        result = run_context.get("KEY")
        assert result == "second"


class TestRunContextFromTask:
    """Tests for cross-task state retrieval."""

    def test_get_from_upstream_task(self, backend):
        """Test getting state from an upstream task."""
        # Upstream task stores a value
        upstream_ti = TaskInstanceRef(
            tenant_id="test_tenant",
            dag_name="test_dag",
            dag_id="run_001",
            task_id="upstream",
            try_number=1,
        )
        upstream_ctx = RunContext(upstream_ti, backend)
        upstream_ctx.set("OCR_RESULT", {"text": "Hello World"})

        # Downstream task retrieves it
        downstream_ti = TaskInstanceRef(
            tenant_id="test_tenant",
            dag_name="test_dag",
            dag_id="run_001",
            task_id="downstream",
            try_number=1,
        )
        downstream_ctx = RunContext(downstream_ti, backend)
        result = downstream_ctx.get("OCR_RESULT", from_task="upstream")

        assert result == {"text": "Hello World"}

    def test_get_from_nonexistent_upstream_task(self, run_context):
        """Test getting from non-existent upstream returns default."""
        result = run_context.get("KEY", from_task="nonexistent", default="fallback")
        assert result == "fallback"


class TestRunContextPushPull:
    """Tests for the advanced push/pull API."""

    def test_push_with_metadata(self, run_context, backend, task_ref):
        """Test push with metadata."""
        run_context.push(
            "PROCESSED",
            {"result": 42},
            metadata={"processor_version": "1.0", "duration_ms": 123},
        )
        # Verify value is retrievable
        result = run_context.pull("PROCESSED")
        assert result == {"result": 42}

    def test_pull_from_tasks_list(self, backend):
        """Test pulling from multiple upstream tasks (first match wins)."""
        # Create multiple upstream tasks - only second one has the key
        for task_id in ["task_a", "task_b", "task_c"]:
            ti = TaskInstanceRef(
                tenant_id="test",
                dag_name="dag",
                dag_id="run",
                task_id=task_id,
                try_number=1,
            )
            ctx = RunContext(ti, backend)
            if task_id == "task_b":
                ctx.set("SHARED_KEY", f"value_from_{task_id}")

        # Downstream pulls from list - should find task_b's value
        downstream_ti = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="downstream",
            try_number=1,
        )
        downstream_ctx = RunContext(downstream_ti, backend)
        result = downstream_ctx.pull(
            "SHARED_KEY", from_tasks=["task_a", "task_b", "task_c"]
        )
        assert result == "value_from_task_b"

    def test_pull_from_tasks_order_matters(self, backend):
        """Test that from_tasks search order matters (first match wins)."""
        # Create tasks - both have the key
        for task_id, value in [("first", "first_value"), ("second", "second_value")]:
            ti = TaskInstanceRef(
                tenant_id="test",
                dag_name="dag",
                dag_id="run",
                task_id=task_id,
                try_number=1,
            )
            RunContext(ti, backend).set("KEY", value)

        # Pull should return first match
        downstream_ti = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="downstream",
            try_number=1,
        )
        ctx = RunContext(downstream_ti, backend)

        result1 = ctx.pull("KEY", from_tasks=["first", "second"])
        assert result1 == "first_value"

        result2 = ctx.pull("KEY", from_tasks=["second", "first"])
        assert result2 == "second_value"

    def test_pull_from_tasks_none_found(self, run_context):
        """Test pull returns default when no task has the key."""
        result = run_context.pull(
            "MISSING", from_tasks=["task_a", "task_b"], default="not_found"
        )
        assert result == "not_found"


class TestRunContextIsolation:
    """Tests for state isolation between tasks/runs."""

    def test_different_tasks_isolated(self, backend):
        """Test that different tasks don't see each other's state."""
        ti1 = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task_1",
            try_number=1,
        )
        ti2 = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task_2",
            try_number=1,
        )

        ctx1 = RunContext(ti1, backend)
        ctx2 = RunContext(ti2, backend)

        ctx1.set("KEY", "value_1")
        ctx2.set("KEY", "value_2")

        assert ctx1.get("KEY") == "value_1"
        assert ctx2.get("KEY") == "value_2"

    def test_different_runs_isolated(self, backend):
        """Test that different DAG runs don't see each other's state."""
        ti1 = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run_1",
            task_id="task",
            try_number=1,
        )
        ti2 = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run_2",
            task_id="task",
            try_number=1,
        )

        ctx1 = RunContext(ti1, backend)
        ctx2 = RunContext(ti2, backend)

        ctx1.set("KEY", "run_1_value")

        assert ctx1.get("KEY") == "run_1_value"
        assert ctx2.get("KEY") is None

    def test_different_tenants_isolated(self, backend):
        """Test that different tenants don't see each other's state."""
        ti1 = TaskInstanceRef(
            tenant_id="tenant_a",
            dag_name="dag",
            dag_id="run",
            task_id="task",
            try_number=1,
        )
        ti2 = TaskInstanceRef(
            tenant_id="tenant_b",
            dag_name="dag",
            dag_id="run",
            task_id="task",
            try_number=1,
        )

        ctx1 = RunContext(ti1, backend)
        ctx2 = RunContext(ti2, backend)

        ctx1.set("KEY", "tenant_a_value")

        assert ctx1.get("KEY") == "tenant_a_value"
        assert ctx2.get("KEY") is None

    def test_different_try_numbers_isolated(self, backend):
        """Test that different retry attempts have isolated state."""
        ti1 = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task",
            try_number=1,
        )
        ti2 = ti1.with_try_number(2)

        ctx1 = RunContext(ti1, backend)
        ctx2 = RunContext(ti2, backend)

        ctx1.set("KEY", "try_1_value")

        assert ctx1.get("KEY") == "try_1_value"
        assert ctx2.get("KEY") is None


class TestRunContextFromAssetDir:
    """Integration tests for RunContext.from_asset_dir() annotation query API."""

    @pytest.fixture
    def asset_dir(self):
        """Create a temporary asset directory with test annotation data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_output = os.path.join(tmpdir, "agent-output")

            # Create annotation directories
            os.makedirs(os.path.join(agent_output, "tables"))
            os.makedirs(os.path.join(agent_output, "claims"))
            os.makedirs(os.path.join(agent_output, "remarks"))

            # Create test JSON files for tables
            with open(os.path.join(agent_output, "tables", "00001.json"), "w") as f:
                json.dump({"rows": [[1, 2, 3]], "cols": ["a", "b", "c"]}, f)
            with open(os.path.join(agent_output, "tables", "00002.json"), "w") as f:
                json.dump({"rows": [[4, 5, 6]], "cols": ["d", "e", "f"]}, f)

            # Create test JSON files for claims
            with open(os.path.join(agent_output, "claims", "00001.json"), "w") as f:
                json.dump({"claim_id": "CLM001", "amount": 100.00}, f)

            # Create test markdown file for remarks
            with open(os.path.join(agent_output, "remarks", "00001.md"), "w") as f:
                f.write("# Page 1 Remarks\n\nThis is a test remark.")

            yield tmpdir

    def test_from_asset_dir_creates_context(self, asset_dir):
        """Test that from_asset_dir creates a valid RunContext."""
        ctx = RunContext.from_asset_dir(asset_dir)

        assert ctx is not None
        assert ctx.ti.dag_name == "filesystem"
        assert ctx.ti.dag_id == "direct"

    def test_list_annotations(self, asset_dir):
        """Test listing available annotations from asset directory."""
        ctx = RunContext.from_asset_dir(asset_dir)

        available = ctx.list_annotations()

        assert isinstance(available, list)
        assert "tables" in available
        assert "claims" in available
        assert "remarks" in available
        # Should be sorted
        assert available == sorted(available)

    def test_list_annotations_empty_dir(self):
        """Test list_annotations with no agent-output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = RunContext.from_asset_dir(tmpdir)
            available = ctx.list_annotations()
            assert available == []

    def test_get_annotation_all_pages(self, asset_dir):
        """Test getting all pages for an annotation."""
        ctx = RunContext.from_asset_dir(asset_dir)

        tables = ctx.get_annotation("tables")

        assert tables is not None
        assert tables["task_id"] == "tables"
        assert "output_dir" in tables
        assert "pages" in tables
        assert "raw_files" in tables
        assert 1 in tables["pages"]
        assert 2 in tables["pages"]

    def test_get_annotation_specific_page(self, asset_dir):
        """Test getting a specific page from an annotation."""
        ctx = RunContext.from_asset_dir(asset_dir)

        page1 = ctx.get_annotation("tables", page=1)

        assert page1 is not None
        assert page1["type"] == "json"
        assert page1["data"]["rows"] == [[1, 2, 3]]
        assert page1["file"] == "00001.json"
        assert "path" in page1
        assert page1["path"].endswith("00001.json")

    def test_get_annotation_includes_absolute_paths(self, asset_dir):
        """Test that annotation results include absolute file paths."""
        ctx = RunContext.from_asset_dir(asset_dir)

        tables = ctx.get_annotation("tables")

        # Check output_dir is absolute
        assert os.path.isabs(tables["output_dir"])
        assert os.path.exists(tables["output_dir"])

        # Check page paths are absolute
        for page_num, page_data in tables["pages"].items():
            assert "path" in page_data
            assert os.path.isabs(page_data["path"])
            assert os.path.exists(page_data["path"])

        # Check raw_files paths are absolute
        for file_info in tables["raw_files"]:
            assert "path" in file_info
            assert os.path.isabs(file_info["path"])
            assert os.path.exists(file_info["path"])

    def test_get_annotation_can_read_files(self, asset_dir):
        """Test that returned paths can be used to read file contents."""
        ctx = RunContext.from_asset_dir(asset_dir)

        page1 = ctx.get_annotation("tables", page=1)

        # Read the file using the returned path
        with open(page1["path"], "r") as f:
            content = json.load(f)

        assert content == page1["data"]

    def test_get_annotation_markdown(self, asset_dir):
        """Test getting markdown annotation results."""
        ctx = RunContext.from_asset_dir(asset_dir)

        remarks = ctx.get_annotation("remarks")

        assert remarks is not None
        assert 1 in remarks["pages"]
        assert remarks["pages"][1]["type"] == "markdown"
        assert "# Page 1 Remarks" in remarks["pages"][1]["data"]

    def test_get_annotation_nonexistent_returns_default(self, asset_dir):
        """Test that nonexistent annotation returns default value."""
        ctx = RunContext.from_asset_dir(asset_dir)

        result = ctx.get_annotation("nonexistent")
        assert result is None

        result_with_default = ctx.get_annotation("nonexistent", default={"empty": True})
        assert result_with_default == {"empty": True}

    def test_get_annotation_nonexistent_page_returns_default(self, asset_dir):
        """Test that nonexistent page returns default value."""
        ctx = RunContext.from_asset_dir(asset_dir)

        result = ctx.get_annotation("tables", page=999)
        assert result is None

        result_with_default = ctx.get_annotation("tables", page=999, default={})
        assert result_with_default == {}

    def test_full_workflow(self, asset_dir):
        """Integration test demonstrating the full annotation query workflow."""
        # Create context from asset directory
        ctx = RunContext.from_asset_dir(asset_dir)

        # List available annotations
        available = ctx.list_annotations()
        assert len(available) >= 3

        # Get all tables data
        tables = ctx.get_annotation("tables")
        assert tables["task_id"] == "tables"

        # Process each page
        for page_num in sorted(tables["pages"].keys()):
            page_data = ctx.get_annotation("tables", page=page_num)
            assert page_data["type"] == "json"

            # Can read the actual file
            with open(page_data["path"]) as f:
                raw_content = json.load(f)
            assert raw_content == page_data["data"]
