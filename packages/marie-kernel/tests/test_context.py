"""
Tests for RunContext API.
"""

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
