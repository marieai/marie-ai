"""
Tests for InMemoryStateBackend.
"""

import pytest

from marie_kernel import TaskInstanceRef
from marie_kernel.backends.memory import InMemoryStateBackend


class TestInMemoryBackendBasic:
    """Tests for basic backend operations."""

    def test_push_and_pull_round_trip(self, backend, task_ref):
        """Test basic push and pull round-trip."""
        backend.push(task_ref, "KEY", "value")
        result = backend.pull(task_ref, "KEY")
        assert result == "value"

    def test_push_dict_value(self, backend, task_ref):
        """Test storing dictionary values."""
        data = {"nested": {"key": [1, 2, 3]}}
        backend.push(task_ref, "DICT_KEY", data)
        result = backend.pull(task_ref, "DICT_KEY")
        assert result == data

    def test_push_with_metadata(self, backend, task_ref):
        """Test push with metadata (metadata is stored but not returned)."""
        backend.push(
            task_ref, "KEY", "value", metadata={"author": "test", "version": 1}
        )
        result = backend.pull(task_ref, "KEY")
        assert result == "value"

    def test_pull_missing_key_returns_default(self, backend, task_ref):
        """Test that missing key returns default."""
        result = backend.pull(task_ref, "MISSING", default="fallback")
        assert result == "fallback"

    def test_pull_missing_key_default_none(self, backend, task_ref):
        """Test that missing key returns None when no default specified."""
        result = backend.pull(task_ref, "MISSING")
        assert result is None

    def test_push_overwrites_existing(self, backend, task_ref):
        """Test that push overwrites existing values (upsert semantics)."""
        backend.push(task_ref, "KEY", "first")
        backend.push(task_ref, "KEY", "second")
        result = backend.pull(task_ref, "KEY")
        assert result == "second"


class TestInMemoryBackendFromTasks:
    """Tests for pulling from upstream tasks."""

    def test_pull_from_single_task(self, backend):
        """Test pulling from a single upstream task."""
        upstream = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="upstream",
            try_number=1,
        )
        downstream = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="downstream",
            try_number=1,
        )

        backend.push(upstream, "RESULT", {"data": 42})
        result = backend.pull(downstream, "RESULT", from_tasks=["upstream"])
        assert result == {"data": 42}

    def test_pull_from_multiple_tasks_first_match(self, backend):
        """Test pulling from multiple tasks returns first match."""
        ti_base = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="base",
            try_number=1,
        )

        # Only task_b has the key
        task_b = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task_b",
            try_number=1,
        )
        backend.push(task_b, "KEY", "from_b")

        result = backend.pull(ti_base, "KEY", from_tasks=["task_a", "task_b", "task_c"])
        assert result == "from_b"

    def test_pull_from_multiple_tasks_order_matters(self, backend):
        """Test that search order determines which value is returned."""
        for task_id in ["first", "second"]:
            ti = TaskInstanceRef(
                tenant_id="test",
                dag_name="dag",
                dag_id="run",
                task_id=task_id,
                try_number=1,
            )
            backend.push(ti, "KEY", f"value_{task_id}")

        base = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="base",
            try_number=1,
        )

        # Order [first, second] -> first
        result1 = backend.pull(base, "KEY", from_tasks=["first", "second"])
        assert result1 == "value_first"

        # Order [second, first] -> second
        result2 = backend.pull(base, "KEY", from_tasks=["second", "first"])
        assert result2 == "value_second"


class TestInMemoryBackendClearForTask:
    """Tests for clear_for_task operation."""

    def test_clear_removes_all_keys_for_task(self, backend):
        """Test that clear removes all keys for a task."""
        ti = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task",
            try_number=1,
        )

        backend.push(ti, "KEY_1", "value_1")
        backend.push(ti, "KEY_2", "value_2")
        backend.push(ti, "KEY_3", "value_3")

        backend.clear_for_task(ti)

        assert backend.pull(ti, "KEY_1") is None
        assert backend.pull(ti, "KEY_2") is None
        assert backend.pull(ti, "KEY_3") is None

    def test_clear_removes_all_try_numbers(self, backend):
        """Test that clear removes state for all try_numbers."""
        ti1 = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task",
            try_number=1,
        )
        ti2 = ti1.with_try_number(2)
        ti3 = ti1.with_try_number(3)

        backend.push(ti1, "KEY", "try_1")
        backend.push(ti2, "KEY", "try_2")
        backend.push(ti3, "KEY", "try_3")

        # Clear using any try_number should clear all
        backend.clear_for_task(ti1)

        assert backend.pull(ti1, "KEY") is None
        assert backend.pull(ti2, "KEY") is None
        assert backend.pull(ti3, "KEY") is None

    def test_clear_does_not_affect_other_tasks(self, backend):
        """Test that clear doesn't affect other tasks in the same run."""
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

        backend.push(ti1, "KEY", "task_1_value")
        backend.push(ti2, "KEY", "task_2_value")

        backend.clear_for_task(ti1)

        assert backend.pull(ti1, "KEY") is None
        assert backend.pull(ti2, "KEY") == "task_2_value"

    def test_clear_does_not_affect_other_runs(self, backend):
        """Test that clear doesn't affect other DAG runs."""
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

        backend.push(ti1, "KEY", "run_1_value")
        backend.push(ti2, "KEY", "run_2_value")

        backend.clear_for_task(ti1)

        assert backend.pull(ti1, "KEY") is None
        assert backend.pull(ti2, "KEY") == "run_2_value"


class TestInMemoryBackendHelpers:
    """Tests for helper methods."""

    def test_len_returns_entry_count(self, backend, task_ref):
        """Test that __len__ returns correct count."""
        assert len(backend) == 0

        backend.push(task_ref, "KEY_1", "value_1")
        assert len(backend) == 1

        backend.push(task_ref, "KEY_2", "value_2")
        assert len(backend) == 2

        # Overwrite doesn't increase count
        backend.push(task_ref, "KEY_1", "new_value")
        assert len(backend) == 2

    def test_clear_removes_all_entries(self, backend):
        """Test that clear() removes all entries."""
        for i in range(5):
            ti = TaskInstanceRef(
                tenant_id="test",
                dag_name="dag",
                dag_id=f"run_{i}",
                task_id="task",
                try_number=1,
            )
            backend.push(ti, "KEY", f"value_{i}")

        assert len(backend) == 5

        backend.clear()

        assert len(backend) == 0

    def test_get_all_for_task(self, backend):
        """Test get_all_for_task returns all state for a task."""
        ti = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task",
            try_number=1,
        )

        backend.push(ti, "KEY_1", "value_1")
        backend.push(ti, "KEY_2", {"nested": True})
        backend.push(ti, "KEY_3", [1, 2, 3])

        result = backend.get_all_for_task(ti)

        assert result == {
            "KEY_1": "value_1",
            "KEY_2": {"nested": True},
            "KEY_3": [1, 2, 3],
        }

    def test_get_all_for_task_respects_try_number(self, backend):
        """Test get_all_for_task only returns state for specific try_number."""
        ti1 = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task",
            try_number=1,
        )
        ti2 = ti1.with_try_number(2)

        backend.push(ti1, "KEY_1", "try_1_value")
        backend.push(ti2, "KEY_2", "try_2_value")

        result1 = backend.get_all_for_task(ti1)
        result2 = backend.get_all_for_task(ti2)

        assert result1 == {"KEY_1": "try_1_value"}
        assert result2 == {"KEY_2": "try_2_value"}
