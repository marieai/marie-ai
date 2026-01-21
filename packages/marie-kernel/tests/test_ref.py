"""
Tests for TaskInstanceRef dataclass.
"""

import pytest
from marie_kernel import TaskInstanceRef


class TestTaskInstanceRefCreation:
    """Tests for TaskInstanceRef creation and initialization."""

    def test_create_with_all_fields(self):
        """Test creating TaskInstanceRef with all fields specified."""
        ti = TaskInstanceRef(
            tenant_id="acme",
            dag_name="document_pipeline",
            dag_id="run_2024_001",
            task_id="extract_text",
            try_number=2,
        )
        assert ti.tenant_id == "acme"
        assert ti.dag_name == "document_pipeline"
        assert ti.dag_id == "run_2024_001"
        assert ti.task_id == "extract_text"
        assert ti.try_number == 2

    def test_default_try_number(self):
        """Test that try_number defaults to 1."""
        ti = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task",
        )
        assert ti.try_number == 1

    def test_frozen_immutability(self):
        """Test that TaskInstanceRef is immutable (frozen)."""
        ti = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task",
        )
        with pytest.raises(AttributeError):
            ti.task_id = "new_task"  # type: ignore

    def test_hashable(self):
        """Test that TaskInstanceRef is hashable (can be used in sets/dicts)."""
        ti1 = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task",
        )
        ti2 = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task",
        )
        # Same values should be equal and hash the same
        assert ti1 == ti2
        assert hash(ti1) == hash(ti2)
        assert len({ti1, ti2}) == 1


class TestTaskInstanceRefFromDict:
    """Tests for TaskInstanceRef.from_dict() factory method."""

    def test_from_dict_new_format(self):
        """Test creating from dict with new field names (dag_name/dag_id)."""
        data = {
            "tenant_id": "acme",
            "dag_name": "my_dag",
            "dag_id": "run_123",
            "task_id": "task_1",
            "try_number": 3,
        }
        ti = TaskInstanceRef.from_dict(data)
        assert ti.tenant_id == "acme"
        assert ti.dag_name == "my_dag"
        assert ti.dag_id == "run_123"
        assert ti.task_id == "task_1"
        assert ti.try_number == 3

    def test_from_dict_missing_fields_uses_defaults(self):
        """Test that missing fields use appropriate defaults."""
        data = {"dag_name": "my_dag"}
        ti = TaskInstanceRef.from_dict(data, tenant_id="default_tenant")
        assert ti.tenant_id == "default_tenant"
        assert ti.dag_name == "my_dag"
        assert ti.dag_id == "my_dag"  # Falls back to dag_name
        assert ti.task_id == ""  # Empty string default
        assert ti.try_number == 1

    def test_from_dict_uses_id_as_task_id_fallback(self):
        """Test that 'id' field is used as task_id fallback."""
        data = {
            "dag_name": "my_dag",
            "dag_id": "run_1",
            "id": "job_123",  # Should be used as task_id
        }
        ti = TaskInstanceRef.from_dict(data)
        assert ti.task_id == "job_123"

    def test_from_dict_task_id_takes_precedence_over_id(self):
        """Test that explicit task_id takes precedence over id."""
        data = {
            "dag_name": "my_dag",
            "dag_id": "run_1",
            "task_id": "explicit_task",
            "id": "job_123",
        }
        ti = TaskInstanceRef.from_dict(data)
        assert ti.task_id == "explicit_task"


class TestTaskInstanceRefFromWorkInfo:
    """Tests for TaskInstanceRef.from_work_info() factory method."""

    def test_from_work_info_basic(self):
        """Test creating from a WorkInfo-like object."""

        class MockWorkInfo:
            id = "job_456"
            dag_id = "my_dag"  # In marie-ai, this is the dag name/type
            data = {
                "tenant_id": "acme",
                "dag_id": "run_789",  # The actual run ID
                "try_number": 2,
            }

        ti = TaskInstanceRef.from_work_info(MockWorkInfo())
        assert ti.tenant_id == "acme"
        assert ti.dag_name == "my_dag"
        assert ti.dag_id == "run_789"
        assert ti.task_id == "job_456"
        assert ti.try_number == 2

    def test_from_work_info_no_data(self):
        """Test creating from WorkInfo with no data."""

        class MockWorkInfo:
            id = "job_123"
            dag_id = "dag_1"
            data = None

        ti = TaskInstanceRef.from_work_info(MockWorkInfo(), tenant_id="fallback")
        assert ti.tenant_id == "fallback"
        assert ti.dag_name == "dag_1"
        assert ti.dag_id == "dag_1"  # Falls back to dag_name when no data
        assert ti.task_id == "job_123"
        assert ti.try_number == 1

    def test_from_work_info_empty_dag_id(self):
        """Test handling of None/empty dag_id."""

        class MockWorkInfo:
            id = "job_123"
            dag_id = None
            data = {"dag_id": "explicit_run"}

        ti = TaskInstanceRef.from_work_info(MockWorkInfo())
        assert ti.dag_name == ""
        assert ti.dag_id == "explicit_run"


class TestTaskInstanceRefWithTryNumber:
    """Tests for TaskInstanceRef.with_try_number() method."""

    def test_with_try_number_creates_new_instance(self):
        """Test that with_try_number creates a new instance."""
        ti1 = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="task",
            try_number=1,
        )
        ti2 = ti1.with_try_number(2)

        assert ti1 is not ti2
        assert ti1.try_number == 1
        assert ti2.try_number == 2

    def test_with_try_number_preserves_other_fields(self):
        """Test that other fields are preserved."""
        ti1 = TaskInstanceRef(
            tenant_id="acme",
            dag_name="my_dag",
            dag_id="run_123",
            task_id="task_1",
            try_number=1,
        )
        ti2 = ti1.with_try_number(3)

        assert ti2.tenant_id == "acme"
        assert ti2.dag_name == "my_dag"
        assert ti2.dag_id == "run_123"
        assert ti2.task_id == "task_1"
        assert ti2.try_number == 3


class TestTaskInstanceRefStr:
    """Tests for TaskInstanceRef string representation."""

    def test_str_representation(self):
        """Test __str__ produces readable output."""
        ti = TaskInstanceRef(
            tenant_id="acme",
            dag_name="my_dag",
            dag_id="run_123",
            task_id="task_1",
            try_number=2,
        )
        s = str(ti)
        assert "acme" in s
        assert "my_dag" in s
        assert "run_123" in s
        assert "task_1" in s
        assert "2" in s
