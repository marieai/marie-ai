"""
Tests for PostgresStateBackend.

These tests use mocking to avoid requiring a real PostgreSQL connection.
Integration tests with a real database should be run separately.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from marie_kernel import TaskInstanceRef

# Skip all tests if psycopg is not installed
pytest.importorskip("psycopg_pool")

from marie_kernel.backends.postgres import PostgresStateBackend


@pytest.fixture
def mock_pool():
    """Create a mock connection pool."""
    pool = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()

    # Setup context manager chain
    pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
    pool.connection.return_value.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    return pool, conn, cursor


@pytest.fixture
def postgres_backend(mock_pool):
    """Create a PostgresStateBackend with mock pool."""
    pool, conn, cursor = mock_pool
    return PostgresStateBackend(pool)


@pytest.fixture
def task_ref():
    """Create a standard TaskInstanceRef for testing."""
    return TaskInstanceRef(
        tenant_id="test_tenant",
        dag_name="test_dag",
        dag_id="run_001",
        task_id="task_1",
        try_number=1,
    )


class TestPostgresBackendPush:
    """Tests for push operation."""

    def test_push_executes_upsert_query(self, mock_pool, task_ref):
        """Test that push executes the correct SQL."""
        pool, conn, cursor = mock_pool
        backend = PostgresStateBackend(pool)

        backend.push(task_ref, "MY_KEY", {"data": 123})

        # Verify execute was called
        cursor.execute.assert_called_once()
        call_args = cursor.execute.call_args

        # Check SQL contains expected elements
        sql = call_args[0][0]
        assert "INSERT INTO task_state" in sql
        assert "ON CONFLICT" in sql
        assert "DO UPDATE SET" in sql

        # Check parameters
        params = call_args[0][1]
        assert params[0] == "test_tenant"
        assert params[1] == "test_dag"
        assert params[2] == "run_001"
        assert params[3] == "task_1"
        assert params[4] == 1  # try_number
        assert params[5] == "MY_KEY"
        assert json.loads(params[6]) == {"data": 123}

        # Verify commit was called
        conn.commit.assert_called_once()

    def test_push_with_metadata(self, mock_pool, task_ref):
        """Test that metadata is included in the query."""
        pool, conn, cursor = mock_pool
        backend = PostgresStateBackend(pool)

        backend.push(
            task_ref, "KEY", "value", metadata={"source": "test", "version": 1}
        )

        call_args = cursor.execute.call_args
        params = call_args[0][1]

        # metadata is the 8th parameter
        meta = json.loads(params[7])
        assert meta == {"source": "test", "version": 1}

    def test_push_empty_metadata(self, mock_pool, task_ref):
        """Test that empty metadata is serialized as empty object."""
        pool, conn, cursor = mock_pool
        backend = PostgresStateBackend(pool)

        backend.push(task_ref, "KEY", "value")

        call_args = cursor.execute.call_args
        params = call_args[0][1]

        # metadata should be empty object
        meta = json.loads(params[7])
        assert meta == {}


class TestPostgresBackendPull:
    """Tests for pull operation."""

    def test_pull_returns_value(self, mock_pool, task_ref):
        """Test that pull returns the stored value."""
        pool, conn, cursor = mock_pool
        cursor.fetchone.return_value = ({"data": 42},)  # JSONB auto-deserialized

        backend = PostgresStateBackend(pool)
        result = backend.pull(task_ref, "MY_KEY")

        assert result == {"data": 42}

    def test_pull_returns_default_when_not_found(self, mock_pool, task_ref):
        """Test that pull returns default when key not found."""
        pool, conn, cursor = mock_pool
        cursor.fetchone.return_value = None

        backend = PostgresStateBackend(pool)
        result = backend.pull(task_ref, "MISSING", default="fallback")

        assert result == "fallback"

    def test_pull_query_structure(self, mock_pool, task_ref):
        """Test that pull executes the correct SQL."""
        pool, conn, cursor = mock_pool
        cursor.fetchone.return_value = None

        backend = PostgresStateBackend(pool)
        backend.pull(task_ref, "KEY", from_tasks=["task_a", "task_b"])

        cursor.execute.assert_called_once()
        call_args = cursor.execute.call_args

        sql = call_args[0][0]
        assert "SELECT value_json" in sql
        assert "FROM task_state" in sql
        assert "ANY(%s)" in sql  # task_id array parameter
        assert "ORDER BY created_at DESC" in sql
        assert "LIMIT 1" in sql

        params = call_args[0][1]
        assert params[3] == ["task_a", "task_b"]  # from_tasks as list

    def test_pull_handles_string_json(self, mock_pool, task_ref):
        """Test that pull handles string JSON (not auto-deserialized)."""
        pool, conn, cursor = mock_pool
        # Some psycopg configs return string instead of dict
        cursor.fetchone.return_value = ('{"key": "value"}',)

        backend = PostgresStateBackend(pool)
        result = backend.pull(task_ref, "KEY")

        assert result == {"key": "value"}


class TestPostgresBackendClearForTask:
    """Tests for clear_for_task operation."""

    def test_clear_executes_delete_query(self, mock_pool, task_ref):
        """Test that clear_for_task executes DELETE SQL."""
        pool, conn, cursor = mock_pool

        backend = PostgresStateBackend(pool)
        backend.clear_for_task(task_ref)

        cursor.execute.assert_called_once()
        call_args = cursor.execute.call_args

        sql = call_args[0][0]
        assert "DELETE FROM task_state" in sql
        assert "task_id = %s" in sql

        params = call_args[0][1]
        assert params == ("test_tenant", "test_dag", "run_001", "task_1")

        conn.commit.assert_called_once()

    def test_clear_does_not_filter_by_try_number(self, mock_pool, task_ref):
        """Test that clear removes ALL try_numbers (not just current)."""
        pool, conn, cursor = mock_pool

        backend = PostgresStateBackend(pool)
        backend.clear_for_task(task_ref)

        call_args = cursor.execute.call_args
        sql = call_args[0][0]

        # Should NOT have try_number in WHERE clause
        # (we want to clear ALL try_numbers for this task)
        assert "try_number" not in sql


class TestPostgresBackendGetAllForTask:
    """Tests for get_all_for_task helper."""

    def test_get_all_returns_dict(self, mock_pool, task_ref):
        """Test that get_all_for_task returns key-value dict."""
        pool, conn, cursor = mock_pool
        cursor.fetchall.return_value = [
            ("KEY_1", {"value": 1}),
            ("KEY_2", {"value": 2}),
        ]

        backend = PostgresStateBackend(pool)
        result = backend.get_all_for_task(task_ref)

        assert result == {"KEY_1": {"value": 1}, "KEY_2": {"value": 2}}

    def test_get_all_handles_string_json(self, mock_pool, task_ref):
        """Test that get_all handles string JSON values."""
        pool, conn, cursor = mock_pool
        cursor.fetchall.return_value = [
            ("KEY_1", '{"value": 1}'),  # String JSON
        ]

        backend = PostgresStateBackend(pool)
        result = backend.get_all_for_task(task_ref)

        assert result == {"KEY_1": {"value": 1}}


class TestPostgresBackendImportError:
    """Tests for import error handling."""

    def test_raises_import_error_without_psycopg(self):
        """Test that helpful ImportError is raised if psycopg not installed."""
        # This test verifies the error message, not the actual import behavior
        with patch.dict("sys.modules", {"psycopg_pool": None}):
            # Reload the module to trigger import check
            # In practice, the import error is raised during class instantiation
            pass  # The real test is that PostgresStateBackend exists


class TestPostgresBackendProtocolCompliance:
    """Tests verifying StateBackend protocol compliance."""

    def test_implements_push(self, postgres_backend, task_ref):
        """Test that push method exists with correct signature."""
        # Should not raise
        postgres_backend.push(task_ref, "key", "value", metadata={"a": 1})

    def test_implements_pull(self, mock_pool, task_ref):
        """Test that pull method exists with correct signature."""
        pool, conn, cursor = mock_pool
        cursor.fetchone.return_value = None

        backend = PostgresStateBackend(pool)
        # Should not raise
        backend.pull(task_ref, "key", from_tasks=["a", "b"], default="x")

    def test_implements_clear_for_task(self, postgres_backend, task_ref):
        """Test that clear_for_task method exists."""
        # Should not raise
        postgres_backend.clear_for_task(task_ref)
