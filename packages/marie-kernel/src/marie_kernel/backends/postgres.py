"""
PostgresStateBackend - PostgreSQL state backend for production use.

This backend stores state in a PostgreSQL database using JSONB columns
for efficient JSON storage and querying.

Requires the 'postgres' optional dependency:
    pip install marie-kernel[postgres]
"""

import json
from typing import Any, Iterable, Optional

from marie_kernel.ref import TaskInstanceRef

try:
    from psycopg_pool import ConnectionPool
except ImportError:
    ConnectionPool = None  # type: ignore


class PostgresStateBackend:
    """
    PostgreSQL backend for task state storage.

    Uses PostgreSQL JSONB columns for efficient storage and querying.
    Supports upsert semantics via ON CONFLICT clause.

    Example:
        ```python
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool("postgresql://user:pass@localhost/marie")
        backend = PostgresStateBackend(pool)

        ti = TaskInstanceRef(
            tenant_id="acme",
            dag_name="document_pipeline",
            dag_id="run_2024_001",
            task_id="extract_text",
            try_number=1,
        )
        backend.push(ti, "result", {"text": "Hello World"})
        ```

    Table Schema:
        See migrations/001_task_state.sql for the required table schema.
    """

    def __init__(self, conn_pool: "ConnectionPool") -> None:
        """
        Initialize with a psycopg connection pool.

        Args:
            conn_pool: psycopg_pool.ConnectionPool instance
        """
        if ConnectionPool is None:
            raise ImportError(
                "psycopg is required for PostgresStateBackend. "
                "Install with: pip install marie-kernel[postgres]"
            )
        self._pool = conn_pool

    def push(
        self,
        ti: TaskInstanceRef,
        key: str,
        value: Any,
        *,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a value in PostgreSQL.

        Uses UPSERT semantics: if the key already exists for this task
        instance, the value is replaced.

        Args:
            ti: Task instance reference
            key: State key
            value: JSON-serializable value to store
            metadata: Optional metadata dict

        Raises:
            TypeError: If value is not JSON-serializable
        """
        value_json = json.dumps(value)
        meta_json = json.dumps(metadata or {})

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO task_state (
                        tenant_id, dag_name, dag_id, task_id, try_number,
                        key, value_json, metadata, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now())
                    ON CONFLICT (tenant_id, dag_name, dag_id, task_id, try_number, key)
                    DO UPDATE SET
                        value_json = EXCLUDED.value_json,
                        metadata = EXCLUDED.metadata,
                        created_at = now()
                    """,
                    (
                        ti.tenant_id,
                        ti.dag_name,
                        ti.dag_id,
                        ti.task_id,
                        ti.try_number,
                        key,
                        value_json,
                        meta_json,
                    ),
                )
            conn.commit()

    def pull(
        self,
        ti: TaskInstanceRef,
        key: str,
        *,
        from_tasks: Optional[Iterable[str]] = None,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a value from PostgreSQL.

        If from_tasks is provided, searches those task IDs and returns
        the most recently created match.

        Args:
            ti: Task instance reference (provides dag context)
            key: State key to retrieve
            from_tasks: Optional list of task IDs to search
            default: Value to return if key not found

        Returns:
            The stored value, or default if not found
        """
        task_ids = list(from_tasks) if from_tasks else [ti.task_id]

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT value_json
                    FROM task_state
                    WHERE tenant_id = %s
                      AND dag_name = %s
                      AND dag_id = %s
                      AND task_id = ANY(%s)
                      AND try_number = %s
                      AND key = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (
                        ti.tenant_id,
                        ti.dag_name,
                        ti.dag_id,
                        task_ids,
                        ti.try_number,
                        key,
                    ),
                )
                row = cur.fetchone()

        if not row:
            return default

        # psycopg3 auto-deserializes JSONB, but handle string case too
        value = row[0]
        if isinstance(value, str):
            return json.loads(value)
        return value

    def clear_for_task(self, ti: TaskInstanceRef) -> None:
        """
        Clear all state for a task instance.

        Removes ALL state for the given (tenant, dag_name, dag_id, task) regardless
        of try_number. This is called BEFORE retry to ensure clean slate.

        Args:
            ti: Task instance reference identifying the task to clear
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM task_state
                    WHERE tenant_id = %s
                      AND dag_name = %s
                      AND dag_id = %s
                      AND task_id = %s
                    """,
                    (ti.tenant_id, ti.dag_name, ti.dag_id, ti.task_id),
                )
            conn.commit()

    def get_all_for_task(self, ti: TaskInstanceRef) -> dict[str, Any]:
        """
        Retrieve all state for a task instance (debugging helper).

        Args:
            ti: Task instance reference

        Returns:
            Dictionary of {key: value} for all state belonging to this task
        """
        result = {}
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT key, value_json
                    FROM task_state
                    WHERE tenant_id = %s
                      AND dag_name = %s
                      AND dag_id = %s
                      AND task_id = %s
                      AND try_number = %s
                    """,
                    (
                        ti.tenant_id,
                        ti.dag_name,
                        ti.dag_id,
                        ti.task_id,
                        ti.try_number,
                    ),
                )
                for row in cur.fetchall():
                    key = row[0]
                    value = row[1]
                    if isinstance(value, str):
                        value = json.loads(value)
                    result[key] = value
        return result
