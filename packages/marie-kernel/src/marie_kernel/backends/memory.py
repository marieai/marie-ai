"""
InMemoryStateBackend - Thread-safe in-memory state backend.

This backend is primarily intended for testing and development.
Data is stored in a dictionary and is lost when the process exits.
"""

from threading import Lock
from typing import Any, Dict, Iterable, Optional, Tuple

from marie_kernel.ref import TaskInstanceRef


class InMemoryStateBackend:
    """
    Thread-safe in-memory backend for testing and development.

    Stores state in a dictionary keyed by (tenant, dag, run, task, try, key).
    All operations are protected by a threading lock for thread-safety.

    Example:
        ```python
        backend = InMemoryStateBackend()
        ti = TaskInstanceRef(
            tenant_id="test",
            dag_name="my_dag",
            dag_id="run_1",
            task_id="task_1",
            try_number=1,
        )
        backend.push(ti, "result", {"data": 123})
        value = backend.pull(ti, "result")  # Returns {"data": 123}
        ```
    """

    def __init__(self) -> None:
        """Initialize empty state store with thread lock."""
        self._store: Dict[Tuple[str, str, str, str, int, str], Dict[str, Any]] = {}
        self._lock = Lock()

    def push(
        self,
        ti: TaskInstanceRef,
        key: str,
        value: Any,
        *,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a value for the given task instance and key.

        Uses upsert semantics: if the key already exists, it is replaced.

        Args:
            ti: Task instance reference
            key: State key
            value: Value to store (should be JSON-serializable for consistency)
            metadata: Optional metadata dict
        """
        pk = (ti.tenant_id, ti.dag_name, ti.dag_id, ti.task_id, ti.try_number, key)
        with self._lock:
            self._store[pk] = {"value": value, "metadata": metadata or {}}

    def pull(
        self,
        ti: TaskInstanceRef,
        key: str,
        *,
        from_tasks: Optional[Iterable[str]] = None,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a value by key.

        If from_tasks is provided, searches those task IDs in order
        and returns the first match. If from_tasks is None, searches
        only the current task.

        Args:
            ti: Task instance reference (provides dag context)
            key: State key to retrieve
            from_tasks: Optional list of task IDs to search
            default: Value to return if key not found

        Returns:
            The stored value, or default if not found
        """
        task_ids = list(from_tasks) if from_tasks else [ti.task_id]

        with self._lock:
            for task_id in task_ids:
                pk = (
                    ti.tenant_id,
                    ti.dag_name,
                    ti.dag_id,
                    task_id,
                    ti.try_number,
                    key,
                )
                if pk in self._store:
                    return self._store[pk]["value"]

        return default

    def clear_for_task(self, ti: TaskInstanceRef) -> None:
        """
        Clear all state for a task instance.

        Removes all keys for the given (tenant, dag, run, task) regardless
        of try_number. This ensures a clean slate before retry.

        Args:
            ti: Task instance reference identifying the task to clear
        """
        with self._lock:
            keys_to_delete = [
                k
                for k in self._store
                if (
                    k[0] == ti.tenant_id
                    and k[1] == ti.dag_name
                    and k[2] == ti.dag_id
                    and k[3] == ti.task_id
                )
            ]
            for k in keys_to_delete:
                del self._store[k]

    def get_all_for_task(self, ti: TaskInstanceRef) -> Dict[str, Any]:
        """
        Retrieve all state for a task instance (debugging helper).

        Args:
            ti: Task instance reference

        Returns:
            Dictionary of {key: value} for all state belonging to this task
        """
        result = {}
        with self._lock:
            for pk, data in self._store.items():
                if (
                    pk[0] == ti.tenant_id
                    and pk[1] == ti.dag_name
                    and pk[2] == ti.dag_id
                    and pk[3] == ti.task_id
                    and pk[4] == ti.try_number
                ):
                    key = pk[5]
                    result[key] = data["value"]
        return result

    def __len__(self) -> int:
        """Return total number of stored entries."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Clear all stored state (useful for test cleanup)."""
        with self._lock:
            self._store.clear()
