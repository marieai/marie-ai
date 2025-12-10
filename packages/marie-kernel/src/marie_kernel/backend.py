"""
StateBackend - Protocol defining the storage backend interface.

All state backends must implement this protocol to be compatible
with the Marie State Kernel.
"""

from typing import Any, Iterable, Optional, Protocol, runtime_checkable

from marie_kernel.ref import TaskInstanceRef


@runtime_checkable
class StateBackend(Protocol):
    """
    Protocol for state storage backends.

    Implementations must provide thread-safe push/pull/clear operations.
    The protocol uses structural subtyping, so any class implementing
    these methods is automatically compatible.

    Key semantics:
    - push: Store or update a value (upsert semantics)
    - pull: Retrieve a value, optionally from upstream tasks
    - clear_for_task: Remove all state for a task (called before retry)

    Example implementation:
        ```python
        class MyBackend:
            def push(self, ti, key, value, *, metadata=None):
                # Store value
                ...

            def pull(self, ti, key, *, from_tasks=None, default=None):
                # Retrieve value
                ...

            def clear_for_task(self, ti):
                # Clear all state for task
                ...
        ```
    """

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

        Uses upsert semantics: if the key already exists for this
        task instance, the value is replaced.

        Args:
            ti: Task instance reference (namespace)
            key: State key
            value: JSON-serializable value to store
            metadata: Optional metadata dict (for debugging/auditing)

        Raises:
            ValueError: If value is not JSON-serializable
        """
        ...

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
        ...

    def clear_for_task(self, ti: TaskInstanceRef) -> None:
        """
        Clear all state for a task instance.

        IMPORTANT: This must be called BEFORE retrying a task to ensure
        the new attempt starts with a clean slate. Clears all keys for
        the given (tenant, dag, run, task) regardless of try_number.

        Args:
            ti: Task instance reference identifying the task to clear
        """
        ...
