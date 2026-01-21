"""
RunContext - Developer-facing API for task state management.

This class provides the primary interface for tasks to store and retrieve
state during DAG execution. It wraps a StateBackend and provides both
simple (set/get) and advanced (push/pull) APIs.
"""

from typing import Any, Iterable, Optional

from marie_kernel.backend import StateBackend
from marie_kernel.ref import TaskInstanceRef


class RunContext:
    """
    Developer-facing API for task state management.

    Primary API: ctx.set() / ctx.get() - simple key-value operations
    Advanced API: ctx.push() / ctx.pull() - with metadata and multi-task pulls

    Example:
        ```python
        def my_task(ctx: RunContext):
            # Read from upstream task
            ocr_lines = ctx.get("CLAIM_OCR_LINES", from_task="ocr")

            # Process and store result
            tables = locate_tables(ocr_lines)
            ctx.set("TABLE_STRUCTS", tables)
        ```

    Attributes:
        ti: The current task instance reference
    """

    def __init__(self, ti: TaskInstanceRef, backend: StateBackend):
        """
        Initialize RunContext.

        Args:
            ti: Task instance reference (identifies current task)
            backend: State storage backend implementation
        """
        self._ti = ti
        self._backend = backend

    @property
    def ti(self) -> TaskInstanceRef:
        """Current task instance reference."""
        return self._ti

    # ========================================================================
    # Primary Developer API (simple key-value)
    # ========================================================================

    def set(self, key: str, value: Any) -> None:
        """
        Store a value for the current task.

        This is the primary API for storing state. The value must be
        JSON-serializable.

        Args:
            key: State key (should be uppercase by convention)
            value: JSON-serializable value to store

        Example:
            ```python
            ctx.set("EXTRACTED_TEXT", "Hello World")
            ctx.set("TABLE_DATA", {"rows": [...], "columns": [...]})
            ```
        """
        return self.push(key, value)

    def get(
        self,
        key: str,
        *,
        from_task: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a value by key.

        This is the primary API for retrieving state. By default retrieves
        from the current task; use from_task to retrieve from an upstream task.

        Args:
            key: The key to retrieve
            from_task: Optional upstream task_id to pull from
            default: Value to return if key not found

        Returns:
            The stored value, or default if not found

        Example:
            ```python
            # Get from current task
            text = ctx.get("EXTRACTED_TEXT")

            # Get from upstream task
            ocr_result = ctx.get("OCR_LINES", from_task="ocr_task")

            # With default value
            config = ctx.get("CONFIG", default={})
            ```
        """
        return self.pull(key, from_task=from_task, default=default)

    # ========================================================================
    # Advanced API (with metadata, multi-task pulls)
    # ========================================================================

    def push(
        self,
        key: str,
        value: Any,
        *,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a value with optional metadata.

        Advanced API that allows attaching metadata for debugging/auditing.
        Use set() for simple cases.

        Args:
            key: State key
            value: JSON-serializable value to store
            metadata: Optional metadata dict (for debugging/auditing)

        Example:
            ```python
            ctx.push(
                "PROCESSED_PAGES",
                pages,
                metadata={"processor_version": "1.2.0", "page_count": len(pages)},
            )
            ```
        """
        return self._backend.push(self._ti, key, value, metadata=metadata)

    def pull(
        self,
        key: str,
        *,
        from_task: Optional[str] = None,
        from_tasks: Optional[Iterable[str]] = None,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a value by key from current or upstream tasks.

        Advanced API that supports pulling from multiple upstream tasks.
        When from_tasks is provided, searches in order and returns first match.
        Use get() for simple cases.

        Args:
            key: The key to retrieve
            from_task: Single upstream task_id to pull from
            from_tasks: Multiple upstream task_ids to search (first match wins)
            default: Value to return if key not found

        Returns:
            The stored value, or default if not found

        Example:
            ```python
            # Pull from first available upstream task
            result = ctx.pull("OCR_RESULT", from_tasks=["ocr_gpu", "ocr_cpu", "ocr_fallback"])
            ```
        """
        if from_tasks is None and from_task is not None:
            from_tasks = [from_task]
        return self._backend.pull(self._ti, key, from_tasks=from_tasks, default=default)

    def __repr__(self) -> str:
        """Debug representation."""
        return f"RunContext(ti={self._ti})"
