"""
TaskInstanceRef - Identity of a task instance within a DAG run.

This dataclass serves as the namespace key for all state operations,
uniquely identifying a task execution attempt.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TaskInstanceRef:
    """
    Identity of a task instance within a DAG run.

    This immutable dataclass uniquely identifies a task execution attempt
    and is used as the namespace for all state operations.

    Attributes:
        tenant_id: Tenant identifier for multi-tenant isolation
        dag_name: DAG name/type grouping related tasks (e.g., "document_processing")
        dag_id: Unique identifier for this DAG execution run (the actual run ID)
        task_id: Unique identifier for the task within the DAG
        try_number: Retry attempt number (1-indexed)

    Example:
        ```python
        ti = TaskInstanceRef(
            tenant_id="acme_corp",
            dag_name="document_processing",
            dag_id="run_2024_001",
            task_id="extract_text",
            try_number=1,
        )
        ```
    """

    tenant_id: str
    dag_name: str
    dag_id: str
    task_id: str
    try_number: int = 1

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], tenant_id: str = "default"
    ) -> "TaskInstanceRef":
        """
        Create TaskInstanceRef from a dictionary.

        Useful for constructing from WorkInfo.data or similar dict sources.

        Args:
            data: Dictionary containing task instance fields
            tenant_id: Default tenant ID if not in data

        Returns:
            TaskInstanceRef instance

        Example:
            ```python
            ti = TaskInstanceRef.from_dict(
                {
                    "dag_name": "my_dag",
                    "dag_id": "run_123",
                    "task_id": "task_1",
                    "try_number": 1,
                }
            )
            ```
        """
        dag_name = data.get("dag_name", "")
        dag_id = data.get("dag_id", dag_name)

        return cls(
            tenant_id=data.get("tenant_id", tenant_id),
            dag_name=dag_name,
            dag_id=dag_id,
            task_id=data.get("task_id", data.get("id", "")),
            try_number=data.get("try_number", 1),
        )

    @classmethod
    def from_work_info(
        cls, work_info: Any, tenant_id: str = "default"
    ) -> "TaskInstanceRef":
        """
        Create TaskInstanceRef from a WorkInfo object.

        Uses Any type to avoid importing from marie-ai, enabling
        marie-kernel to be used independently.

        Args:
            work_info: WorkInfo object with id, dag_id, and data attributes
            tenant_id: Default tenant ID if not in work_info.data

        Returns:
            TaskInstanceRef instance

        Example:
            ```python
            # In marie-ai scheduler/worker
            ti = TaskInstanceRef.from_work_info(job)
            ```
        """
        data = work_info.data if work_info.data else {}
        # work_info.dag_id in marie-ai is the dag name/type
        dag_name = work_info.dag_id or ""
        # The actual run ID comes from data, falling back to dag_name
        dag_id = data.get("dag_id", dag_name)

        return cls(
            tenant_id=data.get("tenant_id", tenant_id),
            dag_name=dag_name,
            dag_id=dag_id,
            task_id=work_info.id,
            try_number=data.get("try_number", 1),
        )

    def with_try_number(self, try_number: int) -> "TaskInstanceRef":
        """
        Return a new TaskInstanceRef with an updated try_number.

        Since TaskInstanceRef is frozen (immutable), this creates a new instance.

        Args:
            try_number: New try number

        Returns:
            New TaskInstanceRef with updated try_number

        Example:
            ```python
            ti = TaskInstanceRef(...)
            ti_retry = ti.with_try_number(2)
            ```
        """
        return TaskInstanceRef(
            tenant_id=self.tenant_id,
            dag_name=self.dag_name,
            dag_id=self.dag_id,
            task_id=self.task_id,
            try_number=try_number,
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"TaskInstanceRef(tenant={self.tenant_id}, dag_name={self.dag_name}, "
            f"dag_id={self.dag_id}, task={self.task_id}, try={self.try_number})"
        )
