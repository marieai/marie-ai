"""
Marie State Kernel - State management for DAG task execution.

This package provides an Airflow-inspired state passing system for Marie AI,
enabling tasks within a DAG run to share state via simple key-value operations.

Example:
    ```python
    from marie_kernel import RunContext, TaskInstanceRef
    from marie_kernel.backends.memory import InMemoryStateBackend

    # Create task instance reference
    ti = TaskInstanceRef(
        tenant_id="default",
        dag_name="my_dag",
        dag_id="run_123",
        task_id="extract_text",
        try_number=1,
    )

    # Create context with backend
    backend = InMemoryStateBackend()
    ctx = RunContext(ti, backend)

    # Use in task
    ctx.set("extracted_text", "Hello World")
    text = ctx.get("extracted_text")
    ```
"""

from marie_kernel.backend import StateBackend
from marie_kernel.context import RunContext
from marie_kernel.factory import create_backend, create_backend_from_url
from marie_kernel.ref import TaskInstanceRef

__version__ = "0.1.0"
__all__ = [
    "TaskInstanceRef",
    "StateBackend",
    "RunContext",
    "create_backend",
    "create_backend_from_url",
]
