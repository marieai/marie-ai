"""
Marie State Kernel - State management for DAG task execution.

This package provides an Airflow-inspired state passing system for Marie AI,
enabling tasks within a DAG run to share state via simple key-value operations.

Example:
    ```python
    from marie.kernel import RunContext, TaskInstanceRef
    from marie.kernel.backends.memory import InMemoryStateBackend

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

from marie.kernel.backend import StateBackend
from marie.kernel.context import RunContext
from marie.kernel.factory import create_backend, create_backend_from_url
from marie.kernel.ref import TaskInstanceRef

__version__ = "0.1.0"
__all__ = [
    "TaskInstanceRef",
    "StateBackend",
    "RunContext",
    "create_backend",
    "create_backend_from_url",
]
