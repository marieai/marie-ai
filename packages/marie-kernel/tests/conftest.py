"""
Pytest configuration and fixtures for marie-kernel tests.
"""

import pytest

from marie_kernel import RunContext, TaskInstanceRef
from marie_kernel.backends.memory import InMemoryStateBackend


@pytest.fixture
def backend():
    """Create a fresh InMemoryStateBackend for each test."""
    return InMemoryStateBackend()


@pytest.fixture
def task_ref():
    """Create a standard TaskInstanceRef for testing."""
    return TaskInstanceRef(
        tenant_id="test_tenant",
        dag_id="test_dag",
        dag_run_id="run_001",
        task_id="task_1",
        try_number=1,
    )


@pytest.fixture
def upstream_task_ref():
    """Create an upstream TaskInstanceRef for cross-task tests."""
    return TaskInstanceRef(
        tenant_id="test_tenant",
        dag_id="test_dag",
        dag_run_id="run_001",
        task_id="upstream_task",
        try_number=1,
    )


@pytest.fixture
def run_context(task_ref, backend):
    """Create a RunContext with the standard task ref and backend."""
    return RunContext(task_ref, backend)
