"""
Tests for concurrent access to state backends.

These tests verify thread-safety of backend implementations.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from marie_kernel import RunContext, TaskInstanceRef
from marie_kernel.backends.memory import InMemoryStateBackend


@pytest.fixture
def backend():
    """Create a fresh InMemoryStateBackend for each test."""
    return InMemoryStateBackend()


@pytest.fixture
def base_task_ref():
    """Create a base TaskInstanceRef for generating test refs."""
    return TaskInstanceRef(
        tenant_id="test",
        dag_name="dag",
        dag_id="run",
        task_id="task",
        try_number=1,
    )


class TestConcurrentPush:
    """Tests for concurrent push operations."""

    def test_concurrent_pushes_to_same_key(self, backend, base_task_ref):
        """Test concurrent pushes to the same key (last write wins)."""
        num_threads = 10
        results = []

        def push_value(thread_id):
            backend.push(base_task_ref, "KEY", f"value_{thread_id}")
            return thread_id

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(push_value, i) for i in range(num_threads)]
            for future in as_completed(futures):
                results.append(future.result())

        # All threads should complete without error
        assert len(results) == num_threads

        # Value should be from one of the threads
        final_value = backend.pull(base_task_ref, "KEY")
        assert final_value.startswith("value_")

    def test_concurrent_pushes_to_different_keys(self, backend, base_task_ref):
        """Test concurrent pushes to different keys (no interference)."""
        num_threads = 20

        def push_value(thread_id):
            key = f"KEY_{thread_id}"
            value = f"value_{thread_id}"
            backend.push(base_task_ref, key, value)
            return (key, value)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(push_value, i) for i in range(num_threads)]
            expected = {future.result() for future in as_completed(futures)}

        # All values should be correctly stored
        for key, expected_value in expected:
            actual_value = backend.pull(base_task_ref, key)
            assert actual_value == expected_value

    def test_concurrent_pushes_different_tasks(self, backend):
        """Test concurrent pushes from different tasks."""
        num_threads = 20

        def push_value(thread_id):
            ti = TaskInstanceRef(
                tenant_id="test",
                dag_name="dag",
                dag_id="run",
                task_id=f"task_{thread_id}",
                try_number=1,
            )
            backend.push(ti, "RESULT", {"thread": thread_id})
            return ti

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(push_value, i) for i in range(num_threads)]
            task_refs = [future.result() for future in as_completed(futures)]

        # Each task should have its own value
        for ti in task_refs:
            result = backend.pull(ti, "RESULT")
            thread_id = int(ti.task_id.split("_")[1])
            assert result == {"thread": thread_id}


class TestConcurrentPull:
    """Tests for concurrent pull operations."""

    def test_concurrent_pulls_same_key(self, backend, base_task_ref):
        """Test concurrent pulls from the same key."""
        # Setup
        backend.push(base_task_ref, "SHARED_KEY", {"data": 42})

        num_threads = 20
        results = []

        def pull_value(thread_id):
            return backend.pull(base_task_ref, "SHARED_KEY")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(pull_value, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # All pulls should return the same value
        assert all(r == {"data": 42} for r in results)

    def test_concurrent_pulls_from_different_tasks(self, backend):
        """Test concurrent pulls from different upstream tasks."""
        # Setup: multiple upstream tasks with state
        for i in range(5):
            ti = TaskInstanceRef(
                tenant_id="test",
                dag_name="dag",
                dag_id="run",
                task_id=f"upstream_{i}",
                try_number=1,
            )
            backend.push(ti, "OUTPUT", f"result_{i}")

        # Downstream task pulls from multiple upstreams concurrently
        downstream = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="downstream",
            try_number=1,
        )

        def pull_from_upstream(upstream_id):
            return backend.pull(
                downstream, "OUTPUT", from_tasks=[f"upstream_{upstream_id}"]
            )

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(pull_from_upstream, i): i for i in range(5)}
            results = {futures[f]: f.result() for f in as_completed(futures)}

        # Each pull should return the correct upstream's value
        for i, result in results.items():
            assert result == f"result_{i}"


class TestConcurrentPushPull:
    """Tests for concurrent push and pull operations."""

    def test_push_while_pulling(self, backend, base_task_ref):
        """Test that pulls during pushes return consistent values."""
        # Initial value
        backend.push(base_task_ref, "KEY", "initial")

        num_operations = 50
        results = {"push": [], "pull": []}
        errors = []

        def do_push(op_id):
            try:
                backend.push(base_task_ref, "KEY", f"pushed_{op_id}")
                results["push"].append(op_id)
            except Exception as e:
                errors.append(e)

        def do_pull(op_id):
            try:
                value = backend.pull(base_task_ref, "KEY")
                results["pull"].append((op_id, value))
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(num_operations):
                if i % 2 == 0:
                    futures.append(executor.submit(do_push, i))
                else:
                    futures.append(executor.submit(do_pull, i))

            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # No errors should have occurred
        assert len(errors) == 0

        # All operations should have completed
        assert len(results["push"]) == num_operations // 2
        assert len(results["pull"]) == num_operations // 2

        # All pulled values should be valid (either initial or a pushed value)
        for op_id, value in results["pull"]:
            assert value == "initial" or value.startswith("pushed_")


class TestConcurrentClear:
    """Tests for concurrent clear operations."""

    def test_clear_during_operations(self, backend):
        """Test that clear works correctly alongside push/pull."""
        num_tasks = 5
        task_refs = [
            TaskInstanceRef(
                tenant_id="test",
                dag_name="dag",
                dag_id="run",
                task_id=f"task_{i}",
                try_number=1,
            )
            for i in range(num_tasks)
        ]

        # Pre-populate
        for ti in task_refs:
            backend.push(ti, "DATA", "initial")

        errors = []
        cleared = []

        def push_to_task(ti, value):
            try:
                backend.push(ti, "DATA", value)
            except Exception as e:
                errors.append(e)

        def clear_task(ti):
            try:
                backend.clear_for_task(ti)
                cleared.append(ti.task_id)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i, ti in enumerate(task_refs):
                # Some push, some clear
                if i % 2 == 0:
                    futures.append(executor.submit(push_to_task, ti, f"new_{i}"))
                else:
                    futures.append(executor.submit(clear_task, ti))

            for future in as_completed(futures):
                future.result()

        # No errors
        assert len(errors) == 0

        # Cleared tasks should have no data
        for ti in task_refs:
            if ti.task_id in cleared:
                assert backend.pull(ti, "DATA") is None


class TestConcurrentRunContext:
    """Tests for concurrent RunContext operations."""

    def test_multiple_contexts_same_backend(self, backend):
        """Test multiple RunContexts sharing a backend."""
        num_contexts = 10

        def task_work(task_id):
            ti = TaskInstanceRef(
                tenant_id="test",
                dag_name="dag",
                dag_id="run",
                task_id=f"task_{task_id}",
                try_number=1,
            )
            ctx = RunContext(ti, backend)

            # Write some state
            ctx.set("TASK_ID", task_id)
            ctx.set("RESULT", {"computed": task_id * 2})

            # Read it back
            read_id = ctx.get("TASK_ID")
            read_result = ctx.get("RESULT")

            return (task_id, read_id, read_result)

        with ThreadPoolExecutor(max_workers=num_contexts) as executor:
            futures = [executor.submit(task_work, i) for i in range(num_contexts)]
            results = [future.result() for future in as_completed(futures)]

        # Each context should see its own state
        for task_id, read_id, read_result in results:
            assert read_id == task_id
            assert read_result == {"computed": task_id * 2}

    def test_fan_in_pattern(self, backend):
        """Test fan-in pattern: multiple upstream tasks, one downstream."""
        num_upstream = 5

        # Upstream tasks write their results
        def upstream_task(task_id):
            ti = TaskInstanceRef(
                tenant_id="test",
                dag_name="dag",
                dag_id="run",
                task_id=f"upstream_{task_id}",
                try_number=1,
            )
            ctx = RunContext(ti, backend)
            ctx.set("PARTIAL_RESULT", {"part": task_id, "data": list(range(task_id))})
            return ti.task_id

        with ThreadPoolExecutor(max_workers=num_upstream) as executor:
            futures = [executor.submit(upstream_task, i) for i in range(num_upstream)]
            upstream_task_ids = [future.result() for future in as_completed(futures)]

        # Downstream task reads all upstream results
        downstream_ti = TaskInstanceRef(
            tenant_id="test",
            dag_name="dag",
            dag_id="run",
            task_id="downstream",
            try_number=1,
        )
        downstream_ctx = RunContext(downstream_ti, backend)

        # Read from all upstreams
        collected = []
        for task_id in upstream_task_ids:
            result = downstream_ctx.get("PARTIAL_RESULT", from_task=task_id)
            collected.append(result)

        # All results should be collected
        assert len(collected) == num_upstream
        assert all(r is not None for r in collected)
        parts = sorted([r["part"] for r in collected])
        assert parts == list(range(num_upstream))
