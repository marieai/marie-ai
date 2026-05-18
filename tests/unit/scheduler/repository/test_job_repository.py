import asyncio

import pytest

from marie.scheduler.repository.job_repository import JobRepository


class FakeLogger:
    def error(self, *args, **kwargs):
        pass


class FailingCursor:
    def __init__(self, error: Exception):
        self.error = error
        self.closed = False

    def execute(self, *_args, **_kwargs):
        raise self.error

    def close(self):
        self.closed = True


class FailingConnection:
    def __init__(self, error: Exception):
        self.error = error
        self.rollback_called = False
        self.closed = False
        self.cursor_instance = FailingCursor(error)

    def cursor(self):
        return self.cursor_instance

    def rollback(self):
        self.rollback_called = True

    def close(self):
        self.closed = True


class SequencedCursor:
    def __init__(self, results: list[list[tuple[object, ...]]]):
        self.results = results
        self.execute_calls: list[tuple[str, object]] = []
        self.result_index = -1
        self.closed = False

    def execute(self, sql: str, params: object = None):
        self.execute_calls.append((sql, params))
        self.result_index += 1

    def fetchall(self):
        return self.results[self.result_index]

    def close(self):
        self.closed = True


class SequencedConnection:
    def __init__(self, results: list[list[tuple[object, ...]]]):
        self.commit_called = False
        self.rollback_called = False
        self.closed = False
        self.cursor_instance = SequencedCursor(results)

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commit_called = True

    def rollback(self):
        self.rollback_called = True

    def close(self):
        self.closed = True


def build_repository(connection) -> JobRepository:
    repository = object.__new__(JobRepository)
    repository.logger = FakeLogger()
    repository._loop = asyncio.get_running_loop()
    repository._db_executor = None
    repository._get_connection = lambda: connection
    repository._close_cursor = lambda cursor: cursor.close() if cursor else None
    repository._close_connection = lambda conn: conn.close() if conn else None
    return repository


@pytest.mark.asyncio
async def test_resolve_dag_state_raises_on_db_error():
    connection = FailingConnection(RuntimeError("resolve failure"))
    repository = build_repository(connection)

    with pytest.raises(RuntimeError, match="resolve failure"):
        await repository.resolve_dag_state("dag-1")

    assert connection.rollback_called is True


@pytest.mark.asyncio
async def test_get_active_dag_ids_raises_on_db_error():
    connection = FailingConnection(RuntimeError("validation failure"))
    repository = build_repository(connection)

    with pytest.raises(RuntimeError, match="validation failure"):
        await repository.get_active_dag_ids(["dag-1"])

    assert connection.rollback_called is True


@pytest.mark.asyncio
async def test_activate_from_lease_reads_attempts_after_activation():
    connection = SequencedConnection(
        [
            [("job-1",)],
            [("job-1", "attempt-1")],
        ]
    )
    repository = build_repository(connection)

    attempts = await repository.activate_from_lease(["job-1"], "owner-1", 300)

    assert attempts == {"job-1": "attempt-1"}
    assert len(connection.cursor_instance.execute_calls) == 2
    assert "activate_from_lease" in connection.cursor_instance.execute_calls[0][0]
    assert "run_attempt_id" in connection.cursor_instance.execute_calls[1][0]
    assert connection.commit_called is True
    assert connection.rollback_called is False


@pytest.mark.asyncio
async def test_activate_from_lease_rejects_missing_attempt_readback():
    connection = SequencedConnection(
        [
            [("job-1",)],
            [("job-1", None)],
        ]
    )
    repository = build_repository(connection)

    with pytest.raises(RuntimeError, match="without run_attempt_id"):
        await repository.activate_from_lease(["job-1"], "owner-1", 300)

    assert connection.rollback_called is True
