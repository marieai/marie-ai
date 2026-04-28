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


def build_repository(connection: FailingConnection) -> JobRepository:
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
