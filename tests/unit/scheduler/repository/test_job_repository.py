import asyncio
from datetime import datetime, timezone

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


class RowsCursor:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class RecoveryCursor:
    def __init__(self, claim_rows):
        self.claim_rows = claim_rows
        self.execute_calls: list[tuple[str, object]] = []
        self._fetchone = None
        self.closed = False

    def execute(self, sql: str, params: object = None):
        self.execute_calls.append((sql, params))
        if "claim_expired_run_leases" not in sql:
            self._fetchone = (params[2],)

    def fetchall(self):
        return self.claim_rows

    def fetchone(self):
        row = self._fetchone
        self._fetchone = None
        return row

    def close(self):
        self.closed = True


class RecoveryConnection:
    def __init__(self, claim_rows):
        self.commit_called = False
        self.rollback_called = False
        self.closed = False
        self.cursor_instance = RecoveryCursor(claim_rows)

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


@pytest.mark.asyncio
async def test_lease_jobs_preserves_input_order_for_sql_call():
    connection = SequencedConnection([])
    repository = build_repository(connection)
    execute_calls = []

    def execute_sql(sql, data=None, **_kwargs):
        execute_calls.append((sql, data))
        return RowsCursor([("job-b",), ("job-a",)])

    repository._execute_sql_gracefully = execute_sql

    leased = await repository.lease_jobs(
        ["job-b", "job-a", "job-b"], "owner-1", 120, "extract"
    )

    assert leased == {"job-a", "job-b"}
    assert execute_calls[0][1][0] == ["job-b", "job-a"]


@pytest.mark.asyncio
async def test_extend_run_lease_passes_attempt_identity():
    connection = SequencedConnection([])
    repository = build_repository(connection)
    execute_calls = []

    def execute_sql(sql, data=None, **_kwargs):
        execute_calls.append((sql, data))
        return RowsCursor([("job-1",)])

    repository._execute_sql_gracefully = execute_sql

    extended = await repository.extend_run_lease(
        ["job-1"], "owner-1", "00000000-0000-0000-0000-000000000001", 300
    )

    assert extended == {"job-1"}
    sql, params = execute_calls[0]
    assert "extend_run_lease" in sql
    assert params[1] == "owner-1"
    assert params[2] == "00000000-0000-0000-0000-000000000001"


@pytest.mark.asyncio
async def test_recover_expired_run_leases_applies_python_policy(monkeypatch):
    start_after = datetime(2026, 5, 18, tzinfo=timezone.utc)
    retry_job_id = "00000000-0000-0000-0000-000000000101"
    failed_job_id = "00000000-0000-0000-0000-000000000102"
    attempt_id = "00000000-0000-0000-0000-000000000201"
    connection = RecoveryConnection(
        [
            (
                retry_job_id,
                "extract",
                "00000000-0000-0000-0000-000000000301",
                "active",
                1,
                2,
                5,
                False,
                start_after,
                "owner-1",
                attempt_id,
                start_after,
            ),
            (
                failed_job_id,
                "extract",
                "00000000-0000-0000-0000-000000000302",
                "active",
                2,
                2,
                5,
                False,
                start_after,
                "owner-1",
                attempt_id,
                start_after,
            ),
        ]
    )
    repository = build_repository(connection)
    monkeypatch.setattr(
        JobRepository,
        "_recovery_start_after",
        staticmethod(lambda **_kwargs: start_after),
    )

    recovered = await repository.recover_expired_run_leases()

    assert [row.recovered_state for row in recovered] == ["retry", "failed"]
    assert recovered[0].start_after == start_after
    retry_sql = connection.cursor_instance.execute_calls[1][0]
    failed_sql = connection.cursor_instance.execute_calls[2][0]
    assert "retry_count" not in retry_sql
    assert "run_owner            = NULL" in retry_sql
    assert "run_owner            = NULL" not in failed_sql
    assert connection.commit_called is True
    assert connection.rollback_called is False
