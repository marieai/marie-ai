import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.repository import JobRepository
from marie.scheduler.state import WorkState


class Cursor:
    def __init__(self, batches: list[list[tuple]]) -> None:
        self.batches = list(batches)
        self.closed = False
        self.itersize = 0
        self.fetch_sizes: list[int] = []
        self.query = ""
        self.params: list[object] = []

    def execute(self, query: str, params: list[object]) -> None:
        self.query = query
        self.params = params

    def fetchmany(self, size: int) -> list[tuple]:
        self.fetch_sizes.append(size)
        return self.batches.pop(0)

    def close(self) -> None:
        self.closed = True


class Connection:
    def __init__(self, cursor: Cursor) -> None:
        self._cursor = cursor
        self.cursor_name = None
        self.committed = False
        self.rollback_calls = 0

    def cursor(self, name: str) -> Cursor:
        self.cursor_name = name
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rollback_calls += 1


def build_repository(
    executor: ThreadPoolExecutor, get_connection: Callable[[], Connection]
) -> JobRepository:
    repository = object.__new__(JobRepository)
    repository._loop = asyncio.get_running_loop()
    repository._db_executor = executor
    repository.logger = MagicMock()
    repository._get_connection = get_connection
    repository._close_cursor = MagicMock(
        side_effect=lambda cursor: cursor.close() if cursor else None
    )
    repository._close_connection = MagicMock()
    repository._record_to_work_info = lambda record: SimpleNamespace(id=str(record[0]))
    return repository


@pytest.mark.asyncio
async def test_list_jobs_fetches_multiple_states_in_bounded_batches() -> None:
    cursor = Cursor([[("job-1",), ("job-2",)], [("job-3",)], []])
    connection = Connection(cursor)

    with ThreadPoolExecutor(max_workers=1) as executor:
        repository = build_repository(executor, lambda: connection)
        jobs = await repository.list_jobs(
            state=[WorkState.ACTIVE, "RETRY"],
            limit=3,
            fetch_size=2,
        )

    assert [job.id for job in jobs] == ["job-1", "job-2", "job-3"]
    assert connection.cursor_name == "job_list_iterator"
    assert cursor.itersize == 2
    assert cursor.fetch_sizes == [2, 2, 2]
    assert "state = ANY(%s::marie_scheduler.job_state[])" in cursor.query
    assert "LIMIT %s" in cursor.query
    assert cursor.params == [["active", "retry"], 3]
    assert connection.committed


@pytest.mark.asyncio
async def test_list_jobs_does_not_block_the_event_loop() -> None:
    cursor = Cursor([[]])
    connection = Connection(cursor)
    entered = threading.Event()
    release = threading.Event()

    def get_connection() -> Connection:
        entered.set()
        release.wait(timeout=1.0)
        return connection

    timer = threading.Timer(0.2, release.set)
    timer.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            repository = build_repository(executor, get_connection)
            list_task = asyncio.create_task(repository.list_jobs())
            while not entered.is_set():
                await asyncio.sleep(0)
            await asyncio.sleep(0.01)
            assert not release.is_set()
            release.set()
            await list_task
    finally:
        timer.cancel()


@pytest.mark.asyncio
async def test_list_jobs_preserves_connection_acquisition_error() -> None:
    def fail_to_connect() -> Connection:
        raise RuntimeError("pool unavailable")

    with ThreadPoolExecutor(max_workers=1) as executor:
        repository = build_repository(executor, fail_to_connect)
        with pytest.raises(RuntimeError, match="pool unavailable"):
            await repository.list_jobs()

    repository._close_connection.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_list_jobs_preserves_query_error_when_rollback_fails() -> None:
    cursor = Cursor([])
    cursor.execute = MagicMock(side_effect=ValueError("query failed"))
    connection = Connection(cursor)
    connection.rollback = MagicMock(side_effect=RuntimeError("rollback failed"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        repository = build_repository(executor, lambda: connection)
        with pytest.raises(ValueError, match="query failed"):
            await repository.list_jobs()

    connection.rollback.assert_called_once_with()
    repository.logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_scheduler_list_jobs_delegates_to_repository() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.repository = SimpleNamespace(
        list_jobs=AsyncMock(
            return_value=[SimpleNamespace(id="job-1"), SimpleNamespace(id="job-2")]
        )
    )

    jobs = await scheduler.list_jobs(state=["ACTIVE", "retry"], batch_size=25)

    scheduler.repository.list_jobs.assert_awaited_once_with(
        state=["active", "retry"], limit=25
    )
    assert list(jobs) == ["job-1", "job-2"]


@pytest.mark.asyncio
async def test_scheduler_list_jobs_rejects_invalid_state() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.repository = SimpleNamespace(list_jobs=AsyncMock())

    with pytest.raises(ValueError, match="Invalid state.*unknown"):
        await scheduler.list_jobs(state="unknown")

    scheduler.repository.list_jobs.assert_not_awaited()
