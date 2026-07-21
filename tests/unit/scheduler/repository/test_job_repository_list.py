from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.repository import JobRepository
from marie.scheduler.state import WorkState


def job_row(job_id: str, state: str) -> tuple:
    now = datetime.now(timezone.utc)
    return (
        job_id,
        "extract",
        1,
        state,
        2,
        now,
        timedelta(seconds=60),
        {},
        0,
        False,
        now + timedelta(days=1),
        "00000000-0000-0000-0000-000000000010",
        0,
        None,
        None,
        None,
        None,
        None,
    )


class ListConnection:
    def __init__(self, rows=None, error: BaseException | None = None) -> None:
        self.rows = rows or []
        self.error = error
        self.query = ""
        self.params = ()

    async def fetch(self, query, *params):
        self.query = query
        self.params = params
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        return self.rows


class ListPool:
    def __init__(
        self,
        connection: ListConnection,
        acquire_error: BaseException | None = None,
    ) -> None:
        self.connection = connection
        self.acquire_error = acquire_error

    @asynccontextmanager
    async def acquire(self):
        if self.acquire_error is not None:
            raise self.acquire_error
        yield self.connection


def build_repository(
    connection: ListConnection, *, acquire_error=None
) -> JobRepository:
    return JobRepository({}, pool=ListPool(connection, acquire_error))


@pytest.mark.asyncio
async def test_list_jobs_filters_multiple_states_and_limit() -> None:
    connection = ListConnection(
        [
            job_row("00000000-0000-0000-0000-000000000001", "active"),
            job_row("00000000-0000-0000-0000-000000000002", "retry"),
        ]
    )
    repository = build_repository(connection)

    jobs = await repository.list_jobs(
        state=[WorkState.ACTIVE, "RETRY"],
        limit=3,
        fetch_size=2,
    )

    assert [job.state for job in jobs] == [WorkState.ACTIVE, WorkState.RETRY]
    assert "state = ANY(%s::marie_scheduler.job_state[])" in connection.query
    assert connection.params == (["active", "retry"], 3)


@pytest.mark.asyncio
async def test_list_jobs_does_not_block_the_event_loop() -> None:
    repository = build_repository(ListConnection([]))
    other_task_ran = False

    async def mark_progress() -> None:
        nonlocal other_task_ran
        await asyncio.sleep(0)
        other_task_ran = True

    await asyncio.gather(repository.list_jobs(), mark_progress())

    assert other_task_ran


@pytest.mark.asyncio
async def test_list_jobs_preserves_connection_acquisition_error() -> None:
    repository = build_repository(
        ListConnection([]), acquire_error=RuntimeError("pool exhausted")
    )

    with pytest.raises(RuntimeError, match="pool exhausted"):
        await repository.list_jobs()


@pytest.mark.asyncio
async def test_list_jobs_preserves_query_error() -> None:
    repository = build_repository(ListConnection(error=ValueError("query failed")))

    with pytest.raises(ValueError, match="query failed"):
        await repository.list_jobs()


@pytest.mark.asyncio
async def test_list_jobs_rejects_nonpositive_fetch_size() -> None:
    repository = build_repository(ListConnection([]))

    with pytest.raises(ValueError, match="fetch_size"):
        await repository.list_jobs(fetch_size=0)


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
