from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from marie.scheduler.psql import PostgreSQLJobScheduler


@pytest.mark.asyncio
async def test_get_job_returns_active_projection_without_database_read() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    work_item = SimpleNamespace(id='job-1')
    scheduler._job_cache = {work_item.id: work_item}
    scheduler.repository = SimpleNamespace(get_job_by_id=AsyncMock())

    assert await scheduler.get_job(work_item.id) is work_item
    scheduler.repository.get_job_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_job_adds_database_result_to_active_projection() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    work_item = SimpleNamespace(id='job-1')
    scheduler._job_cache = {}
    scheduler.repository = SimpleNamespace(
        get_job_by_id=AsyncMock(return_value=work_item)
    )

    assert await scheduler.get_job(work_item.id) is work_item
    assert scheduler._job_cache == {work_item.id: work_item}
    scheduler.repository.get_job_by_id.assert_awaited_once_with(work_item.id)
