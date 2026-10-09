from datetime import datetime, timedelta, timezone, tzinfo

import pytest

import marie.scheduler.memory_frontier as frontier_module
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState


@pytest.fixture
def frontier_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MemoryFrontier, list[datetime]]:
    now = [datetime(2026, 10, 9, tzinfo=timezone.utc)]

    class ClockDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return now[0].astimezone(tz) if tz else now[0].replace(tzinfo=None)

    monkeypatch.setattr(frontier_module, 'datetime', ClockDatetime)
    frontier = MemoryFrontier()
    frontier._now = lambda: now[0].timestamp()
    return frontier, now


def job(job_id: str, now: datetime, *, priority: int = 1) -> WorkInfo:
    return WorkInfo(
        id=job_id,
        dag_id='dag-1',
        name='extract',
        priority=priority,
        data={'metadata': {'on': 'extract://default'}},
        state=WorkState.CREATED,
        retry_limit=3,
        retry_delay=2,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=3600,
        keep_until=now + timedelta(days=1),
    )


@pytest.mark.parametrize('reader', ['peek_ready', 'capture_ready', 'select_ready'])
async def test_early_selection_preserves_delayed_retry_visibility(
    frontier_clock: tuple[MemoryFrontier, list[datetime]], reader: str
) -> None:
    frontier, now = frontier_clock
    retry = job('retry', now[0], priority=2)
    runnable = job('runnable', now[0])
    await frontier.add_dag(None, [retry, runnable])
    await frontier.take(['retry'], lease_ttl=10)
    await frontier.on_job_retry('retry', retry)

    assert [wi.id for wi in await frontier.select_ready(1)] == ['runnable']
    assert await frontier.select_ready(1) == []
    now[0] += timedelta(seconds=1)
    assert await frontier.select_ready(1) == []

    now[0] += timedelta(seconds=1)
    if reader == 'capture_ready':
        visible = (await frontier.capture_ready(1, {'extract': 1})).jobs
    else:
        visible = await getattr(frontier, reader)(1)

    assert [wi.id for wi in visible] == ['retry']
    if reader != 'select_ready':
        assert [wi.id for wi in await frontier.select_ready(1)] == ['retry']
    assert await frontier.select_ready(1) == []


@pytest.mark.parametrize('reap', [True, False])
async def test_leased_entry_survives_selection_until_expiry(
    frontier_clock: tuple[MemoryFrontier, list[datetime]], reap: bool
) -> None:
    frontier, now = frontier_clock
    leased = job('leased', now[0])
    await frontier.add_dag(None, [leased])
    await frontier.mark_leased('leased', ttl_s=2)

    assert await frontier.select_ready(1) == []
    assert await frontier.select_ready(1) == []
    now[0] += timedelta(seconds=2)
    if reap:
        assert await frontier.reap_expired_soft_leases() == 1

    assert [wi.id for wi in await frontier.peek_ready(2)] == ['leased']
    capture = await frontier.capture_ready(2, {'extract': 1})
    assert [wi.id for wi in capture.jobs] == ['leased']
    assert capture.eligible_by_executor == {'extract': 1}
    assert [wi.id for wi in await frontier.select_ready(2)] == ['leased']
    assert await frontier.select_ready(2) == []


@pytest.mark.parametrize('state', [WorkState.ACTIVE, WorkState.COMPLETED, WorkState.FAILED])
async def test_selection_does_not_restore_unschedulable_entries(
    frontier_clock: tuple[MemoryFrontier, list[datetime]], state: WorkState
) -> None:
    frontier, now = frontier_clock
    tracked = job('tracked', now[0])
    await frontier.add_dag(None, [tracked])
    await frontier.update_job_state('tracked', state)

    assert await frontier.select_ready(1) == []
    assert frontier._ready_heap == []
    assert await frontier.peek_ready(1) == []
