from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.services.dag_management_service import DAGManagementService
from marie.scheduler.state import WorkState


class FakeRepository:
    def __init__(self, priorities, hydratable_dags):
        self._priorities = priorities
        self._hydratable_dags = hydratable_dags
        self._closed = []

    async def get_job_priorities(self, job_ids):
        return {jid: self._priorities[jid] for jid in job_ids if jid in self._priorities}

    def _get_connection(self):
        return self

    def cursor(self):
        return self

    def execute(self, query):
        self._query = query

    def fetchall(self):
        return self._hydratable_dags

    def commit(self):
        return None

    def rollback(self):
        return None

    @property
    def closed(self):
        return False

    def _close_cursor(self, cursor):
        self._closed.append(("cursor", cursor is not None))

    def _close_connection(self, conn):
        self._closed.append(("conn", conn is not None))


@pytest.mark.asyncio
async def test_refresh_frontier_priorities_updates_memory_and_hydrates_missing(monkeypatch):
    now = datetime.now(timezone.utc)
    frontier = MemoryFrontier(higher_priority_wins=True, default_lease_ttl=0.25)
    existing = WorkInfo(
        id="job-1",
        dag_id="dag-1",
        name="test",
        priority=1,
        data={"metadata": {"on": "exe://default"}},
        state=WorkState.CREATED,
        retry_limit=0,
        retry_delay=0,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=3600,
        keep_until=now + timedelta(days=1),
        job_level=1,
    )
    await frontier.add_dag(None, [existing])

    repo = FakeRepository(
        priorities={"job-1": 99},
        hydratable_dags=[("dag-1", {}), ("dag-2", {"nodes": []})],
    )
    active_dags = {"dag-1": object()}
    service = DAGManagementService(
        repository=repo,
        frontier=frontier,
        active_dags=active_dags,
    )

    hydrated = []

    async def fake_hydrate_single_dag(dag_id: str) -> bool:
        hydrated.append(dag_id)
        return True

    monkeypatch.setattr(service, "hydrate_single_dag", fake_hydrate_single_dag)

    stats = await service.refresh_frontier_priorities(hydrate_missing_limit=10)

    assert stats == {
        "tracked": 1,
        "fetched": 1,
        "changed": 1,
        "hydrated_missing": 1,
    }
    assert frontier.jobs_by_id["job-1"].priority == 99
    assert hydrated == ["dag-2"]


@pytest.mark.asyncio
async def test_refresh_frontier_priorities_skips_hydration_when_active_dag_limit_is_full(
    monkeypatch,
):
    frontier = MemoryFrontier(higher_priority_wins=True, default_lease_ttl=0.25)
    repo = FakeRepository(
        priorities={},
        hydratable_dags=[("dag-2", {"nodes": []}), ("dag-3", {"nodes": []})],
    )
    service = DAGManagementService(
        repository=repo,
        frontier=frontier,
        active_dags={"dag-1": object()},
        max_active_dags=1,
    )

    hydrated = []

    async def fake_hydrate_single_dag(dag_id: str) -> bool:
        hydrated.append(dag_id)
        return True

    monkeypatch.setattr(service, "hydrate_single_dag", fake_hydrate_single_dag)

    stats = await service.refresh_frontier_priorities(hydrate_missing_limit=10)

    assert stats == {
        "tracked": 0,
        "fetched": 0,
        "changed": 0,
        "hydrated_missing": 0,
    }
    assert hydrated == []


@pytest.mark.asyncio
async def test_handle_state_change_treats_active_as_live_state():
    frontier = MemoryFrontier(higher_priority_wins=True, default_lease_ttl=0.25)
    repo = FakeRepository(priorities={}, hydratable_dags=[])
    service = DAGManagementService(
        repository=repo,
        frontier=frontier,
        active_dags={},
    )
    service.logger.warning = MagicMock()

    await service.handle_state_change(
        {"op": "UPDATE", "dag_id": "dag-active", "state": "active"}
    )

    service.logger.warning.assert_called_once_with(
        "DAG dag-active is in 'active' state but not in active_dags. "
        "It will be hydrated on next scheduler cycle."
    )
