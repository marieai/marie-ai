from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from marie.query_planner.base import QueryPlan
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.services.dag_management_service import DAGManagementService
from marie.scheduler.state import WorkState


class FakeRepository:
    def __init__(
        self,
        priorities,
        hydratable_dags,
        hydratable_jobs=None,
        mark_dag_active_result=True,
    ):
        self._priorities = priorities
        self._hydratable_dags = hydratable_dags
        self._hydratable_jobs = hydratable_jobs or {}
        self._mark_dag_active_result = mark_dag_active_result
        self.marked_active_dags = []
        self._closed = []
        self._rows = []

    async def get_job_priorities(self, job_ids):
        return {jid: self._priorities[jid] for jid in job_ids if jid in self._priorities}

    async def mark_dag_as_active(self, dag_id):
        self.marked_active_dags.append(dag_id)
        return self._mark_dag_active_result

    async def load_dag_and_jobs(self, dag_id):
        for candidate_dag_id, serialized_dag in self._hydratable_dags:
            if str(candidate_dag_id) == str(dag_id):
                rows = [
                    (dag_id, job)
                    for job in self._hydratable_jobs.get(str(dag_id), [])
                ]
                return serialized_dag, rows
        return None, []

    def _get_connection(self):
        return self

    def cursor(self, *args, **kwargs):
        return self

    def execute(self, query, params=None):
        self._query = query
        if "hydrate_frontier_dags" in query:
            self._rows = list(self._hydratable_dags)
            return
        if "hydrate_frontier_jobs" in query:
            dag_ids = [str(dag_id) for dag_id in (params[0] if params else [])]
            rows = []
            for dag_id in dag_ids:
                for job in self._hydratable_jobs.get(dag_id, []):
                    rows.append((dag_id, job))
            self._rows = rows
            return
        self._rows = []

    def fetchall(self):
        return self._hydratable_dags

    def __iter__(self):
        return iter(self._rows)

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


def make_wi(
    job_id: str,
    dag_id: str,
    entrypoint: str,
    *,
    deps=None,
    state=WorkState.CREATED,
    level: int = 0,
    soft_sla=None,
    hard_sla=None,
):
    now = datetime.now(timezone.utc)
    wi = WorkInfo(
        id=job_id,
        dag_id=dag_id,
        name=job_id,
        priority=1,
        data={"metadata": {"on": entrypoint}},
        state=state,
        retry_limit=0,
        retry_delay=0,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=3600,
        keep_until=now + timedelta(days=1),
        job_level=level,
        soft_sla=soft_sla,
        hard_sla=hard_sla,
    )
    wi.dependencies = list(deps or [])
    return wi


def serialize_wi(wi: WorkInfo) -> dict:
    return {
        "id": wi.id,
        "name": wi.name,
        "priority": wi.priority,
        "state": wi.state.value if isinstance(wi.state, WorkState) else wi.state,
        "retry_limit": wi.retry_limit,
        "start_after": wi.start_after,
        "data": wi.data,
        "retry_delay": wi.retry_delay,
        "retry_backoff": wi.retry_backoff,
        "keep_until": wi.keep_until,
        "job_level": wi.job_level,
        "soft_sla": wi.soft_sla,
        "hard_sla": wi.hard_sla,
        "dependencies": list(wi.dependencies or []),
    }


def make_service(slots, *, max_active_dags=32, repo=None):
    frontier = MemoryFrontier(higher_priority_wins=True, default_lease_ttl=0.25)
    repository = repo or FakeRepository(priorities={}, hydratable_dags=[])
    active_dags = {}
    service = DAGManagementService(
        repository=repository,
        frontier=frontier,
        active_dags=active_dags,
        max_active_dags=max_active_dags,
        slot_snapshot_provider=lambda: slots,
    )
    return service, frontier, active_dags


@pytest.mark.asyncio
async def test_hydrate_single_dag_preserves_sla_fields():
    now = datetime.now(timezone.utc)
    soft_sla = now + timedelta(seconds=15)
    hard_sla = now + timedelta(seconds=45)
    dag_id = "dag-sla-single"
    job = make_wi(
        "job-sla-single",
        dag_id,
        "annotator_llm://default",
        soft_sla=soft_sla,
        hard_sla=hard_sla,
    )
    repo = FakeRepository(
        priorities={},
        hydratable_dags=[(dag_id, {"nodes": []})],
        hydratable_jobs={dag_id: [serialize_wi(job)]},
    )
    service, frontier, active_dags = make_service({"annotator_llm": 1}, repo=repo)

    hydrated = await service.hydrate_single_dag(dag_id)

    assert hydrated is True
    assert dag_id in active_dags
    assert frontier.jobs_by_id["job-sla-single"].soft_sla == soft_sla
    assert frontier.jobs_by_id["job-sla-single"].hard_sla == hard_sla


@pytest.mark.asyncio
async def test_hydrate_bulk_preserves_sla_fields():
    now = datetime.now(timezone.utc)
    soft_sla = now + timedelta(seconds=15)
    hard_sla = now + timedelta(seconds=45)
    dag_id = "dag-sla-bulk"
    job = make_wi(
        "job-sla-bulk",
        dag_id,
        "annotator_llm://default",
        soft_sla=soft_sla,
        hard_sla=hard_sla,
    )
    repo = FakeRepository(
        priorities={},
        hydratable_dags=[(dag_id, {"nodes": []})],
        hydratable_jobs={dag_id: [serialize_wi(job)]},
    )
    service, frontier, active_dags = make_service({"annotator_llm": 1}, repo=repo)

    await service.hydrate_bulk(dag_batch_size=10, itersize=10, log_every_seconds=60.0)

    assert dag_id in active_dags
    assert frontier.jobs_by_id["job-sla-bulk"].soft_sla == soft_sla
    assert frontier.jobs_by_id["job-sla-bulk"].hard_sla == hard_sla


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
async def test_refresh_frontier_priorities_skips_dags_already_in_frontier(monkeypatch):
    frontier = MemoryFrontier(higher_priority_wins=True, default_lease_ttl=0.25)
    await frontier.add_dag(
        None,
        [make_wi("job-frontier", "dag-frontier", "exe://default")],
    )
    repo = FakeRepository(
        priorities={},
        hydratable_dags=[
            ("dag-frontier", {"nodes": []}),
            ("dag-missing", {"nodes": []}),
        ],
    )
    service = DAGManagementService(
        repository=repo,
        frontier=frontier,
        active_dags={},
    )

    hydrated = []

    async def fake_hydrate_single_dag(dag_id: str) -> bool:
        hydrated.append(dag_id)
        return True

    monkeypatch.setattr(service, "hydrate_single_dag", fake_hydrate_single_dag)

    stats = await service.refresh_frontier_priorities(hydrate_missing_limit=10)

    assert stats["hydrated_missing"] == 1
    assert hydrated == ["dag-missing"]


@pytest.mark.asyncio
async def test_refresh_frontier_priorities_updates_frontier_tracked_job_priority(
    monkeypatch,
):
    frontier = MemoryFrontier(higher_priority_wins=True, default_lease_ttl=0.25)
    await frontier.add_dag(
        None,
        [make_wi("job-frontier", "dag-frontier", "exe://default")],
    )
    repo = FakeRepository(
        priorities={"job-frontier": 42},
        hydratable_dags=[("dag-frontier", {"nodes": []})],
    )
    service = DAGManagementService(
        repository=repo,
        frontier=frontier,
        active_dags={},
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
        "hydrated_missing": 0,
    }
    assert frontier.jobs_by_id["job-frontier"].priority == 42
    assert hydrated == []


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
    service.logger.debug = MagicMock()
    service.logger.warning = MagicMock()

    await service.handle_state_change(
        {"op": "UPDATE", "dag_id": "dag-active", "state": "active"}
    )

    service.logger.debug.assert_called_once_with(
        "DAG dag-active is 'active' in DB but is not local to this scheduler yet; "
        "it may be admitted by the current cycle, owned by another scheduler, "
        "or hydrated later."
    )
    service.logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_hydrated_dag_with_only_unavailable_mock_ready_work_is_not_admitted():
    repo = FakeRepository(priorities={}, hydratable_dags=[])
    service, frontier, active_dags = make_service(
        {"annotator_llm": 1, "annotator_parser": 1}, repo=repo
    )

    dag_id = "dag-mock"
    nodes = [
        make_wi("job-1", dag_id, "mock_executor_a://document/process", deps=[]),
        make_wi("job-2", dag_id, "mock_executor_b://document/process", deps=["job-1"]),
    ]

    admitted, reason = await service._admit_hydrated_dag(
        dag_id, QueryPlan(nodes=[]), nodes, source="hydrate_bulk"
    )

    assert admitted is False
    assert reason == "executor_capacity"
    assert repo.marked_active_dags == []
    assert dag_id not in active_dags
    assert dag_id not in frontier.dag_nodes


@pytest.mark.asyncio
async def test_hydrated_dag_with_ready_annotator_llm_work_is_admitted():
    repo = FakeRepository(priorities={}, hydratable_dags=[])
    service, frontier, active_dags = make_service(
        {"annotator_llm": 1, "annotator_parser": 1}, repo=repo
    )

    dag_id = "dag-llm"
    nodes = [
        make_wi("job-1", dag_id, "annotator_llm://default", deps=[]),
        make_wi("job-2", dag_id, "annotator_parser://default", deps=["job-1"]),
    ]

    admitted, reason = await service._admit_hydrated_dag(
        dag_id, QueryPlan(nodes=[]), nodes, source="hydrate_bulk"
    )

    assert admitted is True
    assert reason == "admitted"
    assert repo.marked_active_dags == [dag_id]
    assert dag_id in active_dags
    assert dag_id in frontier.dag_nodes


@pytest.mark.asyncio
async def test_hydrated_dag_with_control_flow_path_to_runnable_work_is_admitted():
    repo = FakeRepository(priorities={}, hydratable_dags=[])
    service, _, active_dags = make_service(
        {"annotator_llm": 1, "annotator_parser": 0}, repo=repo
    )

    dag_id = "dag-control"
    nodes = [
        make_wi("root", dag_id, "noop://noop", deps=[]),
        make_wi("real", dag_id, "annotator_llm://default", deps=["root"]),
    ]

    admitted, reason = await service._admit_hydrated_dag(
        dag_id, QueryPlan(nodes=[]), nodes, source="hydrate_bulk"
    )

    assert admitted is True
    assert reason == "admitted"
    assert repo.marked_active_dags == [dag_id]
    assert dag_id in active_dags


@pytest.mark.asyncio
async def test_hydrated_dag_with_only_control_flow_work_is_admitted():
    repo = FakeRepository(priorities={}, hydratable_dags=[])
    service, _, active_dags = make_service(
        {"annotator_llm": 0, "annotator_parser": 0}, repo=repo
    )

    dag_id = "dag-noop"
    nodes = [
        make_wi("root", dag_id, "noop://noop", deps=[]),
        make_wi("end", dag_id, "merger://noop", deps=["root"]),
    ]

    admitted, reason = await service._admit_hydrated_dag(
        dag_id, QueryPlan(nodes=[]), nodes, source="hydrate_bulk"
    )

    assert admitted is True
    assert reason == "admitted"
    assert repo.marked_active_dags == [dag_id]
    assert dag_id in active_dags


@pytest.mark.asyncio
async def test_hydrated_dag_is_not_admitted_when_database_activation_fails():
    repo = FakeRepository(
        priorities={},
        hydratable_dags=[],
        mark_dag_active_result=False,
    )
    service, frontier, active_dags = make_service(
        {"annotator_llm": 1, "annotator_parser": 1}, repo=repo
    )

    dag_id = "dag-db-fail"
    nodes = [make_wi("job-1", dag_id, "annotator_llm://default", deps=[])]

    admitted, reason = await service._admit_hydrated_dag(
        dag_id, QueryPlan(nodes=[]), nodes, source="hydrate_bulk"
    )

    assert admitted is False
    assert reason == "db_activation_failed"
    assert repo.marked_active_dags == [dag_id]
    assert dag_id not in active_dags
    assert dag_id not in frontier.dag_nodes


@pytest.mark.asyncio
async def test_hydrate_bulk_skips_incompatible_dags_and_admits_later_compatible_ones():
    dag_rows = [
        ("dag-mock-1", {"nodes": []}),
        ("dag-mock-2", {"nodes": []}),
        ("dag-llm-1", {"nodes": []}),
        ("dag-parser-1", {"nodes": []}),
    ]
    hydratable_jobs = {
        "dag-mock-1": [
            serialize_wi(
                make_wi("mock-root-1", "dag-mock-1", "noop://noop", deps=[])
            ),
            serialize_wi(
                make_wi(
                    "mock-real-1",
                    "dag-mock-1",
                    "mock_executor_a://document/process",
                    deps=["mock-root-1"],
                )
            ),
        ],
        "dag-mock-2": [
            serialize_wi(
                make_wi("mock-root-2", "dag-mock-2", "noop://noop", deps=[])
            ),
            serialize_wi(
                make_wi(
                    "mock-real-2",
                    "dag-mock-2",
                    "mock_executor_b://document/process",
                    deps=["mock-root-2"],
                )
            ),
        ],
        "dag-llm-1": [
            serialize_wi(make_wi("llm-job", "dag-llm-1", "annotator_llm://default", deps=[]))
        ],
        "dag-parser-1": [
            serialize_wi(
                make_wi("parser-job", "dag-parser-1", "annotator_parser://default", deps=[])
            )
        ],
    }
    repo = FakeRepository(
        priorities={},
        hydratable_dags=dag_rows,
        hydratable_jobs=hydratable_jobs,
    )
    service, frontier, active_dags = make_service(
        {"annotator_llm": 1, "annotator_parser": 1},
        max_active_dags=2,
        repo=repo,
    )

    await service.hydrate_bulk(dag_batch_size=10, itersize=10, log_every_seconds=60.0)

    assert set(active_dags.keys()) == {"dag-llm-1", "dag-parser-1"}
    assert "dag-mock-1" not in active_dags
    assert "dag-mock-2" not in active_dags
    assert set(frontier.dag_nodes.keys()) == {"dag-llm-1", "dag-parser-1"}
