from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from marie.excepts import BadConfigSource
from marie.query_planner.base import QueryPlan
from marie.scheduler.in_memory_scheduling_engine import InMemorySchedulingEngine
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.postgres_scheduler_config import PostgreSQLSchedulerConfig
from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.services.dag_management_service import DAGManagementService
from marie.scheduler.state import WorkState


def test_omitted_queue_names_consumes_all() -> None:
    assert PostgreSQLSchedulerConfig.from_dict({}).queue_names is None


async def test_empty_frontier_sync_requests_durable_admission() -> None:
    service = DAGManagementService(
        repository=SimpleNamespace(),
        frontier=MemoryFrontier(),
        active_dags={},
        subscribed_queues=None,
    )
    await service.sync_once()
    assert service._admission_event.is_set()


async def test_admission_provenance_follows_resident_plan_and_eviction() -> None:
    repository = SimpleNamespace(
        get_dag_queue_identity=AsyncMock(return_value=("extract", True)),
        mark_dag_as_active=AsyncMock(return_value=True),
    )
    service = DAGManagementService(
        repository=repository,
        frontier=MemoryFrontier(),
        active_dags={},
        subscribed_queues=frozenset({"extract"}),
    )
    plan = QueryPlan(nodes=[])
    assert not service.is_admitted("dag")
    assert await service.admit_dag("dag", plan, source="test")
    assert service.is_admitted("dag")
    service.active_dags["dag"] = QueryPlan(nodes=[])
    assert not service.is_admitted("dag")
    assert await service.admit_dag("dag", plan, source="test")
    assert service.is_admitted("dag")
    await service.evict_dag("dag", reason="test")
    assert not service.is_admitted("dag")


async def test_foreign_resident_dag_is_evicted_before_sync_resolution() -> None:
    repository = SimpleNamespace(
        resolve_dag_state=AsyncMock(return_value="failed"),
        get_active_dag_ids=AsyncMock(return_value=set()),
        get_dag_queue_identity=AsyncMock(return_value=("foreign", True)),
    )
    service = DAGManagementService(
        repository=repository,
        frontier=MemoryFrontier(),
        active_dags={"dag": object()},
        subscribed_queues=frozenset({"extract"}),
    )
    await service.sync_once()
    assert service.active_dags == {}
    repository.resolve_dag_state.assert_not_awaited()


@pytest.mark.parametrize("value", [[], None, "extract", [None], [""], [1]])
def test_invalid_queue_names_do_not_enable_all(value: object) -> None:
    with pytest.raises(BadConfigSource):
        PostgreSQLSchedulerConfig.from_dict({"queue_names": value})


async def test_foreign_queue_never_reaches_repository_lease() -> None:
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.subscribed_queues = None
    scheduler.subscribed_queues = frozenset({"extract"})
    scheduler.repository = SimpleNamespace(lease_jobs=AsyncMock(return_value={"job"}))
    scheduler.lease_owner = "owner"
    scheduler.lease_ttl_seconds = 30
    with pytest.raises(ValueError):
        await scheduler._lease_jobs_db("kb_indexing", ["job"])
    scheduler.repository.lease_jobs.assert_not_awaited()


@pytest.mark.parametrize(
    "scope, expected", [(None, {"a", "b"}), (frozenset({"extract"}), {"a"})]
)
async def test_selection_excludes_foreign_queue_before_take(scope, expected) -> None:
    frontier = MemoryFrontier()
    jobs = [
        WorkInfo.model_construct(
            id=job_id,
            dag_id=dag_id,
            name=queue,
            priority=1,
            state=WorkState.CREATED,
            data={"metadata": {"on": "extract://default"}},
            start_after=datetime.now(timezone.utc),
            dependencies=[],
            job_level=0,
        )
        for job_id, dag_id, queue in [
            ("a", "dag-a", "extract"),
            ("b", "dag-b", "kb_indexing"),
        ]
    ]
    for job in jobs:
        await frontier.add_dag(SimpleNamespace(nodes=[], job_id=job.dag_id), [job])
    engine = InMemorySchedulingEngine(
        frontier, sla_priority_interval_seconds=60, subscribed_queues=scope
    )
    selection = await engine.select_ready(
        slots_by_executor={"extract": 2},
        batch_size=32,
        dispatch_capacity=2,
        lease_ttl=10,
        resident_dag_ids={"dag-a", "dag-b"},
        max_resident_dags=2,
    )
    assert {job.id for job in selection.selected} == expected
    assert ("b" in frontier.leased_until) == ("b" in expected)
    assert set(selection.rejected_dag_ids) == (
        {"dag-b"} if scope is not None else set()
    )
