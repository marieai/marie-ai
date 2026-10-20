from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio

from marie.query_planner.base import QueryPlan
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.services.dag_management_service import DAGManagementService
from tests.integration.scheduler._queue_consumer_harness import owned_database, seed_dag


@pytest_asyncio.fixture
async def database() -> AsyncIterator[tuple]:
    async with owned_database() as value:
        yield value


async def test_new_admission_uses_activation_guard_without_extra_identity_query(
    database, monkeypatch
) -> None:
    repo, conn, _ = database
    dag_id, _ = await seed_dag(repo, conn, "extract")
    plan = await repo.get_dag_by_id(dag_id)
    assert plan is not None
    identity_reads: list[str] = []
    original_identity = repo.get_dag_queue_identity

    async def read_identity(dag_id: str) -> tuple[str | None, bool] | None:
        identity_reads.append(dag_id)
        return await original_identity(dag_id)

    monkeypatch.setattr(repo, "get_dag_queue_identity", read_identity)
    service = DAGManagementService(
        repository=repo,
        frontier=MemoryFrontier(),
        active_dags={},
        subscribed_queues=frozenset({"extract"}),
    )

    assert await service.admit_dag(dag_id, plan, source="test")
    assert service.is_admitted(dag_id)
    assert (
        await conn.fetchval(
            "SELECT state::text FROM marie_scheduler.dag WHERE id=%s", dag_id
        )
        == "active"
    )
    assert identity_reads == []


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize(
    "queue,projection,mismatch,reason",
    [
        ("foreign", "same", False, "queue_not_subscribed"),
        ("extract", None, False, "invalid_queue_identity"),
        ("extract", "same", True, "invalid_queue_identity"),
    ],
)
async def test_admission_rejects_foreign_or_invalid_identity(
    database, resident, queue, projection, mismatch, reason
) -> None:
    repo, conn, _ = database
    dag_id, jobs = await seed_dag(repo, conn, queue, projection=projection, children=2)
    if mismatch:
        await conn.execute(
            "UPDATE marie_scheduler.job SET name='other',state='completed' WHERE id=%s",
            jobs[1],
        )
    plan = await repo.get_dag_by_id(dag_id)
    assert plan is not None
    service = DAGManagementService(
        repository=repo,
        frontier=MemoryFrontier(),
        active_dags={dag_id: plan} if resident else {},
        subscribed_queues=frozenset({"extract"}),
    )

    assert await service._admit_dag(dag_id, plan, source="test") == (False, reason)
    assert service.active_dags == {}
    assert not service.is_admitted(dag_id)
    assert (
        await conn.fetchval(
            "SELECT state::text FROM marie_scheduler.dag WHERE id=%s", dag_id
        )
        == "created"
    )
    assert await conn.fetchval("SELECT count(*) FROM marie_scheduler.job_attempt") == 0


async def test_foreign_backlog_filtered_before_limit(database) -> None:
    repo, conn, _ = database
    for _ in range(5):
        await seed_dag(repo, conn, "foreign", priority=100)
    allowed, _ = await seed_dag(repo, conn, "extract")
    rows = await repo.discover_admission_candidates(
        subscribed_queues=frozenset({"extract"}), limit=1, sla_interval_seconds=900
    )
    assert [row[0] for row in rows] == [allowed]
    assert (
        await repo.load_hydratable_jobs(
            [allowed], subscribed_queues=frozenset({"foreign"})
        )
        == []
    )
    assert await repo.load_dag_and_jobs(
        allowed, subscribed_queues=frozenset({"foreign"})
    ) == (None, [])
    assert not await repo.mark_dag_as_active(
        allowed, subscribed_queues=frozenset({"foreign"})
    )
    assert (
        await conn.fetchval(
            "SELECT state::text FROM marie_scheduler.dag WHERE id=%s", allowed
        )
        == "created"
    )
    assert await repo.mark_dag_as_active(
        allowed, subscribed_queues=frozenset({"extract"})
    )


@pytest.mark.parametrize(
    "projection,mismatch", [(None, False), ("", False), ("same", True)]
)
async def test_restricted_identity_includes_terminal_children(
    database, projection, mismatch
) -> None:
    repo, conn, _ = database
    dag_id, jobs = await seed_dag(
        repo, conn, "extract", projection=projection, children=2
    )
    if mismatch:
        await conn.execute(
            "UPDATE marie_scheduler.job SET name='other', state='completed' WHERE id=%s",
            jobs[1],
        )
    assert (
        await repo.discover_admission_candidates(
            subscribed_queues=frozenset({"extract"}), limit=10, sla_interval_seconds=900
        )
        == []
    )
    assert (
        await repo.load_hydratable_jobs(
            [dag_id], subscribed_queues=frozenset({"extract"})
        )
        == []
    )
    assert await repo.load_dag_and_jobs(
        dag_id, subscribed_queues=frozenset({"extract"})
    ) == (None, [])
    assert not await repo.mark_dag_as_active(
        dag_id, subscribed_queues=frozenset({"extract"})
    )
    identity = await repo.get_dag_queue_identity(dag_id)
    assert identity is not None and identity[1] is False
    assert [
        row[0]
        for row in await repo.discover_admission_candidates(
            subscribed_queues=None, limit=10, sla_interval_seconds=900
        )
    ] == [dag_id]
    assert (await repo.load_dag_and_jobs(dag_id, subscribed_queues=None))[0] is not None
    assert await repo.mark_dag_as_active(dag_id, subscribed_queues=None)


async def test_combined_scope_order_exclusions_and_new_queues(database) -> None:
    repo, conn, _ = database
    low, _ = await seed_dag(repo, conn, "extract", priority=1)
    high, _ = await seed_dag(repo, conn, "kb_indexing", priority=5)
    kwargs = dict(
        subscribed_queues=frozenset({"extract", "kb_indexing"}),
        limit=10,
        sla_interval_seconds=900,
    )
    assert [row[0] for row in await repo.discover_admission_candidates(**kwargs)] == [
        high,
        low,
    ]
    assert [
        row[0]
        for row in await repo.discover_admission_candidates(
            **kwargs, excluded_dag_ids=[high]
        )
    ] == [low]
    new, _ = await seed_dag(repo, conn, "new_queue", priority=10)
    assert [
        row[0]
        for row in await repo.discover_admission_candidates(
            subscribed_queues=None, limit=1, sla_interval_seconds=900
        )
    ] == [new]


@pytest.mark.parametrize("scope", [[], [None], [""], [["extract"]]])
async def test_malformed_sql_scope_fails_closed(database, scope) -> None:
    repo, conn, _ = database
    await seed_dag(repo, conn, "extract")
    assert (
        await conn.fetch(
            "SELECT * FROM marie_scheduler.admission_candidate_dags(%s::text[],10,900,ARRAY[]::uuid[])",
            scope,
        )
        == []
    )
    assert (
        await conn.fetch(
            "SELECT * FROM marie_scheduler.claim_expired_run_leases(%s::text[],10)",
            scope,
        )
        == []
    )


async def test_packaged_install_and_readiness_reject_retired_signatures(
    database, monkeypatch
) -> None:
    from pathlib import Path

    import marie.scheduler.repository.async_job_repository as module
    from marie.excepts import RuntimeFailToStart

    repo, conn, _ = database
    packaged = (
        Path(__file__).parents[4] / "deploy/helm/charts/marie/charts/server/files/psql"
    )
    monkeypatch.setattr(
        module,
        "_scheduler_sql_paths",
        lambda: (str(packaged), str(packaged / "schema")),
    )
    await repo.create_tables()
    await repo.validate_durable_scheduler_schema()
    assert await repo.is_installed()
    old_definition = (
        Path(__file__).parents[4]
        / "config/psql/schema/lease/010_claim_expired_run_leases.sql"
    ).read_text()
    await conn.execute(old_definition.replace("{schema}", "marie_scheduler"))
    with pytest.raises(RuntimeFailToStart, match="retired signature"):
        await repo.validate_durable_scheduler_schema()
    await repo.create_tables()
    await repo.validate_durable_scheduler_schema()


async def test_old_admission_signature_removed_on_replay(database) -> None:
    repo, conn, _ = database
    await repo.create_tables()
    assert (
        await conn.fetchval(
            "SELECT to_regprocedure('marie_scheduler.admission_candidate_dags(integer,integer,uuid[])')"
        )
        is None
    )
    assert (
        await conn.fetchval(
            "SELECT to_regprocedure('marie_scheduler.admission_candidate_dags(text[],integer,integer,uuid[])')"
        )
        is not None
    )


@pytest.mark.parametrize("mismatch", ["queue", "dag_id", "empty"])
async def test_create_dag_rejects_mixed_children_before_writing(
    database, mismatch
) -> None:
    repo, conn, _ = database
    source, ids = await seed_dag(repo, conn, "extract")
    node = await repo.get_job_by_id(ids[0])
    dag_id = str(uuid4())
    node = node.model_copy(update={"id": str(uuid4()), "dag_id": dag_id})
    root = node.model_copy()
    if mismatch == "queue":
        node.name = "other"
    if mismatch == "dag_id":
        node.dag_id = source
    with pytest.raises(ValueError):
        await repo.create_dag_with_jobs(
            dag_id, QueryPlan(nodes=[]), [] if mismatch == "empty" else [node], root
        )
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM marie_scheduler.dag WHERE id=%s", dag_id
        )
        == 0
    )


async def test_create_job_rejects_parent_queue_mismatch(database) -> None:
    repo, conn, _ = database
    _, ids = await seed_dag(repo, conn, "extract")
    node = await repo.get_job_by_id(ids[0])
    await repo.create_queue("other")
    foreign = node.model_copy(update={"id": str(uuid4()), "name": "other"})
    assert await repo.create_job(foreign) is False
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM marie_scheduler.job WHERE id=%s", foreign.id
        )
        == 0
    )
    valid = node.model_copy(update={"id": str(uuid4())})
    assert await repo.create_job(valid) is True


@pytest.mark.parametrize("exhausted", [False, True])
async def test_recovery_scope_preserves_foreign_attempts(database, exhausted) -> None:
    repo, conn, _ = database
    a, a_ids = await seed_dag(repo, conn, "extract")
    b, b_ids = await seed_dag(repo, conn, "kb_indexing")
    for queue, ids in [("extract", a_ids), ("kb_indexing", b_ids)]:
        assert await repo.lease_jobs(
            ids, owner="dead", ttl_seconds=10, job_name=queue
        ) == set(ids)
        assert set(await repo.activate_from_lease(ids, "dead", 10, "gateway")) == set(
            ids
        )
    await conn.execute(
        "UPDATE marie_scheduler.job SET run_lease_expires_at=now()-interval '1 minute', retry_limit=%s",
        0 if exhausted else 3,
    )
    recovered = await repo.recover_expired_run_leases(
        subscribed_queues=frozenset({"extract"}), limit=10
    )
    assert [item.id for item in recovered] == a_ids
    assert recovered[0].recovered_state == ("failed" if exhausted else "retry")
    assert (
        await conn.fetchval(
            "SELECT state::text FROM marie_scheduler.job WHERE id=%s", b_ids[0]
        )
        == "active"
    )
    assert (
        await conn.fetchval(
            "SELECT run_owner FROM marie_scheduler.job WHERE id=%s", b_ids[0]
        )
        == "dead"
    )
    assert [
        item.id
        for item in await repo.recover_expired_run_leases(subscribed_queues=None)
    ] == b_ids
