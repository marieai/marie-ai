from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from marie.query_planner.base import QueryPlan
from marie.scheduler.models import WorkInfo
from marie.scheduler.repository import JobRepository
from marie.scheduler.state import WorkState


class FakeTransaction:
    def __init__(self) -> None:
        self.entered = False
        self.error: BaseException | None = None

    async def __aenter__(self) -> None:
        self.entered = True

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        self.error = exc
        return False


class FakeConnection:
    def __init__(
        self,
        *,
        fetch: list[Any] | None = None,
        fetchrow: list[Any] | None = None,
        fetchval: list[Any] | None = None,
        execute_errors: list[BaseException | None] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.fetch_results = deque(fetch or [])
        self.fetchrow_results = deque(fetchrow or [])
        self.fetchval_results = deque(fetchval or [])
        self.execute_errors = deque(execute_errors or [])
        self.error = error
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.transactions: list[FakeTransaction] = []

    def _raise_if_configured(self) -> None:
        if self.error is not None:
            raise self.error

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.calls.append(("fetch", query, args))
        self._raise_if_configured()
        return self.fetch_results.popleft()

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.calls.append(("fetchrow", query, args))
        self._raise_if_configured()
        return self.fetchrow_results.popleft()

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.calls.append(("fetchval", query, args))
        self._raise_if_configured()
        return self.fetchval_results.popleft()

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append(("execute", query, args))
        self._raise_if_configured()
        if self.execute_errors:
            error = self.execute_errors.popleft()
            if error is not None:
                raise error
        return "UPDATE 1"

    async def executemany(self, query: str, args: Any) -> None:
        self.calls.append(("executemany", query, tuple(args)))
        self._raise_if_configured()

    def transaction(self) -> FakeTransaction:
        transaction = FakeTransaction()
        self.transactions.append(transaction)
        return transaction


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.acquire_error: BaseException | None = None
        self.closed = False
        self.autocommit: bool | None = None

    @asynccontextmanager
    async def acquire(self):
        if self.acquire_error is not None:
            raise self.acquire_error
        yield self.connection

    async def initialize(self, config, *, row_factory, autocommit=False) -> None:
        self.autocommit = autocommit

    async def close(self) -> None:
        self.closed = True


def build_repository(connection: FakeConnection) -> JobRepository:
    return JobRepository({}, pool=FakePool(connection))


def build_dag_job(dag_id: str, index: int) -> WorkInfo:
    now = datetime.now(timezone.utc)
    return WorkInfo(
        id=f"00000000-0000-0000-0001-{index:012d}",
        dag_id=dag_id,
        name="extract",
        priority=index,
        data={
            "metadata": {
                "on": "extract://processor",
                "planner": "test-planner",
                "ref_id": f"document-{index}",
            }
        },
        state=WorkState.CREATED,
        retry_limit=2,
        retry_delay=3,
        retry_backoff=True,
        start_after=now,
        expire_in_seconds=300,
        keep_until=now + timedelta(days=1),
        dependencies=[] if index == 1 else [f"node-{index - 1}"],
        job_level=index - 1,
    )


@pytest.mark.asyncio
async def test_repository_initializes_pool_in_autocommit_mode() -> None:
    pool = FakePool(FakeConnection())
    repository = JobRepository({}, pool=pool)

    await repository.initialize()

    assert pool.autocommit is True


@pytest.mark.asyncio
async def test_create_tables_includes_gateway_runtime_tables() -> None:
    connection = FakeConnection(fetchval=[True, True])
    repository = build_repository(connection)

    await repository.create_tables()

    schema_query = "\n".join(
        query for method, query, _args in connection.calls if method == "execute"
    )
    assert "create table if not exists marie_scheduler.sensor_tick" in schema_query
    assert (
        "CREATE TABLE IF NOT EXISTS marie_scheduler.llm_queue_fabric_config"
        in schema_query
    )
    assert "CREATE TABLE IF NOT EXISTS marie_scheduler.job_attempt" in schema_query
    assert (
        "CREATE TABLE IF NOT EXISTS marie_scheduler.resource_workflow_binding"
        in schema_query
    )
    assert (
        "CREATE OR REPLACE FUNCTION marie_scheduler.admission_candidate_dags("
        in schema_query
    )
    assert "VALUES ('default')" in schema_query
    assert "VALUES ('72')" in schema_query


@pytest.mark.asyncio
async def test_create_tables_installs_safe_queue_partition_deletion() -> None:
    connection = FakeConnection(fetchval=[True, True])
    repository = build_repository(connection)

    await repository.create_tables()

    delete_queue_query = next(
        query
        for method, query, _args in connection.calls
        if method == "execute"
        and "CREATE OR REPLACE FUNCTION marie_scheduler.delete_queue" in query
    )
    delete_jobs = delete_queue_query.index("DELETE FROM marie_scheduler.job AS job")
    detach_partition = delete_queue_query.index("DETACH PARTITION")
    drop_partition = delete_queue_query.index("DROP TABLE IF EXISTS")
    delete_registration = delete_queue_query.index(
        "DELETE FROM marie_scheduler.queue AS queue"
    )

    assert delete_jobs < detach_partition < drop_partition < delete_registration
    assert "CASCADE" not in delete_queue_query


@pytest.mark.asyncio
async def test_create_tables_commits_enum_before_dependent_seed() -> None:
    connection = FakeConnection(fetchval=[True, True])
    repository = build_repository(connection)

    await repository.create_tables()

    migration_queries = [
        query
        for method, query, _args in connection.calls
        if method == "execute" and not query.startswith("SET LOCAL")
    ]
    enum_index = next(
        index
        for index, query in enumerate(migration_queries)
        if "ADD VALUE IF NOT EXISTS 'data_sink'" in query
    )
    seed_index = next(
        index
        for index, query in enumerate(migration_queries)
        if "kb-document-sensor" in query
    )

    assert enum_index < seed_index
    assert len(connection.transactions) == len(migration_queries) + 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tables", "current_version", "installed"),
    [
        (("version", "sensor_tick", "job_attempt", "llm_config"), True, True),
        (("version", "sensor_tick", "job_attempt", "llm_config"), False, False),
        (("version", "sensor_tick", None, "llm_config"), False, False),
        (("version", "sensor_tick", "job_attempt", None), False, False),
    ],
)
async def test_is_installed_requires_runtime_tables(
    tables: tuple[str | None, ...], current_version: bool, installed: bool
) -> None:
    connection = FakeConnection(fetchrow=[tables], fetchval=[current_version])
    repository = build_repository(connection)

    assert await repository.is_installed() is installed


@pytest.mark.asyncio
async def test_delete_job_uses_async_connection() -> None:
    job_id = "00000000-0000-0000-0000-000000000001"
    connection = FakeConnection(fetchrow=[(job_id,)])
    repository = build_repository(connection)

    assert await repository.delete_job(job_id) is True
    assert connection.calls[0][0] == "fetchrow"


@pytest.mark.asyncio
async def test_create_job_uses_shared_batch_statement() -> None:
    dag_id = "00000000-0000-0000-0000-000000000099"
    job = build_dag_job(dag_id, 1)
    connection = FakeConnection()
    repository = build_repository(connection)

    assert await repository.create_job(job) is True

    method, query, params = connection.calls[0]
    assert method == "execute"
    assert "jsonb_to_recordset(%s::jsonb)" in query
    assert params[0].obj == [job.model_dump(mode="json")]


@pytest.mark.asyncio
async def test_bulk_cancel_groups_jobs_by_queue_in_one_transaction() -> None:
    first = "00000000-0000-0000-0000-000000000001"
    second = "00000000-0000-0000-0000-000000000002"
    connection = FakeConnection(
        fetch=[[('extract', [first]), ('parse', [second])]],
        fetchval=[1, 1],
    )
    repository = build_repository(connection)

    cancelled = await repository.cancel_jobs([first, second])

    assert cancelled == 2
    assert connection.transactions[0].error is None
    assert [call[0] for call in connection.calls] == [
        "fetch",
        "fetchval",
        "fetchval",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("node_count", [1, 25])
async def test_create_dag_with_jobs_uses_bounded_batch_statements(
    node_count: int,
) -> None:
    dag_id = "00000000-0000-0000-0000-000000000099"
    jobs = [build_dag_job(dag_id, index + 1) for index in range(node_count)]
    connection = FakeConnection(fetchrow=[(dag_id,)])
    repository = build_repository(connection)

    result = await repository.create_dag_with_jobs(
        dag_id,
        QueryPlan(nodes=[]),
        jobs,
        jobs[0],
    )

    assert result == (True, dag_id)
    assert [call[0] for call in connection.calls] == [
        "fetchrow",
        "execute",
        "execute",
    ]
    assert connection.transactions[0].error is None

    _, job_query, job_params = connection.calls[1]
    job_batch = job_params[0].obj
    assert "jsonb_to_recordset(%s::jsonb)" in job_query
    assert "JOIN marie_scheduler.queue queue ON job.name = queue.name" in job_query
    assert "make_interval(secs => job.expire_in_seconds)" in job_query
    assert "COALESCE(job.retry_limit, queue.retry_limit, 2)" in job_query
    assert "COALESCE(job.dependencies, '[]'::jsonb)" in job_query
    assert "ON CONFLICT DO NOTHING" in job_query
    assert len(job_batch) == node_count
    assert [row["id"] for row in job_batch] == [job.id for job in jobs]
    assert job_batch[-1]["dependencies"] == jobs[-1].dependencies

    _, search_query, search_params = connection.calls[2]
    search_batch = search_params[0].obj
    assert "jsonb_to_recordset(%s::jsonb)" in search_query
    assert "ON CONFLICT (queue_name, job_id) DO UPDATE" in search_query
    assert len(search_batch) == node_count
    assert [row["job_id"] for row in search_batch] == [job.id for job in jobs]
    assert search_batch[-1]["ref_id"] == f"document-{node_count}"


@pytest.mark.asyncio
async def test_create_dag_with_jobs_stops_when_dag_conflicts() -> None:
    dag_id = "00000000-0000-0000-0000-000000000099"
    job = build_dag_job(dag_id, 1)
    connection = FakeConnection(fetchrow=[None])
    repository = build_repository(connection)

    result = await repository.create_dag_with_jobs(
        dag_id,
        QueryPlan(nodes=[]),
        [job],
        job,
    )

    assert result == (False, None)
    assert [call[0] for call in connection.calls] == ["fetchrow"]
    assert connection.transactions[0].error is None


@pytest.mark.asyncio
async def test_create_dag_with_jobs_rolls_back_batch_failure() -> None:
    dag_id = "00000000-0000-0000-0000-000000000099"
    jobs = [build_dag_job(dag_id, 1), build_dag_job(dag_id, 2)]
    connection = FakeConnection(
        fetchrow=[(dag_id,)],
        execute_errors=[None, RuntimeError("search batch failed")],
    )
    repository = build_repository(connection)

    with pytest.raises(RuntimeError, match="search batch failed"):
        await repository.create_dag_with_jobs(
            dag_id,
            QueryPlan(nodes=[]),
            jobs,
            jobs[0],
        )

    assert [call[0] for call in connection.calls] == [
        "fetchrow",
        "execute",
        "execute",
    ]
    assert isinstance(connection.transactions[0].error, RuntimeError)


@pytest.mark.asyncio
async def test_resolve_dag_state_raises_on_db_error() -> None:
    repository = build_repository(FakeConnection(error=RuntimeError("db busy")))

    with pytest.raises(RuntimeError, match="db busy"):
        await repository.resolve_dag_state("00000000-0000-0000-0000-000000000001")


@pytest.mark.asyncio
async def test_get_active_dag_ids_raises_on_db_error() -> None:
    repository = build_repository(FakeConnection(error=RuntimeError("db busy")))

    with pytest.raises(RuntimeError, match="db busy"):
        await repository.get_active_dag_ids(["00000000-0000-0000-0000-000000000001"])


@pytest.mark.asyncio
async def test_load_hydratable_jobs_uses_database_function_contract() -> None:
    dag_id = "00000000-0000-0000-0000-000000000001"
    connection = FakeConnection(fetch=[[]])
    repository = build_repository(connection)

    assert await repository.load_hydratable_jobs([dag_id]) == []

    _, query, params = connection.calls[0]
    assert "FROM marie_scheduler.hydrate_frontier_jobs(%s::uuid[])" in query
    assert "JOIN marie_scheduler.dag" not in query
    assert "NOT EXISTS" not in query
    assert params == ([dag_id],)


@pytest.mark.asyncio
async def test_discover_hydratable_dags_uses_database_function_contract() -> None:
    connection = FakeConnection(fetch=[[]])
    repository = build_repository(connection)

    assert await repository.discover_hydratable_dags(limit=10) == []

    _, query, params = connection.calls[0]
    assert "FROM marie_scheduler.hydrate_frontier_dags()" in query
    assert "NOT EXISTS" not in query
    assert params == (10,)


@pytest.mark.asyncio
async def test_discover_admission_candidates_uses_database_function_contract() -> None:
    excluded = "00000000-0000-0000-0000-000000000001"
    connection = FakeConnection(fetch=[[]])
    repository = build_repository(connection)

    assert await repository.discover_admission_candidates(
        limit=25,
        sla_interval_seconds=900,
        excluded_dag_ids=[excluded],
    ) == []

    _, query, params = connection.calls[0]
    assert "FROM marie_scheduler.admission_candidate_dags(" in query
    assert "JOIN marie_scheduler.dag" not in query
    assert "WITH ORDINALITY" in query
    assert "ORDER BY candidate.admission_rank" in query
    assert "priority" not in query
    assert "soft_sla" not in query
    assert "hard_sla" not in query
    assert params == (25, 900, [excluded])


@pytest.mark.asyncio
async def test_load_dag_and_jobs_uses_database_function_contract() -> None:
    dag_id = "00000000-0000-0000-0000-000000000001"
    connection = FakeConnection(fetchrow=[None])
    repository = build_repository(connection)

    assert await repository.load_dag_and_jobs(dag_id) == (None, [])

    _, query, params = connection.calls[0]
    assert "FROM marie_scheduler.hydrate_frontier_dags()" in query
    assert "NOT EXISTS" not in query
    assert params == (dag_id,)


@pytest.mark.asyncio
async def test_dag_activation_diagnostic_includes_root_failure() -> None:
    dag_id = "00000000-0000-0000-0000-000000000001"
    job_id = "00000000-0000-0000-0000-000000000002"
    attempt_id = "00000000-0000-0000-0000-000000000003"
    now = datetime.now(timezone.utc)
    connection = FakeConnection(
        fetchrow=[("failed", now, now, 4, 2, 0)],
        fetch=[
            [("completed", 1), ("created", 2), ("failed", 1)],
            [
                (
                    job_id,
                    "extract",
                    "failed",
                    2,
                    2,
                    now,
                    {"error_message": "processor crashed"},
                    attempt_id,
                    "failed",
                    None,
                    "job_event",
                    "failed",
                    None,
                    "gateway-1",
                    "extractor",
                    "gateway-1",
                    None,
                )
            ],
            [
                (
                    job_id,
                    "extract",
                    "failed",
                    2,
                    2,
                    now,
                    {"error_message": "processor crashed"},
                    attempt_id,
                    now,
                )
            ],
            [("failed", now), ("active", now)],
        ],
    )
    repository = build_repository(connection)

    diagnostic = await repository.diagnose_dag_activation_failure(dag_id)

    assert diagnostic["reason"] == "dag_state_not_activatable"
    assert diagnostic["job_states"] == {
        "completed": 1,
        "created": 2,
        "failed": 1,
    }
    assert diagnostic["blocking_jobs"] == [
        {
            "job_id": job_id,
            "queue": "extract",
            "state": "failed",
            "retry_count": 2,
            "retry_limit": 2,
            "completed_on": now,
            "output": {"error_message": "processor crashed"},
            "run_attempt_id": attempt_id,
            "attempt_state": "failed",
            "dispatch_error": None,
            "terminal_source": "job_event",
            "terminal_work_state": "failed",
            "recovery_reason": None,
            "gateway_instance_id": "gateway-1",
            "executor": "extractor",
            "terminal_gateway_instance_id": "gateway-1",
            "terminal_reject_reason": None,
        }
    ]
    assert diagnostic["historical_blocking_jobs"] == [
        {
            "job_id": job_id,
            "queue": "extract",
            "state": "failed",
            "retry_count": 2,
            "retry_limit": 2,
            "completed_on": now,
            "output": {"error_message": "processor crashed"},
            "run_attempt_id": attempt_id,
            "changed_on": now,
        }
    ]
    assert diagnostic["dag_state_history"][0]["state"] == "failed"


@pytest.mark.asyncio
async def test_activate_from_lease_returns_attempts_in_one_database_call() -> None:
    job_id = "00000000-0000-0000-0000-000000000001"
    attempt_id = "00000000-0000-0000-0000-000000000003"
    connection = FakeConnection(fetch=[[(job_id, attempt_id)]])
    repository = build_repository(connection)

    attempts = await repository.activate_from_lease(
        [job_id], "owner", 60, gateway_instance_id="gateway-1"
    )

    assert attempts == {job_id: attempt_id}
    assert len(connection.calls) == 1
    method, query, params = connection.calls[0]
    assert method == "fetch"
    assert "SELECT job_id, run_attempt_id" in query
    assert params == ([job_id], "owner", "60 seconds", "gateway-1")


@pytest.mark.asyncio
async def test_activate_from_lease_returns_empty_mapping() -> None:
    connection = FakeConnection(fetch=[[]])
    repository = build_repository(connection)

    assert (
        await repository.activate_from_lease(
            ["00000000-0000-0000-0000-000000000001"], "owner", 60
        )
        == {}
    )
    assert len(connection.calls) == 1


def test_activate_from_lease_sql_owns_attempt_audit_atomically() -> None:
    project_root = Path(__file__).parents[4]
    sql = (
        project_root / "config/psql/schema/lease/002_activate_from_lease.sql"
    ).read_text()

    core, compatibility_wrapper = sql.split(
        "CREATE OR REPLACE FUNCTION {schema}.activate_from_lease(", maxsplit=2
    )[1:]
    assert "_gateway_instance_id text" in core
    assert "RETURNS TABLE(job_id uuid, run_attempt_id uuid)" in core
    assert "UPDATE {schema}.job" in core
    assert "INSERT INTO {schema}.job_attempt" in core
    assert "FROM activated" in core
    assert "JOIN audited" in core
    assert "RETURNS uuid[]" in compatibility_wrapper
    assert "FROM {schema}.activate_from_lease(" in compatibility_wrapper


@pytest.mark.asyncio
async def test_schema_validation_requires_atomic_activation_contract() -> None:
    connection = FakeConnection(
        fetchval=[
            True,
            True,
            True,
            True,
            (
                "run_attempt_id lease_owner = _run_owner INSERT INTO "
                "marie_scheduler.job_attempt _gateway_instance_id"
            ),
            "_run_attempt_id",
        ],
        fetch=[[]],
    )
    repository = build_repository(connection)

    await repository.validate_durable_scheduler_schema()

    _, function_query, function_params = connection.calls[3]
    assert "to_regprocedure" in function_query
    assert function_params == (
        "marie_scheduler.admission_candidate_dags(integer,integer,uuid[])",
    )

    _, activation_query, _ = connection.calls[4]
    assert "p.pronargs = 4" in activation_query
    assert all(
        "scheduler_attempt_invariant_checks" not in query
        for _, query, _ in connection.calls
    )


@pytest.mark.asyncio
async def test_lease_jobs_preserves_input_order_for_sql_call() -> None:
    job_ids = [
        "00000000-0000-0000-0000-000000000002",
        "00000000-0000-0000-0000-000000000001",
    ]
    connection = FakeConnection(fetch=[[(job_ids[0],), (job_ids[1],)]])
    repository = build_repository(connection)

    leased = await repository.lease_jobs(job_ids, "owner", 5, "extract")

    _, _, params = connection.calls[0]
    assert params[0] == job_ids
    assert leased == set(job_ids)


@pytest.mark.asyncio
async def test_mark_jobs_as_skipped_returns_only_committed_ids() -> None:
    committed = "00000000-0000-0000-0000-000000000001"
    missing = "00000000-0000-0000-0000-000000000002"
    connection = FakeConnection(fetch=[[(committed,)]])
    repository = build_repository(connection)

    skipped = await repository.mark_jobs_as_skipped([committed, missing], "extract")

    assert skipped == {committed}


@pytest.mark.asyncio
async def test_mark_jobs_as_skipped_propagates_database_errors() -> None:
    repository = build_repository(FakeConnection(error=RuntimeError("write failed")))

    with pytest.raises(RuntimeError, match="write failed"):
        await repository.mark_jobs_as_skipped(
            ["00000000-0000-0000-0000-000000000001"], "extract"
        )


@pytest.mark.asyncio
async def test_commit_guardrail_route_is_atomic() -> None:
    job_id = "00000000-0000-0000-0000-000000000001"
    skipped_id = "00000000-0000-0000-0000-000000000002"
    connection = FakeConnection(
        fetchrow=[(1,), (job_id,)],
        fetch=[[(skipped_id,)]],
    )
    repository = build_repository(connection)

    result = await repository.commit_guardrail_route(
        job_id=job_id,
        queue_name="extract",
        run_owner="owner",
        run_attempt_id="00000000-0000-0000-0000-000000000003",
        branch_metadata={
            "outcome": "VALID",
            "evaluated_at": "2026-07-21T00:00:00Z",
            "selected_path_ids": ["valid"],
            "report_asset": {
                "asset_key": f"guardrail/report/{job_id}",
                "asset_version": "v1",
                "partition_key": None,
            },
        },
        skipped_job_ids=[skipped_id],
    )

    assert result == (True, {skipped_id}, None)
    assert connection.transactions[0].error is None


@pytest.mark.asyncio
async def test_get_guardrail_report_decision_is_attempt_scoped() -> None:
    job_id = "00000000-0000-0000-0000-000000000001"
    connection = FakeConnection(
        fetch=[
            [
                (
                    f"guardrail/report/{job_id}",
                    "v1",
                    None,
                    "VALID",
                    "2026-07-21T00:00:00Z",
                )
            ]
        ]
    )
    repository = build_repository(connection)

    decision = await repository.get_guardrail_report_decision(
        job_id=job_id,
        run_attempt_id="00000000-0000-0000-0000-000000000003",
    )

    assert decision is not None
    assert decision["outcome"] == "VALID"
    _, query, params = connection.calls[0]
    assert "metadata->>'run_attempt_id' = %s" in query
    assert params[-1] == "00000000-0000-0000-0000-000000000003"


@pytest.mark.asyncio
async def test_commit_guardrail_route_rolls_back_partial_skip() -> None:
    job_id = "00000000-0000-0000-0000-000000000001"
    connection = FakeConnection(
        fetchrow=[(1,), (job_id,)],
        fetch=[[]],
    )
    repository = build_repository(connection)

    committed, skipped, reason = await repository.commit_guardrail_route(
        job_id=job_id,
        queue_name="extract",
        run_owner="owner",
        run_attempt_id="00000000-0000-0000-0000-000000000003",
        branch_metadata={
            "outcome": "INVALID",
            "evaluated_at": "2026-07-21T00:00:00Z",
            "selected_path_ids": ["invalid"],
            "report_asset": {
                "asset_key": f"guardrail/report/{job_id}",
                "asset_version": "v1",
            },
        },
        skipped_job_ids=["00000000-0000-0000-0000-000000000002"],
    )

    assert (committed, skipped, reason) == (False, set(), "skip_state_conflict")
    assert isinstance(connection.transactions[0].error, RuntimeError)


@pytest.mark.asyncio
async def test_extend_run_lease_passes_attempt_identity() -> None:
    job_id = "00000000-0000-0000-0000-000000000001"
    attempt_id = "00000000-0000-0000-0000-000000000003"
    connection = FakeConnection(fetch=[[(job_id,)]])
    repository = build_repository(connection)

    extended = await repository.extend_run_lease([job_id], "owner", attempt_id, 30)

    _, query, params = connection.calls[0]
    assert "extend_run_lease" in query
    assert params == ([job_id], "owner", attempt_id, "30 seconds")
    assert extended == {job_id}


@pytest.mark.asyncio
async def test_recover_expired_run_leases_applies_python_policy(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    job_id = "00000000-0000-0000-0000-000000000001"
    dag_id = "00000000-0000-0000-0000-000000000002"
    attempt_id = "00000000-0000-0000-0000-000000000003"
    connection = FakeConnection(
        fetch=[
            [
                (
                    job_id,
                    "extract",
                    dag_id,
                    "active",
                    0,
                    2,
                    5,
                    False,
                    now,
                    "owner",
                    attempt_id,
                    now,
                )
            ]
        ],
        fetchrow=[(job_id,)],
    )
    repository = build_repository(connection)

    recovered = await repository.recover_expired_run_leases(limit=10)

    assert len(recovered) == 1
    assert recovered[0].id == job_id
    assert recovered[0].recovered_state == "retry"
    assert connection.transactions[0].error is None
