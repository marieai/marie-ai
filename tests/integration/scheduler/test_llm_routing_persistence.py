from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from marie.query_planner.base import (
    LlmQueryDefinition,
    Query,
    QueryPlan,
    QueryType,
)
from marie.scheduler.llm_routing import PlannedLlmRoute
from marie.scheduler.models import WorkInfo
from marie.scheduler.repository.async_job_repository import AsyncJobRepository
from marie.scheduler.state import WorkState


def _work(work_id: str, dag_id: str) -> WorkInfo:
    now = datetime.now(timezone.utc)
    return WorkInfo(
        id=work_id,
        dag_id=dag_id,
        name='extract',
        data={'metadata': {'planner': 'test'}},
        state=WorkState.CREATED,
        retry_limit=1,
        retry_delay=1,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=60,
        keep_until=now + timedelta(days=1),
    )


def _submission():
    dag_id = '018fa1f1-0000-7000-8000-000000000001'
    node_id = '018fa1f1-0000-7000-8000-000000000002'
    root = _work(dag_id, dag_id)
    node = _work(node_id, dag_id)
    plan = QueryPlan(
        nodes=[
            Query(
                task_id=node_id,
                query_str='Extract',
                dependencies=[],
                node_type=QueryType.COMPUTE,
                definition=LlmQueryDefinition(
                    model_name='mock',
                    endpoint='annotator_llm://extract',
                    params={'layout': 'mock'},
                ),
            )
        ]
    )
    route = PlannedLlmRoute(
        job_id=dag_id,
        work_unit_id=node_id,
        fabric_group_id='default',
        policy_generation=2,
        policy_digest='a' * 64,
        rule_digest='b' * 64,
        normalized_fact_digest='c' * 64,
        effective_page_count=3,
        pool_id='document-small',
        logical_endpoint_group_id='primary',
        endpoint_revision='r1',
    )
    return dag_id, root, node, plan, route


class _Connection:
    def __init__(self, fail_routes: bool = False) -> None:
        self.fail_routes = fail_routes
        self.committed = False
        self.rolled_back = False
        self.calls = []

    def transaction(self):
        connection = self

        class Transaction:
            async def __aenter__(self):
                return self

            async def __aexit__(self, error_type, error, traceback):
                if error_type is None:
                    connection.committed = True
                else:
                    connection.rolled_back = True

        return Transaction()

    async def fetchrow(self, query, *args):
        self.calls.append(('fetchrow', query, args))
        return (args[0],)

    async def execute(self, query, *args):
        self.calls.append(('execute', query, args))
        return 'INSERT 0 1'

    async def fetch(self, query, *args):
        self.calls.append(('fetch', query, args))
        if self.fail_routes:
            raise RuntimeError('route insert failed')
        return [('018fa1f1-0000-7000-8000-000000000002',)]


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


@pytest.mark.asyncio
async def test_route_and_outbox_commit_with_jobs() -> None:
    dag_id, root, node, plan, route = _submission()
    connection = _Connection()
    repository = AsyncJobRepository({}, pool=_Pool(connection))

    created, _ = await repository.create_dag_with_jobs(
        dag_id, plan, [node], root, (route,)
    )

    assert created is True
    assert connection.committed is True
    jobs_call = next(
        call
        for call in connection.calls
        if call[0] == 'execute' and '.job (' in call[1]
    )
    assert jobs_call[2][0].obj[0]['llm_routing_ready'] is False
    route_call = next(call for call in connection.calls if call[0] == 'fetch')
    record = route_call[2][0].obj[0]
    assert record['payload']['pool_id'] == 'document-small'
    assert len(record['route_digest']) == 64
    assert 'prompt' not in record['payload']


@pytest.mark.asyncio
async def test_route_insert_failure_rolls_back_dag() -> None:
    dag_id, root, node, plan, route = _submission()
    connection = _Connection(fail_routes=True)
    repository = AsyncJobRepository({}, pool=_Pool(connection))

    with pytest.raises(RuntimeError, match='route insert failed'):
        await repository.create_dag_with_jobs(dag_id, plan, [node], root, (route,))

    assert connection.committed is False
    assert connection.rolled_back is True


def test_invalid_or_duplicate_route_identity_is_rejected() -> None:
    dag_id, _root, node, _plan, route = _submission()

    with pytest.raises(ValueError, match='routing_manifest_invalid'):
        AsyncJobRepository._routing_records(
            dag_id,
            [node],
            [route, route.model_copy(update={'pool_id': 'other'})],
        )


def test_admission_candidates_wait_for_every_route_projection() -> None:
    sql = (
        __import__('pathlib').Path(
            'config/psql/schema/lease/017_admission_candidate_dags.sql'
        )
    ).read_text()

    assert 'NOT routing_pending.llm_routing_ready' in sql
    assert "routing_pending.state IN ('created', 'retry')" in sql
