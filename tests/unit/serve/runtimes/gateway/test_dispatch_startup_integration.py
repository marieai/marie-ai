"""Startup activation against disposable PostgreSQL and real queue stores."""

import asyncio
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from uuid import uuid4

import psycopg
import pytest
from marie.engine.llm_queue.config import LlmQueueRuntimeConfig
from marie.engine.llm_queue.scheduler_config import DatabaseSchedulerConfigSource
from marie.engine.llm_queue.store import RequestStore, StoreLimits
from psycopg.types.json import Jsonb

from marie.excepts import RuntimeFailToStart
from marie.scheduler.repository.plans import insert_llm_routes
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.gateway.marie.llm_scheduler_config import (
    PostgresSchedulerConfigRepository,
)
from tests.unit.serve.runtimes.gateway.test_dispatch_policy import persisted
from tests.unit.serve.runtimes.gateway.test_dispatch_startup_policy import scheduled


@pytest.fixture
def database():
    port = os.getenv('MARIE_LLM_DISPATCH_TEST_POSTGRES_PORT')
    if not port:
        pytest.skip('Owned PostgreSQL fixture required')
    schema = 'startup_' + uuid4().hex
    config = dict(
        hostname='127.0.0.1',
        port=int(port),
        username='postgres',
        password='',
        database='postgres',
        schema=schema,
        min_pool_size=1,
        max_pool_size=2,
    )
    conn = psycopg.connect(
        host='127.0.0.1',
        port=int(port),
        user='postgres',
        dbname='postgres',
        autocommit=True,
    )
    repository = None
    try:
        conn.execute(f'CREATE SCHEMA {schema}')
        conn.execute(
            f'CREATE TABLE {schema}.job (id uuid PRIMARY KEY, state text NOT NULL)'
        )
        root = Path(__file__).resolve().parents[5]
        for name in [
            '066_llm_queue_scheduler.sql',
            '089_llm_queue_admission_routing.sql',
            '090_llm_routing_operator_audit.sql',
        ]:
            conn.execute(
                (root / 'config/psql/schema' / name)
                .read_text()
                .replace('{schema}', schema)
            )
        data = persisted()
        data['enabled'] = True
        data['metadata']['llm_dispatch']['endpoints'][0].pop('credential_env')
        conn.execute(
            f'INSERT INTO {schema}.llm_queue_fabric_config(fabric_group_id,policy,total_concurrent_dispatch,metadata) VALUES(%s,%s,%s,%s)',
            ('default', 'drr', 4, Jsonb(data['metadata'])),
        )
        conn.execute(
            f'INSERT INTO {schema}.llm_queue_pool(fabric_group_id,pool_id,quantum,min_concurrent,max_concurrent,metadata) VALUES(%s,%s,%s,%s,%s,%s)',
            ('default', 'default', 3, 1, 2, Jsonb(data['lanes'][0]['metadata'])),
        )
        repository = PostgresSchedulerConfigRepository(config)
        yield conn, repository, data
    finally:
        if repository:
            repository.postgreSQL_pool.close()
        conn.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
        conn.close()


def save_request(conn, repository, data, generation=None):
    marker = scheduled(data, generation)
    marker['request_id'] = uuid4().hex
    conn.execute(
        f'UPDATE {repository.config_schema}.llm_queue_fabric_config SET metadata=jsonb_set(metadata,\'{{llm_dispatch_startup}}\',%s)',
        (Jsonb(marker),),
    )
    return repository.load_startup_dispatch_policy('default')


def route_record(activation):
    return dict(
        work_unit_id=str(uuid4()),
        job_id=str(uuid4()),
        fabric_group_id='default',
        policy_generation=activation.generation,
        policy_digest=activation.policy_digest,
        rule_digest='a' * 64,
        normalized_fact_digest='b' * 64,
        effective_page_count=1,
        pool_id='default',
        logical_endpoint_group_id='primary',
        endpoint_revision='r1',
        estimator_version='page-count-v1',
        routing_source='automatic',
        routing_actor=None,
        routing_reason=None,
        route_digest='c' * 64,
        payload={},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('engine', ['redis', 'valkey'])
@pytest.mark.parametrize('change', ['transport', 'identity'])
@pytest.mark.parametrize('provisioning', ['sql-only', 'saved'])
async def test_restart_activates_captured_target_only_after_owner_and_work_drain(
    database, engine, change, provisioning, monkeypatch
):
    conn, repository, data = database
    manifest = json.loads(Path(os.environ['MARIE_LLM_QUEUE_TEST_STORES']).read_text())
    queue_url = manifest[engine]['url']
    if provisioning == 'saved':
        save_request(conn, repository, data)
    runtime = GatewayLlmDispatchRuntime(
        queue_config=LlmQueueRuntimeConfig(
            enabled=True, fabric_group_id='default', queue_url=queue_url
        ),
        scheduler_config_source=DatabaseSchedulerConfigSource(repository, 'default'),
        config={'llm_dispatch': {'policy_source': 'database'}},
    )
    cleanup = None
    try:
        await runtime.start()
        for _ in range(200):
            if runtime._dispatcher._owner_ready:
                break
            await asyncio.sleep(0.01)
        assert runtime._dispatcher._owner_ready
        assert runtime._dispatcher.policy_generation == 1
        assert repository.load_startup_dispatch_policy('default') is None
        # Exercise the same admission reader used by DAG submissions.
        from marie.scheduler.repository import async_job_repository

        monkeypatch.setattr(
            async_job_repository, 'DEFAULT_SCHEMA', repository.config_schema
        )
        scheduler_repository = async_job_repository.AsyncJobRepository(
            {
                'hostname': conn.info.host,
                'port': conn.info.port,
                'username': conn.info.user,
                'password': '',
                'database': conn.info.dbname,
                'min_pool_size': 1,
                'max_pool_size': 2,
            }
        )
        await scheduler_repository.initialize()
        try:
            admission = await scheduler_repository.load_active_admission_policy(
                'default'
            )
            decision = admission.match({'workload.kind': 'document'})
            assert decision.pool_id == 'default'
            assert decision.policy_generation == runtime._dispatcher.policy_generation
            assert decision.policy_digest == runtime._dispatcher.policy_digest
        finally:
            await scheduler_repository.close()
        from marie.serve.runtimes.gateway.marie.operator_routes import (
            read_operator_runtime_snapshot,
        )
        from tools.stress.llm_dispatch_stress_suite import (
            validate_clean_dispatch_runtime,
        )

        snapshot = await read_operator_runtime_snapshot(
            fabric_group_id='default',
            limit=10,
            policy_repository=repository,
        )
        assert validate_clean_dispatch_runtime(snapshot)['running_dispatchers'] == 1
        cleanup = RequestStore.for_producer(queue_url, fabric_id='default')
        assert cleanup.resolve_route('default')
        await runtime.stop()

        captured = copy.deepcopy(data)
        if change == 'transport':
            captured['metadata']['llm_dispatch']['endpoints'][0][
                'base_url'
            ] = 'https://replacement.example/v1'
        else:
            captured['metadata']['llm_dispatch']['endpoints'][0][
                'endpoint_id'
            ] = 'primary-1'
            captured['lanes'][0]['metadata']['llm_dispatch'][
                'endpoint_id'
            ] = 'primary-1'
        request = save_request(conn, repository, captured, 1)
        # A subsequent draft edit must not change the scheduled snapshot.
        conn.execute(f'UPDATE {repository.config_schema}.llm_queue_pool SET quantum=99')
        other_owner = cleanup.acquire_owner('other-gateway', lease_ms=10_000)
        try:
            with pytest.raises(RuntimeFailToStart, match='dispatch_startup_owner_busy'):
                await runtime.start()
        finally:
            cleanup.release_owner(other_owner)
        assert (
            repository.load_startup_dispatch_policy('default').request_id
            == request.request_id
        )

        activation = repository.load_runtime_dispatch_policy('default')
        from types import SimpleNamespace

        record = route_record(
            SimpleNamespace(generation=1, policy_digest=activation['policy_digest'])
        )
        conn.execute(
            f'INSERT INTO {repository.config_schema}.job VALUES(%s,%s)',
            (record['work_unit_id'], 'created'),
        )
        assert (
            len(
                conn.execute(
                    insert_llm_routes(repository.config_schema), (Jsonb([record]),)
                ).fetchall()
            )
            == 1
        )
        with pytest.raises(RuntimeFailToStart, match='dispatch_startup_work_pending'):
            await runtime.start()
        assert (
            repository.load_runtime_dispatch_policy('default')['policy_generation'] == 1
        )
        assert (
            repository.load_startup_dispatch_policy('default').request_id
            == request.request_id
        )
        conn.execute(f'UPDATE {repository.config_schema}.job SET state=\'completed\'')

        await runtime.start()
        for _ in range(200):
            if runtime._dispatcher._owner_ready:
                break
            await asyncio.sleep(0.01)
        assert runtime._dispatcher._owner_ready
        assert runtime._dispatcher.policy_generation == 2
        if change == 'transport':
            assert (
                runtime._dispatcher.endpoints['primary'].base_url
                == 'https://replacement.example/v1'
            )
        else:
            assert set(runtime._dispatcher.endpoints) == {'primary-1'}
            assert (
                runtime._dispatcher.endpoints['primary-1'].base_url
                == 'https://inference.example/v1'
            )
        assert runtime._dispatcher.lanes[0].quantum == 3
        assert repository.load_startup_dispatch_policy('default') is None
    finally:
        await runtime.stop()
        if cleanup:
            keys = list(cleanup.client.scan_iter(match=cleanup.keys.prefix + '*'))
            if keys:
                cleanup.client.delete(*keys)
            cleanup.close()


def test_route_insert_waits_for_activation_and_rejects_stale_generation(database):
    conn, repository, data = database
    first = repository.activate_admission_policy('default', 'operator')
    changed = copy.deepcopy(data)
    changed['lanes'][0]['quantum'] = 7
    request = save_request(conn, repository, changed, first.generation)
    record = route_record(first)
    conn.execute(
        f'INSERT INTO {repository.config_schema}.job VALUES(%s,%s)',
        (record['work_unit_id'], 'created'),
    )
    locked, release = Event(), Event()

    def store_check():
        locked.set()
        assert release.wait(5)

    inserting = psycopg.connect(conn.info.dsn, autocommit=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            activation = workers.submit(
                repository.activate_startup_policy, 'default', request, store_check
            )
            assert locked.wait(3)
            insert = workers.submit(
                lambda: inserting.execute(
                    insert_llm_routes(repository.config_schema), (Jsonb([record]),)
                ).fetchall()
            )
            try:
                for _ in range(200):
                    row = conn.execute(
                        'SELECT wait_event_type FROM pg_stat_activity WHERE pid=%s',
                        (inserting.info.backend_pid,),
                    ).fetchone()
                    if row and row[0] == 'Lock':
                        break
                    time.sleep(0.01)
                assert row[0] == 'Lock'
                assert not insert.done()
            finally:
                release.set()
            assert activation.result(timeout=5).generation == 2
            assert insert.result(timeout=5) == []
            assert (
                conn.execute(
                    f'SELECT COUNT(*) FROM {repository.config_schema}.llm_job_route'
                ).fetchone()[0]
                == 0
            )
    finally:
        release.set()
        inserting.close()


def test_ownership_loss_before_commit_rolls_back_startup_activation(database):
    conn, repository, data = database
    first = repository.activate_admission_policy('default', 'operator')
    changed = copy.deepcopy(data)
    changed['lanes'][0]['quantum'] = 7
    request = save_request(conn, repository, changed, first.generation)
    checks = []

    def store_check():
        checks.append(True)
        if len(checks) == 2:
            raise RuntimeError('ownership lost before commit')

    with pytest.raises(RuntimeError, match='ownership lost'):
        repository.activate_startup_policy('default', request, store_check)
    assert (
        repository.load_runtime_dispatch_policy('default')['policy_generation']
        == first.generation
    )
    assert (
        repository.load_startup_dispatch_policy('default').request_id
        == request.request_id
    )
    assert (
        conn.execute(
            f'SELECT COUNT(*) FROM {repository.config_schema}.llm_queue_policy_generation'
        ).fetchone()[0]
        == 1
    )


@pytest.mark.asyncio
async def test_cancelling_startup_waits_for_activation_rollback_before_releasing_owner(
    database, monkeypatch
):
    conn, repository, data = database
    first = repository.activate_admission_policy('default', 'operator')
    changed = copy.deepcopy(data)
    changed['lanes'][0]['quantum'] = 7
    request = save_request(conn, repository, changed, first.generation)
    url = json.loads(Path(os.environ['MARIE_LLM_QUEUE_TEST_STORES']).read_text())[
        'valkey'
    ]['url']
    cleanup = RequestStore(
        url, fabric_id='default', limits=StoreLimits(max_execution_items=4)
    )
    entered, release, finished = Event(), Event(), Event()
    activate = repository.activate_startup_policy

    def blocked_activation(*args):
        entered.set()
        try:
            assert release.wait(5)
            return activate(*args)
        finally:
            finished.set()

    monkeypatch.setattr(repository, 'activate_startup_policy', blocked_activation)
    runtime = GatewayLlmDispatchRuntime(
        queue_config=LlmQueueRuntimeConfig(
            enabled=True, fabric_group_id='default', queue_url=url
        ),
        scheduler_config_source=DatabaseSchedulerConfigSource(repository, 'default'),
        config={'llm_dispatch': {'policy_source': 'database'}},
    )
    task = asyncio.create_task(runtime.start())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        try:
            await asyncio.sleep(0.05)
            assert not task.done()
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
            await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
        assert (
            repository.load_runtime_dispatch_policy('default')['policy_generation']
            == first.generation
        )
        assert (
            repository.load_startup_dispatch_policy('default').request_id
            == request.request_id
        )
        next_owner = cleanup.acquire_owner('next-gateway', lease_ms=1000)
        assert next_owner is not None
        cleanup.release_owner(next_owner)
    finally:
        release.set()
        await runtime.stop()
        keys = list(cleanup.client.scan_iter(match=cleanup.keys.prefix + '*'))
        if keys:
            cleanup.client.delete(*keys)
        cleanup.close()
