"""Owned PostgreSQL schema066 plus actual V3 startup and producer policy join."""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import psycopg
import pytest
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.scheduler_config import DatabaseSchedulerConfigSource
from marie.engine.llm_queue.store import RequestStore
from psycopg.types.json import Jsonb

from marie.excepts import RuntimeFailToStart
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.gateway.marie.llm_scheduler_config import (
    PostgresSchedulerConfigRepository,
)
from tests.unit.serve.runtimes.gateway.test_dispatch_policy import persisted


class _PolicyDatabase:
    def __init__(self) -> None:
        self.data = persisted()
        self.data['enabled'] = True
        self.data['admission_mode'] = 'shadow'
        self.active_generation = None
        self.generations: list[tuple[int, str, dict, str, datetime]] = []
        self.commits = 0
        self.rollbacks = 0

    def connection(self):
        database = self

        class Cursor:
            result = None
            rows = None

            def execute(self, query, params=None):
                compact = ' '.join(query.split())
                if compact.startswith('SET LOCAL'):
                    return
                if 'SELECT config.active_policy_generation' in compact:
                    if database.active_generation is None:
                        self.result = (None, None, None)
                    else:
                        found = next(
                            row
                            for row in database.generations
                            if row[0] == database.active_generation
                        )
                        self.result = (found[0], found[1], found[2])
                elif 'SELECT policy, total_concurrent_dispatch, enabled' in compact:
                    self.result = (
                        database.data['policy'],
                        database.data['total_concurrent_dispatch'],
                        database.data['enabled'],
                        database.data['metadata'],
                        database.data['admission_mode'],
                        database.active_generation,
                    )
                elif 'FROM marie_scheduler.llm_queue_pool' in compact:
                    self.rows = [
                        (
                            lane['pool_id'],
                            lane.get('display_name'),
                            lane.get('endpoint_url'),
                            lane.get('quantum', 1),
                            lane.get('min_concurrent', 0),
                            lane.get('max_concurrent'),
                            lane.get('max_burst_per_visit'),
                            lane.get('enabled', True),
                            lane.get('metadata', {}),
                        )
                        for lane in database.data['lanes']
                    ]
                elif 'SELECT generation, activated_by, created_on' in compact:
                    digest = params[1]
                    found = next(
                        (row for row in database.generations if row[1] == digest),
                        None,
                    )
                    self.result = (
                        (found[0], found[3], found[4]) if found is not None else None
                    )
                elif 'SELECT COALESCE(MAX(generation), 0)' in compact:
                    self.result = (
                        max((row[0] for row in database.generations), default=0),
                    )
                elif (
                    'INSERT INTO marie_scheduler.llm_queue_policy_generation' in compact
                ):
                    created_on = datetime(2026, 9, 21, tzinfo=timezone.utc)
                    generation, digest, snapshot, actor = params[1:]
                    database.generations.append(
                        (generation, digest, snapshot.obj, actor, created_on)
                    )
                    self.result = (created_on,)
                elif 'UPDATE marie_scheduler.llm_queue_fabric_config' in compact:
                    database.active_generation = params[0]
                elif 'SELECT active_policy_generation' in compact:
                    self.result = (database.active_generation,)
                elif 'SELECT policy_digest, policy_snapshot' in compact:
                    generation = params[1]
                    found = next(
                        row for row in database.generations if row[0] == generation
                    )
                    self.result = (found[1], found[2])
                else:
                    raise AssertionError(compact)

            def fetchone(self):
                return self.result

            def fetchall(self):
                return self.rows

            def close(self):
                pass

        return SimpleNamespace(
            cursor=Cursor,
            commit=lambda: setattr(database, 'commits', database.commits + 1),
            rollback=lambda: setattr(database, 'rollbacks', database.rollbacks + 1),
        )


def _memory_repository(database: _PolicyDatabase) -> PostgresSchedulerConfigRepository:
    repository = object.__new__(PostgresSchedulerConfigRepository)
    repository.config_schema = 'marie_scheduler'
    repository._get_connection = database.connection
    repository._close_cursor = lambda cursor: cursor.close()
    repository._close_connection = lambda connection: None
    return repository


def test_invalid_activation_keeps_previous_generation() -> None:
    database = _PolicyDatabase()
    repository = _memory_repository(database)

    first = repository.activate_admission_policy('default', 'operator-1')
    conflicting = persisted()['lanes'][0]
    conflicting['pool_id'] = 'also-default'
    conflicting['metadata']['admission']['match'] = {
        'workload.kind': {'eq': 'document'}
    }
    database.data['lanes'].append(conflicting)

    with pytest.raises(ValueError, match='routing_policy_ambiguous'):
        repository.activate_admission_policy('default', 'operator-1')

    active = repository.load_active_admission_policy('default')
    assert first.generation == 1
    assert active.generation == first.generation
    assert database.active_generation == 1
    assert database.rollbacks == 1


def test_reactivating_same_policy_reuses_generation() -> None:
    database = _PolicyDatabase()
    repository = _memory_repository(database)

    first = repository.activate_admission_policy('default', 'operator-1')
    second = repository.activate_admission_policy('default', 'operator-2')

    assert second.generation == first.generation
    assert second.policy_digest == first.policy_digest
    assert len(database.generations) == 1


def test_runtime_uses_immutable_activated_dispatch_snapshot() -> None:
    database = _PolicyDatabase()
    repository = _memory_repository(database)
    activated = repository.activate_admission_policy('default', 'operator-1')
    database.data['lanes'][0]['quantum'] = 99

    runtime_policy = repository.load_runtime_dispatch_policy('default')

    assert runtime_policy['policy_generation'] == activated.generation
    assert runtime_policy['policy_digest'] == activated.policy_digest
    assert runtime_policy['lanes'][0]['quantum'] != 99


@pytest.mark.asyncio
@pytest.mark.parametrize('engine', ['redis', 'valkey'])
async def test_actual_postgres_policy_selects_v3_startup_and_producer_limits(engine):
    port = os.getenv('MARIE_LLM_DISPATCH_TEST_POSTGRES_PORT')
    if not port:
        pytest.skip('Owned task7 PostgreSQL fixture required')
    manifest = json.loads(Path(os.environ['MARIE_LLM_QUEUE_TEST_STORES']).read_text())
    queue_url = manifest[engine]['url']
    identity = 'pg-' + uuid4().hex
    schema = 'task7_' + uuid4().hex
    config = {
        'hostname': '127.0.0.1',
        'port': int(port),
        'username': 'postgres',
        'password': '',
        'database': 'postgres',
        'schema': schema,
        'min_pool_size': 1,
        'max_pool_size': 1,
    }
    connection = psycopg.connect(
        host='127.0.0.1',
        port=int(port),
        user='postgres',
        dbname='postgres',
        autocommit=True,
    )
    repository = runtime = cleanup = None
    try:
        connection.execute(f'CREATE SCHEMA {schema}')
        ddl = (
            Path(__file__).resolve().parents[5]
            / 'config/psql/schema/066_llm_queue_scheduler.sql'
        ).read_text()
        connection.execute(ddl.replace('{schema}', schema))
        routing_ddl = (
            Path(__file__).resolve().parents[5]
            / 'config/psql/schema/089_llm_queue_admission_routing.sql'
        ).read_text()
        connection.execute(routing_ddl.replace('{schema}', schema))
        data = persisted()
        data['metadata']['llm_dispatch']['endpoints'][0].pop('credential_env')
        connection.execute(
            f'INSERT INTO {schema}.llm_queue_fabric_config(fabric_group_id,policy,total_concurrent_dispatch,metadata) VALUES(%s,%s,%s,%s)',
            (identity, 'drr', 4, Jsonb(data['metadata'])),
        )
        lane = data['lanes'][0]
        connection.execute(
            f'INSERT INTO {schema}.llm_queue_pool(fabric_group_id,pool_id,quantum,min_concurrent,max_concurrent,metadata,endpoint_url) VALUES(%s,%s,%s,%s,%s,%s,%s)',
            (
                identity,
                'default',
                3,
                1,
                2,
                Jsonb(lane['metadata']),
                'https://untrusted-old-column.example',
            ),
        )
        repository = PostgresSchedulerConfigRepository(config)
        loaded = repository.load_scheduler_config(identity)
        assert loaded['metadata'] == data['metadata']
        assert loaded['lanes'][0]['quantum'] == 3
        activated = repository.activate_admission_policy(identity, 'operator-test')
        active_policy = repository.load_active_admission_policy(identity)
        assert activated.generation == 1
        assert active_policy.policy_digest == activated.policy_digest
        assert active_policy.match({'workload.kind': 'document'}).pool_id == 'default'
        binding = active_policy.endpoint_binding('default')
        assert binding.endpoint_group_id == 'primary'
        assert binding.revision == 'r1'
        runtime = GatewayLlmDispatchRuntime(
            queue_config=LlmQueueConfig(
                enabled=True,
                queue_contract_version='v3',
                fabric_group_id=identity,
                queue_url=queue_url,
            ),
            scheduler_config_source=DatabaseSchedulerConfigSource(repository, identity),
            config={'llm_dispatch': {'policy_source': 'database'}},
        )
        await runtime.start()
        joined = RequestStore.for_producer(queue_url, fabric_id=identity)
        try:
            assert joined.limits.max_execution_items == 4
            for _ in range(100):
                route = joined.resolve_route('default')
                if route:
                    break
                await asyncio.sleep(0.01)
            assert route['endpoint_id'] == 'primary'
            assert runtime._dispatcher.lanes[0].execution_limit == 2
            connection.execute(
                f'UPDATE {schema}.llm_queue_pool SET quantum=7,min_concurrent=0,enabled=false WHERE fabric_group_id=%s',
                (identity,),
            )
            runtime._dispatcher._next_refresh = 0
            for _ in range(200):
                if (
                    not runtime._dispatcher.lanes[0].enabled
                    and runtime._dispatcher._config_revision >= 2
                ):
                    break
                await asyncio.sleep(0.01)
            assert runtime._dispatcher.lanes[0].quantum == 7
            assert runtime._dispatcher.lanes[0].enabled is False
            assert joined.resolve_route('default') is None
            blocker = psycopg.connect(
                host='127.0.0.1', port=int(port), user='postgres', dbname='postgres'
            )
            try:
                blocker.execute(
                    f'LOCK TABLE {schema}.llm_queue_fabric_config IN ACCESS EXCLUSIVE MODE'
                )
                runtime._dispatcher._next_refresh = 0
                for _ in range(300):
                    if runtime._dispatcher._policy_paused:
                        break
                    await asyncio.sleep(0.01)
                assert runtime._dispatcher._policy_paused
                assert runtime._dispatcher._category == 'policy_refresh_unavailable'
                assert runtime._dispatcher.owner is not None
                assert not runtime._dispatcher._runner.done()
            finally:
                blocker.rollback()
                blocker.close()
            runtime._dispatcher._next_refresh = 0
            for _ in range(200):
                if not runtime._dispatcher._policy_paused:
                    break
                await asyncio.sleep(0.01)
            assert not runtime._dispatcher._policy_paused
            assert (
                runtime._dispatcher.endpoints['primary'].base_url
                == 'https://inference.example/v1'
            )
        finally:
            joined.close()
        await runtime.stop()
        data['metadata']['llm_dispatch']['limits']['max_execution_items'] = 5
        connection.execute(
            f'UPDATE {schema}.llm_queue_fabric_config SET metadata=%s WHERE fabric_group_id=%s',
            (Jsonb(data['metadata']), identity),
        )
        with pytest.raises(RuntimeFailToStart):
            await runtime.start()
        cleanup = RequestStore.for_producer(queue_url, fabric_id=identity)
        assert cleanup.limits.max_execution_items == 4
    finally:
        if runtime:
            await runtime.stop()
        if cleanup:
            keys = list(cleanup.client.scan_iter(match=cleanup.keys.prefix + '*'))
            if keys:
                cleanup.client.delete(*keys)
            cleanup.close()
        if repository:
            repository.postgreSQL_pool.close()
        connection.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
        connection.close()
