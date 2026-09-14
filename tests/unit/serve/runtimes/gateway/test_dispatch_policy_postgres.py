"""Owned PostgreSQL schema066 plus actual V3 startup and producer policy join."""

import asyncio
import json
import os
from pathlib import Path
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
