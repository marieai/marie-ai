import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from psycopg.types.json import Jsonb

import marie.scheduler.repository.async_job_repository as repository_module
from marie.scheduler.repository import JobRepository
from marie.scheduler.state import WorkState
from marie.storage.database.postgres_pool import AsyncPostgresConnection

OLDER_ID = '00000000-0000-0000-0000-000000000001'
NEWER_ID = '00000000-0000-0000-0000-000000000002'
UNRELATED_ID = '00000000-0000-0000-0000-000000000003'


@pytest_asyncio.fixture
async def database(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[JobRepository, AsyncPostgresConnection, str]]:
    port = os.environ.get('MARIE_SCHEDULER_SQL_TEST_POSTGRES_PORT')
    if not port:
        pytest.skip(
            'Set MARIE_SCHEDULER_SQL_TEST_POSTGRES_PORT to an owned test database'
        )
    schema = 'job_policy_' + uuid4().hex
    monkeypatch.setattr(repository_module, 'DEFAULT_SCHEMA', schema)
    async with await psycopg.AsyncConnection.connect(
        host='127.0.0.1',
        port=int(port),
        user='postgres',
        dbname='postgres',
        autocommit=True,
    ) as raw:
        async with raw.transaction(force_rollback=True):
            conn = AsyncPostgresConnection(raw)
            await conn.execute(f'''
                CREATE SCHEMA {schema};
                CREATE TABLE {schema}.job (
                    id uuid PRIMARY KEY, name text DEFAULT 'extract', priority int DEFAULT 0,
                    state text, retry_limit int DEFAULT 3, start_after timestamptz DEFAULT now(),
                    expire_in interval DEFAULT '1 minute', data jsonb,
                    retry_delay int DEFAULT 2, retry_backoff boolean DEFAULT false,
                    keep_until timestamptz DEFAULT now() + interval '1 day', dag_id uuid,
                    job_level int DEFAULT 0, soft_sla timestamptz, hard_sla timestamptz,
                    run_owner text, run_attempt_id uuid, branch_metadata jsonb,
                    created_on timestamptz NOT NULL
                );
            ''')

            class Pool:
                @asynccontextmanager
                async def acquire(self) -> AsyncIterator[AsyncPostgresConnection]:
                    yield conn

            yield JobRepository(config={}, pool=Pool()), conn, schema


@pytest.mark.parametrize('same_timestamp', [False, True])
async def test_policy_lookup_selects_latest_matching_job(
    database: tuple[JobRepository, AsyncPostgresConnection, str],
    same_timestamp: bool,
) -> None:
    repository, conn, schema = database
    metadata = Jsonb({'metadata': {'ref_type': 'invoice', 'ref_id': "ref'123"}})
    older_timestamp = (
        '2026-10-09T00:00:00Z' if same_timestamp else '2026-10-08T00:00:00Z'
    )
    await conn.execute(
        f'''INSERT INTO {schema}.job (id, state, data, created_on)
            VALUES (%s::uuid, 'completed', %s, %s::timestamptz),
                   (%s::uuid, 'active', %s, '2026-10-09T00:00:00Z'),
                   (%s::uuid, 'created', %s, '2026-10-10T00:00:00Z')''',
        OLDER_ID,
        metadata,
        older_timestamp,
        NEWER_ID,
        metadata,
        UNRELATED_ID,
        Jsonb({'metadata': {'ref_type': 'other', 'ref_id': "ref'123"}}),
    )

    job = await repository.get_job_by_policy('invoice', "ref'123")

    assert job is not None
    assert job.id == NEWER_ID
    assert job.state is WorkState.ACTIVE
    assert job.data['metadata'] == {'ref_type': 'invoice', 'ref_id': "ref'123"}
    assert await repository.get_job_by_policy('invoice', 'missing') is None


async def test_policy_lookup_returns_none_when_no_jobs_exist(
    database: tuple[JobRepository, AsyncPostgresConnection, str],
) -> None:
    repository, _, _ = database
    assert await repository.get_job_by_policy('invoice', 'missing') is None
