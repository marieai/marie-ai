import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio

from marie.scheduler.repository import JobRepository
from marie.storage.database.postgres_pool import AsyncPostgresConnection

BRANCH_ID = '00000000-0000-0000-0000-000000000001'
CHILD_ID = '00000000-0000-0000-0000-000000000002'
ATTEMPT_ID = '00000000-0000-0000-0000-000000000003'
OTHER_ID = '00000000-0000-0000-0000-000000000004'


@pytest_asyncio.fixture
async def database() -> AsyncIterator[
    tuple[JobRepository, AsyncPostgresConnection, str]
]:
    port = os.environ.get('MARIE_SCHEDULER_SQL_TEST_POSTGRES_PORT')
    if not port:
        pytest.skip(
            'Set MARIE_SCHEDULER_SQL_TEST_POSTGRES_PORT to an owned test database'
        )
    schema = 'branch_route_' + uuid4().hex
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
                    id uuid PRIMARY KEY, name text, state text,
                    completed_on timestamptz, output jsonb, branch_metadata jsonb,
                    lease_owner text, lease_expires_at timestamptz,
                    lease_epoch int DEFAULT 0, run_owner text, run_attempt_id uuid,
                    run_lease_expires_at timestamptz
                );
            ''')
            lease_sql = Path(
                'config/psql/schema/lease/001_lease_jobs_by_id.sql'
            ).read_text()
            await conn.execute(lease_sql.replace('{schema}', schema))
            await conn.execute(
                f'''INSERT INTO {schema}.job (id, name, state, run_owner, run_attempt_id)
                    VALUES (%s::uuid, 'extract', 'active', 'owner', %s::uuid),
                           (%s::uuid, 'extract', 'created', NULL, NULL)''',
                BRANCH_ID,
                ATTEMPT_ID,
                CHILD_ID,
            )

            class Pool:
                @asynccontextmanager
                async def acquire(self) -> AsyncIterator[AsyncPostgresConnection]:
                    yield conn

            yield JobRepository(config={}, pool=Pool()), conn, schema


async def commit(
    repository: JobRepository,
    schema: str,
    *,
    owner: str = 'owner',
    attempt: str = ATTEMPT_ID,
    skipped_ids: list[str] | None = None,
) -> tuple[bool, set[str]]:
    return await repository.commit_branch_route(
        job_id=BRANCH_ID,
        queue_name='extract',
        run_owner=owner,
        run_attempt_id=attempt,
        branch_metadata={'selected_path_ids': ['selected']},
        skipped_job_ids=[CHILD_ID] if skipped_ids is None else skipped_ids,
        skip_metadata={'skip_reason': {'branch_node_id': BRANCH_ID}},
        schema=schema,
    )


@pytest.mark.parametrize('initial', ['created', 'retry', 'skipped'])
async def test_branch_completion_commits_skips_and_rejects_their_leases(
    database, initial: str
) -> None:
    repository, conn, schema = database
    await conn.execute(
        f'UPDATE {schema}.job SET state = %s WHERE id = %s::uuid', initial, CHILD_ID
    )

    assert await commit(repository, schema) == (True, {CHILD_ID})
    assert await conn.fetch(
        f'SELECT state, run_owner FROM {schema}.job ORDER BY id'
    ) == [('completed', None), ('skipped', None)]
    assert await conn.fetchval(
        f'SELECT branch_metadata FROM {schema}.job WHERE id = %s::uuid',
        BRANCH_ID,
    ) == {'selected_path_ids': ['selected']}
    assert (
        str(
            await conn.fetchval(
                f'SELECT run_attempt_id FROM {schema}.job WHERE id = %s::uuid',
                BRANCH_ID,
            )
        )
        == ATTEMPT_ID
    )
    assert await conn.fetchval(
        f'SELECT branch_metadata FROM {schema}.job WHERE id = %s::uuid',
        CHILD_ID,
    ) == {'skipped': True, 'skip_reason': {'branch_node_id': BRANCH_ID}}
    assert (
        await conn.fetchval(
            f'SELECT {schema}.lease_jobs_by_id(%s::uuid[])',
            [CHILD_ID],
        )
        == []
    )


@pytest.mark.parametrize('mismatch', ['owner', 'attempt'])
async def test_stale_branch_attempt_changes_no_rows(database, mismatch: str) -> None:
    repository, conn, schema = database
    assert await commit(
        repository,
        schema,
        owner='stale' if mismatch == 'owner' else 'owner',
        attempt=OTHER_ID if mismatch == 'attempt' else ATTEMPT_ID,
    ) == (False, set())
    assert await conn.fetch(
        f'SELECT state, branch_metadata FROM {schema}.job ORDER BY id'
    ) == [('active', None), ('created', None)]


async def test_skip_database_error_rolls_back_branch_completion(database) -> None:
    repository, conn, schema = database
    await conn.execute(f'''
        CREATE FUNCTION {schema}.reject_skip() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.state = 'skipped' THEN RAISE EXCEPTION 'injected skip failure'; END IF;
            RETURN NEW;
        END; $$;
        CREATE TRIGGER reject_skip BEFORE UPDATE ON {schema}.job
        FOR EACH ROW EXECUTE FUNCTION {schema}.reject_skip();
    ''')
    with pytest.raises(psycopg.errors.RaiseException, match='injected skip failure'):
        await commit(repository, schema)
    assert await conn.fetch(
        f'SELECT state, branch_metadata, completed_on FROM {schema}.job ORDER BY id'
    ) == [('active', None, None), ('created', None, None)]

    await conn.execute(f'DROP TRIGGER reject_skip ON {schema}.job')
    assert await commit(repository, schema) == (True, {CHILD_ID})


async def test_partial_skip_conflict_rolls_back_all_changes(database) -> None:
    repository, conn, schema = database
    await conn.execute(
        f"INSERT INTO {schema}.job (id, name, state) VALUES (%s::uuid, 'extract', 'active')",
        OTHER_ID,
    )
    with pytest.raises(RuntimeError, match='could not skip all unselected jobs'):
        await commit(repository, schema, skipped_ids=[CHILD_ID, OTHER_ID])
    assert await conn.fetch(
        f'SELECT state, branch_metadata FROM {schema}.job ORDER BY id'
    ) == [('active', None), ('created', None), ('active', None)]


async def test_branch_with_no_unselected_jobs_can_complete(database) -> None:
    repository, conn, schema = database
    assert await commit(repository, schema, skipped_ids=[]) == (True, set())
    assert (
        await conn.fetchval(
            f'SELECT state FROM {schema}.job WHERE id = %s::uuid',
            BRANCH_ID,
        )
        == 'completed'
    )
