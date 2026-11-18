import json
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest

from marie.kb.code.repository_operations import RepositoryOperations
from marie.kb.item_state import StaleIndexingRun
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool
from tests.unit.runtime.test_git_provider_schema import install
from tests.unit.runtime.test_runtime_schema import database

SCHEMAS = Path(__file__).parents[3] / 'config/psql/schema'


@pytest.fixture
async def operations(database, tmp_path, monkeypatch):
    monkeypatch.setenv('FIXTURE_GIT_TOKEN', 'test-only-token')
    install(database)
    database.execute((SCHEMAS / '105_git_repository_discovery.sql').read_text())
    database.execute('CREATE SCHEMA marie_scheduler')
    for filename in ['007_dag.sql', '002_enums.sql', '005_job.sql']:
        database.execute(
            (SCHEMAS / filename).read_text().replace('{schema}', 'marie_scheduler')
        )
    org, user, connection_id = uuid4(), uuid4(), uuid4()
    database.execute(
        "INSERT INTO marie_runtime.organization(id,name,slug,owner_id) VALUES(%s,'Fixture','fixture',%s)",
        (org, user),
    )
    database.execute(
        '''INSERT INTO marie_runtime.git_provider_connections(id,organization_id,owner_key,name,provider,provider_url,secret_ref)
        VALUES(%s,%s,%s,'Fixture','gitea','https://git.test','env:FIXTURE_GIT_TOKEN')''',
        (connection_id, org, f'organization:{org}'),
    )
    info = database.info
    pool = AsyncPostgresConnectionPool()
    await pool.initialize(
        {
            'hostname': info.host,
            'port': info.port,
            'username': info.user,
            'password': '',
            'database': info.dbname,
        },
        autocommit=True,
    )
    worker = RepositoryOperations(pool, [str(tmp_path)])

    async def execution():
        params = {
            'job_id': str(uuid4()),
            'run_attempt_id': str(uuid4()),
            'run_owner': 'fixture',
        }
        dag = uuid4()
        metadata = {
            'planner': 'repository_management',
            'repository_operation': 'discover',
            'connection_id': str(connection_id),
            'organization_id': str(org),
            'user_id': str(user),
        }
        await pool.execute(
            "INSERT INTO marie_scheduler.dag(id,state,name) VALUES($1,'active','repository_management')",
            dag,
        )
        await pool.execute(
            "INSERT INTO marie_scheduler.job(id,dag_id,state,data,run_attempt_id,run_owner,run_lease_expires_at,name) VALUES($1,$2,'active',$3::jsonb,$4,'fixture',clock_timestamp()+interval '5 minutes','repository_management')",
            UUID(params['job_id']),
            dag,
            json.dumps({'metadata': metadata}),
            UUID(params['run_attempt_id']),
        )
        return params, dag

    yield worker, pool, connection_id, execution
    await pool.close()


async def test_complete_snapshot_is_immutable_and_older_run_cannot_replace_newer(
    operations, monkeypatch
):
    worker, pool, connection_id, execution = operations
    monkeypatch.setenv('FIXTURE_GIT_TOKEN', 'test-only-token')
    content = [{'id': '1'}]

    async def discover(*args):
        return list(content)

    monkeypatch.setattr(
        'marie.kb.code.repository_operations.discover_repositories', discover
    )
    older, old_dag = await execution()
    newer, new_dag = await execution()
    await worker.execute(newer)
    content[:] = [{'id': '2'}]
    await worker.execute(older)
    await worker.execute(newer)
    assert (
        await pool.fetchval(
            'SELECT discovered_dag_id FROM marie_runtime.git_provider_connections WHERE id=$1',
            connection_id,
        )
        == new_dag
    )
    assert await pool.fetchval(
        'SELECT repositories FROM marie_runtime.git_repository_discovery WHERE connection_id=$1 AND dag_id=$2',
        connection_id,
        new_dag,
    ) == [{'id': '1'}]


async def test_partial_failure_and_revocation_retain_prior_complete_snapshot(
    operations, monkeypatch
):
    worker, pool, connection_id, execution = operations
    monkeypatch.setenv('FIXTURE_GIT_TOKEN', 'test-only-token')

    async def first(*args):
        return [{'id': 'complete'}]

    monkeypatch.setattr(
        'marie.kb.code.repository_operations.discover_repositories', first
    )
    params, dag = await execution()
    await worker.execute(params)

    async def failed(*args):
        raise RuntimeError('Provider page failed')

    monkeypatch.setattr(
        'marie.kb.code.repository_operations.discover_repositories', failed
    )
    params, _ = await execution()
    with pytest.raises(RuntimeError, match='page failed'):
        await worker.execute(params)

    async def revoked(*args):
        await pool.execute(
            'UPDATE marie_runtime.git_provider_connections SET is_enabled=false WHERE id=$1',
            connection_id,
        )
        return [{'id': 'revoked'}]

    monkeypatch.setattr(
        'marie.kb.code.repository_operations.discover_repositories', revoked
    )
    with pytest.raises(PermissionError):
        await worker.execute(params)
    assert (
        await pool.fetchval(
            'SELECT discovered_dag_id FROM marie_runtime.git_provider_connections WHERE id=$1',
            connection_id,
        )
        == dag
    )
    assert (
        await pool.fetchval(
            'SELECT count(*) FROM marie_runtime.git_repository_discovery'
        )
        == 1
    )


async def test_stale_job_lease_cannot_read_a_connection(operations):
    worker, pool, _, execution = operations
    params, _ = await execution()
    await pool.execute(
        "UPDATE marie_scheduler.job SET run_lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=$1",
        UUID(params['job_id']),
    )
    with pytest.raises(StaleIndexingRun):
        await worker.execute(params)
