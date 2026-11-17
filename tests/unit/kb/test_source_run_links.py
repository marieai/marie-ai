import os
from uuid import uuid4

import pytest

from tests.unit.kb.test_catalog import catalog as catalog
from tests.unit.kb.test_catalog import scope

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_SENSOR_TEST_PORT'),
    reason='Requires an owned disposable PostgreSQL database',
)


@pytest.mark.parametrize('task_state', ['present', 'missing', 'unavailable'])
async def test_repository_run_links_identify_the_latest_scoped_refresh(
    catalog, task_state
):
    service, pool, schema = catalog
    identity = scope()
    index = await service.call('create_index', {**identity, 'name': 'code'})
    source = (
        await service.call(
            'save_source',
            {
                **identity,
                'indexId': index['id'],
                'workspaceId': str(uuid4()),
                'expectedConfigurationRevision': 0,
                'sourceType': 'repository',
                'name': 'Repository',
            },
        )
    )['source']
    await pool.execute(
        f'''CREATE TABLE {schema}.dag (
        id uuid PRIMARY KEY,project_id text,ref_id text,planner text,
        state text,created_on timestamptz,completed_on timestamptz)'''
    )
    older, latest, foreign, other_planner = [uuid4() for _ in range(4)]
    for run_id, tenant, planner, created, state in [
        (
            older,
            identity['tenant_id'],
            'kb_repository_refresh',
            '2026-11-16Z',
            'completed',
        ),
        (
            latest,
            identity['tenant_id'],
            'kb_repository_refresh',
            '2026-11-17Z',
            'failed',
        ),
        (foreign, str(uuid4()), 'kb_repository_refresh', '2026-11-18Z', 'completed'),
        (
            other_planner,
            identity['tenant_id'],
            'kb_datasource_sync',
            '2026-11-19Z',
            'completed',
        ),
    ]:
        await pool.execute(
            f'''INSERT INTO {schema}.dag VALUES($1,$2,$3,$4,$5,$6::text::timestamptz,NULL)''',
            run_id,
            tenant,
            source['id'],
            planner,
            state,
            created,
        )
    task = uuid4()
    if task_state != 'unavailable':
        await pool.execute(
            f'''CREATE TABLE {schema}.job_search_document (
            dag_id uuid,job_id uuid,endpoint text)'''
        )
        for dag_id, job_id, endpoint in [
            (older, uuid4(), 'repository_executor://refresh_repository'),
            (latest, uuid4(), 'noop://noop'),
            (
                latest,
                task,
                (
                    'repository_executor://refresh_repository'
                    if task_state == 'present'
                    else 'noop://noop'
                ),
            ),
        ]:
            await pool.execute(
                f'INSERT INTO {schema}.job_search_document VALUES($1,$2,$3)',
                dag_id,
                job_id,
                endpoint,
            )
    sources = await service.call(
        'list_sources', {**identity, 'indexId': index['id'], 'sourceType': 'repository'}
    )
    assert sources[0]['latestRun'] == {
        'dagId': str(latest),
        'jobId': str(task) if task_state == 'present' else None,
        'state': 'failed',
        'completedOn': None,
    }
