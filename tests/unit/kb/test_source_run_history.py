import json
import os
from uuid import uuid4

import pytest

from tests.unit.kb.test_catalog import catalog as catalog
from tests.unit.kb.test_catalog import scope

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_SENSOR_TEST_PORT'),
    reason='Requires an owned disposable PostgreSQL database',
)


async def repository(service, identity):
    index = await service.call(
        'create_index', {**identity, 'name': 'history', 'workspaceId': str(uuid4())}
    )
    source = (
        await service.call(
            'save_source',
            {
                **identity,
                'indexId': index['id'],
                'expectedConfigurationRevision': 0,
                'sourceType': 'repository',
                'name': 'Repository',
            },
        )
    )['source']
    return index, source


async def test_history_uses_scoped_scheduler_runs_and_original_trigger_metadata(
    catalog,
):
    service, pool, schema = catalog
    identity = scope()
    index, source = await repository(service, identity)
    await pool.execute(
        f'''CREATE TABLE {schema}.dag (
        id uuid PRIMARY KEY,project_id text,ref_id text,planner text,state text,
        created_on timestamptz,started_on timestamptz,completed_on timestamptz);
        CREATE TABLE {schema}.job(id uuid PRIMARY KEY,dag_id uuid,data jsonb);'''
    )
    ids = [uuid4() for _ in range(6)]
    sensor = str(uuid4())
    for number, dag_id in enumerate(ids):
        await pool.execute(
            f'''INSERT INTO {schema}.dag VALUES($1,$2,$3,$4,'completed',
            '2026-11-17Z'::timestamptz + ($5::int * interval '1 minute'),NULL,NULL)''',
            dag_id,
            str(uuid4()) if number == 4 else identity['tenant_id'],
            str(uuid4()) if number == 5 else source['id'],
            'kb_datasource_sync' if number == 3 else 'kb_repository_refresh',
            number,
        )
    manual_job = uuid4()
    for dag_id, job_id, data in [
        (ids[0], uuid4(), {'metadata': {'on': 'noop://noop'}}),
        (
            ids[0],
            manual_job,
            {
                'metadata': {
                    'on': 'repository_executor://refresh_repository',
                    'user_id': identity['actor_id'],
                    'trigger': 'manual',
                    'private': 'must-not-leak',
                }
            },
        ),
        (
            ids[1],
            uuid4(),
            {
                'trigger': 'schedule',
                'source_sensor_id': sensor,
                'sensor_name': 'Repository refresh',
                'run_config': {'scheduled_at': '2026-11-17T00:01:00Z'},
                'metadata': {
                    'on': 'repository_executor://refresh_repository',
                    'user_id': identity['actor_id'],
                },
            },
        ),
    ]:
        await pool.execute(
            f'INSERT INTO {schema}.job VALUES($1,$2,$3::jsonb)',
            job_id,
            dag_id,
            json.dumps(data),
        )
    command = {
        **identity,
        'indexId': index['id'],
        'sourceId': source['id'],
        'pageSize': 2,
    }
    first = await service.call('source_runs', command)
    assert first['total'] == 3
    assert [run['dagId'] for run in first['runs']] == [str(ids[2]), str(ids[1])]
    assert first['runs'][0]['trigger'] == 'unknown'
    assert first['runs'][0]['jobId'] is None
    assert first['runs'][1]['trigger'] == 'schedule'
    assert first['runs'][1]['sensorId'] == sensor
    assert first['runs'][1]['sensorName'] == 'Repository refresh'
    second = await service.call('source_runs', {**command, 'page': 2})
    assert second['runs'][0]['dagId'] == str(ids[0])
    assert second['runs'][0]['jobId'] == str(manual_job)
    assert second['runs'][0]['trigger'] == 'manual'
    assert second['runs'][0]['actorId'] == identity['actor_id']
    assert 'must-not-leak' not in json.dumps([first, second])
    from marie.kb.catalog import CatalogError

    for mismatch in [
        {'tenant_id': str(uuid4())},
        {'organization_id': str(uuid4())},
        {'sourceId': str(uuid4())},
    ]:
        with pytest.raises(CatalogError) as error:
            await service.call('source_runs', {**command, **mismatch})
        assert error.value.status_code == 404


async def test_empty_history_is_available_without_scheduler_tables(catalog):
    service, _pool, _schema = catalog
    identity = scope()
    index, source = await repository(service, identity)
    assert await service.call(
        'source_runs', {**identity, 'indexId': index['id'], 'sourceId': source['id']}
    ) == {'runs': [], 'total': 0}


async def test_saved_schedule_uses_the_existing_sensor_and_submission_envelope(
    catalog, monkeypatch
):
    from datetime import datetime, timezone

    from marie.sensors.context import SensorEvaluationContext
    from marie.sensors.daemon.worker import build_work_info, repository_sensor
    from marie.sensors.definitions import schedule_sensor as module
    from marie.sensors.definitions.schedule_sensor import ScheduleSensor
    from marie.sensors.types import SensorType

    service, pool, schema = catalog
    await pool.execute(
        f'''CREATE TABLE {schema}.sensor (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),external_id uuid UNIQUE,name text,config jsonb,
        target_job_name text,target_dag_id uuid,cursor text,minimum_interval_seconds int DEFAULT 30,
        sensor_type text,status text DEFAULT 'inactive',updated_at timestamptz DEFAULT now());'''
    )
    identity = scope()
    index = await service.call(
        'create_index', {**identity, 'name': 'scheduled', 'workspaceId': str(uuid4())}
    )
    saved = await service.call(
        'save_source',
        {
            **identity,
            'indexId': index['id'],
            'expectedConfigurationRevision': 0,
            'sourceType': 'repository',
            'name': 'Repository',
            'datasetInputs': {'repositoryId': str(uuid4())},
            'syncMode': 'scheduled',
            'schedule': {'cron': '0 * * * *', 'timezone': 'UTC'},
        },
    )
    source = saved['source']
    assert source['schedule']['status'] == 'inactive'
    await service.call(
        'set_source_schedule',
        {
            **identity,
            'indexId': index['id'],
            'sourceId': source['id'],
            'expectedConfigurationRevision': 1,
            'action': 'activate',
        },
    )
    record = await pool.fetchrow(
        f'SELECT * FROM {schema}.sensor WHERE external_id=$1', source['triggerConfigId']
    )
    assert record['status'] == 'active'
    assert repository_sensor(record)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 17, 10, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(module, 'datetime', Clock)

    class SourcePool:
        async def fetchrow(self, statement, *arguments):
            return await pool.fetchrow(
                statement.replace('marie_scheduler.', f'{schema}.'), *arguments
            )

    context = SensorEvaluationContext(
        str(record['id']),
        record['name'],
        SensorType.SCHEDULE,
        cursor='2026-11-17T09:01:00+00:00',
    )
    context.resources['postgres_pool'] = SourcePool()
    result = await ScheduleSensor(record).evaluate(context)
    assert len(result.run_requests) == 1
    request = result.run_requests[0]
    assert request.job_name == 'kb_repository_refresh'
    work = build_work_info(
        request, str(record['id']), record['name'], request.job_name, None
    )
    assert work.data['trigger'] == 'schedule'
    assert work.data['source_sensor_id'] == str(record['id'])
    assert work.data['metadata']['source_id'] == source['id']
    assert work.data['metadata']['ref_id'] == source['id']
    assert work.data['metadata']['planner'] == 'kb_repository_refresh'
    assert work.data['run_config']['scheduled_at'] == '2026-11-17T10:00:00+00:00'
    context.cursor = result.cursor
    assert not (await ScheduleSensor(record).evaluate(context)).run_requests
