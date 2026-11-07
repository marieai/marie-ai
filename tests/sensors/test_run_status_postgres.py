import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from marie.sensors.definitions.run_status_sensor import RunStatusSensor
from marie.sensors.types import SensorType
from tests.sensors.test_sensor_postgres import runtime

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_SENSOR_TEST_PORT'),
    reason='Requires an owned disposable scheduler PostgreSQL service',
)


@pytest.fixture
async def completions(runtime):
    directory = Path(__file__).parents[2] / 'config/psql/schema'
    for name in ('008_dag_history.sql', '013_dag_history_trigger.sql', '050_kv_store_worker.sql'):
        await runtime.pool.execute((directory / name).read_text().replace('{schema}', 'marie_scheduler'))
    await runtime.pool.execute('TRUNCATE marie_scheduler.dag_history')
    runtime.worker._registry.register(SensorType.RUN_STATUS, RunStatusSensor)
    sid = await runtime.storage.create_sensor({
        'id': str(uuid4()),
        'external_id': str(uuid4()),
        'name': 'after extraction',
        'sensor_type': 'run_status',
        'status': 'active',
        'target_job_name': 'mock_connector_tool',
        'config': {
            'monitored_job_names': ['extract'], 'status_types': ['completed'],
            'pipeline_mode': True,
            'source_scope': {'project_ids': ['project-one']},
        },
    })
    yield runtime, await runtime.storage.get_sensor(sid)


async def source(connection, *, project='project-one', terminal=True):
    dag_id = uuid4()
    await connection.execute(
        "INSERT INTO marie_scheduler.dag(id,name,planner,state,project_id) VALUES($1,'source','extract','active',$2)",
        dag_id, project,
    )
    metadata = {
        'planner': 'extract', 'project_id': project,
        'uri': 's3://fixture/one.pdf', 'ref_id': str(dag_id), 'ref_type': 'document',
    }
    await connection.execute(
        """INSERT INTO marie_scheduler.job(id,name,dag_id,state,data,output)
        VALUES($1,'mock_connector_tool',$2,'completed',$3::jsonb,$4::jsonb)""",
        uuid4(), dag_id, json.dumps({'metadata': metadata}),
        json.dumps({'records': [{'moves': ['e4']}]}),
    )
    if terminal:
        await connection.execute(
            "UPDATE marie_scheduler.dag SET state='completed',completed_on=clock_timestamp() WHERE id=$1", dag_id
        )
    return str(dag_id)


async def test_observes_whole_workflow_completion_not_a_completed_node(completions):
    runtime, sensor = completions
    dag_id = await source(runtime.pool, terminal=False)
    assert await runtime.storage.get_completed_workflows(sensor, limit=10) == []
    await runtime.pool.execute(
        "UPDATE marie_scheduler.dag SET state='completed',completed_on=clock_timestamp() WHERE id=$1", UUID(dag_id)
    )
    rows = await runtime.storage.get_completed_workflows(sensor, limit=10)
    assert len(rows) == 1
    assert rows[0]['dag_id'] == dag_id
    assert rows[0]['output'] == {'records': [{'moves': ['e4']}]}


async def test_scope_filter_precedes_the_batch_limit(completions):
    runtime, sensor = completions
    await source(runtime.pool, project='another-project')
    allowed = await source(runtime.pool)
    rows = await runtime.storage.get_completed_workflows(sensor, limit=1)
    assert [row['dag_id'] for row in rows] == [allowed]


async def test_reads_typed_worker_output_only_from_the_matching_durable_attempt(completions):
    from marie.constants import KV_NAMESPACE_JOB
    from marie.job.common import JOB_INFO_KEY_PREFIX
    runtime, sensor = completions
    dag_id = await source(runtime.pool)
    job_id = await runtime.pool.fetchval('SELECT id FROM marie_scheduler.job WHERE dag_id=$1', UUID(dag_id))
    attempt = uuid4()
    await runtime.pool.execute('UPDATE marie_scheduler.job SET run_attempt_id=$1 WHERE id=$2', attempt, job_id)
    value = {
        'run_attempt_id': str(attempt),
        'runtime_env_json': json.dumps({'output': {'positions': [{'fen': 'opaque', 'ply': 3}]}}),
    }
    await runtime.pool.execute(
        'INSERT INTO marie_scheduler.kv_store_worker(namespace,key,value) VALUES($1,$2,$3::jsonb)',
        KV_NAMESPACE_JOB.decode(), JOB_INFO_KEY_PREFIX + str(job_id), json.dumps(value),
    )
    row = (await runtime.storage.get_completed_workflows(sensor, limit=10))[0]
    assert row['output'] == {'positions': [{'fen': 'opaque', 'ply': 3}]}
    value['run_attempt_id'] = str(uuid4())
    await runtime.pool.execute('UPDATE marie_scheduler.kv_store_worker SET value=$1::jsonb', json.dumps(value))
    row = (await runtime.storage.get_completed_workflows(sensor, limit=10))[0]
    assert row['output'] == {'records': [{'moves': ['e4']}]}


async def test_durable_receipt_deduplicates_and_in_place_reset_is_a_new_occurrence(completions):
    runtime, sensor = completions
    dag_id = await source(runtime.pool)
    first = (await runtime.storage.get_completed_workflows(sensor, limit=10))[0]
    key = f"run_status:{sensor['id']}:{first['completion_id']}"
    await runtime.storage.record_run_key(sensor['id'], key, str(uuid4()))
    assert await runtime.storage.get_completed_workflows(sensor, limit=10) == []
    await runtime.pool.execute(
        "UPDATE marie_scheduler.dag SET state='created',completed_on=NULL WHERE id=$1", UUID(dag_id)
    )
    await runtime.pool.execute(
        "UPDATE marie_scheduler.dag SET state='completed',completed_on=clock_timestamp() WHERE id=$1", UUID(dag_id)
    )
    second = (await runtime.storage.get_completed_workflows(sensor, limit=10))[0]
    assert second['completion_id'] != first['completion_id']


async def test_late_committing_completion_is_not_lost_behind_a_newer_cursor(completions):
    runtime, sensor = completions
    async with runtime.pool.acquire() as connection:
        async with connection.transaction():
            late = await source(connection)
            early = await source(runtime.pool)
            first = await runtime.storage.get_completed_workflows(sensor, limit=10)
            assert [row['dag_id'] for row in first] == [early]
            key = f"run_status:{sensor['id']}:{first[0]['completion_id']}"
            await runtime.storage.record_run_key(sensor['id'], key, str(uuid4()))
            sensor['cursor'] = first[0]['completion_id']
    assert [row['dag_id'] for row in await runtime.storage.get_completed_workflows(sensor, limit=10)] == [late]


async def test_old_unacknowledged_completion_recovers_and_retains_its_receipt(completions):
    runtime, sensor = completions
    directory = Path(__file__).parents[2] / 'config/psql/schema'
    for name in ('094_kb_item_state.sql', '098_kb_workflow_inputs.sql'):
        await runtime.pool.execute((directory / name).read_text().replace('{schema}', 'marie_scheduler'))
    dag_id = await source(runtime.pool)
    await runtime.pool.execute("UPDATE marie_scheduler.dag SET completed_on=clock_timestamp()-interval '45 days' WHERE id=$1", UUID(dag_id))
    rows = await runtime.storage.get_completed_workflows(sensor)
    assert len(rows) == 1
    key = f"run_status:{sensor['id']}:{rows[0]['completion_id']}"
    await runtime.storage.record_run_key(sensor['id'], key, str(uuid4()))
    await runtime.pool.execute("UPDATE marie_scheduler.sensor_run_key SET created_at=clock_timestamp()-interval '45 days'")
    await runtime.storage.cleanup_old_run_keys()
    assert await runtime.storage.get_completed_workflows(sensor) == []


async def test_worker_submits_and_recovers_one_normal_downstream_dag(completions, monkeypatch):
    runtime, sensor = completions
    await source(runtime.pool)
    original = runtime.storage.record_run_key

    async def interrupted(*args):
        raise RuntimeError('interrupted after durable downstream admission')

    monkeypatch.setattr(runtime.storage, 'record_run_key', interrupted)
    await runtime.worker._evaluate_sensor_with_tick(sensor)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='mock_connector_tool'") == 1
    monkeypatch.setattr(runtime.storage, 'record_run_key', original)
    await runtime.worker._evaluate_sensor_with_tick(await runtime.storage.get_sensor(sensor['id']))
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='mock_connector_tool'") == 1
    assert await runtime.storage.get_completed_workflows(sensor, limit=10) == []
