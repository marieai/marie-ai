"""Current native item admission through real PostgreSQL scheduler persistence."""

import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from marie.kb.item_state import KbItemScope, KbItemStore
from marie.sensors.context import SensorEvaluationContext
from marie.sensors.daemon.worker import SensorWorker
from marie.sensors.definitions.kb_document_sensor import KbDocumentSensor
from marie.sensors.types import SensorType, TickStatus
from tests.sensors.test_sensor_postgres import runtime

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_SENSOR_TEST_PORT'),
    reason='Requires disposable scheduler PostgreSQL',
)


async def test_native_admission_recovers_after_scheduler_acceptance(
    runtime, monkeypatch
):
    from marie.query_planner.base import QueryPlanRegistry
    from marie.query_planner.kb_indexing_planner import query_planner_kb_indexing

    QueryPlanRegistry.register('kb_indexing', query_planner_kb_indexing)
    sql = Path(__file__).parents[2] / 'config/psql/schema/094_kb_item_state.sql'
    await runtime.pool.execute((sql.parent / '068_data_sink_enum.sql').read_text())
    await runtime.pool.execute(sql.read_text().replace('{schema}', 'marie_scheduler'))
    for name in ['098_kb_workflow_inputs.sql','099_kb_text_index.sql','100_kb_indexing_dag_names.sql']:
        await runtime.pool.execute((sql.parent/name).read_text().replace('{schema}','marie_scheduler'))
    await runtime.pool.execute('TRUNCATE marie_scheduler.kb_item_state')
    await runtime.pool.execute(
        "INSERT INTO marie_scheduler.queue(name) VALUES('kb_indexing') ON CONFLICT DO NOTHING"
    )
    scope = KbItemScope(str(uuid4()), str(uuid4()), str(uuid4()), 'document:one')
    attempt = str(uuid4())
    items = KbItemStore(runtime.pool)
    await items.register(
        scope,
        indexing_dag_id=attempt,
        owner_type='kb_document',
        owner_id='one',
        uri='s3://fixture/one.pdf',
        run_params={},
        backend='pgvector',
        store_name='kb_vectors',
    )
    sid = await runtime.storage.create_sensor(
        {
            'name': 'kb-native',
            'external_id': str(uuid4()),
            'sensor_type': 'data_sink',
            'status': 'active',
            'target_job_name': 'kb_indexing',
            'config': {'subtype': 'kb_document'},
        }
    )
    sensor = await runtime.storage.get_sensor(sid)
    evaluator = KbDocumentSensor(sensor)
    request = (
        await evaluator.evaluate(
            SensorEvaluationContext(
                sensor_id=sid,
                sensor_name='kb-native',
                sensor_type=SensorType.DATA_SINK,
                resources={'postgres_pool': runtime.pool},
            )
        )
    ).run_requests[0]
    tick = await runtime.storage.create_tick(
        sid, TickStatus.STARTED, [request.to_dict()], [attempt]
    )
    original = runtime.storage.record_run_key

    async def interrupted(*args):
        raise RuntimeError('interrupted after scheduler acceptance')

    monkeypatch.setattr(runtime.storage, 'record_run_key', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        await runtime.worker._submit_run_request(sensor, request)
    assert await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 1
    monkeypatch.setattr(runtime.storage, 'record_run_key', original)
    restarted = SensorWorker({}, job_scheduler=runtime.scheduler)
    restarted.set_storage(runtime.storage)
    await restarted._evaluate_sensor_with_tick(sensor)
    assert await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 1
    assert (await runtime.storage.get_tick(tick))['run_ids'] == [attempt]
    assert (await items.get(scope))['admitted_at'] is not None
    jobs = await runtime.pool.fetch(
        'SELECT data FROM marie_scheduler.job WHERE dag_id=$1', UUID(attempt)
    )
    assert len(jobs) == 4
    assert all(row['data']['metadata']['item_id'] == scope.item_id for row in jobs)
    assert all(row['data']['metadata']['kb_indexing_dag_id'] == attempt for row in jobs)
    assert all(row['data']['metadata']['ref_id'].endswith(attempt) for row in jobs)
    newer = str(uuid4())
    await items.reindex(scope, expected_indexing_dag_id=attempt, indexing_dag_id=newer)
    await restarted._evaluate_sensor_with_tick(await runtime.storage.get_sensor(sid))
    assert await runtime.pool.fetchval('SELECT count(*) FROM marie_scheduler.dag') == 2
    assert (await items.get(scope))['admitted_at'] is not None
