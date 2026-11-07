import asyncio
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from marie.kb.item_state import KbItemScope, KbItemStore
from marie.query_planner.base import QueryPlanRegistry
from marie.query_planner.kb_indexing_planner import (
    query_planner_kb_index_content,
    query_planner_kb_indexing,
)
from marie.query_planner.ocr_planner import query_planner_extract
from marie.sensors.definitions.submission_document_sensor import (
    SubmissionDocumentSensor,
)
from marie.sensors.types import SensorType
from tests.sensors.test_run_status_postgres import completions
from tests.sensors.test_sensor_postgres import runtime

pytestmark = pytest.mark.skipif(not os.environ.get('MARIE_SENSOR_TEST_PORT'), reason='Requires owned disposable PostgreSQL')


@pytest.fixture
async def handoff(completions):
    runtime, definition = completions
    directory = Path(__file__).parents[2] / 'config/psql/schema'
    await runtime.pool.execute((directory / '094_kb_item_state.sql').read_text().replace('{schema}', 'marie_scheduler'))
    migration = directory / '098_kb_workflow_inputs.sql'
    assert migration.exists(), 'Persist the upstream input reference before enabling workflow handoff'
    await runtime.pool.execute(migration.read_text().replace('{schema}', 'marie_scheduler'))
    for name in ['099_kb_text_index.sql','100_kb_indexing_dag_names.sql']:
        await runtime.pool.execute((directory/name).read_text().replace('{schema}','marie_scheduler'))
    await runtime.pool.execute('TRUNCATE marie_scheduler.kb_item_state')
    for name, planner in [('extract', query_planner_extract), ('kb_indexing', query_planner_kb_indexing), ('kb_index_content', query_planner_kb_index_content)]:
        if name not in QueryPlanRegistry.list_planners():
            QueryPlanRegistry.register(name, planner)
        await runtime.pool.execute('INSERT INTO marie_scheduler.queue(name) VALUES($1) ON CONFLICT DO NOTHING', name)
    definition['config'] = {'subtype': 'kb_workflow', 'monitored_job_names': ['extract'], 'status_types': ['completed']}
    definition['target_job_name'] = 'kb_indexing'
    await runtime.storage.update_sensor(definition['id'], {'config': definition['config'], 'target_job_name': 'kb_indexing'})
    definition = await runtime.storage.get_sensor(definition['id'])
    store = KbItemStore(runtime.pool)
    tenant = str(uuid4())
    uri = 's3://fixture/submission/document.pdf'
    scopes = [KbItemScope(tenant, str(uuid4()), str(uuid4()), 'document-one') for _ in range(2)]
    for scope in scopes:
        await store.register(scope, indexing_dag_id=str(uuid4()), owner_type='submission_document', owner_id='document-one', uri=uri,
            run_params={'chunk_size': 100, 'chunk_overlap': 0}, backend='pgvector', store_name='kb_vectors')
        await runtime.pool.execute(
            'UPDATE marie_scheduler.kb_item_state SET input_ref=$1::jsonb WHERE index_id=$2',
            json.dumps({'kind': 'upstream', 'planner': 'extract', 'processing_workflow': 'kb_index_content'}), UUID(scope.index_id),
        )
    return runtime, definition, store, scopes, uri


async def extraction(runtime, scope, uri, *, fabric=None):
    from marie.sensors.daemon.worker import build_work_info
    from marie.sensors.types import RunRequest
    work = build_work_info(RunRequest(job_name='extract', run_config={
        'planner': 'extract', 'project_id': scope.tenant_id, 'uri': uri, 'ref_id': 'saved-output', 'ref_type': 'document',
        **({'fabric_group_id': fabric} if fabric else {}),
    }), str(uuid4()), 'source', 'extract', None)
    work.policy = 'allow_duplicate'
    await runtime.scheduler.submit_job(work)
    return work.id


async def finish(runtime, dag_id):
    reference = {'kind': 'extraction', 'ref_id': 'saved-output', 'ref_type': 'document', 'sha256': 'a' * 64}
    await runtime.pool.execute(
        "UPDATE marie_scheduler.job SET output=$1::jsonb WHERE dag_id=$2 AND strpos(data->'metadata'->>'on','extract_executor:')=1",
        json.dumps({'input_ref': reference}), UUID(dag_id),
    )
    await runtime.pool.execute("UPDATE marie_scheduler.dag SET state='completed',completed_on=clock_timestamp() WHERE id=$1", UUID(dag_id))


async def test_admission_binds_the_upstream_dag_and_native_sensor_waits(handoff):
    runtime, definition, store, scopes, uri = handoff
    dag_id = await extraction(runtime, scopes[0], uri)
    assert (await store.get(scopes[0]))['input_ref']['dag_id'] == dag_id
    sensor = SubmissionDocumentSensor(definition)
    context = await runtime.worker._build_context({**definition, 'sensor_type': 'data_sink'})
    assert not (await sensor.evaluate(context)).run_requests


async def test_one_extraction_fans_out_into_two_normal_index_only_dags(handoff):
    from marie.sensors.definitions.kb_workflow_sensor import KbWorkflowSensor
    runtime, definition, store, scopes, uri = handoff
    runtime.worker._registry.register(SensorType.RUN_STATUS, KbWorkflowSensor, subtype='kb_workflow')
    dag_id = await extraction(runtime, scopes[0], uri)
    await finish(runtime, dag_id)
    await runtime.worker._evaluate_sensor_with_tick(definition)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 2
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.job WHERE data->'metadata'->>'on' LIKE 'extract_executor:%'") == 1
    for scope in scopes:
        item = await store.get(scope)
        assert item['admitted_at'] is not None
        assert item['input_ref']['artifact']['ref_id'] == 'saved-output'
    await runtime.worker._evaluate_sensor_with_tick(definition)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 2


async def test_superseded_extraction_and_removed_destination_cannot_create_indexing_work(handoff):
    from marie.sensors.definitions.kb_workflow_sensor import KbWorkflowSensor
    runtime, definition, store, scopes, uri = handoff
    runtime.worker._registry.register(SensorType.RUN_STATUS, KbWorkflowSensor, subtype='kb_workflow')
    old = await extraction(runtime, scopes[0], uri)
    current = await extraction(runtime, scopes[0], uri)
    await finish(runtime, old)
    await runtime.worker._evaluate_sensor_with_tick(definition)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 0
    await store.remove(scopes[1])
    await finish(runtime, current)
    await runtime.worker._evaluate_sensor_with_tick(definition)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 1


async def test_a_new_extraction_refreshes_each_kb_using_a_new_normal_indexing_run(handoff):
    from marie.sensors.definitions.kb_workflow_sensor import KbWorkflowSensor
    runtime, definition, store, scopes, uri = handoff
    runtime.worker._registry.register(SensorType.RUN_STATUS, KbWorkflowSensor, subtype='kb_workflow')
    first = await extraction(runtime, scopes[0], uri)
    await finish(runtime, first)
    await runtime.worker._evaluate_sensor_with_tick(definition)
    previous = []
    for scope in scopes:
        item = await store.get(scope)
        previous.append(str(item['indexing_dag_id']))
        async with store.write(scope, str(item['indexing_dag_id']), record_count=1):
            pass
    refreshed = await extraction(runtime, scopes[0], uri)
    for index, scope in enumerate(scopes):
        item = await store.get(scope)
        assert str(item['indexing_dag_id']) != previous[index]
        assert str(item['indexed_dag_id']) == previous[index]
        assert item['input_ref']['dag_id'] == refreshed
    await finish(runtime, refreshed)
    await runtime.worker._evaluate_sensor_with_tick(definition)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 4


async def test_in_place_source_reset_is_a_new_handoff_and_fences_the_old_input(handoff):
    from marie.kb.item_state import StaleIndexingRun
    from marie.sensors.definitions.kb_workflow_sensor import KbWorkflowSensor
    runtime, definition, store, scopes, uri = handoff
    runtime.worker._registry.register(SensorType.RUN_STATUS, KbWorkflowSensor, subtype='kb_workflow')
    source_id = await extraction(runtime, scopes[0], uri)
    await finish(runtime, source_id)
    await runtime.worker._evaluate_sensor_with_tick(definition)
    first = await store.get(scopes[0])
    await runtime.pool.execute("UPDATE marie_scheduler.dag SET state='created',completed_on=NULL WHERE id=$1", UUID(source_id))
    with pytest.raises(StaleIndexingRun):
        async with store.write(scopes[0], str(first['indexing_dag_id']), record_count=1):
            pass
    await finish(runtime, source_id)
    await runtime.worker._evaluate_sensor_with_tick(definition)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 4


async def test_partial_fanout_recovers_after_scheduler_acceptance(handoff, monkeypatch):
    from marie.sensors.definitions.kb_workflow_sensor import KbWorkflowSensor
    runtime, definition, store, scopes, uri = handoff
    runtime.worker._registry.register(SensorType.RUN_STATUS, KbWorkflowSensor, subtype='kb_workflow')
    dag_id = await extraction(runtime, scopes[0], uri)
    await finish(runtime, dag_id)
    original = runtime.storage.record_run_key

    async def interrupted(*args):
        raise RuntimeError('interrupted after first destination admission')

    monkeypatch.setattr(runtime.storage, 'record_run_key', interrupted)
    await runtime.worker._evaluate_sensor_with_tick(definition)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 1
    monkeypatch.setattr(runtime.storage, 'record_run_key', original)
    await runtime.worker._evaluate_sensor_with_tick(await runtime.storage.get_sensor(definition['id']))
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 2
    for scope in scopes:
        assert (await store.get(scope))['admitted_at'] is not None


async def test_source_reset_after_acceptance_before_receipt_starts_fresh_indexing(handoff, monkeypatch):
    from marie.sensors.definitions.kb_workflow_sensor import KbWorkflowSensor
    runtime, definition, store, scopes, uri = handoff
    runtime.worker._registry.register(SensorType.RUN_STATUS, KbWorkflowSensor, subtype='kb_workflow')
    source_id = await extraction(runtime, scopes[0], uri)
    await finish(runtime, source_id)
    original = runtime.storage.record_run_key

    async def interrupted(*args):
        raise RuntimeError('interrupted after scheduler acceptance')

    monkeypatch.setattr(runtime.storage, 'record_run_key', interrupted)
    await runtime.worker._evaluate_sensor_with_tick(definition)
    previous = await runtime.pool.fetchval("SELECT id FROM marie_scheduler.dag WHERE planner='kb_index_content'")
    await runtime.pool.execute("UPDATE marie_scheduler.dag SET state='created',completed_on=NULL WHERE id=$1", UUID(source_id))
    await finish(runtime, source_id)
    monkeypatch.setattr(runtime.storage, 'record_run_key', original)
    await KbWorkflowSensor(definition).evaluate(await runtime.worker._build_context(definition))
    await runtime.worker._evaluate_sensor_with_tick(await runtime.storage.get_sensor(definition['id']))
    await runtime.worker._evaluate_sensor_with_tick(await runtime.storage.get_sensor(definition['id']))
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 3
    for scope in scopes:
        assert (await store.get(scope))['indexing_dag_id'] != previous


async def test_fanout_continues_after_a_bounded_tick(handoff):
    from marie.sensors.definitions.kb_workflow_sensor import KbWorkflowSensor
    runtime, definition, store, scopes, uri = handoff
    runtime.worker._registry.register(SensorType.RUN_STATUS, KbWorkflowSensor, subtype='kb_workflow')
    config = {**definition['config'], 'max_destinations_per_tick': 1}
    await runtime.storage.update_sensor(definition['id'], {'config': config})
    definition = await runtime.storage.get_sensor(definition['id'])
    await finish(runtime, await extraction(runtime, scopes[0], uri))
    await runtime.worker._evaluate_sensor_with_tick(definition)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 1
    await runtime.worker._evaluate_sensor_with_tick(await runtime.storage.get_sensor(definition['id']))
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 2


async def test_missing_saved_output_fails_destinations_without_reextracting(handoff):
    from marie.sensors.definitions.kb_workflow_sensor import KbWorkflowSensor
    runtime, definition, store, scopes, uri = handoff
    runtime.worker._registry.register(SensorType.RUN_STATUS, KbWorkflowSensor, subtype='kb_workflow')
    dag_id = await extraction(runtime, scopes[0], uri)
    await runtime.pool.execute("UPDATE marie_scheduler.dag SET state='completed',completed_on=clock_timestamp() WHERE id=$1", UUID(dag_id))
    await runtime.worker._evaluate_sensor_with_tick(definition)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 0
    for scope in scopes:
        item = await store.get(scope)
        assert item['state'] == 'error'
        assert 'unambiguous saved content' in item['error']


async def test_native_and_completion_producers_share_one_scheduler_receipt(handoff):
    from marie.sensors.definitions.kb_workflow_sensor import KbWorkflowSensor
    runtime, definition, store, scopes, uri = handoff
    await finish(runtime, await extraction(runtime, scopes[0], uri))
    context = await runtime.worker._build_context(definition)
    requests = (await KbWorkflowSensor(definition).evaluate(context)).run_requests
    native = SubmissionDocumentSensor({**definition, 'id': str(uuid4())})
    native_request = native._request(await store.get(scopes[0]))
    matching = next(request for request in requests if request.run_config['index_id'] == scopes[0].index_id)
    await runtime.worker._submit_run_request(definition, matching)
    native_id = await runtime.storage.create_sensor({
        'external_id': str(uuid4()), 'name': 'native', 'sensor_type': 'data_sink',
        'status': 'active', 'target_job_name': 'kb_indexing', 'config': {'subtype': 'submission_document'},
    })
    await runtime.worker._submit_run_request(await runtime.storage.get_sensor(native_id), native_request)
    assert await runtime.pool.fetchval("SELECT count(*) FROM marie_scheduler.dag WHERE planner='kb_index_content'") == 1


async def test_binding_requires_an_allowed_fabric_and_pins_it_for_refresh(handoff):
    runtime, definition, store, scopes, uri = handoff
    await runtime.pool.execute("UPDATE marie_scheduler.kb_item_state SET input_ref=input_ref || '{\"fabric_group_ids\":[\"fabric-a\"]}'::jsonb")
    await extraction(runtime, scopes[0], uri, fabric='fabric-b')
    assert (await store.get(scopes[0]))['input_ref'].get('dag_id') is None
    allowed = await extraction(runtime, scopes[0], uri, fabric='fabric-a')
    assert (await store.get(scopes[0]))['input_ref']['dag_id'] == allowed
    await extraction(runtime, scopes[0], uri, fabric='fabric-b')
    assert (await store.get(scopes[0]))['input_ref']['dag_id'] == allowed


async def test_waiting_memberships_can_recover_an_old_retained_completion(handoff):
    runtime, definition, store, scopes, uri = handoff
    dag_id = await extraction(runtime, scopes[0], uri)
    await finish(runtime, dag_id)
    await runtime.pool.execute("UPDATE marie_scheduler.dag SET completed_on=clock_timestamp()-interval '45 days' WHERE id=$1", UUID(dag_id))
    rows = await runtime.storage.get_completed_workflows(definition)
    assert len(rows) == 1
    assert len(await store.complete_upstream(rows[0])) == 2


async def test_source_admission_waits_for_overlapping_registration(handoff):
    runtime, definition, store, scopes, uri = handoff
    source = None
    async with runtime.pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(jsonb_build_array('kb_upstream'::text,$1::text,$2::text,$3::text)::text,0))",
                scopes[0].tenant_id, 'extract', uri,
            )
            source = asyncio.create_task(extraction(runtime, scopes[0], uri))
            blocked = False
            for _ in range(100):
                blocked = await runtime.pool.fetchval("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND wait_event='advisory')")
                if blocked or source.done():
                    break
                await asyncio.sleep(0.005)
            assert blocked
    dag_id = await source
    assert (await store.get(scopes[0]))['input_ref']['dag_id'] == dag_id
