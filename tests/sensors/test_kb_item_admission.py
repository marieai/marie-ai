import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.daemon.worker import SensorWorker, build_work_info
from marie.sensors.definitions.data_sink.base import FileObject
from marie.sensors.definitions.kb_document_sensor import KbDocumentSensor
from marie.sensors.definitions.submission_document_sensor import (
    SubmissionDocumentSensor,
)
from marie.sensors.types import SensorType
from tests.unit.kb.test_item_state import state

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable KB PostgreSQL'
)


def context(pool):
    return SensorEvaluationContext(
        sensor_id=str(uuid4()),
        sensor_name='kb',
        sensor_type=SensorType.DATA_SINK,
        resources={'postgres_pool': pool},
        cursor='2099-01-01T00:00:00+00:00',
    )


def sensor(cls=KbDocumentSensor):
    instance = cls(
        {
            'id': str(uuid4()),
            'name': 'kb',
            'config': {'provider': 's3'},
            'target_job_name': 'kb_indexing',
        }
    )
    instance.list_objects = AsyncMock(return_value=[])
    return instance


async def test_registered_pending_work_survives_past_the_s3_cursor(state):
    _, pool, scope, attempt = state
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET owner_type='kb_document',backend='pgvector',store_name='kb_vectors'"
    )
    result = await sensor().evaluate(context(pool))
    assert len(result.run_requests) == 1
    request = result.run_requests[0]
    assert request.run_config['item_id'] == scope.item_id
    assert request.run_config['kb_indexing_dag_id'] == attempt
    assert request.run_config['uri'] == 's3://fixture/games/one.json'
    work = build_work_info(request, str(uuid4()), 'kb', 'kb_indexing', None)
    assert work.id == attempt
    assert work.data['metadata']['item_id'] == scope.item_id
    assert work.data['metadata']['kb_indexing_dag_id'] == attempt


async def test_an_object_key_cannot_register_its_own_item(state):
    _, pool, scope, _ = state
    key = f'tenants/{scope.tenant_id}/kb-indexes/{scope.index_id}/sources/{scope.source_id}/unregistered.pdf'
    obj = FileObject(
        key=key, size=1, etag='new', last_modified=datetime.now(timezone.utc)
    )
    assert await sensor().build_run_request(context(pool), obj, 'fixture') is None
    assert await pool.fetchval('SELECT count(*) FROM marie_scheduler.kb_item_state') == 1


async def test_storage_callback_cannot_admit_an_item_waiting_for_its_workflow(state):
    _, pool, scope, _ = state
    key = f'tenants/{scope.tenant_id}/kb-indexes/{scope.index_id}/sources/{scope.source_id}/waiting.pdf'
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET owner_type='kb_document',backend='pgvector',uri=$1,input_ref=$2::jsonb",
        f's3://fixture/{key}', '{"kind":"upstream","processing_workflow":"kb_index_content"}',
    )
    obj = FileObject(key=key, size=1, etag='new', last_modified=datetime.now(timezone.utc))
    assert await sensor().build_run_request(context(pool), obj, 'fixture') is None
    assert (
        await pool.fetchval('SELECT count(*) FROM marie_scheduler.kb_item_state') == 1
    )


async def test_each_native_producer_uses_only_its_registered_owner_type(state):
    _, pool, _, _ = state
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET owner_type='submission_document',backend='pgvector',store_name='kb_vectors'"
    )
    assert not (await sensor().evaluate(context(pool))).run_requests
    requests = (
        await sensor(SubmissionDocumentSensor).evaluate(context(pool))
    ).run_requests
    assert len(requests) == 1
    assert requests[0].run_config['owner_type'] == 'submission_document'


async def test_admitted_work_is_not_recreated_after_scheduler_history_pruning(state):
    _, pool, _, _ = state
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET owner_type='kb_document',backend='pgvector',admitted_at=NOW()"
    )
    assert not (await sensor().evaluate(context(pool))).run_requests


async def test_durable_worker_receipt_marks_the_native_item_admitted(state):
    _, pool, scope, attempt = state
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET owner_type='kb_document',backend='pgvector'"
    )
    request = (await sensor().evaluate(context(pool))).run_requests[0]
    worker = object.__new__(SensorWorker)
    worker._storage = SimpleNamespace(
        pool=pool,
        get_run_key_job=AsyncMock(return_value=None),
        record_run_key=AsyncMock(),
    )
    worker._job_scheduler = SimpleNamespace(
        submit_sensor_job=AsyncMock(return_value=attempt),
        get_sensor_job_state=AsyncMock(return_value='created'),
    )
    assert (
        await worker._submit_run_request({'id': str(uuid4()), 'name': 'kb'}, request)
        == attempt
    )
    assert (await state[0].get(scope))['admitted_at'] is not None


async def test_unknown_cached_receipt_does_not_recreate_a_pruned_execution(state):
    _, pool, scope, attempt = state
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET owner_type='kb_document',backend='pgvector'"
    )
    request = (await sensor().evaluate(context(pool))).run_requests[0]
    worker = object.__new__(SensorWorker)
    worker._storage = SimpleNamespace(
        pool=pool,
        get_run_key_job=AsyncMock(return_value=attempt),
        record_run_key=AsyncMock(),
    )
    worker._job_scheduler = SimpleNamespace(
        submit_sensor_job=AsyncMock(), get_sensor_job_state=AsyncMock(return_value=None)
    )
    with pytest.raises(RuntimeError, match='receipt'):
        await worker._submit_run_request({'id': str(uuid4()), 'name': 'kb'}, request)
    worker._job_scheduler.submit_sensor_job.assert_not_awaited()
    row = await state[0].get(scope)
    assert row['state'] == 'error'
    assert row['admitted_at'] is None


async def test_completed_item_survives_pruned_receipts_and_late_admission_failure(
    state,
):
    store, pool, scope, attempt = state
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET owner_type='kb_document',backend='pgvector'"
    )
    request = (await sensor().evaluate(context(pool))).run_requests[0]
    async with store.write(scope, attempt, record_count=0):
        pass
    await store.fail(scope, attempt, 'late receipt failure')
    worker = object.__new__(SensorWorker)
    worker._storage = SimpleNamespace(
        pool=pool, get_run_key_job=AsyncMock(return_value=None)
    )
    worker._job_scheduler = SimpleNamespace(submit_sensor_job=AsyncMock())
    assert (
        await worker._submit_run_request({'id': str(uuid4()), 'name': 'kb'}, request)
        == attempt
    )
    worker._job_scheduler.submit_sensor_job.assert_not_awaited()
    assert (await store.get(scope))['state'] == 'ready'
