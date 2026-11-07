from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.data_sink.base import FileObject
from marie.sensors.definitions.kb_document_sensor import KB_KEY_RE, KbDocumentSensor
from marie.sensors.types import SensorType

KEY = "tenants/11111111-1111-4111-8111-111111111111/kb-indexes/22222222-2222-4222-8222-222222222222/sources/33333333-3333-4333-8333-333333333333/report.pdf"
TENANT = "11111111-1111-4111-8111-111111111111"
SOURCE = "33333333-3333-4333-8333-333333333333"
INDEX_UNBOUND = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
INDEX_BOUND = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
NON_KB_KEY = "tenants/t1/submissions/s1/file.pdf"
UNBOUND_KEY = (
    f"tenants/{TENANT}/kb-indexes/{INDEX_UNBOUND}/sources/{SOURCE}/unbound.pdf"
)
BOUND_KEY = f"tenants/{TENANT}/kb-indexes/{INDEX_BOUND}/sources/{SOURCE}/bound.pdf"
BOUND_KEY_2 = f"tenants/{TENANT}/kb-indexes/{INDEX_BOUND}/sources/{SOURCE}/bound2.pdf"


def _context(pool=None) -> SensorEvaluationContext:
    return SensorEvaluationContext(
        sensor_id="sid",
        sensor_name="kb-document-sensor",
        sensor_type=SensorType.DATA_SINK,
        resources={"postgres_pool": pool} if pool is not None else {},
    )


def test_key_regex_parses_ids():
    m = KB_KEY_RE.match(KEY)
    assert m.group("tenant_id").startswith("1111")
    assert m.group("index_id").startswith("2222")
    assert m.group("source_id").startswith("3333")


def _sensor(**config_overrides):
    return KbDocumentSensor(
        {
            "id": "sid",
            "name": "kb-document-sensor",
            "config": {
                "subtype": "kb_document",
                "provider": "s3",
                "prefix": "tenants/",
                **config_overrides,
            },
        }
    )


def registered_item():
    return {
        'tenant_id': TENANT,
        'index_id': INDEX_BOUND,
        'source_id': SOURCE,
        'item_id': 'opaque:document',
        'indexing_dag_id': UUID('99999999-9999-4999-8999-999999999999'),
        'owner_type': 'kb_document',
        'owner_id': 'document',
        'uri': 's3://registered/report.pdf',
        'run_params': {'parse_mode': 'agent', 'multimodal': True},
    }


@pytest.mark.asyncio
async def test_run_request_uses_registered_identity_and_attempt_artifacts():
    sensor = _sensor()
    obj = FileObject(
        key=KEY, size=10, last_modified=datetime.now(timezone.utc), etag='x'
    )
    pool = AsyncMock()
    pool.fetchrow.return_value = item = registered_item()
    rr = await sensor.build_run_request(_context(pool), obj, bucket='marie')
    assert rr.job_name == 'kb_indexing'
    rc = rr.run_config
    assert rc['uri'] == item['uri']
    assert rc['ref_id'] != KEY
    assert rc['ref_id'].endswith(str(item['indexing_dag_id']))
    assert rc['ref_type'] == 'kb_item'
    assert rc['item_id'] == item['item_id']
    assert rc['index_name'] == rc['index_id']
    assert rc['run_params'] == item['run_params']


@pytest.mark.asyncio
async def test_unregistered_object_does_not_become_a_run():
    pool = AsyncMock()
    pool.fetchrow.return_value = None
    obj = FileObject(
        key=KEY, size=1, last_modified=datetime.now(timezone.utc), etag='x'
    )
    assert (
        await _sensor().build_run_request(_context(pool), obj, bucket='marie') is None
    )


def test_non_kb_key_ignored():
    assert KB_KEY_RE.match(NON_KB_KEY) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('bucket', ['', 'stale-bucket'])
async def test_saved_runtime_uri_wins_over_stale_sensor_storage_config(bucket):
    pool = AsyncMock()
    pool.fetch.return_value = [registered_item()]
    sensor = _sensor(bucket=bucket, endpoint_url='http://stale.invalid')
    with patch(
        'boto3.client', side_effect=AssertionError('No S3 scanning for membership')
    ):
        result = await sensor.evaluate(_context(pool))
    assert result.run_requests[0].run_config['uri'] == 's3://registered/report.pdf'


@pytest.mark.asyncio
async def test_cursor_does_not_hide_registered_pending_work():
    pool = AsyncMock()
    pool.fetch.return_value = [registered_item()]
    context = _context(pool)
    context.cursor = '2099-01-01T00:00:00+00:00'
    result = await _sensor().evaluate(context)
    assert len(result.run_requests) == 1
    assert result.cursor == context.cursor


@pytest.mark.asyncio
async def test_evaluate_requires_runtime_pool():
    with pytest.raises(RuntimeError, match='Sensor PostgreSQL pool'):
        await _sensor().evaluate(_context())


@pytest.mark.asyncio
async def test_empty_pending_set_skips_without_moving_cursor():
    pool = AsyncMock()
    pool.fetch.return_value = []
    context = _context(pool)
    result = await _sensor().evaluate(context)
    assert not result.run_requests
    assert result.cursor == context.cursor


@pytest.mark.asyncio
async def test_pending_query_is_bounded():
    pool = AsyncMock()
    pool.fetch.return_value = []
    await _sensor(max_files_per_tick=12).evaluate(_context(pool))
    assert pool.fetch.await_args.args[-1] == 12
    with pytest.raises(ValueError, match='between'):
        await _sensor(max_files_per_tick=1001).evaluate(_context(pool))
