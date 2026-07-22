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


def _sensor():
    return KbDocumentSensor(
        {
            "id": "sid",
            "name": "kb-document-sensor",
            "config": {
                "subtype": "kb_document",
                "provider": "s3",
                "prefix": "tenants/",
            },
        }
    )


@pytest.mark.asyncio
async def test_run_request_shape():
    sensor = _sensor()
    obj = FileObject(
        key=KEY, size=10, last_modified=datetime.now(timezone.utc), etag="x"
    )
    binding = {
        "workflow_name": "kb_indexing",
        "run_params": {"parse_mode": "agent", "multimodal": True},
    }
    pool = AsyncMock()
    pool.fetchrow.return_value = binding

    rr = await sensor.build_run_request(_context(pool), obj, bucket="marie")

    assert rr.job_name == "kb_indexing"
    assert rr.run_key == f"kb_indexing:22222222-2222-4222-8222-222222222222:{KEY}"
    rc = rr.run_config
    assert rc["uri"] == f"s3://marie/{KEY}"
    assert rc["ref_id"] == KEY
    assert rc["ref_type"] == "kb_document"
    assert rc["project_id"] == rc["tenant_id"]
    assert rc["index_name"] == rc["index_id"]
    assert rc["run_params"] == binding["run_params"]


@pytest.mark.asyncio
async def test_missing_binding_skips():
    sensor = _sensor()
    obj = FileObject(
        key=KEY, size=10, last_modified=datetime.now(timezone.utc), etag="x"
    )
    pool = AsyncMock()
    pool.fetchrow.return_value = None

    assert await sensor.build_run_request(_context(pool), obj, bucket="marie") is None


def test_non_kb_key_ignored():
    assert KB_KEY_RE.match("tenants/t1/submissions/s1/file.pdf") is None


@pytest.mark.asyncio
async def test_evaluate_holds_cursor_at_recoverable_skip_minimum():
    """Non-KB key advances past; unbound KB key holds the cursor at its own
    timestamp (not the max) so it gets re-listed and retried next tick."""
    sensor = _sensor()
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 3, tzinfo=timezone.utc)

    non_kb = FileObject(key=NON_KB_KEY, size=1, last_modified=t0, etag="x")
    unbound = FileObject(key=UNBOUND_KEY, size=1, last_modified=t1, etag="x")
    bound = FileObject(key=BOUND_KEY, size=1, last_modified=t2, etag="x")
    binding = {"workflow_name": "kb_indexing", "run_params": {}}

    def _load_binding(_context, index_id):
        return binding if index_id == INDEX_BOUND else None

    with (
        patch.object(
            sensor, "list_objects", new=AsyncMock(return_value=[non_kb, unbound, bound])
        ),
        patch.object(sensor, "load_binding", side_effect=_load_binding),
    ):
        result = await sensor.evaluate(_context())

    assert len(result.run_requests) == 1
    assert result.run_requests[0].run_config["index_id"] == INDEX_BOUND
    assert result.cursor == t1.isoformat()


@pytest.mark.asyncio
async def test_evaluate_all_bound_advances_cursor_to_max():
    sensor = _sensor()
    t1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 3, tzinfo=timezone.utc)

    obj1 = FileObject(key=BOUND_KEY, size=1, last_modified=t1, etag="x")
    obj2 = FileObject(key=BOUND_KEY_2, size=1, last_modified=t2, etag="x")
    binding = {"workflow_name": "kb_indexing", "run_params": {}}

    with (
        patch.object(sensor, "list_objects", new=AsyncMock(return_value=[obj1, obj2])),
        patch.object(sensor, "load_binding", return_value=binding) as load_binding,
    ):
        result = await sensor.evaluate(_context())

    assert len(result.run_requests) == 2
    assert result.cursor == t2.isoformat()
    assert load_binding.await_count == 1


@pytest.mark.asyncio
async def test_evaluate_missing_shared_pool_raises():
    sensor = _sensor()
    obj = FileObject(
        key=BOUND_KEY,
        size=1,
        last_modified=datetime(2026, 1, 2, tzinfo=timezone.utc),
        etag="x",
    )

    with patch.object(sensor, "list_objects", new=AsyncMock(return_value=[obj])):
        with pytest.raises(RuntimeError, match="Sensor PostgreSQL pool"):
            await sensor.evaluate(_context())


@pytest.mark.asyncio
async def test_load_binding_uses_shared_pool():
    sensor = _sensor()
    pool = AsyncMock()
    pool.fetchrow.return_value = None

    await sensor.load_binding(_context(pool), INDEX_BOUND)

    pool.fetchrow.assert_awaited_once()
    query, index_id = pool.fetchrow.call_args.args
    assert "FROM marie_scheduler.resource_workflow_binding" in query
    assert "resource_type = 'kb_index'" in query
    assert "$1" in query
    assert index_id == UUID(INDEX_BOUND)


@pytest.mark.asyncio
async def test_load_binding_requires_shared_pool():
    sensor = _sensor()

    with pytest.raises(RuntimeError, match="Sensor PostgreSQL pool"):
        await sensor.load_binding(_context(), INDEX_BOUND)
