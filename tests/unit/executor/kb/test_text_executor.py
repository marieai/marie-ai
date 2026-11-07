import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from docarray import DocList
from docarray.documents import TextDoc

from marie.kb.item_state import item_run_parameters
from marie.storage import StorageManager
from tests.unit.executor.kb.test_vector_store_executor_embed import _make_executor
from tests.unit.kb.test_item_state import state
from tests.unit.kb.test_text_index import reference, text_store

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable KB PostgreSQL'
)


async def prepared(state, text_store, monkeypatch):
    items, pool, scope, attempt = state
    text = '# Benefits\nEmergency treatment is covered.'
    ref = reference(text)
    await pool.execute(
        'UPDATE marie_scheduler.kb_item_state SET input_ref=$1::jsonb', json.dumps(ref)
    )
    item = await items.get(scope)
    parameters = {
        **item_run_parameters(item),
        'job_id': str(uuid4()),
        'run_attempt_id': str(uuid4()),
        'run_owner': 'fixture',
        'payload': {'op_params': {'input_mode': 'saved_content'}},
    }
    await pool.execute('''CREATE TABLE IF NOT EXISTS marie_scheduler.job (
        id uuid PRIMARY KEY,dag_id uuid,state text,data jsonb,run_attempt_id uuid,run_owner text,run_lease_expires_at timestamptz)''')
    await pool.execute(
        'INSERT INTO marie_scheduler.job(id,dag_id,state,data,run_attempt_id,run_owner,run_lease_expires_at) VALUES($1,$2,$3,$4::jsonb,$5,$6,$7)',
        UUID(parameters['job_id']),
        UUID(attempt),
        'active',
        json.dumps({'metadata': item_run_parameters(item)}),
        UUID(parameters['run_attempt_id']),
        'fixture',
        datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    executor = _make_executor()
    executor._text_store = text_store
    executor._ensure_store = AsyncMock(
        side_effect=AssertionError('No vector initialization for full text')
    )
    monkeypatch.setattr(StorageManager, 'read', lambda uri, **kwargs: text.encode())
    return executor, parameters, text


async def test_saved_content_job_indexes_queries_and_opens_without_embedding(
    state, text_store, monkeypatch
):
    executor, params, text = await prepared(state, text_store, monkeypatch)
    await executor.index_text(DocList[TextDoc]([TextDoc()]), params)
    rows = (
        await executor.text_search(DocList[TextDoc](), {**params, 'query': 'emergency'})
    )['results']
    assert len(rows) == 1
    result = await executor.open_text(DocList[TextDoc](), {**params, **rows[0]})
    assert result['outcome'] == 'ready' and result['text'] == text
    executor._embeddings.embed_text.assert_not_called()
    executor._ensure_store.assert_not_awaited()


async def test_changed_saved_bytes_cannot_replace_or_open_current_output(
    state, text_store, monkeypatch
):
    executor, params, _ = await prepared(state, text_store, monkeypatch)
    await executor.index_text(DocList[TextDoc]([TextDoc()]), params)
    row = (
        await text_store.search(params['tenant_id'], params['index_id'], 'emergency')
    )[0]
    monkeypatch.setattr(StorageManager, 'read', lambda uri, **kw: b'different bytes')
    result = await executor.open_text(DocList[TextDoc](), {**params, **row})
    assert result['outcome'] == 'changed' and 'text' not in result


async def test_removal_during_source_read_blocks_delivery(
    state, text_store, monkeypatch
):
    executor, params, _ = await prepared(state, text_store, monkeypatch)
    await executor.index_text(DocList[TextDoc]([TextDoc()]), params)
    row = (
        await text_store.search(params['tenant_id'], params['index_id'], 'emergency')
    )[0]
    original = text_store.passage
    calls = 0

    async def revoking(*args, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            await state[0].remove(state[2])
        return await original(*args, **kw)

    monkeypatch.setattr(text_store, 'passage', revoking)
    result = await executor.open_text(DocList[TextDoc](), {**params, **row})
    assert result['outcome'] == 'unavailable' and 'text' not in result


async def test_expired_job_and_changed_input_cannot_publish(
    state, text_store, monkeypatch
):
    from marie.kb.item_state import StaleIndexingRun

    executor, params, _ = await prepared(state, text_store, monkeypatch)
    await state[1].execute(
        "UPDATE marie_scheduler.job SET run_lease_expires_at=NOW()-INTERVAL '1 minute' WHERE id=$1",
        UUID(params['job_id']),
    )
    with pytest.raises(StaleIndexingRun):
        await executor.index_text(DocList[TextDoc](), params)
    assert not await text_store.search(
        params['tenant_id'], params['index_id'], 'emergency'
    )
    with pytest.raises(StaleIndexingRun):
        await executor.index_text(
            DocList[TextDoc](),
            {
                **params,
                'input_ref': {**params['input_ref'], 'uri': 's3://fixture/changed.md'},
            },
        )


async def test_missing_saved_source_has_explicit_outcome(
    state, text_store, monkeypatch
):
    executor, params, _ = await prepared(state, text_store, monkeypatch)
    await executor.index_text(DocList[TextDoc](), params)
    row = (
        await text_store.search(params['tenant_id'], params['index_id'], 'emergency')
    )[0]

    def missing(*args, **kwargs):
        raise FileNotFoundError('fixture removed')

    monkeypatch.setattr(StorageManager, 'read', missing)
    result = await executor.open_text(DocList[TextDoc](), {**params, **row})
    assert result['outcome'] == 'unavailable' and 'text' not in result


async def test_missing_execution_is_rejected_before_reading_source(
    state, text_store, monkeypatch
):
    executor, params, _ = await prepared(state, text_store, monkeypatch)

    def unread(*args, **kwargs):
        raise AssertionError('No source read for malformed execution')

    monkeypatch.setattr(StorageManager, 'read', unread)
    with pytest.raises(ValueError, match='scheduler execution'):
        await executor.index_text(
            DocList[TextDoc](), {**params, 'run_attempt_id': None}
        )


def test_shared_ocr_reader_tolerates_empty_and_malformed_pages():
    from marie.executor.kb.vector_store_executor import VectorStoreExecutor

    assert (
        VectorStoreExecutor._flatten_ocr_pages(
            [{'lines': None}, None, {'lines': [None, {'text': 'covered'}]}]
        )[0]
        == '\ncovered'
    )


async def test_real_text_pool_status_and_index_name_cleanup(
    state, text_store, monkeypatch
):
    import asyncio

    executor, params, _ = await prepared(state, text_store, monkeypatch)
    await executor.index_text(DocList[TextDoc](), params)
    executor._text_store = None
    executor._text_store_lock = asyncio.Lock()
    executor._connection_string = (
        f"postgresql://postgres@127.0.0.1:{os.environ['MARIE_KB_TEST_PORT']}/postgres"
    )
    executor._vector_store = None
    try:
        stores = await asyncio.gather(
            executor._ensure_text_store(), executor._ensure_text_store()
        )
        assert stores[0] is stores[1]
        assert (await executor.item_status(DocList[TextDoc](), params))[
            'state'
        ] == 'ready'
        result = await executor.delete_text_index(
            DocList[TextDoc](),
            {'tenant_id': params['tenant_id'], 'index_name': params['index_id']},
        )
        assert result['deleted_count'] == 1
        pool = stores[0].pool._pool
    finally:
        await executor.close()
    assert pool.closed
    executor._embeddings.embed_text.assert_not_called()


@pytest.mark.skipif(
    not os.environ.get('MARIE_KB_LEASE_BENCHMARK'),
    reason='Explicit local lease-budget qualification',
)
async def test_maximum_section_batch_with_five_second_lease(
    state, text_store, monkeypatch
):
    import time

    from marie.kb.item_state import scheduler_execution

    _, params, _ = await prepared(state, text_store, monkeypatch)
    items, pool, scope, attempt = state
    length = 1600 + 9999 * 1400
    text = ('Emergency coverage and authorization information. ' * 300_000)[:length]
    assert len(text) == length
    ref = reference(text)
    await pool.execute(
        'UPDATE marie_scheduler.kb_item_state SET input_ref=$1::jsonb', json.dumps(ref)
    )
    item = await items.get(scope)
    await pool.execute(
        "UPDATE marie_scheduler.job SET data=$2::jsonb,run_lease_expires_at=NOW()+INTERVAL '5 seconds' WHERE id=$1",
        UUID(params['job_id']),
        json.dumps({'metadata': item_run_parameters(item)}),
    )
    started = time.perf_counter()
    count = await text_store.replace(
        scope,
        attempt,
        text,
        ref,
        chunk_size=1600,
        chunk_overlap=200,
        **scheduler_execution(params),
    )
    elapsed = (time.perf_counter() - started) * 1000
    print(
        json.dumps(
            {
                'input_bytes': len(text.encode()),
                'sections': count,
                'elapsed_ms': round(elapsed, 2),
                'lease_ms': 5000,
            }
        )
    )
    assert count == 10_000 and (await items.get(scope))['state'] == 'ready'
