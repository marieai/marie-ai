import hashlib
import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from marie.kb.item_state import KbItemScope, StaleIndexingRun
from marie.kb.text import TextIndexStore
from tests.unit.kb.test_item_state import state

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable KB PostgreSQL'
)


@pytest.fixture
async def text_store(state):
    _, pool, _, _ = state
    await pool.execute(
        (Path(__file__).parents[3] / 'config/psql/schema/099_kb_text_index.sql')
        .read_text()
        .replace('{schema}', 'marie_scheduler')
    )
    await pool.execute('TRUNCATE marie_scheduler.kb_text_section')
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET backend='postgresql_text',store_name='kb_text_section'"
    )
    return TextIndexStore(pool)


def reference(text):
    return {
        'kind': 'text',
        'uri': 's3://fixture/one.md',
        'media_type': 'text/markdown',
        'sha256': hashlib.sha256(text.encode()).hexdigest(),
    }


async def test_saved_markdown_is_searchable_without_vectors_and_has_exact_offsets(
    state, text_store
):
    _, _, scope, attempt = state
    text = '# Coverage\nThe plan covers emergency treatment.\n'
    await text_store.replace(
        scope, attempt, text, reference(text), chunk_size=100, chunk_overlap=5
    )
    rows = await text_store.search(scope.tenant_id, scope.index_id, 'emergency')
    assert rows and all(row['item_id'] == scope.item_id for row in rows)
    row = rows[0]
    assert text[row['start_offset'] : row['end_offset']] == row['content']
    assert row['source_digest'] == reference(text)['sha256']
    assert (
        await text_store.passage(
            scope, row['section_id'], attempt, row['source_digest']
        )
        is not None
    )
    assert not await text_store.search(str(uuid4()), scope.index_id, 'emergency')


async def test_shorter_and_empty_replacement_keep_other_items_and_old_writer_is_rejected(
    state, text_store
):
    items, _, scope, attempt = state
    other = KbItemScope(scope.tenant_id, scope.index_id, scope.source_id, 'second')
    other_attempt = str(uuid4())
    await items.register(
        other,
        indexing_dag_id=other_attempt,
        owner_type='record',
        owner_id='two',
        uri='s3://fixture/two.md',
        run_params={},
        backend='postgresql_text',
        store_name='kb_text_section',
    )
    await text_store.replace(
        scope, attempt, 'emergency care old', reference('emergency care old')
    )
    await text_store.replace(
        other,
        other_attempt,
        'emergency care sibling',
        reference('emergency care sibling'),
    )
    first = (await text_store.search(scope.tenant_id, scope.index_id, 'old'))[0]
    newer = str(uuid4())
    await items.reindex(scope, expected_indexing_dag_id=attempt, indexing_dag_id=newer)
    assert await text_store.search(scope.tenant_id, scope.index_id, 'old')
    await text_store.replace(scope, newer, '', reference(''))
    assert (
        await text_store.passage(
            scope, first['section_id'], attempt, first['source_digest']
        )
        is None
    )
    assert [
        row['item_id']
        for row in await text_store.search(scope.tenant_id, scope.index_id, 'emergency')
    ] == ['second']
    with pytest.raises(StaleIndexingRun):
        await text_store.replace(scope, attempt, 'late', reference('late'))


async def test_failed_replacement_and_removal_preserve_boundaries(
    state, text_store, monkeypatch
):
    items, pool, scope, attempt = state
    await text_store.replace(
        scope, attempt, 'emergency previous', reference('emergency previous')
    )
    newer = str(uuid4())
    await items.reindex(scope, expected_indexing_dag_id=attempt, indexing_dag_id=newer)
    await pool.execute(
        "ALTER TABLE marie_scheduler.kb_text_section ADD CONSTRAINT reject_bad CHECK(content <> 'invalid')"
    )
    try:
        import psycopg

        with pytest.raises(psycopg.errors.CheckViolation):
            await text_store.replace(scope, newer, 'invalid', reference('invalid'))
    finally:
        await pool.execute(
            'ALTER TABLE marie_scheduler.kb_text_section DROP CONSTRAINT reject_bad'
        )
    assert (await text_store.search(scope.tenant_id, scope.index_id, 'previous'))[0][
        'indexed_dag_id'
    ] == attempt
    await items.remove(scope)
    assert not await text_store.search(scope.tenant_id, scope.index_id, 'previous')
    await text_store.delete_item(scope)
    assert (await items.get(scope))['cleanup_state'] == 'complete'
    with pytest.raises(StaleIndexingRun):
        await text_store.replace(scope, newer, 'late', reference('late'))


async def test_index_cleanup_batches_and_captured_cleanup_cannot_delete_relinked_output(
    state, text_store
):
    items, pool, scope, attempt = state
    await text_store.replace(
        scope, attempt, 'emergency old', reference('emergency old')
    )
    removed = await items.remove(scope)
    new_attempt = str(uuid4())
    # Simulate Studio's authorized membership transition after a completed withdrawal.
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET state='pending',indexing_dag_id=$1 WHERE tenant_id=$2 AND index_id=$3 AND source_id=$4 AND item_id=$5",
        UUID(new_attempt),
        *scope.values(),
    )
    await text_store.replace(
        scope, new_attempt, 'emergency new', reference('emergency new')
    )
    assert (
        await text_store._cleanup_removed(scope.tenant_id, scope.index_id, [removed])
        == 0
    )
    assert (await text_store.search(scope.tenant_id, scope.index_id, 'new'))[0][
        'indexed_dag_id'
    ] == new_attempt
    await pool.execute(
        """INSERT INTO marie_scheduler.kb_item_state
        (tenant_id,index_id,source_id,item_id,indexing_dag_id,owner_type,owner_id,uri,run_params,backend,store_name,state,indexed_dag_id)
        SELECT tenant_id,index_id,source_id,'extra-' || n,indexing_dag_id,owner_type,owner_id,uri,run_params,backend,store_name,state,indexed_dag_id
        FROM marie_scheduler.kb_item_state CROSS JOIN generate_series(1,501) n WHERE item_id=$1""",
        scope.item_id,
    )
    await pool.execute("""INSERT INTO marie_scheduler.kb_text_section
        (tenant_id,index_id,source_id,item_id,section_id,indexed_dag_id,content,source_digest,input_ref,start_offset,end_offset)
        SELECT k.tenant_id,k.index_id,k.source_id,k.item_id,s.section_id,k.indexing_dag_id,s.content,s.source_digest,s.input_ref,s.start_offset,s.end_offset
        FROM marie_scheduler.kb_item_state k JOIN marie_scheduler.kb_text_section s USING (tenant_id,index_id,source_id)
        WHERE k.item_id<>s.item_id""")
    assert await text_store.delete_index(scope.tenant_id, scope.index_id) == 502
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM marie_scheduler.kb_item_state WHERE cleanup_state <> 'complete'"
        )
        == 0
    )
    assert not await text_store.search(scope.tenant_id, scope.index_id, 'emergency')
    assert await text_store.delete_index(scope.tenant_id, scope.index_id) == 0
    assert (await items.get(scope))['cleanup_state'] == 'complete'


async def test_foreign_backend_cannot_be_deleted_by_text_cleanup(state, text_store):
    _, pool, scope, _ = state
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET backend='pgvector',store_name='kb_vectors'"
    )
    with pytest.raises(PermissionError):
        await text_store.delete_index(scope.tenant_id, scope.index_id)
    assert (
        await pool.fetchval('SELECT state FROM marie_scheduler.kb_item_state')
        == 'pending'
    )


async def test_pathological_overlap_is_rejected_before_building_sections(
    state, text_store
):
    _, _, scope, attempt = state
    text = 'coverage ' * 20_000
    with pytest.raises(ValueError, match='too many sections'):
        await text_store.replace(
            scope, attempt, text, reference(text), chunk_size=8000, chunk_overlap=7999
        )
    assert not await text_store.search(scope.tenant_id, scope.index_id, 'coverage')


async def test_section_limit_rejects_oversized_expansion_without_changing_current_output(
    state, text_store
):
    _, _, scope, attempt = state
    text = 'coverage ' * 112_000
    with pytest.raises(ValueError, match='too many sections'):
        await text_store.replace(
            scope, attempt, text, reference(text), chunk_size=100, chunk_overlap=0
        )
    assert not await text_store.search(scope.tenant_id, scope.index_id, 'coverage')


async def test_removal_waits_for_replacement_then_withdraws_it(
    state, text_store, monkeypatch
):
    import asyncio

    from marie.storage.database.postgres_pool import AsyncPostgresConnection

    items, pool, scope, attempt = state
    inserted = asyncio.Event()
    release = asyncio.Event()
    original = AsyncPostgresConnection.executemany

    async def paused(connection, query, *args):
        await original(connection, query, *args)
        inserted.set()
        await release.wait()

    monkeypatch.setattr(AsyncPostgresConnection, 'executemany', paused)
    writing = asyncio.create_task(
        text_store.replace(
            scope, attempt, 'emergency covered', reference('emergency covered')
        )
    )
    await asyncio.wait_for(inserted.wait(), 2)
    removing = asyncio.create_task(text_store.delete_item(scope))
    try:
        for _ in range(100):
            if await pool.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock')"
            ):
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail('Removal did not wait on the checked replacement')
        assert not removing.done()
    finally:
        release.set()
        await asyncio.gather(writing, removing)
    assert not await text_store.search(scope.tenant_id, scope.index_id, 'emergency')
    assert (await items.get(scope))['cleanup_state'] == 'complete'


async def test_rerun_upstream_cannot_replace_previous_text(state, text_store):
    from datetime import datetime, timezone

    items, pool, scope, attempt = state
    await text_store.replace(
        scope, attempt, 'emergency old', reference('emergency old')
    )
    new_attempt = str(uuid4())
    await items.reindex(scope, expected_indexing_dag_id=attempt, indexing_dag_id=new_attempt)
    upstream = uuid4()
    await pool.execute(
        'CREATE TABLE IF NOT EXISTS marie_scheduler.dag (id uuid PRIMARY KEY,state text,completed_on timestamptz)'
    )
    await pool.execute(
        "INSERT INTO marie_scheduler.dag(id,state,completed_on) VALUES($1,'created',NULL)",
        upstream,
    )
    ref = {
        'kind': 'upstream',
        'dag_id': str(upstream),
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'artifact': reference('emergency new'),
    }
    import json

    await pool.execute(
        'UPDATE marie_scheduler.kb_item_state SET input_ref=$1::jsonb', json.dumps(ref)
    )
    with pytest.raises(StaleIndexingRun, match='Upstream'):
        await text_store.replace(
            scope, new_attempt, 'emergency new', reference('emergency new')
        )
    assert (await text_store.search(scope.tenant_id, scope.index_id, 'old'))[0][
        'indexed_dag_id'
    ] == attempt
