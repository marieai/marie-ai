import os
from uuid import uuid4

import pytest
from psycopg import ProgrammingError

from marie.kb.item_state import KbItemScope, StaleIndexingRun
from marie.vector_stores.pgvector.pgvector import PGVectorStore
from tests.unit.kb.test_item_state import state

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable KB PostgreSQL'
)


@pytest.fixture
async def vector(state):
    _, pool, _, _ = state
    await pool.execute('DROP VIEW IF EXISTS kb_f02a_vectors_current')
    await pool.execute('DROP TABLE IF EXISTS kb_f02a_vectors')
    await pool.execute('''CREATE TABLE kb_f02a_vectors (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(), node_id text UNIQUE NOT NULL,
        index_name text, source_id text, ref_doc_id text, content text, node_type text,
        metadata jsonb, embedding text, created_at timestamptz DEFAULT NOW())''')
    store = PGVectorStore(
        connection_string='postgresql://unused',
        table_name='kb_f02a_vectors',
        embedding_dim=2,
        guarded=True,
    )
    store._pool = pool
    store._initialized = True
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET backend='pgvector',store_name='kb_f02a_vectors'"
    )
    await store.initialize_current_view()
    return store


def node(record_id, text='record'):
    return {
        'node_id': record_id,
        'content': text,
        'embedding': [0.1, 0.2],
        'node_type': 'text',
        'metadata': {},
    }


async def test_shorter_replacement_removes_only_its_owned_output(state, vector):
    items, pool, scope, attempt = state
    await vector.replace_item_nodes(scope, attempt, [node('a'), node('b')])
    other = KbItemScope(
        scope.tenant_id, scope.index_id, scope.source_id, 'another-item'
    )
    other_attempt = str(uuid4())
    await items.register(
        other,
        indexing_dag_id=other_attempt,
        owner_type='record',
        owner_id='another',
        uri='s3://fixture/other',
        run_params={},
        backend='pgvector',
        store_name=vector.table_name,
    )
    await vector.replace_item_nodes(other, other_attempt, [node('a', 'other')])
    next_attempt = str(uuid4())
    await items.reindex(scope, expected_indexing_dag_id=attempt, indexing_dag_id=next_attempt)
    await vector.replace_item_nodes(scope, next_attempt, [node('a', 'new')])
    rows = await pool.fetch(
        'SELECT content FROM kb_f02a_vectors_current ORDER BY content'
    )
    assert [r['content'] for r in rows] == ['new', 'other']


async def test_cross_index_local_ids_do_not_overwrite_each_other(state, vector):
    items, pool, scope, attempt = state
    other = KbItemScope(scope.tenant_id, str(uuid4()), scope.source_id, scope.item_id)
    second = str(uuid4())
    await items.register(
        other,
        indexing_dag_id=second,
        owner_type='record',
        owner_id='another',
        uri='s3://fixture/other',
        run_params={},
        backend='pgvector',
        store_name=vector.table_name,
    )
    await vector.replace_item_nodes(scope, attempt, [node('same', 'first')])
    await vector.replace_item_nodes(other, second, [node('same', 'second')])
    assert await pool.fetchval('SELECT count(*) FROM kb_f02a_vectors_current') == 2


async def test_removed_output_is_hidden_before_physical_cleanup(state, vector):
    items, pool, scope, attempt = state
    await vector.replace_item_nodes(scope, attempt, [node('record')])
    await items.remove(scope)
    assert await pool.fetchval('SELECT count(*) FROM kb_f02a_vectors') == 1
    assert await pool.fetchval('SELECT count(*) FROM kb_f02a_vectors_current') == 0
    with pytest.raises(StaleIndexingRun):
        await vector.replace_item_nodes(scope, attempt, [node('late')])


async def test_stale_attempt_cannot_delete_the_current_result(state, vector):
    items, pool, scope, attempt = state
    second = str(uuid4())
    await items.reindex(scope, expected_indexing_dag_id=attempt, indexing_dag_id=second)
    await vector.replace_item_nodes(scope, second, [node('current')])
    with pytest.raises(StaleIndexingRun):
        await vector.replace_item_nodes(scope, attempt, [])
    assert await pool.fetchval('SELECT count(*) FROM kb_f02a_vectors_current') == 1


async def test_adapter_failure_rolls_back_entire_output_and_completion(state, vector):
    items, pool, scope, attempt = state
    await vector.replace_item_nodes(scope, attempt, [node('previous', 'old')])
    second = str(uuid4())
    await items.reindex(scope, expected_indexing_dag_id=attempt, indexing_dag_id=second)
    with pytest.raises(ProgrammingError):
        await vector.replace_item_nodes(
            scope,
            second,
            [node('ok'), {**node('bad'), 'content': {'invalid': 'sql text'}}],
        )
    assert (await pool.fetchrow('SELECT content FROM kb_f02a_vectors_current'))[
        'content'
    ] == 'old'
    assert str((await items.get(scope))['indexed_dag_id']) == attempt


async def test_managed_reads_and_counts_hide_removed_items(state, vector):
    items, _, scope, attempt = state
    await vector.replace_item_nodes(scope, attempt, [node('record')])
    assert await vector.count_nodes(index_name=scope.index_id) == 1
    await items.remove(scope)
    assert await vector.count_nodes(index_name=scope.index_id) == 0
    assert await vector.aget_nodes(index_name=scope.index_id) == []


async def test_unchecked_writes_and_partial_deletes_are_not_managed_operations(
    state, vector
):
    with pytest.raises(PermissionError):
        await vector.add_nodes_batch(
            [node('unregistered')], source_id='source', index_name='index'
        )
    with pytest.raises(PermissionError):
        await vector.adelete_nodes(node_ids=['record'])
    with pytest.raises(PermissionError):
        await vector.aclear()


async def test_document_cleanup_withdraws_one_item_and_preserves_its_sibling(
    state, vector
):
    items, pool, scope, attempt = state
    other = KbItemScope(scope.tenant_id, scope.index_id, scope.source_id, 'sibling')
    second = str(uuid4())
    await items.register(
        other,
        indexing_dag_id=second,
        owner_type='record',
        owner_id='sibling',
        uri='s3://fixture/sibling',
        run_params={},
        backend='pgvector',
        store_name=vector.table_name,
    )
    await vector.replace_item_nodes(scope, attempt, [node('record', 'first')])
    await vector.replace_item_nodes(other, second, [node('record', 'sibling')])
    await vector.delete_item(scope)
    assert [
        r['content']
        for r in await pool.fetch('SELECT content FROM kb_f02a_vectors_current')
    ] == ['sibling']
    assert (await items.get(scope))['cleanup_state'] == 'complete'
    await vector.delete_item(scope)
    assert (await items.get(other))['state'] == 'ready'


async def test_wrong_backend_table_cannot_complete_or_delete_an_item(state, vector):
    _, pool, scope, attempt = state
    wrong = PGVectorStore(
        connection_string='postgresql://unused',
        table_name='kb_unowned_table',
        embedding_dim=2,
        guarded=True,
    )
    wrong._pool = pool
    wrong._initialized = True
    with pytest.raises(PermissionError, match='backend'):
        await wrong.replace_item_nodes(scope, attempt, [node('forged')])
    with pytest.raises(PermissionError, match='backend'):
        await wrong.delete_item(scope)


async def test_reads_are_bound_to_the_configured_backend_store(state, vector):
    _, pool, scope, attempt = state
    await vector.replace_item_nodes(scope, attempt, [node('record')])
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET store_name='different_store'"
    )
    assert await vector.count_nodes(index_name=scope.index_id) == 0


async def test_scope_delete_does_not_claim_cleanup_for_another_backend(state, vector):
    _, pool, scope, _ = state
    await pool.execute("UPDATE marie_scheduler.kb_item_state SET backend='code'")
    with pytest.raises(PermissionError, match='unsupported backend'):
        await vector.delete_scope(scope.tenant_id, scope.index_id)


async def test_scope_cleanup_resumes_only_unfinished_batches_and_preserves_relinks(
    state, vector, monkeypatch
):
    items, pool, scope, attempt = state
    await vector.replace_item_nodes(scope, attempt, [node('one')])
    original = PGVectorStore._cleanup_removed_batch
    captured = []

    async def interrupted(self, tenant, index, rows):
        captured.extend(rows)
        raise RuntimeError('cleanup interrupted')

    monkeypatch.setattr(PGVectorStore, '_cleanup_removed_batch', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        await vector.delete_scope(scope.tenant_id, scope.index_id)
    assert (await items.get(scope))['cleanup_state'] == 'pending'
    linked = KbItemScope(scope.tenant_id, scope.index_id, scope.source_id, 'new-link')
    linked_attempt = str(uuid4())
    await items.register(
        linked,
        indexing_dag_id=linked_attempt,
        owner_type='record',
        owner_id='one',
        uri='s3://fixture/one',
        run_params={},
        backend='pgvector',
        store_name=vector.table_name,
    )
    await vector.replace_item_nodes(linked, linked_attempt, [node('one', 'relinked')])
    assert await original(vector, scope.tenant_id, scope.index_id, captured) == 1
    assert [
        r['content']
        for r in await pool.fetch('SELECT content FROM kb_f02a_vectors_current')
    ] == ['relinked']
    seen = []

    async def recording(self, tenant, index, rows):
        seen.extend(row['item_id'] for row in rows)
        return await original(self, tenant, index, rows)

    monkeypatch.setattr(PGVectorStore, '_cleanup_removed_batch', recording)
    await vector.delete_scope(scope.tenant_id, scope.index_id)
    assert seen == ['new-link']
    assert await vector.delete_scope(scope.tenant_id, scope.index_id) == 0
