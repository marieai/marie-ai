import hashlib
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from docarray import DocList
from docarray.documents import TextDoc

from marie.executor.kb.vector_store_executor import VectorStoreExecutor
from marie.kb.item_state import StaleIndexingRun, item_run_parameters
from marie.storage import StorageManager
from tests.unit.executor.kb.test_vector_store_executor_embed import _make_executor


async def test_saved_extraction_uses_upstream_reference_and_preserves_destination(current):
    reference = {'kind': 'extraction', 'ref_id': 'upstream-output', 'ref_type': 'document'}
    executor, row, params = current
    row['input_ref'] = reference
    params['input_ref'] = reference
    from unittest.mock import Mock
    resolver = Mock(return_value=('saved text', [(0, 10, 0)]))
    executor._resolve_extraction_output = resolver
    await executor.embed_and_store(DocList[TextDoc]([TextDoc()]), params)
    assert resolver.call_args.args[:2] == ('upstream-output', 'document')
    scope = executor._vector_store.replace_item_nodes.await_args.args[0]
    assert scope.index_id == row['index_id']


@pytest.mark.parametrize('kind', ['text', 'extraction'])
async def test_saved_input_retry_reuses_captured_bytes_and_rejects_changed_content(current, monkeypatch, kind):
    executor, row, params = current
    payload = b'# Saved content\nA document.' if kind == 'text' else json.dumps({'ocr': [{'lines': [{'text': 'Saved extraction'}]}]}).encode()
    reference = {'kind': kind, 'sha256': hashlib.sha256(payload).hexdigest()}
    if kind == 'text':
        reference.update(uri='s3://fixture/captured/content.md', media_type='text/markdown')
    else:
        reference.update(ref_id='source.pdf', ref_type='document')
    row['input_ref'] = params['input_ref'] = reference
    reads = []

    def read(uri):
        reads.append(uri)
        return payload

    monkeypatch.setattr(StorageManager, 'read', read)
    monkeypatch.setattr(StorageManager, 'ensure_connection', lambda *a, **kw: True)
    monkeypatch.setattr(StorageManager, 'exists', lambda *a, **kw: True)
    embeddings = executor._embeddings.embed_text.side_effect
    executor._embeddings.embed_text.side_effect = RuntimeError('temporary embedding failure')
    with pytest.raises(RuntimeError, match='temporary embedding'):
        await executor.embed_and_store(DocList[TextDoc]([TextDoc()]), params)
    executor._vector_store.replace_item_nodes.assert_not_awaited()
    executor._embeddings.embed_text.side_effect = embeddings
    await executor.embed_and_store(DocList[TextDoc]([TextDoc()]), params)
    assert len(reads) == 2 and reads[0] == reads[1]
    executor._vector_store.replace_item_nodes.assert_awaited_once()
    payload = b'changed after capture'
    with pytest.raises(ValueError, match='changed'):
        await executor.embed_and_store(DocList[TextDoc]([TextDoc(text='must not bypass saved input')]), params)
    executor._vector_store.replace_item_nodes.assert_awaited_once()


async def test_unsupported_domain_input_never_falls_back_to_document_extraction(current):
    executor, row, params = current
    row['input_ref'] = {'kind': 'chess', 'uri': 's3://fixture/game.json'}
    params['input_ref'] = row['input_ref']
    with pytest.raises(ValueError, match='workflow|Unsupported'):
        await executor.embed_and_store(DocList[TextDoc]([TextDoc(text='do not coerce')]), params)
    executor._embeddings.embed_text.assert_not_called()


async def test_saved_content_workflow_cannot_use_an_unpinned_request_reference(current):
    executor, row, params = current
    params['input_ref'] = {'kind': 'extraction', 'ref_id': 'injected', 'ref_type': 'document'}
    with pytest.raises(ValueError, match='reference|input'):
        await executor.embed_and_store(DocList[TextDoc]([TextDoc()]), params)


def test_managed_table_contract_is_checked_before_executor_startup():
    with pytest.raises(ValueError, match='kb_vectors'):
        VectorStoreExecutor(table_name='custom_unqualified_store')


@pytest.fixture
def current():
    executor = _make_executor()
    row = {
        'tenant_id': str(uuid4()),
        'index_id': str(uuid4()),
        'source_id': str(uuid4()),
        'item_id': str(uuid4()),
        'indexing_dag_id': uuid4(),
        'owner_type': 'kb_document',
        'owner_id': str(uuid4()),
        'uri': 's3://fixture/document.pdf',
        'state': 'pending',
        'run_params': {'chunk_size': 10, 'chunk_overlap': 0},
        'backend': 'pgvector',
        'store_name': 'kb_vectors',
    }
    executor._table_name = 'kb_vectors'
    executor._vector_store.guarded = True
    executor._vector_store.client = AsyncMock()
    executor._vector_store.client.fetchrow.return_value = row
    executor._vector_store.replace_item_nodes = AsyncMock(
        side_effect=lambda scope, attempt, nodes, **kw: len(nodes)
    )
    executor._vector_store.delete_scope = AsyncMock(return_value=2)
    executor._vector_store.delete_item = AsyncMock(return_value=1)
    executor._vector_store.aclear = AsyncMock()
    return (
        executor,
        row,
        {
            **item_run_parameters(row),
            'job_id': str(uuid4()),
            'run_attempt_id': str(uuid4()),
            'run_owner': 'scheduler',
        },
    )


async def test_current_item_prepares_everything_before_one_checked_write(current):
    executor, row, params = current
    docs = DocList[TextDoc]([TextDoc(text='one'), TextDoc(text='two')])
    await executor.embed_and_store(docs, params)
    executor._vector_store.replace_item_nodes.assert_awaited_once()
    nodes = executor._vector_store.replace_item_nodes.await_args.args[2]
    assert [node['content'] for node in nodes] == ['one', 'two']
    digest = executor._vector_store.replace_item_nodes.await_args.kwargs[
        'content_digest'
    ]
    assert len(digest) == 64
    assert all(node['metadata']['artifact_digest'] == digest for node in nodes)
    assert nodes[0]['metadata']['source_uri'] == row['uri']
    executor._vector_store.add_node.assert_not_awaited()
    executor._vector_store.add_nodes_batch.assert_not_awaited()


async def test_failed_preparation_never_stores_partial_current_output(current):
    executor, _, params = current
    executor._embeddings.embed_text.side_effect = RuntimeError('embedding unavailable')
    with pytest.raises(RuntimeError, match='embedding unavailable'):
        await executor.embed_and_store(
            DocList[TextDoc]([TextDoc(text='one'), TextDoc(text='two')]), params
        )
    executor._vector_store.replace_item_nodes.assert_not_awaited()


async def test_stale_attempt_is_rejected_before_embedding(current):
    executor, row, params = current
    row['indexing_dag_id'] = uuid4()
    with pytest.raises(StaleIndexingRun):
        await executor.embed_and_store(DocList[TextDoc]([TextDoc(text='late')]), params)
    executor._embeddings.embed_text.assert_not_called()


async def test_unregistered_direct_ingest_cannot_write_to_managed_store(current):
    executor, _, _ = current
    with pytest.raises(ValueError, match='KB item'):
        await executor.ingest_batch(
            DocList[TextDoc](), {'source_id': 'untrusted', 'texts': ['injected']}
        )
    executor._vector_store.add_nodes_batch.assert_not_awaited()


async def test_incomplete_embedding_batch_cannot_become_current(current):
    executor, _, params = current
    executor._embeddings.embed_text.side_effect = lambda *args, **kw: []
    with pytest.raises(RuntimeError, match='embedding'):
        await executor.embed_and_store(DocList[TextDoc]([TextDoc(text='one')]), params)
    executor._vector_store.replace_item_nodes.assert_not_awaited()


async def test_direct_write_cannot_omit_scheduler_execution(current):
    executor, row, _ = current
    with pytest.raises(ValueError, match='scheduler execution'):
        await executor.embed_and_store(
            DocList[TextDoc]([TextDoc(text='injected')]), item_run_parameters(row)
        )
    executor._vector_store.replace_item_nodes.assert_not_awaited()


async def test_repeated_completed_delivery_does_not_recompute_or_overwrite(current):
    executor, row, params = current
    row['indexed_dag_id'] = row['indexing_dag_id']
    row['state'] = 'ready'
    await executor.embed_and_store(
        DocList[TextDoc]([TextDoc(text='duplicate')]), params
    )
    executor._embeddings.embed_text.assert_not_called()
    executor._vector_store.replace_item_nodes.assert_not_awaited()


async def test_direct_read_requires_tenant_and_index_scope(current):
    executor, row, _ = current
    with pytest.raises(ValueError, match='scope'):
        await executor.search(
            DocList[TextDoc](), {'query': 'secret', 'index_name': row['index_id']}
        )


async def test_clear_is_scoped_to_registered_items_in_one_index(current):
    executor, row, params = current
    await executor.clear_store(DocList[TextDoc](), {**params, 'confirm': True})
    executor._vector_store.delete_scope.assert_awaited_once_with(
        row['tenant_id'], row['index_id']
    )
    executor._vector_store.aclear.assert_not_awaited()


async def test_delete_item_uses_exact_registered_identity(current):
    executor, row, params = current
    result = await executor.delete_item(DocList[TextDoc](), params)
    assert result['item_id'] == row['item_id']
    assert result['cleanup_state'] == 'complete'
    scope = executor._vector_store.delete_item.await_args.args[0]
    assert scope.source_id == row['source_id']
    assert scope.index_id == row['index_id']


async def test_empty_index_removal_is_scoped_without_requiring_an_item(current):
    executor, row, params = current
    executor._vector_store.client.fetchrow.return_value = None
    executor._vector_store.delete_scope.return_value = 0
    result = await executor.delete_index(DocList[TextDoc](), params)
    assert result['deleted_count'] == 0
    executor._vector_store.delete_scope.assert_awaited_once_with(
        row['tenant_id'], row['index_id']
    )
