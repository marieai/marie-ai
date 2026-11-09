import json
import os
from uuid import uuid4

import pytest

from marie.kb.item_state import KbItemScope, StaleIndexingRun, item_run_parameters
from marie.query_planner.base import PlannerInfo
from marie.query_planner.kb_indexing_planner import (
    query_planner_kb_index_content,
    query_planner_kb_indexing,
)
from tests.unit.executor.kb.test_vector_store_executor_embed import _make_executor
from tests.unit.kb.test_item_state import state

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable KB PostgreSQL'
)


@pytest.mark.parametrize('saved_content', [False, True])
async def test_registered_indexing_resolves_saved_settings_before_planning(
    state, saved_content
):
    store, pool, scope, dag = state
    saved = {
        'parse_mode': 'auto',
        'chunk_size': 500,
        'chunk_overlap': 20,
        'retrieval_backend': 'postgresql_text',
        'multimodal': False,
    }
    await pool.execute(
        'UPDATE marie_scheduler.kb_item_state SET run_params=$1::jsonb',
        json.dumps(saved),
    )
    if saved_content:
        await pool.execute(
            'UPDATE marie_scheduler.kb_item_state SET input_ref=$1::jsonb',
            json.dumps(
                {
                    'kind': 'text',
                    'uri': 's3://fixture/saved.md',
                    'media_type': 'text/markdown',
                    'sha256': 'a' * 64,
                    'processing_workflow': 'kb_index_content',
                }
            ),
        )
    item = await store.get(scope)
    params = item_run_parameters(item)
    params.pop('run_params')
    resolved = await store.resolve_indexing_parameters(params)
    assert resolved['run_params'] == saved
    planner = (
        query_planner_kb_index_content if saved_content else query_planner_kb_indexing
    )
    plan = planner(
        PlannerInfo(name=resolved['planner'], base_id=dag, metadata=resolved)
    )
    if not saved_content:
        assert plan.nodes[1].definition.params['parse_mode'] == 'auto'
    assert plan.nodes[-2].definition.endpoint == 'vector_store_executor://index_text'


async def test_registered_indexing_rejects_caller_processing_override(state):
    store, _, scope, _ = state
    params = item_run_parameters(await store.get(scope))
    params['run_params'] = {'parse_mode': 'forged'}
    with pytest.raises(ValueError, match='saved'):
        await store.resolve_indexing_parameters(params)


async def test_registered_indexing_rejects_stale_dag(state):
    store, _, scope, dag = state
    params = item_run_parameters(await store.get(scope))
    await store.remove(scope)
    with pytest.raises(StaleIndexingRun):
        await store.resolve_indexing_parameters(params)


@pytest.mark.parametrize('legacy_state', ['pending', 'removed'])
async def test_search_prefers_a_live_saved_profile_over_legacy_or_removed_rows(
    state, legacy_state
):
    store, pool, scope, _ = state
    await pool.execute(
        "UPDATE marie_scheduler.kb_item_state SET backend='pgvector',store_name='kb_vectors',state=$1",
        legacy_state,
    )
    await store.register(
        KbItemScope(scope.tenant_id, scope.index_id, scope.source_id, 'game:two'),
        indexing_dag_id=str(uuid4()),
        owner_type='kb_document',
        owner_id='two',
        uri='s3://fixture/two',
        run_params={'embedding_model': 'different-model', 'embedding_dim': 2},
        backend='pgvector',
        store_name='kb_vectors',
    )
    executor = _make_executor()
    executor._table_name = 'kb_vectors'
    executor._embedding_model = 'runtime-model'
    executor._embedding_dim = 2
    executor._vector_store.guarded = True
    executor._vector_store.client = pool
    with pytest.raises(ValueError, match='embedding profile'):
        await executor._require_index_scope(
            {'tenant_id': scope.tenant_id, 'index_id': scope.index_id}, embedding=True
        )
