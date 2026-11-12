from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from marie.kb.submission import KbSubmissionHooks


async def test_preparation_resolves_current_item_in_kb_owned_integration():
    item = {
        'tenant_id': UUID(int=1),
        'index_id': UUID(int=2),
        'source_id': 'source',
        'item_id': 'item',
        'indexing_dag_id': UUID(int=3),
        'owner_type': 'document',
        'owner_id': 'doc',
        'uri': 's3://fixture/item.pdf',
        'run_params': {'parse_mode': 'auto'},
        'state': 'pending',
    }
    pool = SimpleNamespace(fetchval=AsyncMock(return_value=item))
    hooks = KbSubmissionHooks(lambda: pool)
    metadata = {
        key: str(item[key]) for key in ('tenant_id', 'index_id', 'source_id', 'item_id')
    }
    metadata.update(
        planner='kb_indexing', kb_indexing_dag_id=str(item['indexing_dag_id'])
    )
    work = SimpleNamespace(name='kb_indexing', data={'metadata': metadata})
    await hooks.prepare(work)
    assert work.data['metadata']['run_params'] == {'parse_mode': 'auto'}
    assert work.data['metadata']['uri'] == item['uri']
    pool.fetchval.assert_awaited_once()


async def test_non_kb_work_is_opaque_to_the_kb_integration():
    pool = SimpleNamespace(fetchval=AsyncMock())
    hooks = KbSubmissionHooks(lambda: pool)
    work = SimpleNamespace(
        id='job', name='calculate', data={'metadata': {'mode': 'sum'}}
    )
    await hooks.prepare(work)
    await hooks.persist(pool, work)
    pool.fetchval.assert_not_awaited()
    assert work.data['metadata'] == {'mode': 'sum'}
