import json
from uuid import UUID

import pytest

from tests.unit.kb.code.test_capture import git, repo
from tests.unit.kb.code.test_store import code_store
from tests.unit.kb.code.test_zoekt import zoekt


async def prepare(code_store):
    store, pool, scope, execution, repository = code_store
    (repository / 'same.py').write_text(
        'def target():\n    return 1\n\nvalue = target()\n'
    )
    (repository / 'other.py').write_text(
        'def target():\n    return 2\n\nvalue = target()\n'
    )
    (repository / 'README.txt').write_text('Exact search remains available.\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'navigation fixture')
    await store.refresh(await execution())
    citation = (await store.search(scope, 'target', 'literal'))['results'][0][
        'citation'
    ]
    return store, pool, scope, execution, repository, citation


async def test_navigation_keeps_duplicate_definitions_and_opens_captured_candidates(
    code_store,
):
    store, pool, scope, execution, repository, citation = await prepare(code_store)
    definitions = await store.navigate(scope, citation, 'definitions', 'target')
    assert definitions['outcome'] == 'ready'
    assert definitions['heuristic'] is True
    assert [row['citation']['path'] for row in definitions['results']] == [
        'other.py',
        'same.py',
    ]
    assert definitions['coverage']['unsupported'] == 1
    for row in definitions['results']:
        opened = await store.open(scope, row['citation'])
        assert 'def target():' in opened['text']
    references = await store.navigate(scope, citation, 'references', 'target')
    assert [row['citation']['startLine'] for row in references['results']] == [4, 4]
    limited = await store.navigate(scope, citation, 'definitions', 'target', limit=1)
    assert len(limited['results']) == 1 and limited['truncated']
    outline = await store.navigate(scope, citation, 'outline')
    assert {row['citation']['path'] for row in outline['results']} == {citation['path']}


async def test_navigation_failed_refresh_preserves_previous_then_retires_old_capture(
    code_store, monkeypatch
):
    store, pool, scope, execution, repository, citation = await prepare(code_store)
    before = await store.navigate(scope, citation, 'definitions', 'target')
    (repository / 'same.py').write_text('def replacement():\n    pass\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'replacement')
    with monkeypatch.context() as patch:

        async def fail(*args):
            raise RuntimeError('build interrupted')

        patch.setattr(store.zoekt, 'build', fail)
        with pytest.raises(RuntimeError, match='interrupted'):
            await store.refresh(await execution())
    assert (await store.navigate(scope, citation, 'definitions', 'target')) == before
    await store.refresh(await execution())
    assert (await store.navigate(scope, citation, 'outline'))['outcome'] == 'changed'
    assert (
        await pool.fetchval(
            'SELECT count(*) FROM marie_scheduler.kb_code_navigation WHERE handle=$1',
            citation['handle'],
        )
        == 0
    )


async def test_navigation_rechecks_authority_and_removal_retires_records(code_store):
    store, pool, scope, execution, repository, citation = await prepare(code_store)
    with pytest.raises(PermissionError):
        await store.navigate(
            {**scope, 'organization_id': '11111111-1111-4111-8111-111111111111'},
            citation,
            'outline',
        )
    await pool.execute(
        "UPDATE marie_scheduler.kb_source SET dataset_inputs=jsonb_set(dataset_inputs,'{branch}','\"other\"')"
    )
    assert (await store.navigate(scope, citation, 'outline'))[
        'outcome'
    ] == 'unavailable'
    await pool.execute(
        "UPDATE marie_scheduler.kb_source SET dataset_inputs=jsonb_set(dataset_inputs,'{branch}','\"main\"')"
    )
    await store.remove(scope)
    assert (
        await pool.fetchval(
            'SELECT count(*) FROM marie_scheduler.kb_code_navigation WHERE source_id=$1',
            UUID(scope['source_id']),
        )
        == 0
    )
    with pytest.raises(PermissionError):
        await store.navigate(scope, citation, 'outline')


async def test_navigation_candidates_always_fit_captured_read_budgets(code_store):
    store, pool, scope, execution, repository = code_store
    (repository / 'large.py').write_text(
        'def large():\n' + ('    value = "' + 'x' * 130 + '"\n') * 80
    )
    (repository / 'giant.py').write_text('def giant(): return "' + 'x' * 9000 + '"\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'large navigation fixture')
    await store.refresh(await execution())
    citation = (await store.search(scope, 'def large', 'literal'))['results'][0][
        'citation'
    ]
    result = await store.navigate(scope, citation, 'definitions', 'large')
    assert len(result['results']) == 1
    candidate = result['results'][0]
    opened = await store.open(scope, candidate['citation'])
    assert opened['outcome'] == 'ready'
    assert len(opened['passage']) <= 8000
    assert candidate['rangeTruncated'] is True
    oversized = await store.navigate(scope, citation, 'definitions', 'giant')
    assert oversized['results'] == [] and oversized['truncated'] is True


@pytest.mark.parametrize('separator', ['\u2028', '\x85', '\v', '\f'])
async def test_navigation_uses_parser_lf_line_numbers_for_original_ranges(
    code_store, separator
):
    store, pool, scope, execution, repository = code_store
    body = f'# separator:{separator}more\r\n\r\ndef target():\r\n    return 1\r\n'
    (repository / 'unicode.py').write_bytes(body.encode())
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'unicode source lines')
    await store.refresh(await execution())
    citation = (await store.search(scope, 'def target', 'literal'))['results'][0][
        'citation'
    ]
    result = await store.navigate(scope, citation, 'definitions', 'target')
    assert len(result['results']) == 1
    candidate = result['results'][0]['citation']
    assert candidate['startLine'] == 3 and candidate['endLine'] == 4
    opened = await store.open(scope, candidate)
    assert opened['passage'] == 'def target():\r\n    return 1\r\n'


async def test_configuration_change_between_anchor_and_navigation_is_unavailable(
    code_store, monkeypatch
):
    store, pool, scope, execution, repository, citation = await prepare(code_store)
    original_open = store.open

    async def change_after_open(*args):
        result = await original_open(*args)
        await pool.execute(
            "UPDATE marie_scheduler.kb_source SET dataset_inputs=jsonb_set(dataset_inputs,'{branch}','\"other\"')"
        )
        return result

    monkeypatch.setattr(store, 'open', change_after_open)
    assert (await store.navigate(scope, citation, 'outline'))[
        'outcome'
    ] == 'unavailable'


async def test_captured_open_validates_line_numbers_after_real_rpc_roundtrip(
    code_store,
):
    from marie.types_core.request.data import DataRequest

    store, pool, scope, execution, repository, citation = await prepare(code_store)
    request = DataRequest()
    request.parameters = {'citation': citation}
    decoded = request.parameters['citation']
    assert isinstance(decoded['startLine'], float)
    opened = await store.open(scope, decoded)
    assert opened['outcome'] == 'ready'
    assert isinstance(opened['code']['startLine'], int)
    for invalid in (1.5, True, '1', float('nan'), float('inf')):
        with pytest.raises(ValueError, match='line numbers'):
            await store.open(scope, {**decoded, 'startLine': invalid})
