import os

import pytest

from tests.unit.kb.code.test_capture import git, repo
from tests.unit.kb.code.test_store import code_store
from tests.unit.kb.code.test_zoekt import zoekt

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable PostgreSQL'
)


async def test_manifest_tree_and_retained_revision_survive_refresh(code_store):
    store, _, scope, execution, repository = code_store
    store.retained_captures = 5
    await store.refresh(await execution())
    first = (await store.search(scope, 'alpha', 'literal'))['results'][0]['citation']
    (repository / 'same.py').write_text('alpha = 3\n')
    git(repository, 'add', 'same.py')
    git(repository, 'commit', '-m', 'second capture')
    await store.refresh(await execution())
    current = (await store.search(scope, 'alpha', 'literal'))['results'][0]['citation']
    assert (await store.open(scope, first, whole_file=True))['text'].startswith(
        'alpha = 1'
    )
    revisions = await store.browse(scope, current, 'revisions')
    assert len(revisions['revisions']) == 2
    assert revisions['revisions'][1]['citation']['commit'] == first['commit']
    tree = await store.browse(scope, current, 'tree')
    assert [entry['path'] for entry in tree['entries']] == ['same.py']
    assert tree['entries'][0]['citation']['handle'] == current['handle']


async def test_retained_revision_is_revoked_by_changed_source_configuration(code_store):
    store, pool, scope, execution, _ = code_store
    await store.refresh(await execution())
    citation = (await store.search(scope, 'alpha', 'literal'))['results'][0]['citation']
    await pool.execute(
        "UPDATE marie_scheduler.kb_source SET dataset_inputs=dataset_inputs||'{\"include\":[\"other.py\"]}'::jsonb"
    )
    assert (await store.browse(scope, citation, 'tree'))['outcome'] != 'ready'


async def test_navigation_continuation_returns_the_remaining_records(code_store):
    store, _, scope, execution, repository = code_store
    (repository / 'functions.py').write_text(
        '\n'.join(f'def fn_{i}(): return {i}' for i in range(12)) + '\n'
    )
    git(repository, 'add', 'functions.py')
    git(repository, 'commit', '-m', 'outline fixture')
    await store.refresh(await execution())
    citation = (await store.search(scope, 'fn_0', 'literal'))['results'][0]['citation']
    first = await store.navigate(scope, citation, 'outline', '', 5, offset=0)
    second = await store.navigate(
        scope, citation, 'outline', '', 5, offset=first['nextOffset']
    )
    assert first['total'] == 12
    assert not {r['name'] for r in first['results']} & {
        r['name'] for r in second['results']
    }


async def test_reference_search_includes_declarations_and_variable_reads_without_a_parser(
    code_store,
):
    store, _, scope, execution, repository = code_store
    (repository / 'notes.txt').write_text(
        'target = configuration\nconsume(target)\nnot_target = 0\n'
    )
    git(repository, 'add', 'notes.txt')
    git(repository, 'commit', '-m', 'lexical reference fixture')
    await store.refresh(await execution())
    citation = (await store.search(scope, 'configuration', 'literal'))['results'][0][
        'citation'
    ]
    first = await store.navigate(scope, citation, 'references', 'target', 1)
    assert first['total'] == 2
    assert first['results'][0]['citation']['startLine'] == 1
    next_page = await store.navigate(
        scope, citation, 'references', 'target', 1, offset=first['nextOffset']
    )
    assert next_page['results'][0]['citation']['startLine'] == 2


async def test_dirty_captures_are_not_presented_as_committed_history(code_store):
    store, pool, scope, execution, repository = code_store
    store.retained_captures = 5
    await pool.execute(
        "UPDATE marie_scheduler.kb_source SET dataset_inputs=dataset_inputs||'{\"dirty\":true}'::jsonb"
    )
    for value in (2, 3):
        (repository / 'same.py').write_text(f'alpha = {value}\n')
        await store.refresh(await execution())
    citation = (await store.search(scope, 'alpha', 'literal'))['results'][0]['citation']
    revisions = await store.browse(scope, citation, 'revisions')
    assert len(revisions['revisions']) == 2
    assert len({row['commit'] for row in revisions['revisions']}) == 1
    history = await store.browse(scope, citation, 'history')
    assert history['outcome'] == 'ready'
    assert history['commits'][0]['citation'] is None


async def test_navigation_available_totals_exclude_uncitable_long_lines(code_store):
    store, _, scope, execution, repository = code_store
    (repository / 'long.py').write_text('def target(): pass  #' + 'x' * 8100 + '\n')
    (repository / 'short.py').write_text('def target(): pass\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'long line fixture')
    await store.refresh(await execution())
    citation = (await store.search(scope, 'alpha', 'literal'))['results'][0]['citation']
    for mode in ('definitions', 'references'):
        result = await store.navigate(scope, citation, mode, 'target', 50)
        assert result['total'] == len(result['results']) == 1
        assert result['matchedTotal'] == 2
        assert result['omitted'] == 1
        assert result['nextOffset'] is None


async def test_definitions_report_repository_parser_gaps_not_just_anchor(code_store):
    store, _, scope, execution, repository = code_store
    (repository / 'short.py').write_text('def target(): pass\n')
    (repository / 'notes.txt').write_text('target has undocumented callers\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'parser gap fixture')
    await store.refresh(await execution())
    citation = (await store.search(scope, 'alpha', 'literal'))['results'][0]['citation']
    result = await store.navigate(scope, citation, 'definitions', 'target', 50)
    assert result['coverage']['unsupported'] == 1
    assert result['status'] == 'partial'
    assert result['totalExact'] is False


async def test_legacy_navigation_totals_check_actual_captured_lines(code_store):
    store, pool, scope, execution, repository = code_store
    (repository / 'long.py').write_text('def target(): pass  #' + 'x' * 8100 + '\n')
    (repository / 'short.py').write_text('def target(): pass\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'legacy navigation fixture')
    await store.refresh(await execution())
    await pool.execute(
        "UPDATE marie_scheduler.kb_code_navigation SET symbols=(SELECT jsonb_agg(entry - 'readable') FROM jsonb_array_elements(symbols) entry) WHERE jsonb_array_length(symbols)>0"
    )
    citation = (await store.search(scope, 'alpha', 'literal'))['results'][0]['citation']
    result = await store.navigate(scope, citation, 'definitions', 'target', 1)
    assert result['total'] == 1
    assert result['omitted'] == 1
    assert result['results'][0]['citation']['path'] == 'short.py'
    assert result['nextOffset'] is None


async def test_git_metadata_failure_leaves_captured_file_readable(code_store):
    store, _, scope, execution, repository = code_store
    await store.refresh(await execution())
    citation = (await store.search(scope, 'alpha', 'literal'))['results'][0]['citation']
    (repository / '.git').rename(repository / '.git-unavailable')
    assert (await store.browse(scope, citation, 'history'))['outcome'] == 'unavailable'
    assert (await store.open(scope, citation, whole_file=True))['outcome'] == 'ready'


async def test_captured_file_list_loads_nested_paths_in_one_bounded_page(code_store):
    store, _, scope, execution, repository = code_store
    folder = repository / 'src' / 'nested'
    folder.mkdir(parents=True)
    (folder / 'one.py').write_text('one = 1\n')
    (folder / 'two.py').write_text('two = 2\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'nested captured tree')
    await store.refresh(await execution())
    citation = (await store.search(scope, 'one', 'literal'))['results'][0]['citation']
    listing = await store.browse(scope, citation, 'files', limit=1000)
    assert listing['mode'] == 'files' and listing['nextOffset'] is None
    assert [row['path'] for row in listing['entries']] == [
        'src/nested/one.py',
        'src/nested/two.py',
        'same.py',
    ]
    assert all(
        row['citation']['handle'] == citation['handle'] for row in listing['entries']
    )
    assert all(
        row['citation']['sourceDigest'] == row['digest'] for row in listing['entries']
    )
    page = await store.browse(scope, citation, 'files', limit=1)
    next_page = await store.browse(
        scope, citation, 'files', limit=1, offset=page['nextOffset']
    )
    assert next_page['entries'][0]['path'] == 'src/nested/two.py'


async def test_directory_history_uses_only_captured_members_and_retained_commit(
    code_store,
):
    import json

    store, pool, scope, execution, repository = code_store
    folder = repository / 'src'
    folder.mkdir()
    (folder / 'visible.py').write_text('visible = 1\n')
    (folder / 'hidden.py').write_text('hidden = 1\n')
    git(repository, 'add', 'src')
    git(repository, 'commit', '-m', 'captured source files')
    (folder / 'hidden.py').write_text('hidden = 2\n')
    git(repository, 'add', 'src/hidden.py')
    git(repository, 'commit', '-m', 'excluded source change')
    await pool.execute(
        'UPDATE marie_scheduler.kb_source SET dataset_inputs=dataset_inputs || $1::jsonb',
        json.dumps({'include': ['same.py', 'src/visible.py']}),
    )
    await store.refresh(await execution())
    citation = (await store.search(scope, 'visible', 'literal'))['results'][0][
        'citation'
    ]
    (folder / 'visible.py').write_text('visible = 3\n')
    git(repository, 'add', 'src/visible.py')
    git(repository, 'commit', '-m', 'later than capture')
    history = await store.browse(
        scope, citation, 'directory_history', path='src', limit=1
    )
    assert history['mode'] == 'directory_history' and history['path'] == 'src'
    assert [row['summary'] for row in history['commits']] == ['captured source files']
    assert history['commits'][0]['citation'] is None
    root = await store.browse(scope, citation, 'directory_history', limit=1)
    assert root['nextOffset'] == 1
    next_page = await store.browse(
        scope, citation, 'directory_history', offset=1, limit=1
    )
    assert [row['summary'] for row in next_page['commits']] == ['initial']
    assert (await store.browse(scope, citation, 'directory_history', path='missing'))[
        'commits'
    ] == []
    with pytest.raises(ValueError):
        await store.browse(scope, citation, 'directory_history', path='../outside')
