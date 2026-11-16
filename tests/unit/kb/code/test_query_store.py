import os

import pytest

from tests.unit.kb.code.test_capture import git, repo
from tests.unit.kb.code.test_query import atom
from tests.unit.kb.code.test_store import code_store
from tests.unit.kb.code.test_zoekt import zoekt

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable PostgreSQL'
)


async def test_filters_case_and_boolean_query_reuse_authorized_capture(code_store):
    store, _, scope, execution, repository = code_store
    (repository / 'target.py').write_text(
        'def target():\n    return "Alpha timeout"\n\ntarget()\n'
    )
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'query fixture')
    await store.refresh(await execution())
    branches = [
        [
            atom('repo', 'fixture'),
            atom('context', 'Engineering'),
            atom('file', r'target\.py$', regex=True),
            atom('lang', 'Python'),
            atom('content', 'alpha'),
        ]
    ]
    assert not (await store.search(scope, 'query', 'query', branches=branches))[
        'results'
    ]
    result = await store.search(
        scope, 'query', 'query', branches=branches, case_sensitive=False
    )
    assert [row['path'] for row in result['results']] == ['target.py']
    assert result['results'][0]['line'] == 2
    assert result['results'][0]['language'] == 'Python'
    assert (await store.open(scope, result['results'][0]['citation']))[
        'outcome'
    ] == 'ready'
    assert not (
        await store.search(
            scope,
            'query',
            'query',
            branches=[[atom('repo', 'other'), atom('content', 'alpha')]],
            case_sensitive=False,
        )
    )['results']
    union = await store.search(
        scope,
        'query',
        'query',
        branches=[[atom('content', 'alpha')], [atom('content', 'timeout')]],
    )
    assert {row['path'] for row in union['results']} == {'same.py', 'target.py'}


async def test_symbol_filter_uses_actual_definitions_and_reports_parser_gaps(
    code_store,
):
    store, _, scope, execution, repository = code_store
    (repository / 'target.py').write_text('def target():\n    return 1\n\ntarget()\n')
    (repository / 'notes.txt').write_text('target documentation\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'symbol query fixture')
    await store.refresh(await execution())
    result = await store.search(
        scope, 'sym:target', 'query', branches=[[atom('sym', 'target')]]
    )
    assert [(row['path'], row['line']) for row in result['results']] == [
        ('target.py', 1)
    ]
    assert result['warnings']
    assert (await store.open(scope, result['results'][0]['citation']))[
        'passage'
    ].startswith('def target')
    negative = await store.search(
        scope,
        'negative symbol',
        'query',
        branches=[[atom('sym', 'target', negated=True), atom('content', 'target')]],
    )
    assert not negative['results']
    union = await store.search(
        scope,
        'symbol or text',
        'query',
        branches=[[atom('sym', 'target')], [atom('content', 'documentation')]],
    )
    assert {row['path'] for row in union['results']} == {'target.py', 'notes.txt'}


async def test_revision_filter_opens_retained_capture_not_checkout_bytes(code_store):
    store, _, scope, execution, repository = code_store
    store.retained_captures = 5
    first = await store.refresh(await execution())
    (repository / 'same.py').write_text('replacement = 2\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'new query capture')
    current = await store.refresh(await execution())
    old = await store.search(
        scope,
        'old revision',
        'query',
        branches=[[atom('rev', first['commit'][:8]), atom('content', 'alpha')]],
    )
    assert old['results'][0]['citation']['handle'] == first['handle']
    assert (
        'alpha' in (await store.open(scope, old['results'][0]['citation']))['passage']
    )
    for revision in ['HEAD', 'main']:
        result = await store.search(
            scope, 'current revision', 'query', branches=[[atom('rev', revision)]]
        )
        assert {row['citation']['handle'] for row in result['results']} == {
            current['handle']
        }


async def test_query_atoms_cannot_inject_a_different_repository_scope(code_store):
    store, _, scope, execution, _ = code_store
    await store.refresh(await execution())
    for branch in [[atom('content', '") OR repo:.*')], [atom('repo', '.*')]]:
        assert not (await store.search(scope, 'injection', 'query', branches=[branch]))[
            'results'
        ]


@pytest.mark.parametrize('legacy', [False, True])
async def test_symbol_quote_preserves_lf_lines_with_unicode_separators(
    code_store, legacy
):
    store, pool, scope, execution, repository = code_store
    (repository / 'target.py').write_text(
        '# note\u2028still first line\n\ndef target():\n    return 1\n'
    )
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'symbol line identity')
    await store.refresh(await execution())
    if legacy:
        await pool.execute(
            """UPDATE marie_scheduler.kb_code_navigation
        SET symbols=(SELECT jsonb_agg((value-'span') || jsonb_build_object('line',value->'span'->'start_line'))
            FROM jsonb_array_elements(symbols)) WHERE path='target.py'"""
        )
    result = await store.search(
        scope, 'sym:target', 'query', branches=[[atom('sym', 'target')]]
    )
    row = result['results'][0]
    assert row['line'] == 3
    assert row['preview'] == 'def target():\n'
    assert (await store.open(scope, row['citation']))['passage'].startswith(
        'def target():'
    )


async def test_negative_symbols_use_complete_file_membership_beyond_definition_budget(
    code_store,
):
    store, _, scope, execution, repository = code_store
    for number in range(9):
        (repository / f'a{number}.py').write_text('def target():\n    return 1\n' * 60)
    (repository / 'z.py').write_text('def target():\n    return "needle"\n')
    (repository / 'safe.py').write_text('def unrelated():\n    return "needle"\n')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'large symbol membership')
    await store.refresh(await execution())
    result = await store.search(
        scope,
        '-sym:target needle',
        'query',
        branches=[[atom('sym', 'TARGET', negated=True), atom('content', 'needle')]],
        case_sensitive=False,
    )
    assert {row['path'] for row in result['results']} == {'safe.py'}


async def test_quoted_content_regex_native_dispatch_with_file_filter(code_store):
    store, _, scope, execution, repository = code_store
    (repository / 'target.py').write_text(
        'timeout_17 = 1\nretry_2 = 2\nunmatched = 0\n'
    )
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'quoted regex fixture')
    await store.refresh(await execution())
    result = await store.search(
        scope,
        'file:target.py content:"(timeout|retry)_[0-9]+"',
        'query',
        branches=[
            [
                atom('file', r'target\.py$', regex=True),
                atom('content', '(timeout|retry)_[0-9]+', regex=True),
            ]
        ],
    )
    assert {row['line'] for row in result['results']} == {1, 2}
