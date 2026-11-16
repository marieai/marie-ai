import pytest

from marie.kb.code.browser import directory_entries, file_entries, git_history
from tests.unit.kb.code.test_capture import git, repo


def test_directory_lists_manifest_members_without_search_and_pages():
    files = {
        'src/a.py': {'sha256': 'a' * 64, 'size': 12},
        'src/nested/b.py': {'sha256': 'b' * 64, 'size': 20},
        'README.md': {'sha256': 'c' * 64, 'size': 30},
    }
    assert directory_entries(files, '', 0, 10) == {
        'entries': [
            {'name': 'src', 'path': 'src', 'kind': 'directory'},
            {
                'name': 'README.md',
                'path': 'README.md',
                'kind': 'file',
                'size': 30,
                'digest': 'c' * 64,
            },
        ],
        'total': 2,
        'nextOffset': None,
    }
    page = directory_entries(files, 'src', 0, 1)
    assert page['entries'][0]['path'] == 'src/nested'
    assert page['nextOffset'] == 1
    assert directory_entries(files, 'src', 1, 1)['entries'][0]['path'] == 'src/a.py'
    with pytest.raises(ValueError):
        directory_entries(files, '../src', 0, 10)


def test_git_history_is_anchored_at_the_capture_commit(repo):
    captured = git(repo, 'rev-parse', 'HEAD')
    (repo / 'same.py').write_text('changed = 2\n')
    git(repo, 'add', 'same.py')
    git(repo, 'commit', '-m', 'later change')
    rows = git_history(repo, captured, 'same.py', 0, 10)
    assert [row['summary'] for row in rows] == ['initial']
    assert rows[0]['commit'] == captured
    assert rows[0]['author'] == 'Fixture'


@pytest.mark.parametrize(
    'commit,path', [('--help', 'same.py'), ('a' * 40, '../same.py'), ('a' * 40, '.env')]
)
def test_git_metadata_rejects_untrusted_revision_and_sensitive_paths(
    repo, commit, path
):
    with pytest.raises(ValueError):
        git_history(repo, commit, path, 0, 10)


def test_file_pages_bound_serialized_size_and_continue_without_missing_paths():
    import json

    citation = {
        'kind': 'code',
        'path': 'src/start.py',
        'handle': 'a' * 64,
        'sourceDigest': 'b' * 64,
    }
    files = {
        ('folder/' + '界' * 200 + f'/{i}.py'): {'size': 1, 'sha256': 'b' * 64}
        for i in range(800)
    }
    rows = []
    offset = 0
    while True:
        page = file_entries(files, citation, offset, 1000)
        assert page['entries']
        assert (
            len(json.dumps(page['entries'], ensure_ascii=True).encode())
            < 1024 * 1024 + 3000
        )
        rows += [entry['path'] for entry in page['entries']]
        if page['nextOffset'] is None:
            break
        assert page['nextOffset'] > offset
        offset = page['nextOffset']
    assert len(rows) == len(set(rows)) == len(files)


def test_directory_history_only_reads_captured_members_at_the_saved_commit(repo):
    from marie.kb.code.browser import git_directory_history

    folder = repo / 'src'
    folder.mkdir()
    (folder / 'visible.py').write_text('visible = 1\n')
    (folder / 'hidden.py').write_text('hidden = 1\n')
    git(repo, 'add', 'src')
    git(repo, 'commit', '-m', 'add source files')
    (folder / 'hidden.py').write_text('hidden = 2\n')
    git(repo, 'add', 'src/hidden.py')
    git(repo, 'commit', '-m', 'excluded file only')
    captured = git(repo, 'rev-parse', 'HEAD')
    (folder / 'visible.py').write_text('visible = 2\n')
    git(repo, 'add', 'src/visible.py')
    git(repo, 'commit', '-m', 'after the capture')
    rows = git_directory_history(repo, captured, ['src/visible.py'], 0, 10)
    assert [row['summary'] for row in rows] == ['add source files']
    root = git_directory_history(repo, captured, ['same.py', 'src/visible.py'], 0, 10)
    assert [row['summary'] for row in root] == ['add source files', 'initial']
    assert (
        git_directory_history(repo, captured, ['same.py', 'src/visible.py'], 1, 1)
        == root[1:]
    )


def test_directory_history_treats_stdin_paths_literally(repo):
    from marie.kb.code.browser import git_directory_history

    paths = ['src/a[1].py', 'src/界 "quote".py', 'src/new\nline.py']
    (repo / 'src').mkdir()
    for path in paths:
        (repo / path).write_text('value = 1\n')
    git(repo, 'add', 'src')
    git(repo, 'commit', '-m', 'literal members')
    (repo / 'src/a1.py').write_text('different = 1\n')
    git(repo, 'add', 'src/a1.py')
    git(repo, 'commit', '-m', 'different member')
    captured = git(repo, 'rev-parse', 'HEAD')
    for path in paths:
        rows = git_directory_history(repo, captured, [path], 0, 10)
        assert [row['summary'] for row in rows] == ['literal members']
    with pytest.raises(ValueError):
        git_directory_history(repo, captured, ['../private.py'], 0, 10)


def test_directory_member_selection_stays_inside_the_captured_folder():
    from marie.kb.code.browser import directory_members

    files = {'src/a.py': {}, 'src/nested/b.py': {}, 'src-other/c.py': {}}
    assert directory_members(files, 'src') == ['src/a.py', 'src/nested/b.py']
    assert directory_members(files, '') == sorted(files)
    assert directory_members(files, 'absent') == []
    with pytest.raises(ValueError):
        directory_members(files, '../src')
