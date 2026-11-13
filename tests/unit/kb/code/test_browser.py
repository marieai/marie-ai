import pytest

from marie.kb.code.browser import directory_entries, git_history
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
