import subprocess
from pathlib import Path

import pytest

from marie.kb.code.capture import capture_repository, resolve_repository_path


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / 'repo'
    path.mkdir()
    git(path, 'init', '-b', 'main')
    git(path, 'config', 'user.name', 'Fixture')
    git(path, 'config', 'user.email', 'fixture@example.invalid')
    (path / 'same.py').write_text('alpha = 1\nsecond = "é"\n')
    (path / 'private.key').write_text('excluded fixture')
    (path / '.env').write_text('excluded fixture')
    git(path, 'add', '.')
    git(path, 'commit', '-m', 'initial')
    return path


def test_capture_uses_fixed_git_bytes_and_excludes_sensitive_names(repo, tmp_path):
    commit = git(repo, 'rev-parse', 'HEAD')
    (repo / 'same.py').write_text('uncommitted = 2\n')
    capture = capture_repository(repo, 'main', tmp_path / 'capture')
    assert capture.commit == commit
    assert list(capture.files) == ['same.py']
    assert (tmp_path / 'capture/same.py').read_text() == 'alpha = 1\nsecond = "é"\n'
    assert capture.omitted['sensitive'] == 2


def test_dirty_capture_digest_changes_at_same_commit(repo, tmp_path):
    first = capture_repository(repo, 'main', tmp_path / 'one', dirty=True)
    (repo / 'same.py').write_text('dirty = 5\n')
    second = capture_repository(repo, 'main', tmp_path / 'two', dirty=True)
    assert first.commit == second.commit
    assert first.digest != second.digest
    assert (tmp_path / 'two/same.py').read_text() == 'dirty = 5\n'


def test_capture_resolves_branches_renames_deletions_and_path_scope(repo, tmp_path):
    git(repo, 'checkout', '-b', 'other')
    git(repo, 'mv', 'same.py', 'new.py')
    git(repo, 'commit', '-m', 'rename')
    main = capture_repository(repo, 'main', tmp_path / 'one')
    other = capture_repository(repo, 'other', tmp_path / 'two')
    assert set(main.files) == {'same.py'}
    assert set(other.files) == {'new.py'}
    excluded = capture_repository(repo, 'other', tmp_path / 'three', exclude=['*.py'])
    assert not excluded.files
    with pytest.raises(ValueError, match='checked-out branch'):
        capture_repository(repo, 'main', tmp_path / 'four', dirty=True)


def test_incomplete_or_oversized_tree_is_failure_not_empty_capture(repo, tmp_path):
    with pytest.raises(ValueError, match='byte budget'):
        capture_repository(repo, 'main', tmp_path / 'one', max_bytes=1)
    with pytest.raises(ValueError, match='branch'):
        capture_repository(repo, '--upload-pack=anything', tmp_path / 'two')
    with pytest.raises(subprocess.CalledProcessError):
        capture_repository(repo, 'missing', tmp_path / 'three')


def test_symlinks_binary_and_unsupported_files_are_disclosed(repo, tmp_path):
    (repo / 'link.py').symlink_to('/etc/passwd')
    (repo / 'bytes.bin').write_bytes(b'\x00binary')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'unsupported')
    capture = capture_repository(repo, 'main', tmp_path / 'one')
    assert set(capture.files) == {'same.py'}
    assert capture.omitted['symlink'] == 1
    assert capture.omitted['binary'] == 1


def test_only_registered_paths_under_configured_mounts_are_read(repo, tmp_path):
    assert resolve_repository_path(str(repo), [str(tmp_path)]) == repo
    with pytest.raises(PermissionError):
        resolve_repository_path(str(repo), ['/unrelated'])
    assert (
        resolve_repository_path(
            '/studio/repos/repo', [str(tmp_path)], {'/studio/repos': str(tmp_path)}
        )
        == repo
    )


def test_indexer_control_directories_are_disclosed_omissions(repo, tmp_path):
    for name in ['.sourcegraph/ignore', 'vendor/.svn/entries', 'vendor/.hg/hgrc']:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('*\n')
    git(repo, 'add', '-f', '.')
    git(repo, 'commit', '-m', 'indexer controls')
    result = capture_repository(repo, 'main', tmp_path / 'captured')
    assert set(result.files) == {'same.py'}
    assert result.omitted['indexer_scope'] == 3
