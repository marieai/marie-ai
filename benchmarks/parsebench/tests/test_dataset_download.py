from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.hf_api import RepoFile


def file_info(path: str, content: bytes, *, lfs: bool = False) -> RepoFile:
    info = {
        'path': path,
        'size': len(content),
        'oid': hashlib.sha1(
            b'blob ' + str(len(content)).encode() + b'\0' + content
        ).hexdigest(),
    }
    if lfs:
        info['lfs'] = {
            'size': len(content),
            'oid': hashlib.sha256(content).hexdigest(),
            'pointerSize': 130,
        }
    return RepoFile(**info)


@pytest.mark.parametrize('lfs', [False, True])
def test_verifies_remote_identity_and_rejects_same_size_corruption(
    tmp_path: Path, lfs: bool
) -> None:
    from marie_parsebench.download import verify_file

    path = tmp_path / 'document.pdf'
    path.write_bytes(b'original')
    info = file_info(path.name, b'original', lfs=lfs)
    assert verify_file(tmp_path, info)
    path.write_bytes(b'corrupt!')
    with pytest.raises(ValueError, match='digest'):
        verify_file(tmp_path, info)


def test_missing_file_can_be_downloaded_but_wrong_size_cannot(tmp_path: Path) -> None:
    from marie_parsebench.download import verify_file

    info = file_info('document.pdf', b'original')
    assert not verify_file(tmp_path, info)
    (tmp_path / info.path).write_bytes(b'partial')
    with pytest.raises(ValueError, match='size'):
        verify_file(tmp_path, info)


@pytest.mark.parametrize(
    'path', ['../escape.pdf', '/tmp/escape.pdf', 'docs/../escape.pdf']
)
def test_rejects_unsafe_remote_paths(tmp_path: Path, path: str) -> None:
    from marie_parsebench.download import verify_file

    with pytest.raises(ValueError, match='path'):
        verify_file(tmp_path, file_info(path, b'original'))


def test_rejects_symlink_escape_before_reading_file(tmp_path: Path) -> None:
    from marie_parsebench.download import verify_file

    root = tmp_path / 'dataset'
    root.mkdir()
    (root / 'docs').symlink_to(tmp_path / 'outside', target_is_directory=True)
    with pytest.raises(ValueError, match='path'):
        verify_file(root, file_info('docs/document.pdf', b'original'))


def test_revision_mismatch_fails_without_creating_dataset(tmp_path: Path) -> None:
    from marie_parsebench.download import download_dataset

    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'revision': 'a' * 40, 'groups': []}))
    root = tmp_path / 'dataset'
    with pytest.raises(ValueError, match='revision'):
        download_dataset(root, inventory, 'b' * 40)
    assert not root.exists()


@pytest.mark.parametrize('corrupt_download', [False, True])
def test_download_verifies_bytes_after_transport_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt_download: bool
) -> None:
    from marie_parsebench import download

    revision = 'a' * 40
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(
        json.dumps({'revision': revision, 'groups': [{'group': 'table'}]})
    )
    root = tmp_path / 'dataset'
    entry = file_info('table.jsonl', b'original')

    class Hub:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def list_repo_tree(self, repo: str, **kwargs: Any) -> list[RepoFile]:
            if repo != 'llamaindex/ParseBench' or kwargs['revision'] != revision:
                raise ValueError('Wrong remote identity')
            return [entry]

    def fetch(repo: str, **kwargs: Any) -> None:
        if kwargs['revision'] != revision or kwargs['allow_patterns'] != [
            'table.jsonl'
        ]:
            raise ValueError('Wrong download selection')
        kwargs['local_dir'].mkdir(parents=True)
        (kwargs['local_dir'] / 'table.jsonl').write_bytes(
            b'corrupt!' if corrupt_download else b'original'
        )

    monkeypatch.setattr(download, 'HfApi', Hub)
    monkeypatch.setattr(download, 'snapshot_download', fetch)
    if corrupt_download:
        with pytest.raises(ValueError, match='digest'):
            download.download_dataset(root, inventory, revision)
    else:
        report = download.download_dataset(root, inventory, revision)
        assert report['files_verified'] == 1
        assert report['bytes_verified'] == 8
        assert (root / 'table.jsonl').read_bytes() == b'original'


def test_corrupt_existing_file_is_not_repaired_by_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marie_parsebench import download

    revision = 'a' * 40
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(
        json.dumps({'revision': revision, 'groups': [{'group': 'table'}]})
    )
    root = tmp_path / 'dataset'
    root.mkdir()
    (root / 'table.jsonl').write_bytes(b'corrupt!')

    class Hub:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def list_repo_tree(self, repo: str, **kwargs: Any) -> list[RepoFile]:
            return [file_info('table.jsonl', b'original')]

    def fetch(repo: str, **kwargs: Any) -> None:
        (root / 'table.jsonl').write_bytes(b'original')

    monkeypatch.setattr(download, 'HfApi', Hub)
    monkeypatch.setattr(download, 'snapshot_download', fetch)
    with pytest.raises(ValueError, match='digest'):
        download.download_dataset(root, inventory, revision)
    assert (root / 'table.jsonl').read_bytes() == b'corrupt!'
