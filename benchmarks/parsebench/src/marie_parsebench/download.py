from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFile

DATASET_REPO = 'llamaindex/ParseBench'


def verify_file(dataset_dir: Path, entry: RepoFile) -> bool:
    """Verify existing bytes against a pinned Hub file identity.

    Return False for missing files; reject changed files and escaping paths.
    """
    relative = PurePosixPath(entry.path)
    root = dataset_dir.resolve()
    path = root / relative
    if (
        relative.is_absolute()
        or '..' in relative.parts
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError(f'Unsafe dataset path: {entry.path}')
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != entry.size:
        raise ValueError(f'Dataset file size mismatch: {entry.path}')

    if entry.lfs:
        digest = hashlib.sha256()
        expected = entry.lfs.sha256
    else:
        digest = hashlib.sha1()
        digest.update(f'blob {entry.size}\0'.encode())
        expected = entry.blob_id
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'Dataset file digest mismatch: {entry.path}')
    return True


def download_dataset(
    dataset_dir: Path, inventory_path: Path, revision: str
) -> dict[str, str | int]:
    """Fetch missing files at the audited revision and verify remote identities.

    Existing corrupt files fail before download; this command never repairs them.
    """
    inventory = json.loads(inventory_path.read_text())
    if not re.fullmatch(r'[0-9a-f]{40}', revision) or revision != inventory['revision']:
        raise ValueError('Dataset revision must match the audited inventory')

    required = {f'{group["group"]}.jsonl' for group in inventory['groups']}
    files = [
        entry
        for entry in HfApi(token=False).list_repo_tree(
            DATASET_REPO, repo_type='dataset', revision=revision, recursive=True
        )
        if isinstance(entry, RepoFile)
        and (
            entry.path in required | {'eval.yaml'}
            or (
                entry.path.startswith('docs/')
                and Path(entry.path).suffix.lower() in {'.pdf', '.png', '.jpg', '.jpeg'}
            )
        )
    ]
    if not required.issubset({entry.path for entry in files}):
        raise ValueError('Pinned dataset is missing required annotation files')
    missing = [entry.path for entry in files if not verify_file(dataset_dir, entry)]
    if missing:
        snapshot_download(
            DATASET_REPO,
            repo_type='dataset',
            revision=revision,
            local_dir=dataset_dir,
            allow_patterns=missing,
            token=False,
            max_workers=8,
        )
    for entry in files:
        if not verify_file(dataset_dir, entry):
            raise FileNotFoundError(dataset_dir / entry.path)
    return {
        'repo': DATASET_REPO,
        'revision': revision,
        'files_verified': len(files),
        'bytes_verified': sum(entry.size for entry in files),
        'verification': 'huggingface_git_blob_sha1_or_lfs_sha256',
    }
