"""Captured directory listings and read-only Git metadata."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

from marie.kb.code.capture import eligible_path, git


def directory_entries(
    files: dict[str, Any], path: str, offset: int, limit: int
) -> dict[str, Any]:
    """List one directory of a captured manifest, directories first."""
    if path and (
        path.startswith('/') or any(part in ('', '.', '..') for part in path.split('/'))
    ):
        raise ValueError('Invalid captured directory')
    prefix = f'{path}/' if path else ''
    entries = {}
    for member, info in files.items():
        if not member.startswith(prefix):
            continue
        relative = member[len(prefix) :]
        name, separator, _ = relative.partition('/')
        entry = {'name': name, 'path': prefix + name}
        entries[name] = (
            {**entry, 'kind': 'directory'}
            if separator
            else {
                **entry,
                'kind': 'file',
                'size': info['size'],
                'digest': info['sha256'],
            }
        )
    ordered = sorted(
        entries.values(), key=lambda item: (item['kind'] != 'directory', item['name'])
    )
    end = min(offset + limit, len(ordered))
    return {
        'entries': ordered[offset:end],
        'total': len(ordered),
        'nextOffset': end if end < len(ordered) else None,
    }


def _validate(commit: str, path: str, offset: int, limit: int) -> None:
    if (
        not re.fullmatch('[a-f0-9]{40,64}', commit)
        or not path
        or path.startswith('-')
        or any(part in ('', '.', '..') for part in path.split('/'))
        or eligible_path(path, ['*'], []) is not None
        or not 0 <= offset <= 100000
        or not 1 <= limit <= 201
    ):
        raise ValueError('Invalid Git metadata request')


def git_history(
    repository: Path, commit: str, path: str, offset: int, limit: int
) -> list[dict[str, Any]]:
    """Return file commits reachable from the captured commit, without fetching."""
    _validate(commit, path, offset, limit)
    data = git(
        repository,
        '--literal-pathspecs',
        'log',
        '--no-show-signature',
        f'--skip={offset}',
        f'--max-count={limit}',
        '--format=%H%x00%an%x00%at%x00%s%x00',
        commit,
        '--',
        path,
        timeout=10,
    ).decode('utf-8', errors='replace')
    fields = data.split('\x00')
    rows = []
    for i in range(0, len(fields) - 3, 4):
        rows.append(
            {
                'commit': fields[i].strip(),
                'author': fields[i + 1][:200],
                'timestamp': int(fields[i + 2]),
                'summary': fields[i + 3][:500],
            }
        )
    return rows


def git_blob_digest(repository: Path, commit: str, path: str) -> str | None:
    """Hash the committed blob only when it fits the captured-file limit."""
    _validate(commit, path, 0, 1)
    try:
        object_name = f'{commit}:{path}'
        size = int(git(repository, 'cat-file', '-s', object_name, timeout=10))
        if size > 2 * 1024 * 1024:
            return None
        return hashlib.sha256(
            git(repository, 'cat-file', 'blob', object_name, timeout=10)
        ).hexdigest()
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
