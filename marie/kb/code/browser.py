"""Captured directory listings and read-only Git metadata."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from marie.kb.code.capture import eligible_path, git


def _directory_prefix(path: str) -> str:
    if path and (
        path.startswith('/') or any(part in ('', '.', '..') for part in path.split('/'))
    ):
        raise ValueError('Invalid captured directory')
    return f'{path}/' if path else ''


def directory_members(files: dict[str, Any], path: str) -> list[str]:
    """Select only captured members beneath one directory."""
    prefix = _directory_prefix(path)
    return sorted(member for member in files if member.startswith(prefix))


def directory_entries(
    files: dict[str, Any], path: str, offset: int, limit: int
) -> dict[str, Any]:
    """List one directory of a captured manifest, directories first."""
    prefix = _directory_prefix(path)
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


def file_entries(
    files: dict[str, Any], citation: dict[str, Any], offset: int, limit: int
) -> dict[str, Any]:
    """Page captured files, prioritizing the opened directory and bounding response size."""
    folder = citation['path'].rpartition('/')[0]
    paths = sorted(
        files, key=lambda member: (member.rpartition('/')[0] != folder, member)
    )
    entries = []
    size = 0
    for member in paths[offset : offset + limit]:
        info = files[member]
        entry = {
            'kind': 'file',
            'name': member.rsplit('/', 1)[-1],
            'path': member,
            'size': info['size'],
            'digest': info['sha256'],
            'citation': {
                **citation,
                'path': member,
                'sourceDigest': info['sha256'],
                'startLine': 1,
                'endLine': 1,
            },
        }
        encoded = len(json.dumps(entry, ensure_ascii=True).encode('utf-8'))
        if entries and size + encoded > 1024 * 1024:
            break
        entries.append(entry)
        size += encoded
    end = offset + len(entries)
    return {
        'entries': entries,
        'total': len(paths),
        'nextOffset': end if end < len(paths) else None,
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
    )
    return _history_rows(data)


def git_directory_history(
    repository: Path, commit: str, paths: list[str], offset: int, limit: int
) -> list[dict[str, Any]]:
    """Read history for captured members without expanding to excluded paths."""
    if not paths:
        return []
    stdin_paths = []
    multiline_paths = []
    for path in paths:
        _validate(commit, path, offset, limit)
        # Git's stdin path list is line-delimited; other paths stay literal argv entries.
        (multiline_paths if '\n' in path or '\r' in path else stdin_paths).append(path)
    data = git(
        repository,
        '--literal-pathspecs',
        'log',
        '--no-show-signature',
        f'--skip={offset}',
        f'--max-count={limit}',
        '--format=%H%x00%an%x00%at%x00%s%x00',
        '--stdin',
        commit,
        '--',
        *multiline_paths,
        input_data=('\n'.join(['--', *stdin_paths]) + '\n').encode('utf-8'),
        timeout=10,
    )
    return _history_rows(data)


def _history_rows(data: bytes) -> list[dict[str, Any]]:
    data = data.decode('utf-8', errors='replace')
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
