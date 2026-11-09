"""Capture eligible repository bytes without checking out or modifying a branch."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath


@dataclass
class RepositoryCapture:
    commit: str
    files: dict[str, dict[str, str | int]]
    digest: str
    omitted: dict[str, int]
    size: int


def resolve_repository_path(
    path: str, roots: list[str], mounts: dict[str, str] | None = None
) -> Path:
    """Resolve a registered Studio path inside explicitly mounted executor roots."""
    source = Path(path)
    for prefix, target in sorted(
        (mounts or {}).items(), key=lambda item: len(item[0]), reverse=True
    ):
        if source.is_relative_to(prefix):
            source = Path(target) / source.relative_to(prefix)
            break
    source = source.resolve(strict=True)
    if not any(source.is_relative_to(Path(root).resolve()) for root in roots):
        raise PermissionError('Repository is outside configured executor roots')
    return source


def git(repository: Path, *arguments: str, timeout: float = 120) -> bytes:
    """Run a bounded, noninteractive Git operation without a shell."""
    return subprocess.check_output(
        ['git', '-C', str(repository), *arguments],
        timeout=timeout,
        env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'},
        stderr=subprocess.PIPE,
    )


def branch_commit(
    repository: Path, branch: str, *, fetch: bool = False, timeout: float = 120
) -> str:
    """Resolve a configured branch once; fetch uses the registered origin credentials."""
    if branch.startswith('-') or not branch or '\x00' in branch:
        raise ValueError('Invalid repository branch')
    git(repository, 'check-ref-format', f'refs/heads/{branch}', timeout=timeout)
    reference = f'refs/heads/{branch}'
    if fetch:
        reference = f'refs/marie-kb/{hashlib.sha256(branch.encode()).hexdigest()}'
        git(
            repository,
            'fetch',
            '--no-tags',
            '--no-write-fetch-head',
            'origin',
            f'+refs/heads/{branch}:{reference}',
            timeout=timeout,
        )
    return (
        git(
            repository,
            'rev-parse',
            '--verify',
            f'{reference}^{{commit}}',
            timeout=timeout,
        )
        .decode()
        .strip()
    )


def eligible_path(path: str, include: list[str], exclude: list[str]) -> str | None:
    parts = PurePosixPath(path).parts
    if not parts or path.startswith('/') or any(part in ('.', '..') for part in parts):
        raise ValueError('Unsafe repository member path')
    if any(
        part in ('secrets', '.git')
        or part.startswith('.env')
        or part.endswith(('.pem', '.key'))
        for part in parts
    ):
        return 'sensitive'
    if any(part in ('.hg', '.svn', '.sourcegraph') for part in parts):
        return 'indexer_scope'
    if not any(fnmatchcase(path, pattern) for pattern in include) or any(
        fnmatchcase(path, pattern) for pattern in exclude
    ):
        return 'scope'
    return None


def capture_repository(
    repository: Path,
    branch: str,
    destination: Path,
    *,
    dirty: bool = False,
    fetch: bool = False,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    max_bytes: int = 128 * 1024 * 1024,
    max_files: int = 100_000,
    file_limit: int = 1024 * 1024,
) -> RepositoryCapture:
    """Save a complete eligible tree or fail; omissions are explicitly counted."""
    commit = branch_commit(repository, branch, fetch=fetch)
    if dirty and (
        fetch
        or git(repository, 'symbolic-ref', '--short', 'HEAD').decode().strip() != branch
    ):
        raise ValueError('Dirty capture requires the checked-out branch without fetch')
    include = include or ['*']
    exclude = exclude or []
    omitted: Counter[str] = Counter()
    entries = git(repository, 'ls-tree', '-rz', '--full-tree', commit).split(b'\x00')
    members: dict[str, tuple[str, str]] = {}
    for entry in filter(None, entries):
        metadata, raw_path = entry.split(b'\t', 1)
        mode, kind, object_id = metadata.decode().split()
        path = raw_path.decode('utf-8')
        members[path] = (mode, object_id)
    if dirty:
        submodules = {path for path, value in members.items() if value[0] == '160000'}
        paths = git(repository, 'ls-files', '-co', '--exclude-standard', '-z')
        members = {
            path.decode('utf-8'): (
                '160000' if path.decode('utf-8') in submodules else 'worktree',
                '',
            )
            for path in paths.split(b'\x00')
            if path
        }
    if len(members) > max_files:
        raise ValueError('Repository exceeds file budget')
    destination.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, str | int]] = {}
    total = 0
    with subprocess.Popen(
        ['git', '-C', str(repository), 'cat-file', '--batch'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as blobs:
        assert blobs.stdin is not None and blobs.stdout is not None
        for path, (mode, object_id) in sorted(members.items()):
            reason = eligible_path(path, include, exclude)
            if reason:
                omitted[reason] += 1
                continue
            if (
                mode == '120000'
                or mode == 'worktree'
                and (repository / path).is_symlink()
            ):
                omitted['symlink'] += 1
                continue
            if mode == '160000':
                omitted['submodule'] += 1
                continue
            if mode == 'worktree':
                original = repository / path
                if not original.exists():
                    continue
                if any(
                    parent.is_symlink()
                    for parent in original.parents
                    if parent != repository.parent
                ):
                    raise ValueError('Worktree member traverses a symlink')
                before = original.stat()
                if before.st_size > file_limit:
                    omitted['large'] += 1
                    continue
                content = original.read_bytes()
                after = original.stat()
                if (
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError('Worktree changed during capture; retry')
            else:
                blobs.stdin.write(f'{object_id}\n'.encode())
                blobs.stdin.flush()
                header = blobs.stdout.readline().split()
                if len(header) != 3 or header[1] != b'blob':
                    raise ValueError('Incomplete repository blob scan')
                size = int(header[2])
                if size > file_limit:
                    # Drain without keeping the excluded blob in memory.
                    remaining = size + 1
                    while remaining:
                        block = blobs.stdout.read(min(remaining, 65536))
                        if not block:
                            raise ValueError('Incomplete repository blob')
                        remaining -= len(block)
                    omitted['large'] += 1
                    continue
                content = blobs.stdout.read(size)
                if len(content) != size or blobs.stdout.read(1) != b'\n':
                    raise ValueError('Incomplete repository blob')
            try:
                content.decode('utf-8')
            except UnicodeDecodeError:
                omitted['binary'] += 1
                continue
            if b'\x00' in content:
                omitted['binary'] += 1
                continue
            total += len(content)
            if total > max_bytes:
                raise ValueError('Repository exceeds byte budget')
            output = destination / path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
            files[path] = {
                'sha256': hashlib.sha256(content).hexdigest(),
                'size': len(content),
            }
        blobs.stdin.close()
        if blobs.wait(timeout=10):
            raise ValueError('Repository blob scan failed')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return RepositoryCapture(commit, files, digest, dict(omitted), total)
