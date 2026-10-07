"""Path containment helpers for candidate packages."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


class CandidatePathError(ValueError):
    """Raised when a candidate artifact path is unsafe."""


def safe_artifact_path(root: Path, value: str) -> Path:
    """Resolve a declared POSIX artifact path inside a candidate root.

    Args:
        root: Candidate package root.
        value: Manifest-declared relative artifact path.

    Returns:
        Resolved artifact path below ``root``.

    Raises:
        CandidatePathError: If the path is absolute, traverses, or is malformed.
    """
    if not value or '\\' in value or '\x00' in value:
        raise CandidatePathError(f'Unsafe candidate artifact path: {value!r}')
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {'', '.', '..'} for part in relative.parts
    ):
        raise CandidatePathError(f'Unsafe candidate artifact path: {value!r}')

    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*relative.parts)).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise CandidatePathError(f'Candidate artifact escapes package root: {value!r}')
    return resolved


def package_files(root: Path) -> set[str]:
    """Return regular package files while rejecting all symlinks.

    Args:
        root: Candidate package root.

    Returns:
        POSIX paths relative to ``root``.

    Raises:
        CandidatePathError: If the root or any entry is a symlink or special file.
    """
    if root.is_symlink() or not root.is_dir():
        raise CandidatePathError(f'Candidate root must be a regular directory: {root}')

    files: set[str] = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise CandidatePathError(f'Candidate package contains a symlink: {path}')
        if path.is_dir():
            continue
        if not path.is_file():
            raise CandidatePathError(
                f'Candidate package contains a special file: {path}'
            )
        files.add(path.relative_to(root).as_posix())
    return files
