from __future__ import annotations

import hashlib
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml
from pydantic import ValidationError

from marie.extension.errors import PackageLoadError, PackageValidationError
from marie.extension.manifest import ExtensionPackage

MANIFEST_NAME = "marie-extension.yaml"
BLOCKED_NAMES = {".env"}
BLOCKED_SUFFIXES = {".pem", ".key"}


@dataclass(frozen=True)
class LoadedPackage:
    manifest: ExtensionPackage
    manifest_path: str
    digest: str
    files: tuple[str, ...]


def load_package(path: str | Path) -> LoadedPackage:
    source = Path(path)
    if source.is_dir():
        return _load_directory(source)
    if source.is_file() and zipfile.is_zipfile(source):
        return _load_zip(source)
    raise PackageLoadError(f"not a Marie extension directory or ZIP archive: {source}")


def _load_directory(root: Path) -> LoadedPackage:
    files: dict[str, bytes] = {}
    for file_path in sorted(
        p for p in root.rglob("*") if p.is_file() or p.is_symlink()
    ):
        rel = _relative_path(file_path, root)
        _reject_blocked_file(rel)
        if file_path.is_symlink():
            target = file_path.resolve()
            if not target.is_relative_to(root.resolve()):
                raise PackageLoadError(f"symlink escapes package: {rel}")
            if target.is_dir():
                continue
        files[rel] = file_path.read_bytes()
    return _load_files(files)


def _load_zip(path: Path) -> LoadedPackage:
    files: dict[str, bytes] = {}
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            rel = _normalize_archive_name(info.filename)
            if rel in seen:
                raise PackageLoadError(f"duplicate archive path: {rel}")
            seen.add(rel)
            _reject_zip_mode(info, rel)
            _reject_blocked_file(rel)
            files[rel] = archive.read(info)
    return _load_files(files)


def _load_files(files: dict[str, bytes]) -> LoadedPackage:
    manifests = [path for path in files if PurePosixPath(path).name == MANIFEST_NAME]
    if not manifests:
        raise PackageLoadError(f"missing {MANIFEST_NAME}")
    if len(manifests) > 1:
        raise PackageLoadError(f"multiple {MANIFEST_NAME} files found")
    manifest_path = manifests[0]
    data = yaml.safe_load(files[manifest_path]) or {}
    try:
        manifest = ExtensionPackage.model_validate(data)
    except ValidationError as exc:
        raise PackageValidationError(str(exc)) from exc
    package_root = str(PurePosixPath(manifest_path).parent)
    if package_root == ".":
        package_root = ""
    available = set(files)
    for ref in _declared_paths(manifest):
        resolved = _resolve_declared_path(ref, package_root)
        if resolved not in available:
            raise PackageLoadError(f"declared file does not exist: {ref}")
    return LoadedPackage(
        manifest=manifest,
        manifest_path=manifest_path,
        digest=_digest(files),
        files=tuple(sorted(files)),
    )


def _relative_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return _normalize_archive_name(rel)


def _normalize_archive_name(name: str) -> str:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise PackageLoadError(f"unsafe package path: {name}")
    normalized = pure.as_posix()
    if normalized in {"", "."}:
        raise PackageLoadError(f"unsafe package path: {name}")
    return normalized


def _reject_zip_mode(info: zipfile.ZipInfo, rel: str) -> None:
    mode = (info.external_attr >> 16) & 0o777777
    if stat.S_ISLNK(mode):
        raise PackageLoadError(f"archive symlink is not allowed: {rel}")
    if mode and (mode & (stat.S_ISUID | stat.S_ISGID | stat.S_IWOTH)):
        raise PackageLoadError(f"unsafe archive mode for {rel}: {oct(mode)}")


def _reject_blocked_file(rel: str) -> None:
    path = PurePosixPath(rel)
    if path.name in BLOCKED_NAMES or path.suffix in BLOCKED_SUFFIXES:
        raise PackageLoadError(f"blocked secret-like file in package: {rel}")


def _resolve_declared_path(ref: str, package_root: str) -> str:
    rel = _normalize_archive_name(ref)
    if package_root:
        rel = f"{package_root}/{rel}"
    return _normalize_archive_name(rel)


def _declared_paths(manifest: ExtensionPackage) -> Iterable[str]:
    metadata = manifest.metadata
    yield from _optional_paths(metadata.icon, metadata.readme, metadata.privacy)
    for provider in manifest.providers:
        yield from _optional_paths(provider.icon)
        if provider.implementation:
            yield from _implementation_paths(provider.implementation)
        for tool in provider.tools:
            if tool.implementation:
                yield from _implementation_paths(tool.implementation)
    for tool in manifest.tools:
        if tool.implementation:
            yield from _implementation_paths(tool.implementation)
    for datasource in manifest.datasources:
        if datasource.implementation:
            yield from _implementation_paths(datasource.implementation)
    for trigger in manifest.triggers:
        if trigger.implementation:
            yield from _implementation_paths(trigger.implementation)
    for endpoint in manifest.endpoints:
        if endpoint.implementation:
            yield from _implementation_paths(endpoint.implementation)


def _implementation_paths(ref) -> Iterable[str]:
    yield from _optional_paths(ref.source, ref.provider_source, *ref.model_sources)


def _optional_paths(*values: str | None) -> Iterable[str]:
    for value in values:
        if value:
            yield value


def _digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[path])
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"
