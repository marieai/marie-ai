from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image
from pypdf import PdfReader

DATASET_REVISION = '2805a1d940f95a203e0ae4b88be9934f7765b3fc'
GROUPS = ('table', 'chart', 'text_content', 'text_formatting', 'layout')
GROUP_DIRECTORIES = {
    'table': 'table',
    'chart': 'chart',
    'text_content': 'text',
    'text_formatting': 'text',
    'layout': 'layout',
}
MEDIA_TYPES = {
    '.pdf': 'application/pdf',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return value


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def _resolve_source(dataset_dir: Path, relative: str, group: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Source path escapes dataset root: {relative}')
    expected_prefix = ('docs', GROUP_DIRECTORIES[group])
    if path.parts[:2] != expected_prefix:
        raise ValueError(f'Source path does not match {group} corpus: {relative}')

    source = dataset_dir.joinpath(*path.parts)
    resolved = source.resolve(strict=False)
    if not resolved.is_relative_to(dataset_dir) or source.is_symlink():
        raise ValueError(f'Source path escapes dataset root: {relative}')
    if not source.is_file():
        raise OSError(f'Missing source file: {relative}')
    return source


def _read_rows(path: Path, group: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'Expected object at {path.name}:{line_number}')
            try:
                row = {key: value[key] for key in ('pdf', 'category', 'id', 'page')}
            except KeyError as error:
                raise ValueError(
                    f'Missing {error.args[0]} at {path.name}:{line_number}'
                ) from error
            if row['category'] != group:
                raise ValueError(
                    f'Row category does not match {group} at {path.name}:{line_number}'
                )
            if not isinstance(row['pdf'], str) or not isinstance(row['id'], str):
                raise ValueError(f'Invalid source or id at {path.name}:{line_number}')
            if row['page'] is not None and (
                not isinstance(row['page'], int) or row['page'] < 1
            ):
                raise ValueError(f'Invalid page at {path.name}:{line_number}')
            rows.append(row)
    return rows


def _pdf_metadata(path: Path) -> dict[str, object]:
    reader = PdfReader(path)
    pages = [
        {'width': float(page.mediabox.width), 'height': float(page.mediabox.height)}
        for page in reader.pages
    ]
    return {'pdf': {'page_count': len(pages), 'pages': pages}}


def _image_metadata(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
    return {'image': {'width': width, 'height': height}}


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _select_smoke(sources: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        for group in source['groups']:
            by_group[group].append(source)

    selections: dict[str, list[dict[str, Any]]] = {
        'table': by_group['table'][:3],
        'chart': by_group['chart'][:3],
        'text': [
            source
            for source in sources
            if {'text_content', 'text_formatting'}.issubset(source['groups'])
        ][:3],
    }
    layout: list[dict[str, Any]] = []
    for extension in ('.jpg', '.pdf', '.png'):
        match = next(
            (
                source
                for source in by_group['layout']
                if Path(source['path']).suffix.lower() == extension
            ),
            None,
        )
        if match is not None:
            layout.append(match)
    selections['layout'] = sorted(layout, key=lambda item: item['path'])

    if any(len(selected) != 3 for selected in selections.values()):
        raise ValueError('Dataset cannot satisfy fixed smoke coverage')
    return {
        'schema_version': 1,
        'revision': DATASET_REVISION,
        'corpora': [
            {
                'corpus': corpus,
                'sources': [
                    {
                        'path': source['path'],
                        'sha256': source['sha256'],
                        'groups': source['groups'],
                    }
                    for source in selected
                ],
            }
            for corpus, selected in sorted(selections.items())
        ],
    }


def _build_manifests(
    dataset_dir: Path, inventory_path: Path, revision: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = _load_json(inventory_path)
    if revision != DATASET_REVISION or inventory.get('revision') != revision:
        raise ValueError('Dataset revision does not match the audited inventory')
    inventory_groups = {item['group']: item for item in inventory.get('groups', [])}
    if set(inventory_groups) != set(GROUPS):
        raise ValueError('Dataset inventory must contain exactly five groups')

    annotations: list[dict[str, object]] = []
    rows_by_group: dict[str, list[dict[str, object]]] = {}
    groups_by_source: dict[str, set[str]] = defaultdict(set)
    for group in GROUPS:
        annotation_path = dataset_dir / f'{group}.jsonl'
        if not annotation_path.is_file():
            raise OSError(f'Missing annotation file: {annotation_path.name}')
        rows = _read_rows(annotation_path, group)
        rows_by_group[group] = rows
        source_paths = {str(row['pdf']) for row in rows}
        source_page_pairs = {(str(row['pdf']), row['page']) for row in rows}
        actual = {
            'group': group,
            'rows': len(rows),
            'source_files': len(source_paths),
            'source_page_pairs': len(source_page_pairs),
            'bytes': annotation_path.stat().st_size,
            'sha256': _sha256(annotation_path),
            'source_extensions': sorted(
                {PurePosixPath(source).suffix for source in source_paths}
            ),
        }
        if actual != inventory_groups[group]:
            raise ValueError(f'{group} annotation does not match audited inventory')
        annotations.append({'path': annotation_path.name, **actual})
        for source in source_paths:
            _resolve_source(dataset_dir, source, group)
            groups_by_source[source].add(group)

    text_shared = groups_by_source.keys() & {
        source
        for source, groups in groups_by_source.items()
        if {'text_content', 'text_formatting'}.issubset(groups)
    }
    group_memberships = sum(len(groups) for groups in groups_by_source.values())
    if len(groups_by_source) != inventory.get('union_source_files'):
        raise ValueError('Union source count does not match audited inventory')
    if len(text_shared) != inventory.get('shared_text_source_files'):
        raise ValueError('Shared text source count does not match audited inventory')

    sources: list[dict[str, Any]] = []
    page_counts: dict[str, int] = {}
    paths_by_hash: dict[str, list[str]] = defaultdict(list)
    for relative, groups in sorted(groups_by_source.items()):
        source_path = _resolve_source(dataset_dir, relative, sorted(groups)[0])
        suffix = source_path.suffix.lower()
        if suffix not in MEDIA_TYPES:
            raise ValueError(f'Unsupported source media: {relative}')
        physical = (
            _pdf_metadata(source_path)
            if suffix == '.pdf'
            else _image_metadata(source_path)
        )
        page_counts[relative] = physical.get('pdf', {}).get('page_count', 1)
        digest = _sha256(source_path)
        paths_by_hash[digest].append(relative)
        sources.append(
            {
                'path': relative,
                'sha256': digest,
                'bytes': source_path.stat().st_size,
                'media_type': MEDIA_TYPES[suffix],
                'groups': sorted(groups),
                'physical': physical,
            }
        )

    evaluations: list[dict[str, Any]] = []
    for group in GROUPS:
        examples_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows_by_group[group]:
            relative = str(row['pdf'])
            page = row['page']
            if page is not None and page > page_counts[relative]:
                raise ValueError(f'Page {page} is outside {relative}')
            examples_by_source[relative].append(
                {
                    'id': row['id'],
                    'page': page,
                    'page_index': None if page is None else page - 1,
                }
            )
        evaluations.extend(
            {'group': group, 'path': relative, 'examples': examples}
            for relative, examples in sorted(examples_by_source.items())
        )

    manifest = {
        'schema_version': 1,
        'revision': revision,
        'inventory': {
            'file': inventory_path.name,
            'bytes': inventory_path.stat().st_size,
            'sha256': _sha256(inventory_path),
        },
        'annotations': sorted(annotations, key=lambda item: item['group']),
        'summary': {
            'source_files': len(sources),
            'group_memberships': group_memberships,
            'shared_text_source_files': len(text_shared),
        },
        'sources': sources,
        'evaluations': sorted(
            evaluations, key=lambda item: (item['group'], item['path'])
        ),
        'duplicate_hashes': [
            {'sha256': digest, 'paths': paths}
            for digest, paths in sorted(paths_by_hash.items())
            if len(paths) > 1
        ],
    }
    return manifest, _select_smoke(sources)


def prepare_dataset(
    dataset_dir: Path, output_dir: Path, inventory_path: Path, revision: str
) -> tuple[Path, Path]:
    """Verify a frozen ParseBench dataset and create deterministic manifests.

    Returns:
        Paths to the input and smoke manifests.
    """
    dataset_dir = dataset_dir.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    inventory_path = inventory_path.resolve(strict=True)
    if output_dir == dataset_dir or output_dir.is_relative_to(dataset_dir):
        raise ValueError('Manifest output must be outside the dataset directory')

    input_manifest, smoke_manifest = _build_manifests(
        dataset_dir, inventory_path, revision
    )
    input_path = output_dir / 'input-manifest.json'
    smoke_path = output_dir / 'smoke-manifest.json'
    contents = (_canonical_bytes(input_manifest), _canonical_bytes(smoke_manifest))

    if input_path.exists() or smoke_path.exists():
        if not input_path.is_file() or not smoke_path.is_file():
            raise ValueError('Both existing manifests must be regular files')
        if (
            input_path.read_bytes() != contents[0]
            or smoke_path.read_bytes() != contents[1]
        ):
            raise ValueError('Dataset does not match existing manifest')
        return input_path, smoke_path

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(input_path, contents[0])
    _atomic_write(smoke_path, contents[1])
    return input_path, smoke_path
