from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from marie_parsebench.dataset import DATASET_REVISION, _atomic_write, prepare_dataset
from PIL import Image
from pypdf import PdfWriter

GROUPS = ('table', 'chart', 'text_content', 'text_formatting', 'layout')


def _write_pdf(path: Path, pages: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for page_number in range(pages):
        writer.add_blank_page(width=200 + page_number, height=300 + page_number)
    with path.open('wb') as stream:
        writer.write(stream)


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, 'white').save(path)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inventory(dataset_dir: Path, inventory_path: Path) -> None:
    groups = []
    all_sources: set[str] = set()
    sources_by_group: dict[str, set[str]] = {}
    for group in GROUPS:
        annotation = dataset_dir / f'{group}.jsonl'
        rows = [json.loads(line) for line in annotation.read_text().splitlines()]
        sources = {row['pdf'] for row in rows}
        sources_by_group[group] = sources
        all_sources.update(sources)
        groups.append(
            {
                'group': group,
                'rows': len(rows),
                'source_files': len(sources),
                'source_page_pairs': len({(row['pdf'], row['page']) for row in rows}),
                'bytes': annotation.stat().st_size,
                'sha256': _sha256(annotation),
                'source_extensions': sorted(
                    {Path(source).suffix for source in sources}
                ),
            }
        )
    inventory_path.write_text(
        json.dumps(
            {
                'revision': DATASET_REVISION,
                'groups': groups,
                'union_source_files': len(all_sources),
                'shared_text_source_files': len(
                    sources_by_group['text_content']
                    & sources_by_group['text_formatting']
                ),
            }
        ),
        encoding='utf-8',
    )


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / 'dataset'
    root.mkdir()

    rows: dict[str, list[dict[str, object]]] = {}
    for group in ('table', 'chart'):
        rows[group] = []
        for number in range(3):
            source = f'docs/{group}/{number}.pdf'
            _write_pdf(root / source)
            rows[group].append(
                {
                    'pdf': source,
                    'category': group,
                    'id': f'{group}-{number}',
                    'page': None,
                }
            )

    shared_text = [f'docs/text/{number}.pdf' for number in range(3)]
    for source in shared_text:
        _write_pdf(root / source)
    rows['text_content'] = [
        {
            'pdf': source,
            'category': 'text_content',
            'id': f'content-{number}',
            'page': 1,
        }
        for number, source in enumerate(shared_text)
    ]
    rows['text_formatting'] = [
        {
            'pdf': source,
            'category': 'text_formatting',
            'id': f'format-{number}',
            'page': 1,
        }
        for number, source in enumerate(shared_text)
    ]

    _write_pdf(root / 'docs/layout/0.pdf', pages=2)
    _write_image(root / 'docs/layout/1.jpg', (31, 41))
    _write_image(root / 'docs/layout/2.png', (32, 42))
    rows['layout'] = [
        {
            'pdf': 'docs/layout/0.pdf',
            'category': 'layout',
            'id': 'layout-pdf',
            'page': 2,
        },
        {
            'pdf': 'docs/layout/1.jpg',
            'category': 'layout',
            'id': 'layout-jpg',
            'page': 1,
        },
        {
            'pdf': 'docs/layout/2.png',
            'category': 'layout',
            'id': 'layout-png',
            'page': 1,
        },
    ]

    for group, group_rows in rows.items():
        _write_rows(root / f'{group}.jsonl', group_rows)

    inventory = tmp_path / 'inventory.json'
    _write_inventory(root, inventory)
    return root, inventory


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def test_package_import_does_not_require_preparation_dependencies() -> None:
    source_dir = Path(__file__).parents[1] / 'src'
    result = subprocess.run(
        [
            sys.executable,
            '-I',
            '-S',
            '-c',
            f'import sys; sys.path.insert(0, {str(source_dir)!r}); import marie_parsebench',
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_atomic_write_preserves_destination_created_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / 'manifest.json'
    real_mkstemp = tempfile.mkstemp

    def create_competing_destination(
        *args: object, **kwargs: object
    ) -> tuple[int, str]:
        descriptor, name = real_mkstemp(*args, **kwargs)
        destination.write_bytes(b'other process')
        return descriptor, name

    monkeypatch.setattr(
        'marie_parsebench.dataset.tempfile.mkstemp', create_competing_destination
    )

    with pytest.raises(FileExistsError):
        _atomic_write(destination, b'our process')

    assert destination.read_bytes() == b'other process'


def test_prepare_dataset_records_physical_sources_and_evaluations_separately(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    root, inventory = dataset
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob('*')
        if path.is_file()
    }

    input_path, smoke_path = prepare_dataset(
        root, tmp_path / 'out', inventory, DATASET_REVISION
    )

    manifest = _load(input_path)
    assert manifest['summary'] == {
        'source_files': 12,
        'group_memberships': 15,
        'shared_text_source_files': 3,
    }
    assert set(manifest) == {
        'schema_version',
        'revision',
        'inventory',
        'annotations',
        'summary',
        'sources',
        'evaluations',
        'duplicate_hashes',
    }
    assert all(
        'ids' not in source and 'pages' not in source for source in manifest['sources']
    )
    assert all(
        forbidden not in json.dumps(manifest)
        for forbidden in ('rules', 'expected_markdown', 'tags')
    )
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob('*')
        if path.is_file()
    } == before
    assert smoke_path.name == 'smoke-manifest.json'


def test_prepare_dataset_preserves_shared_text_ids_and_maps_pages(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    root, inventory = dataset
    input_path, _ = prepare_dataset(root, tmp_path / 'out', inventory, DATASET_REVISION)
    manifest = _load(input_path)

    shared_source = next(
        source for source in manifest['sources'] if source['path'] == 'docs/text/0.pdf'
    )
    assert shared_source['groups'] == ['text_content', 'text_formatting']
    memberships = {
        (item['group'], item['path']): item['examples']
        for item in manifest['evaluations']
    }
    assert memberships[('text_content', 'docs/text/0.pdf')] == [
        {'id': 'content-0', 'page': 1, 'page_index': 0}
    ]
    assert memberships[('text_formatting', 'docs/text/0.pdf')] == [
        {'id': 'format-0', 'page': 1, 'page_index': 0}
    ]
    assert memberships[('table', 'docs/table/0.pdf')] == [
        {'id': 'table-0', 'page': None, 'page_index': None}
    ]
    assert memberships[('layout', 'docs/layout/0.pdf')] == [
        {'id': 'layout-pdf', 'page': 2, 'page_index': 1}
    ]


def test_prepare_dataset_records_pdf_and_image_dimensions(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    root, inventory = dataset
    input_path, _ = prepare_dataset(root, tmp_path / 'out', inventory, DATASET_REVISION)
    sources = {item['path']: item for item in _load(input_path)['sources']}

    assert sources['docs/layout/0.pdf']['physical']['pdf'] == {
        'page_count': 2,
        'pages': [
            {'height': 300.0, 'width': 200.0},
            {'height': 301.0, 'width': 201.0},
        ],
    }
    assert sources['docs/layout/1.jpg']['physical']['image'] == {
        'height': 41,
        'width': 31,
    }
    assert sources['docs/layout/2.png']['physical']['image'] == {
        'height': 42,
        'width': 32,
    }


def test_smoke_manifest_has_three_sources_per_corpus_and_required_media(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    root, inventory = dataset
    _, smoke_path = prepare_dataset(root, tmp_path / 'out', inventory, DATASET_REVISION)
    smoke = _load(smoke_path)

    by_corpus = {item['corpus']: item['sources'] for item in smoke['corpora']}
    assert {corpus: len(sources) for corpus, sources in by_corpus.items()} == {
        'chart': 3,
        'layout': 3,
        'table': 3,
        'text': 3,
    }
    assert {Path(item['path']).suffix for item in by_corpus['layout']} == {
        '.jpg',
        '.pdf',
        '.png',
    }
    assert all(
        item['groups'] == ['text_content', 'text_formatting']
        for item in by_corpus['text']
    )


@pytest.mark.parametrize(
    'bad_path', ['/tmp/escape.pdf', '../escape.pdf', 'docs/table/missing.pdf']
)
def test_prepare_dataset_rejects_unsafe_or_missing_source(
    dataset: tuple[Path, Path], tmp_path: Path, bad_path: str
) -> None:
    root, inventory = dataset
    rows = [
        json.loads(line) for line in (root / 'table.jsonl').read_text().splitlines()
    ]
    rows[0]['pdf'] = bad_path
    _write_rows(root / 'table.jsonl', rows)
    _write_inventory(root, inventory)

    with pytest.raises((ValueError, OSError)):
        prepare_dataset(root, tmp_path / 'out', inventory, DATASET_REVISION)


def test_prepare_dataset_rejects_symlink_escape(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    root, inventory = dataset
    outside = tmp_path / 'outside.pdf'
    _write_pdf(outside)
    (root / 'docs/table/0.pdf').unlink()
    (root / 'docs/table/0.pdf').symlink_to(outside)

    with pytest.raises(ValueError, match='escapes'):
        prepare_dataset(root, tmp_path / 'out', inventory, DATASET_REVISION)


def test_prepare_dataset_rejects_changed_annotation_or_source(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    root, inventory = dataset
    (root / 'chart.jsonl').write_text('{}\n', encoding='utf-8')

    with pytest.raises(ValueError):
        prepare_dataset(root, tmp_path / 'out', inventory, DATASET_REVISION)

    _write_rows(
        root / 'chart.jsonl',
        [
            {
                'pdf': f'docs/chart/{number}.pdf',
                'category': 'chart',
                'id': f'chart-{number}',
                'page': None,
            }
            for number in range(3)
        ],
    )
    _write_inventory(root, inventory)
    with (root / 'docs/chart/0.pdf').open('ab') as stream:
        stream.write(b'damage')
    output = tmp_path / 'out-source'
    first = prepare_dataset(root, output, inventory, DATASET_REVISION)
    assert first[0].exists()
    with (root / 'docs/chart/0.pdf').open('ab') as stream:
        stream.write(b'more-damage')
    with pytest.raises(ValueError, match='existing manifest'):
        prepare_dataset(root, output, inventory, DATASET_REVISION)


def test_prepare_dataset_rejects_group_revision_and_page_errors(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    root, inventory = dataset
    rows = [
        json.loads(line) for line in (root / 'layout.jsonl').read_text().splitlines()
    ]
    rows[0]['category'] = 'chart'
    _write_rows(root / 'layout.jsonl', rows)
    _write_inventory(root, inventory)
    with pytest.raises(ValueError, match='category'):
        prepare_dataset(root, tmp_path / 'group', inventory, DATASET_REVISION)

    root, inventory = dataset
    with pytest.raises(ValueError, match='revision'):
        prepare_dataset(root, tmp_path / 'revision', inventory, 'wrong-revision')

    rows = [
        json.loads(line) for line in (root / 'layout.jsonl').read_text().splitlines()
    ]
    rows[0]['category'] = 'layout'
    rows[0]['page'] = 3
    _write_rows(root / 'layout.jsonl', rows)
    _write_inventory(root, inventory)
    with pytest.raises(ValueError, match='[Pp]age'):
        prepare_dataset(root, tmp_path / 'page', inventory, DATASET_REVISION)


def test_repeat_verifies_both_manifests_without_rewriting(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    root, inventory = dataset
    output = tmp_path / 'out'
    paths = prepare_dataset(root, output, inventory, DATASET_REVISION)
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]

    assert prepare_dataset(root, output, inventory, DATASET_REVISION) == paths
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before

    paths[1].write_text('{}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='existing manifest'):
        prepare_dataset(root, output, inventory, DATASET_REVISION)


def test_prepare_dataset_refuses_output_inside_dataset(
    dataset: tuple[Path, Path],
) -> None:
    root, inventory = dataset
    with pytest.raises(ValueError, match='output'):
        prepare_dataset(root, root / 'manifests', inventory, DATASET_REVISION)
