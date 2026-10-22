from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip('marie')
pytest.importorskip('marie.executor.request_util')

import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas

from marie.api.docs import DOC_KEY_PAGE_NUMBER
from marie.utils.docs import docs_from_file

TOOL_PATH = Path(__file__).parents[2] / 'tools/inspect-runtime-preparation.py'
SPEC = importlib.util.spec_from_file_location('inspect_runtime_preparation', TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifests(
    root: Path, sources: list[Path], *, smoke_path: str | None = None
) -> tuple[Path, Path]:
    revision = 'fixture-revision'
    records = [
        {
            'path': source.relative_to(root).as_posix(),
            'sha256': _sha256(source),
            'bytes': source.stat().st_size,
            'media_type': {
                '.pdf': 'application/pdf',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
            }[source.suffix],
            'physical': {},
        }
        for source in sources
    ]
    input_manifest = root / 'input-manifest.json'
    input_manifest.write_text(
        json.dumps({'schema_version': 1, 'revision': revision, 'sources': records})
    )
    selected_paths = [smoke_path] if smoke_path else [item['path'] for item in records]
    smoke_sources = []
    for selected_path in selected_paths:
        selected = next(
            (item for item in records if item['path'] == selected_path), None
        )
        smoke_sources.append(
            {
                'path': selected_path,
                'sha256': selected['sha256'] if selected else '0' * 64,
                'groups': ['fixture'],
            }
        )
    smoke_manifest = root / 'smoke-manifest.json'
    smoke_manifest.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'revision': revision,
                'corpora': [{'corpus': 'fixture', 'sources': smoke_sources}],
            }
        )
    )
    return input_manifest, smoke_manifest


def _fixture_sources(root: Path) -> list[Path]:
    docs = root / 'docs'
    docs.mkdir(parents=True)
    pdf = docs / 'rotated.pdf'
    writer = canvas.Canvas(str(pdf), pagesize=(144, 288))
    writer.setPageRotation(90)
    writer.drawString(20, 40, 'rotated page')
    writer.showPage()
    writer.setPageRotation(0)
    writer.setPageSize((300, 150))
    writer.drawString(20, 40, 'landscape page')
    writer.save()
    Image.new('RGB', (41, 23), 'red').save(docs / 'sample.png')
    Image.new('RGB', (37, 29), 'blue').save(docs / 'sample.jpg')
    return [pdf, docs / 'sample.png', docs / 'sample.jpg']


def _fixture_case(tmp_path: Path) -> tuple[Path, list[Path], Path, Path]:
    dataset = tmp_path / 'dataset'
    sources = _fixture_sources(dataset)
    input_manifest, smoke_manifest = _write_manifests(dataset, sources)
    return dataset, sources, input_manifest, smoke_manifest


def test_inspects_real_runtime_frames_and_preserves_sources(tmp_path: Path) -> None:
    dataset, sources, input_manifest, smoke_manifest = _fixture_case(tmp_path)
    before = {path: _sha256(path) for path in sources}

    report = TOOL.inspect_preparation(
        dataset, input_manifest, smoke_manifest, tmp_path / 'output'
    )

    assert report['scope'] == 'local_runtime_functions'
    assert report['gateway_submission'] == 'not_executed'
    assert report['ocr_execution'] == 'not_executed'
    assert report['input_manifest']['sha256'] == _sha256(input_manifest)
    assert report['smoke_manifest']['sha256'] == _sha256(smoke_manifest)
    assert report['renderer']['pdf_dpi'] == 200
    assert [
        frame['physical_page_index'] for frame in report['sources'][0]['frames']
    ] == [0, 1]
    assert report['sources'][0]['source']['pdf']['pages'][0]['rotation'] == 90
    assert (
        report['sources'][0]['frames'][0]['before']['width']
        != report['sources'][0]['frames'][0]['before']['height']
    )
    assert len(list((tmp_path / 'output' / 'frames').glob('*.png'))) == 4
    with Image.open(tmp_path / 'output' / 'frames' / '01-0000.png') as frame:
        assert frame.getpixel((0, 0)) == (255, 0, 0)
    assert {path: _sha256(path) for path in sources} == before


def test_rejects_tampered_source(tmp_path: Path) -> None:
    dataset, sources, input_manifest, smoke_manifest = _fixture_case(tmp_path)
    sources[0].write_bytes(sources[0].read_bytes() + b'tampered')

    with pytest.raises(ValueError, match='sha256'):
        TOOL.inspect_preparation(
            dataset, input_manifest, smoke_manifest, tmp_path / 'output'
        )


def test_rejects_source_path_that_escapes_dataset(tmp_path: Path) -> None:
    dataset, sources, input_manifest, smoke_manifest = _fixture_case(tmp_path)
    _, smoke_manifest = _write_manifests(dataset, sources, smoke_path='../outside.pdf')

    with pytest.raises(ValueError, match='escapes dataset root'):
        TOOL.inspect_preparation(
            dataset, input_manifest, smoke_manifest, tmp_path / 'output'
        )


def test_refuses_any_existing_output_directory(tmp_path: Path) -> None:
    dataset, _, input_manifest, smoke_manifest = _fixture_case(tmp_path)
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'partial-frame.png').write_bytes(b'partial')

    with pytest.raises(FileExistsError, match='Output directory already exists'):
        TOOL.inspect_preparation(dataset, input_manifest, smoke_manifest, output)


@pytest.mark.parametrize('overlap', ['dataset', 'input-parent', 'smoke-parent'])
def test_rejects_output_path_overlapping_protected_inputs(
    tmp_path: Path, overlap: str
) -> None:
    dataset = tmp_path / 'dataset'
    sources = _fixture_sources(dataset)
    input_parent = tmp_path / 'input-manifest-dir'
    smoke_parent = tmp_path / 'smoke-manifest-dir'
    input_parent.mkdir()
    smoke_parent.mkdir()
    generated_input, generated_smoke = _write_manifests(dataset, sources)
    input_manifest = input_parent / 'input.json'
    smoke_manifest = smoke_parent / 'smoke.json'
    input_manifest.write_bytes(generated_input.read_bytes())
    smoke_manifest.write_bytes(generated_smoke.read_bytes())
    output = {
        'dataset': dataset / 'probe-output',
        'input-parent': input_manifest.parent,
        'smoke-parent': smoke_manifest.parent,
    }[overlap]

    with pytest.raises(ValueError, match='overlaps protected path'):
        TOOL.inspect_preparation(dataset, input_manifest, smoke_manifest, output)


def test_real_loader_preserves_selected_pdf_page_identity(tmp_path: Path) -> None:
    sources = _fixture_sources(tmp_path)
    all_frames = TOOL.get_frames_from_file(str(sources[0]))
    selected = TOOL.get_frames_from_file(str(sources[0]), pages=[1, 0])
    docs = docs_from_file(str(sources[0]), pages=[1, 0])

    assert [doc.tags[DOC_KEY_PAGE_NUMBER] for doc in docs] == [1, 0]
    np.testing.assert_array_equal(selected[0], all_frames[1])
    np.testing.assert_array_equal(selected[1], all_frames[0])


def test_probe_records_actual_runtime_resize(tmp_path: Path) -> None:
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    source = dataset / 'oversized.png'
    Image.new('RGB', (3200, 4000), 'red').save(source)
    input_manifest, smoke_manifest = _write_manifests(dataset, [source])

    report = TOOL.inspect_preparation(
        dataset, input_manifest, smoke_manifest, tmp_path / 'output'
    )

    frame = report['sources'][0]['frames'][0]
    assert frame['before'] == {'width': 3200, 'height': 4000}
    assert frame['after'] == {'width': 2932, 'height': 3665}
    assert frame['scale'] == {'x': 2932 / 3200, 'y': 3665 / 4000}
    with Image.open(tmp_path / 'output' / frame['artifact']) as saved:
        assert saved.size == (2932, 3665)
        assert saved.getpixel((0, 0)) == (255, 0, 0)


def test_rejects_empty_smoke_selection(tmp_path: Path) -> None:
    dataset, _, input_manifest, smoke_manifest = _fixture_case(tmp_path)
    smoke = json.loads(smoke_manifest.read_text())
    smoke['corpora'] = []
    smoke_manifest.write_text(json.dumps(smoke))

    with pytest.raises(ValueError, match='no sources'):
        TOOL.inspect_preparation(
            dataset, input_manifest, smoke_manifest, tmp_path / 'output'
        )
    assert not (tmp_path / 'output').exists()


def test_concurrent_output_creation_does_not_overwrite_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, _, input_manifest, smoke_manifest = _fixture_case(tmp_path)
    output = tmp_path / 'output'
    original_load = TOOL._load_json
    competing_frame = output / 'frames' / '00-0000.png'

    def create_competing_output(path: Path) -> dict:
        if not output.exists():
            competing_frame.parent.mkdir(parents=True)
            competing_frame.write_bytes(b'competing run')
        return original_load(path)

    monkeypatch.setattr(TOOL, '_load_json', create_competing_output)
    with pytest.raises(FileExistsError):
        TOOL.inspect_preparation(dataset, input_manifest, smoke_manifest, output)
    assert competing_frame.read_bytes() == b'competing run'
