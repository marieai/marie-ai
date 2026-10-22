#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
from PIL import ExifTags, Image
from PyPDF4 import PdfFileReader

from marie.executor.request_util import get_frames_from_file
from marie.utils.docs import load_document


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


def _resolve_source(dataset_dir: Path, relative: str) -> Path:
    posix_path = PurePosixPath(relative)
    if posix_path.is_absolute() or '..' in posix_path.parts:
        raise ValueError(f'Source path escapes dataset root: {relative}')
    source = dataset_dir.joinpath(*posix_path.parts)
    resolved = source.resolve(strict=False)
    if not resolved.is_relative_to(dataset_dir) or source.is_symlink():
        raise ValueError(f'Source path escapes dataset root: {relative}')
    if not source.is_file():
        raise OSError(f'Missing source file: {relative}')
    return source


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _box(box: Any) -> list[float]:
    return [float(value) for value in box]


def _source_geometry(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == '.pdf':
        with path.open('rb') as stream:
            reader = PdfFileReader(stream)
            pages = [
                {
                    'physical_page_index': index,
                    'rotation': int(page.get('/Rotate', 0) or 0) % 360,
                    'media_box': _box(page.mediaBox),
                    'crop_box': _box(page.cropBox),
                }
                for index, page in enumerate(reader.pages)
            ]
        return {'pdf': {'page_count': len(pages), 'pages': pages}}

    with Image.open(path) as image:
        orientation = image.getexif().get(ExifTags.Base.Orientation)
        return {
            'image': {
                'width': image.width,
                'height': image.height,
                'exif_orientation': orientation,
            }
        }


def _size(frame: Any) -> dict[str, int]:
    height, width = frame.shape[:2]
    return {'width': width, 'height': height}


def inspect_preparation(
    dataset_dir: Path,
    input_manifest: Path,
    smoke_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Inspect Marie's local document loading and page-size cap for smoke sources."""
    dataset_dir = dataset_dir.resolve()
    input_manifest = input_manifest.resolve()
    smoke_manifest = smoke_manifest.resolve()
    output_dir = output_dir.resolve()
    for protected_path in (dataset_dir, input_manifest, smoke_manifest):
        if _paths_overlap(output_dir, protected_path):
            raise ValueError(
                f'Output directory overlaps protected path: {protected_path}'
            )
    if output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    report_path = output_dir / 'preparation-report.json'

    protected = {
        input_manifest: _sha256(input_manifest),
        smoke_manifest: _sha256(smoke_manifest),
    }
    input_data = _load_json(input_manifest)
    smoke_data = _load_json(smoke_manifest)
    if smoke_data.get('revision') != input_data.get('revision'):
        raise ValueError('Smoke revision does not match input revision')

    input_sources = {source['path']: source for source in input_data.get('sources', [])}
    selected: list[tuple[str, dict[str, Any], Path]] = []
    seen: set[str] = set()
    for corpus in smoke_data.get('corpora', []):
        for smoke_source in corpus.get('sources', []):
            relative = smoke_source.get('path')
            if not isinstance(relative, str):
                raise ValueError('Smoke source path must be a string')
            source_path = _resolve_source(dataset_dir, relative)
            input_source = input_sources.get(relative)
            if input_source is None:
                raise ValueError(
                    f'Smoke source is absent from input manifest: {relative}'
                )
            actual_hash = _sha256(source_path)
            if actual_hash != input_source.get('sha256'):
                raise ValueError(
                    f'Source sha256 does not match input manifest: {relative}'
                )
            if actual_hash != smoke_source.get('sha256'):
                raise ValueError(
                    f'Source sha256 does not match smoke manifest: {relative}'
                )
            protected[source_path] = actual_hash
            if relative not in seen:
                selected.append((relative, input_source, source_path))
                seen.add(relative)

    if not selected:
        raise ValueError('Smoke manifest selects no sources')

    output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir()
    source_reports = []
    for source_number, (relative, input_source, source_path) in enumerate(selected):
        before_frames = load_document(str(source_path))['frames']
        after_frames = get_frames_from_file(str(source_path))
        if len(before_frames) != len(after_frames):
            raise RuntimeError(f'Runtime changed page count while preparing {relative}')
        frame_reports = []
        for page_index, (before, after) in enumerate(zip(before_frames, after_frames)):
            before_size = _size(before)
            after_size = _size(after)
            frame_name = f'{source_number:02d}-{page_index:04d}.png'
            encoded_frame = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(frames_dir / frame_name), encoded_frame):
                raise OSError(f'Could not write frame {frame_name}')
            frame_reports.append(
                {
                    'physical_page_index': page_index,
                    'before': before_size,
                    'after': after_size,
                    'scale': {
                        'x': after_size['width'] / before_size['width'],
                        'y': after_size['height'] / before_size['height'],
                    },
                    'artifact': f'frames/{frame_name}',
                }
            )
        source_reports.append(
            {
                'path': relative,
                'sha256': input_source['sha256'],
                'bytes': input_source.get('bytes'),
                'media_type': input_source.get('media_type'),
                'source': _source_geometry(source_path),
                'frames': frame_reports,
            }
        )

    for path, digest in protected.items():
        if _sha256(path) != digest:
            raise RuntimeError(f'Protected input changed during inspection: {path}')

    report = {
        'schema_version': 1,
        'revision': input_data.get('revision'),
        'input_manifest': {
            'path': str(input_manifest),
            'sha256': protected[input_manifest],
        },
        'smoke_manifest': {
            'path': str(smoke_manifest),
            'sha256': protected[smoke_manifest],
        },
        'scope': 'local_runtime_functions',
        'gateway_submission': 'not_executed',
        'ocr_execution': 'not_executed',
        'renderer': {
            'loader': 'marie.utils.docs.load_document',
            'preparation': 'marie.executor.request_util.get_frames_from_file',
            'pdf_backend': 'pdf2image.convert_from_path',
            'pdf_dpi': 200,
            'page_size_cap': 'marie.utils.image_utils.ensure_max_page_size',
        },
        'sources': source_reports,
    }
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True, type=Path)
    parser.add_argument('--input-manifest', required=True, type=Path)
    parser.add_argument('--smoke-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    inspect_preparation(
        args.dataset_dir, args.input_manifest, args.smoke_manifest, args.output_dir
    )


if __name__ == '__main__':
    main()
