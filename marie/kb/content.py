"""Input references accepted by the document-content indexing workflow."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from marie.storage import StorageManager
from marie.utils.asset_util import s3_asset_path, split_filename


def document_content_reference(reference: Any) -> dict[str, Any]:
    """Resolve supported saved content without converting domain records to text."""
    if not isinstance(reference, dict):
        raise ValueError('A saved content reference is required')
    if reference.get('kind') == 'upstream':
        reference = reference.get('artifact')
        if not isinstance(reference, dict):
            raise ValueError('Upstream content is not ready')
    kind = reference.get('kind')
    if kind == 'extraction':
        if not all(
            isinstance(reference.get(key), str) and reference[key]
            for key in ('ref_id', 'ref_type')
        ):
            raise ValueError('Saved extraction content requires ref_id and ref_type')
    elif kind == 'text':
        if reference.get('media_type') not in ('text/plain', 'text/markdown'):
            raise ValueError(
                'Unsupported content format; select a compatible processing workflow'
            )
        if not isinstance(reference.get('uri'), str) or not reference['uri'].startswith(
            's3://'
        ):
            raise ValueError('Saved text content requires an S3 artifact reference')
        if not reference.get('sha256'):
            raise ValueError('Saved text content requires its captured digest')
    else:
        raise ValueError(
            'Unsupported content kind; select a compatible processing workflow'
        )
    digest = reference.get('sha256')
    if digest is not None and (
        not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest)
    ):
        raise ValueError('Saved content digest must be a SHA-256 hex string')
    return dict(reference)


def load_document_text(reference: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Read captured text and return its digest-pinned storage reference."""
    reference = document_content_reference(reference)
    if reference['kind'] == 'extraction':
        filename, _, _ = split_filename(reference['ref_id'])
        uri = f"{s3_asset_path(reference['ref_id'], reference['ref_type'])}/{filename}.meta.json"
    else:
        uri = reference['uri']
    raw = StorageManager.read(uri, max_bytes=32 * 1024 * 1024)
    if raw is None:
        raise FileNotFoundError('Saved source is unavailable')
    payload = raw.encode('utf-8') if isinstance(raw, str) else raw
    if len(payload) > 32 * 1024 * 1024:
        raise ValueError('Saved source exceeds the supported artifact size')
    digest = hashlib.sha256(payload).hexdigest()
    if reference.get('sha256') and reference['sha256'] != digest:
        raise ValueError('Saved source changed after capture')
    text = payload.decode('utf-8')
    if reference['kind'] == 'extraction':
        metadata = json.loads(text)
        if not isinstance(metadata, dict) or not isinstance(metadata.get('ocr'), list):
            raise ValueError('Saved extraction has no supported text output')
        text, _ = flatten_ocr_pages(metadata['ocr'])
    return text, {**reference, 'sha256': digest}


def flatten_ocr_pages(
    ocr_results: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, int, int]]]:
    parts = []
    page_ranges = []
    cursor = 0
    for page_idx, page_result in enumerate(ocr_results):
        if not isinstance(page_result, dict):
            continue
        lines = page_result.get('lines') or []
        if not isinstance(lines, list):
            raise ValueError('Saved extraction has invalid OCR lines')
        page_text = '\n'.join(
            line['text']
            for line in lines
            if isinstance(line, dict) and isinstance(line.get('text'), str)
        )
        if parts:
            parts.append('\n')
            cursor += 1
        start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        page_ranges.append((page_idx, start, cursor))
    return ''.join(parts), page_ranges
