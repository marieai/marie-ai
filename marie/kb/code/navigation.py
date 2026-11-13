"""Bounded syntax artifacts derived from digest-checked repository captures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def parse_file(path: Path, digest: str, *, max_records: int = 1000) -> dict[str, Any]:
    """Extract navigation records without making grammar availability a search gate."""
    if not 1 <= max_records <= 10000:
        raise ValueError('Navigation record limit must be between 1 and 10000')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('Captured file digest changed before parsing')
    result: dict[str, Any] = {
        'status': 'unavailable',
        'language': None,
        'provider_version': None,
        'symbols': [],
        'references': [],
        'imports': [],
    }
    try:
        from marie_plugins.document_extraction.detection import detect_format
        from marie_plugins.document_extraction.providers.tree_sitter import (
            TreeSitterProvider,
        )
    except ModuleNotFoundError as error:
        if not (error.name or '').startswith('marie_plugins'):
            raise
        return result
    try:
        language = detect_format(str(path)).canonical_format
    except ValueError:
        return {**result, 'status': 'unsupported'}
    result['language'] = language
    provider = TreeSitterProvider()
    if language not in provider.formats:
        return {**result, 'status': 'unsupported'}
    if not provider.is_ready(language):
        return result
    document = provider.extract(
        str(path), language, {'include_references': True}, 'json'
    )
    body = json.loads(document.content)
    result['parse_status'] = body['parse_status']
    lines = content.decode('utf-8').split('\n')
    remaining = max_records
    truncated = False
    characters = 0
    for key in ('symbols', 'references', 'imports'):
        values = body.get(key, [])
        for value in values:
            if remaining == 0:
                truncated = True
                break
            if key == 'imports':
                if len(value) > 500:
                    truncated = True
                    continue
                record = value
            else:
                if any(
                    len(value.get(field, '')) > limit
                    for field, limit in (
                        ('name', 200),
                        ('qualified_name', 1000),
                        ('kind', 100),
                    )
                ):
                    truncated = True
                    continue
                record = {
                    field: value[field]
                    for field in (
                        'name',
                        'qualified_name',
                        'kind',
                        'signature',
                        'span',
                        'line',
                        'col',
                    )
                    if field in value
                }
                record['parse_status'] = body['parse_status']
                start = value.get('line') or value['span']['start_line']
                record['readable'] = (
                    1 <= start <= len(lines)
                    and len(lines[start - 1]) + (start < len(lines)) <= 8000
                )
            size = len(json.dumps(record, ensure_ascii=False))
            if characters + size > 128000:
                truncated = True
                break
            characters += size
            remaining -= 1
            result[key].append(record)
    return {
        **result,
        'status': 'truncated' if truncated else body['parse_status'],
        'provider_version': document.provider_version,
    }
