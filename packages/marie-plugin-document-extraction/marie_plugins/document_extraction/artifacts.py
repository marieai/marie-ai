"""Controlled extraction result artifact handling."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path

from .models import ArtifactDescriptor

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


def write_document_artifact(
    content: str, *, output_dir: str | None, media_type: str
) -> ArtifactDescriptor:
    """Write one immutable extraction result beneath the requested output root."""
    root = _output_root(output_dir)
    suffixes = {
        'text/markdown': '.md',
        'text/html': '.html',
        'text/plain': '.txt',
        'application/json': '.json',
        'application/x-ndjson': '.jsonl',
    }
    suffix = suffixes.get(media_type)
    if suffix is None:
        raise ValueError(f'Unsupported artifact media type: {media_type}')
    path = root / f'document-{uuid.uuid4().hex}{suffix}'
    data = content.encode('utf-8')
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError(
            f'Extraction result exceeds {MAX_ARTIFACT_BYTES} byte artifact limit'
        )
    path.write_bytes(data)
    path.chmod(0o440)
    return ArtifactDescriptor(
        path=str(path),
        media_type=media_type,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _output_root(output_dir: str | None) -> Path:
    if output_dir:
        root = Path(os.path.expanduser(output_dir)).resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(tempfile.mkdtemp(prefix='document_extraction_')).resolve()
    if not root.is_dir():
        raise ValueError(f'Output root is not a directory: {root}')
    return root
