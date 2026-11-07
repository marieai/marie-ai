"""Validation and loading for plugin-produced extraction artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from marie.extraction.models import ExtractionSuccess
from marie.utils.json import to_json

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
SUPPORTED_MEDIA_TYPES = frozenset({"text/markdown", "application/json"})


def saved_extraction_reference(
    ref_id: str, ref_type: str, metadata: dict[str, Any]
) -> dict[str, str]:
    """Describe the metadata already persisted by the extraction asset writer."""
    return {
        'kind': 'extraction',
        'ref_id': ref_id,
        'ref_type': ref_type,
        'sha256': hashlib.sha256(to_json(metadata).encode('utf-8')).hexdigest(),
    }


def read_extraction_artifact(result: ExtractionSuccess, output_root: str) -> str:
    """Read a bounded artifact after validating its path, size, and digest."""
    root = Path(output_root).resolve(strict=True)
    path = Path(result.artifact.path)
    if path.is_symlink():
        raise ValueError("Extraction artifact must not be a symbolic link")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("Extraction artifact is outside the assigned output directory")
    if not resolved.is_file():
        raise ValueError("Extraction artifact is not a regular file")
    if result.artifact.media_type not in SUPPORTED_MEDIA_TYPES:
        raise ValueError(
            f"Unsupported extraction artifact media type: {result.artifact.media_type}"
        )

    size = resolved.stat().st_size
    if size != result.artifact.size_bytes:
        raise ValueError("Extraction artifact size does not match its descriptor")
    if size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"Extraction artifact exceeds {MAX_ARTIFACT_BYTES} bytes")

    data = resolved.read_bytes()
    if hashlib.sha256(data).hexdigest() != result.artifact.sha256:
        raise ValueError("Extraction artifact digest does not match its descriptor")
    return data.decode("utf-8")
