"""Adapt provider-neutral plugin results to Marie extraction metadata."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

from marie.extraction.models import ExtractionSuccess
from marie.utils.asset_util import split_filename, store_assets
from marie.utils.json import store_json_object


def build_extraction_metadata(
    content: str, result: ExtractionSuccess
) -> dict[str, Any]:
    """Build metadata compatible with downstream OCR-result consumers."""
    plugin_metadata = result.metadata
    page_count = plugin_metadata.get("page_count") or 1
    extraction: dict[str, Any] = {
        "engine": result.provenance.provider,
        "provider": result.provenance.provider,
        "provider_version": result.provenance.provider_version,
        "backend": result.provenance.backend,
        "format": result.provenance.canonical_format,
        "result_kind": result.result_kind,
        "route": "plugin",
        "ocr_invoked": False,
        "char_count": len(content),
        "page_count": page_count,
        "warnings": result.warnings,
    }
    return {
        "ocr": [{"lines": [{"text": line} for line in content.split("\n")]}],
        "extraction": extraction,
    }


def write_extraction_metadata(
    ref_id: str,
    ref_type: str,
    metadata: dict[str, Any],
) -> None:
    """Persist adapted metadata through the existing extraction asset path."""
    filename, _, _ = split_filename(ref_id)
    root_asset_dir = tempfile.mkdtemp(prefix="document_extraction_artifact_")
    try:
        metadata_path = os.path.join(root_asset_dir, f"{filename}.meta.json")
        store_json_object(metadata, metadata_path)
        store_assets(ref_id, ref_type, root_asset_dir)
    finally:
        shutil.rmtree(root_asset_dir, ignore_errors=True)
