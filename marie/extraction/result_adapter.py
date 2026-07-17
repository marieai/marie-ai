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
    pages = _pages(content, result)
    page_count = max(int(plugin_metadata.get("page_count") or 1), len(pages))
    pages.extend([""] * (page_count - len(pages)))
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
        "pages": str(page_count),
        "ocr": [_ocr_page(text, page) for page, text in enumerate(pages)],
        "extraction": extraction,
    }


def _pages(content: str, result: ExtractionSuccess) -> list[str]:
    if result.provenance.canonical_format != "pdf":
        return [content]

    pages = content.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages or [""]


def _ocr_page(content: str, page: int) -> dict[str, Any]:
    text_lines = [line.strip() for line in content.splitlines() if line.strip()]
    width = max([1200, *(len(line) * 8 + 16 for line in text_lines)])
    line_boxes = [
        [8, index * 24 + 8, max(len(line) * 8, 8), 20]
        for index, line in enumerate(text_lines)
    ]
    words = [
        {
            "id": index,
            "text": line,
            "confidence": 1.0,
            "box": line_boxes[index],
            "line": index,
            "word_index": index,
        }
        for index, line in enumerate(text_lines)
    ]
    lines = [
        {
            "line": index,
            "wordids": [index],
            "text": line,
            "bbox": line_boxes[index],
            "confidence": 1.0,
        }
        for index, line in enumerate(text_lines)
    ]
    return {
        "words": words,
        "lines": lines,
        "meta": {
            "page": page,
            "lines": list(range(len(lines))),
            "lines_bboxes": line_boxes,
            "imageSize": {"width": width, "height": max(len(lines) * 24 + 16, 32)},
            "format": "xywh",
        },
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
