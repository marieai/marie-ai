import os
import shutil
import tempfile
from typing import Any, Optional

from marie.utils.asset_util import split_filename, store_assets
from marie.utils.json import store_json_object


def build_markitdown_metadata(
    markdown: str,
    fmt: str,
    page_count: int,
    fallback_reason: Optional[str] = None,
) -> dict[str, Any]:
    """Build the ``{ref_id}.meta.json`` payload for a markitdown extraction.

    ``metadata["ocr"]`` is a single synthetic page whose line records reproduce
    the source markdown byte-for-byte when flattened by
    ``VectorStoreExecutor._flatten_ocr_pages``: every line from
    ``markdown.split("\\n")`` becomes one ``{"text": line}`` record, blank lines
    included as ``{"text": ""}``. A single page is used because the writer only
    receives a page count, not per-page text boundaries; the flattener joins
    pages with an extra ``"\\n"``, so any multi-page split would need exact line
    boundaries to stay lossless.
    """
    lines = [{"text": line} for line in markdown.split("\n")]
    extraction: dict[str, Any] = {
        "engine": "markitdown",
        "format": fmt,
        "char_count": len(markdown),
        "page_count": page_count,
    }
    if fallback_reason is not None:
        extraction["fallback_reason"] = fallback_reason
    return {
        "ocr": [{"lines": lines}],
        "extraction": extraction,
    }


def write_markitdown_artifact(
    ref_id: str,
    ref_type: str,
    markdown: str,
    fmt: str,
    page_count: int,
    fallback_reason: Optional[str] = None,
) -> None:
    """Persist a markitdown extraction as the same ``{ref_id}.meta.json`` artifact
    ``ExtractPipeline`` writes for OCR, so the EMBED stage reads it identically.

    The payload is written into a working dir and uploaded via ``store_assets``
    (the same helpers ``ExtractPipeline.store_metadata()``/``store_assets()`` use),
    landing the object at ``s3_asset_path(ref_id, ref_type)/{filename}.meta.json``.
    """
    metadata = build_markitdown_metadata(markdown, fmt, page_count, fallback_reason)

    filename, _, _ = split_filename(ref_id)
    root_asset_dir = tempfile.mkdtemp(prefix="markitdown_artifact_")
    try:
        metadata_path = os.path.join(root_asset_dir, f"{filename}.meta.json")
        store_json_object(metadata, metadata_path)
        store_assets(ref_id, ref_type, root_asset_dir)
    finally:
        shutil.rmtree(root_asset_dir, ignore_errors=True)
