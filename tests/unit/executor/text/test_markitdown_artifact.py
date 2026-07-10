import json
import os

import pytest

from marie.executor.kb.vector_store_executor import VectorStoreExecutor
from marie.executor.text.markitdown_artifact import (
    build_markitdown_metadata,
    write_markitdown_artifact,
)
from marie.storage import StorageManager
from marie.utils.asset_util import s3_asset_path, split_filename


def _flatten(ocr_pages):
    """Reproduce the writer's contract with the downstream reader: join line
    texts exactly as VectorStoreExecutor._flatten_ocr_pages does."""
    return "".join(
        "\n".join(line["text"] for line in page["lines"]) for page in ocr_pages
    )


# --- losslessness ----------------------------------------------------------

MARKDOWN_SAMPLES = [
    "hello world",
    "# Title\n\nParagraph one.\n\nParagraph two.\n",
    "\n\nleading blanks then text",
    "trailing newline\n",
    "a\n\n\nb",  # consecutive blank lines
    "",  # empty document
    "| col a | col b |\n| --- | --- |\n| 1 | 2 |\n",
]


@pytest.mark.parametrize("markdown", MARKDOWN_SAMPLES)
def test_losslessness_roundtrip(markdown):
    metadata = build_markitdown_metadata(markdown, "md", page_count=1)
    lines = metadata["ocr"][0]["lines"]
    assert "\n".join(line["text"] for line in lines) == markdown


def test_blank_lines_included_as_empty_text():
    metadata = build_markitdown_metadata("a\n\nb", "md", page_count=1)
    texts = [line["text"] for line in metadata["ocr"][0]["lines"]]
    assert texts == ["a", "", "b"]


# --- extraction metadata shape ---------------------------------------------


def test_extraction_metadata_shape_without_fallback():
    metadata = build_markitdown_metadata("abc\ndef", "docx", page_count=3)
    assert metadata["extraction"] == {
        "engine": "markitdown",
        "format": "docx",
        "char_count": len("abc\ndef"),
        "page_count": 3,
    }
    assert "fallback_reason" not in metadata["extraction"]


def test_extraction_metadata_includes_fallback_reason_when_set():
    metadata = build_markitdown_metadata(
        "x", "pdf", page_count=2, fallback_reason="low_text_yield"
    )
    assert metadata["extraction"]["fallback_reason"] == "low_text_yield"


# --- cross-contract: real flattener reproduces the markdown -----------------


@pytest.mark.parametrize("markdown", MARKDOWN_SAMPLES)
def test_real_flattener_reproduces_markdown(markdown):
    metadata = build_markitdown_metadata(markdown, "md", page_count=1)
    full_text, page_ranges = VectorStoreExecutor._flatten_ocr_pages(metadata["ocr"])
    assert full_text == markdown
    assert len(page_ranges) == 1


# --- artifact location ------------------------------------------------------


def test_write_artifact_lands_at_expected_s3_key(monkeypatch):
    ref_id = "tenants/t1/kb-indexes/i1/sources/s1/doc.pdf"
    ref_type = "kb_document"
    markdown = "# Heading\n\nBody line.\n"

    uploaded: dict[str, str] = {}

    def fake_copy_dir(src, dst, relative_to_dir=None, match_wildcard="*"):
        base = relative_to_dir or src
        for root, _, files in os.walk(src):
            for name in files:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, base)
                with open(path) as fh:
                    uploaded[f"{dst}/{rel}"] = fh.read()

    monkeypatch.setattr(StorageManager, "ensure_connection", lambda *a, **k: True)
    monkeypatch.setattr(StorageManager, "copy_dir", staticmethod(fake_copy_dir))
    monkeypatch.setattr(
        StorageManager, "list", staticmethod(lambda *a, **k: list(uploaded))
    )

    write_markitdown_artifact(ref_id, ref_type, markdown, "pdf", page_count=1)

    filename, _, _ = split_filename(ref_id)
    expected_key = f"{s3_asset_path(ref_id, ref_type)}/{filename}.meta.json"
    assert expected_key in uploaded

    # The persisted payload must read back the way the EMBED stage reads it.
    persisted = json.loads(uploaded[expected_key])
    full_text, _ = VectorStoreExecutor._flatten_ocr_pages(persisted["ocr"])
    assert full_text == markdown
    assert persisted["extraction"]["engine"] == "markitdown"
