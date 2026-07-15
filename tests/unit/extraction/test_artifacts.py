import hashlib
from pathlib import Path

import pytest

from marie.extraction.artifacts import read_extraction_artifact
from marie.extraction.models import ExtractionSuccess, parse_extraction_result
from marie.extraction.result_adapter import build_extraction_metadata


def _result(path: Path, data: bytes, **artifact_overrides) -> ExtractionSuccess:
    artifact = {
        "path": str(path),
        "media_type": "text/markdown",
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "role": "document",
        **artifact_overrides,
    }
    result = parse_extraction_result(
        {
            "schema_version": "1.0",
            "outcome": "success",
            "result_kind": "semantic_document",
            "artifact": artifact,
            "provenance": {
                "provider": "docling",
                "provider_version": "2.111.0",
                "canonical_format": "docx",
                "backend": "MsWordDocumentBackend",
            },
            "metadata": {"page_count": 2},
            "warnings": ["warning"],
        }
    )
    assert isinstance(result, ExtractionSuccess)
    return result


def test_read_extraction_artifact_validates_and_loads_content(tmp_path):
    data = b"# Extracted\n\nBody"
    path = tmp_path / "document.md"
    path.write_bytes(data)

    assert read_extraction_artifact(_result(path, data), str(tmp_path)) == data.decode()


def test_read_extraction_artifact_rejects_path_escape(tmp_path):
    outside = tmp_path.parent / "outside.md"
    outside.write_text("outside")

    with pytest.raises(ValueError, match="outside"):
        read_extraction_artifact(_result(outside, outside.read_bytes()), str(tmp_path))


def test_read_extraction_artifact_rejects_digest_mismatch(tmp_path):
    data = b"content"
    path = tmp_path / "document.md"
    path.write_bytes(data)

    with pytest.raises(ValueError, match="digest"):
        read_extraction_artifact(_result(path, data, sha256="0" * 64), str(tmp_path))


def test_result_adapter_preserves_provider_provenance(tmp_path):
    data = b"first\nsecond"
    path = tmp_path / "document.md"
    path.write_bytes(data)
    result = _result(path, data)

    metadata = build_extraction_metadata(data.decode(), result)

    assert metadata["extraction"] == {
        "engine": "docling",
        "provider": "docling",
        "provider_version": "2.111.0",
        "backend": "MsWordDocumentBackend",
        "format": "docx",
        "result_kind": "semantic_document",
        "route": "plugin",
        "ocr_invoked": False,
        "char_count": 12,
        "page_count": 2,
        "warnings": ["warning"],
    }
    assert metadata["ocr"] == [{"lines": [{"text": "first"}, {"text": "second"}]}]
    assert metadata["extraction"]["route"] == "plugin"
    assert metadata["extraction"]["ocr_invoked"] is False
