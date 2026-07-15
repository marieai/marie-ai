import pytest
from pydantic import ValidationError

from marie.extraction import FormatRouter


def _capabilities(*formats: str) -> dict:
    return {
        "schema_version": "1.0",
        "plugin_version": "0.2.0",
        "ready": bool(formats),
        "formats": [
            {
                "canonical_format": canonical,
                "aliases": ["htm"] if canonical == "html" else [],
                "extensions": [canonical],
                "mime_types": [],
                "intents": ["semantic"],
                "result_kinds": ["semantic_document"],
                "providers": ["test-provider"],
            }
            for canonical in formats
        ],
    }


def test_router_ingests_plugin_formats_and_aliases() -> None:
    router = FormatRouter()
    router.ingest_capabilities(_capabilities("docx", "html"))

    assert router.route("DOCX", None, ocr_supported=False) == "plugin"
    assert router.route("htm", None, ocr_supported=False) == "plugin"
    assert router.route("png", None, ocr_supported=True) == "ocr"
    assert router.route("rst", None, ocr_supported=False) == "unsupported"


def test_router_replaces_snapshot_without_stale_formats() -> None:
    router = FormatRouter()
    router.ingest_capabilities(_capabilities("docx"))
    router.ingest_capabilities(_capabilities("html"))

    assert router.plugin_formats == frozenset({"html"})
    assert router.route("docx", None, ocr_supported=False) == "unsupported"


def test_router_ocr_mode_only_routes_ocr_loadable_inputs() -> None:
    router = FormatRouter()
    router.ingest_capabilities(_capabilities("docx", "pdf"))

    assert router.route("pdf", "ocr", ocr_supported=True) == "ocr"
    assert router.route("docx", "ocr", ocr_supported=False) == "unsupported"


def test_router_rejects_an_invalid_snapshot() -> None:
    router = FormatRouter()
    with pytest.raises(ValidationError):
        router.ingest_capabilities({"schema_version": "2.0"})
