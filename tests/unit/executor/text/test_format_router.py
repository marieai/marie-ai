import hashlib
import logging
from pathlib import Path

import pytest
from docarray import DocList

from marie.api.docs import AssetKeyDoc
from marie.executor.text import text_extraction_executor as tee
from marie.executor.text.text_extraction_executor import TextExtractionExecutor
from marie.extraction import FormatRouter

PACKAGE = "marie/document-extraction"


def _capabilities(*formats):
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


class _StubPlugins:
    def __init__(
        self,
        *,
        formats=("docx", "pdf"),
        content="# Title\n\nBody paragraph.",
        not_extractable=False,
        packages=(PACKAGE,),
    ):
        self._snapshot = _capabilities(*formats)
        self._content = content
        self._not_extractable = not_extractable
        self.configured_packages = list(packages)
        self.capability_requests = []
        self.invoked = []
        self.runtime_generation = 1

    def capabilities(self, package):
        self.capability_requests.append(package)
        return self._snapshot

    def invoke(self, package, action, payload):
        self.invoked.append((package, action, payload))
        if self._not_extractable:
            return {
                "schema_version": "1.0",
                "outcome": "not_extractable",
                "canonical_format": payload["format"],
                "reason": "providers_exhausted",
                "attempted_providers": ["test-provider"],
                "warnings": [],
            }

        artifact = Path(payload["output_dir"]) / "document.md"
        data = self._content.encode()
        artifact.write_bytes(data)
        return {
            "schema_version": "1.0",
            "outcome": "success",
            "result_kind": "semantic_document",
            "artifact": {
                "path": str(artifact),
                "media_type": "text/markdown",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "role": "document",
            },
            "provenance": {
                "provider": "test-provider",
                "provider_version": "1.2.3",
                "canonical_format": payload["format"],
                "backend": "TestBackend",
            },
            "metadata": {"page_count": 1},
            "warnings": [],
        }


class _StubPipeline:
    def __init__(self, metadata):
        self.pipeline_name = "default"
        self._metadata = metadata
        self.executed = False

    def execute(self, **kwargs):
        self.executed = True
        return self._metadata


def _bare_executor(*, plugins, pipeline=None, enabled=True):
    ex = object.__new__(TextExtractionExecutor)
    ex.logger = logging.getLogger("test-extract")
    ex.runtime_info = {"name": "test"}
    ex.embedded_plugins = plugins
    ex.pipeline = pipeline
    ex._document_extraction_enabled = enabled
    ex._document_extraction_package = PACKAGE
    ex._capabilities_loaded = False
    ex._capability_generation = None
    ex._document_extraction_gate_warned = False
    ex.format_router = FormatRouter()
    ex.show_error = True
    ex.persisted = []
    ex.persist = lambda **kwargs: ex.persisted.append(kwargs)
    return ex


def _docs(asset_key):
    return DocList[AssetKeyDoc]([AssetKeyDoc(asset_key=asset_key, pages=None)])


def _params(**overrides):
    base = {"job_id": "job-1", "ref_id": "ref-1", "ref_type": "extract", "payload": {}}
    base.update(overrides)
    return base


def test_extract_plugin_success_is_terminal(monkeypatch):
    plugins = _StubPlugins(formats=("docx",))
    pipeline = _StubPipeline({"ocr": []})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline)
    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/doc.docx", "docx")
    )
    written = []
    monkeypatch.setattr(
        tee,
        "write_extraction_metadata",
        lambda ref_id, ref_type, metadata: written.append(metadata),
    )

    response = ex.extract(_docs("s3://bucket/doc.docx"), _params())

    assert response["status"] == "succeeded"
    assert response["metadata"]["extraction"]["engine"] == "test-provider"
    assert response["metadata"]["extraction"]["format"] == "docx"
    assert plugins.capability_requests == [PACKAGE]
    assert plugins.invoked[0][0:2] == (PACKAGE, "extract")
    assert "output_dir" in plugins.invoked[0][2]
    assert pipeline.executed is False
    assert written and ex.persisted


def test_executor_replaces_capabilities_after_runtime_restart():
    plugins = _StubPlugins(formats=("docx",))
    ex = _bare_executor(plugins=plugins)
    ex._load_document_extraction_capabilities()
    assert ex.format_router.plugin_formats == frozenset({"docx"})

    plugins._snapshot = _capabilities("html")
    plugins.runtime_generation += 1
    ex._load_document_extraction_capabilities()

    assert ex.format_router.plugin_formats == frozenset({"html"})
    assert plugins.capability_requests == [PACKAGE, PACKAGE]


def test_low_text_pdf_plugin_success_does_not_run_ocr(monkeypatch):
    plugins = _StubPlugins(formats=("pdf",), content="x")
    pipeline = _StubPipeline({"ocr": [{"lines": []}]})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline)
    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/scan.pdf", "pdf")
    )
    monkeypatch.setattr(tee, "write_extraction_metadata", lambda *args: None)

    response = ex.extract(_docs("s3://bucket/scan.pdf"), _params())

    assert response["status"] == "succeeded"
    assert pipeline.executed is False


def test_not_extractable_pdf_falls_back_to_ocr(monkeypatch):
    plugins = _StubPlugins(formats=("pdf",), not_extractable=True)
    pipeline = _StubPipeline({"ocr": [{"lines": []}]})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline)
    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/scan.pdf", "pdf")
    )
    monkeypatch.setattr(tee, "get_frames_from_file", lambda path, pages: ["frame"])

    response = ex.extract(_docs("s3://bucket/scan.pdf"), _params())

    assert response["status"] == "succeeded"
    assert plugins.invoked
    assert pipeline.executed is True


def test_not_extractable_semantic_document_does_not_run_ocr(monkeypatch):
    plugins = _StubPlugins(formats=("docx",), not_extractable=True)
    pipeline = _StubPipeline({"ocr": [{"lines": []}]})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline)
    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/doc.docx", "docx")
    )

    response = ex.extract(_docs("s3://bucket/doc.docx"), _params())

    assert response["status"] == "error"
    assert pipeline.executed is False


def test_unavailable_plugin_rejects_non_ocr_format(monkeypatch):
    plugins = _StubPlugins(packages=())
    pipeline = _StubPipeline({"ocr": [{"lines": []}]})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline)
    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/doc.docx", "docx")
    )

    response = ex.extract(_docs("s3://bucket/doc.docx"), _params())

    assert response["status"] == "error"
    assert plugins.capability_requests == []
    assert pipeline.executed is False


def test_parse_mode_forces_ocr_without_loading_capabilities(monkeypatch):
    plugins = _StubPlugins(formats=("pdf",))
    pipeline = _StubPipeline({"ocr": [{"lines": []}]})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline)
    monkeypatch.setattr(tee, "fetch_asset_to_temp", lambda key: ("/tmp/doc.pdf", "pdf"))
    monkeypatch.setattr(tee, "get_frames_from_file", lambda path, pages: ["frame"])

    response = ex.extract(
        _docs("s3://bucket/doc.pdf"),
        _params(run_params={"parse_mode": "ocr"}),
    )

    assert response["status"] == "succeeded"
    assert plugins.capability_requests == []
    assert plugins.invoked == []
    assert pipeline.executed is True


def test_extract_asset_fetch_failure_raises(monkeypatch):
    plugins = _StubPlugins()
    ex = _bare_executor(plugins=plugins, pipeline=_StubPipeline({"ocr": []}))

    def _boom(key):
        raise OSError("transient S3 read failure")

    monkeypatch.setattr(tee, "fetch_asset_to_temp", _boom)
    with pytest.raises(OSError, match="transient S3 read failure"):
        ex.extract(_docs("s3://bucket/doc.docx"), _params())


def test_extract_frame_parse_failure_raises(monkeypatch):
    plugins = _StubPlugins(packages=())
    pipeline = _StubPipeline({"ocr": [{"lines": []}]})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline)
    monkeypatch.setattr(tee, "fetch_asset_to_temp", lambda key: ("/tmp/doc.png", "png"))

    def _boom(path, pages):
        raise ValueError("malformed document")

    monkeypatch.setattr(tee, "get_frames_from_file", _boom)
    with pytest.raises(ValueError, match="malformed document"):
        ex.extract(_docs("s3://bucket/doc.png"), _params())
    assert pipeline.executed is False
