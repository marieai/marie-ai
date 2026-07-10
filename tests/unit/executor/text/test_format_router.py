import logging

import pytest
from docarray import DocList

from marie.api.docs import AssetKeyDoc
from marie.executor.text import text_extraction_executor as tee
from marie.executor.text.format_router import FormatRouter, pdf_yield_ok
from marie.executor.text.text_extraction_executor import TextExtractionExecutor

# --- routing matrix --------------------------------------------------------

DIGITAL_FORMATS = [
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "html",
    "htm",
    "md",
    "markdown",
    "csv",
    "rtf",
    "odt",
    "epub",
    "eml",
    "msg",
]

IMAGE_FORMATS = ["tiff", "png", "jpeg", "jpg", "webp", "bmp", "gif", "heif", "djvu"]

# Formats intentionally left on the OCR/rasterize path.
OTHER_OCR_FORMATS = ["doc", "xls", "ppt", "ods", "odp", "xml", "tsv", "latex", "rst", ""]


@pytest.mark.parametrize("fmt", DIGITAL_FORMATS)
def test_digital_formats_route_to_markitdown(fmt):
    assert FormatRouter.route(fmt, parse_mode=None) == "markitdown"


@pytest.mark.parametrize("fmt", IMAGE_FORMATS + OTHER_OCR_FORMATS)
def test_non_digital_formats_route_to_ocr(fmt):
    assert FormatRouter.route(fmt, parse_mode=None) == "ocr"


def test_route_is_case_insensitive():
    assert FormatRouter.route("PDF", parse_mode=None) == "markitdown"
    assert FormatRouter.route("DOCX", parse_mode=None) == "markitdown"


@pytest.mark.parametrize("fmt", DIGITAL_FORMATS)
def test_parse_mode_ocr_forces_ocr(fmt):
    assert FormatRouter.route(fmt, parse_mode="ocr") == "ocr"


def test_parse_mode_other_values_do_not_force_ocr():
    assert FormatRouter.route("pdf", parse_mode="auto") == "markitdown"
    assert FormatRouter.route("pdf", parse_mode="") == "markitdown"


# --- pdf yield floor -------------------------------------------------------


@pytest.mark.parametrize(
    "chars,pages,expected",
    [
        (199, 1, False),
        (200, 1, True),
        (201, 1, True),
        (399, 2, False),
        (400, 2, True),
        (401, 2, True),
    ],
)
def test_pdf_yield_floor_boundary(chars, pages, expected):
    assert pdf_yield_ok("x" * chars, pages, floor_chars_per_page=200) is expected


def test_pdf_yield_page_count_clamped_to_one():
    # A zero/None page_count must not divide the floor away.
    assert pdf_yield_ok("x" * 199, 0, floor_chars_per_page=200) is False
    assert pdf_yield_ok("x" * 200, 0, floor_chars_per_page=200) is True
    assert pdf_yield_ok("x" * 200, None, floor_chars_per_page=200) is True


def test_pdf_yield_empty_markdown_fails():
    assert pdf_yield_ok("", 1) is False
    assert pdf_yield_ok(None, 3) is False


def test_pdf_yield_custom_floor():
    assert pdf_yield_ok("x" * 50, 1, floor_chars_per_page=50) is True
    assert pdf_yield_ok("x" * 49, 1, floor_chars_per_page=50) is False


# --- extract() integration shape (no models) -------------------------------


class _StubPlugins:
    def __init__(self, result, packages=("marie/markitdown",)):
        self._result = result
        self.configured_packages = list(packages)
        self.invoked = []

    def invoke(self, package, action, payload):
        self.invoked.append((package, action, payload))
        return self._result


class _StubPipeline:
    def __init__(self, metadata):
        self.pipeline_name = "default"
        self._metadata = metadata
        self.executed = False

    def execute(self, **kwargs):
        self.executed = True
        return self._metadata


def _bare_executor(
    *,
    plugins,
    pipeline=None,
    markitdown_enabled=True,
    floor=200,
):
    """Construct a TextExtractionExecutor without running __init__ (no models)."""
    ex = object.__new__(TextExtractionExecutor)
    ex.logger = logging.getLogger("test-extract")
    ex.runtime_info = {"name": "test"}
    ex.embedded_plugins = plugins
    ex.pipeline = pipeline
    ex._markitdown_enabled = markitdown_enabled
    ex._markitdown_floor = floor
    ex._markitdown_gate_warned = False
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


def test_extract_markitdown_success_shape(monkeypatch):
    plugins = _StubPlugins(
        {"markdown": "# Title\n\nBody paragraph.", "metadata": {"page_count": 1}}
    )
    ex = _bare_executor(plugins=plugins, pipeline=_StubPipeline({"ocr": []}))

    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/doc.docx", "docx")
    )
    written = {}
    monkeypatch.setattr(
        tee,
        "write_markitdown_artifact",
        lambda ref_id, ref_type, markdown, fmt, page_count, **kw: written.update(
            ref_id=ref_id, markdown=markdown, fmt=fmt, page_count=page_count
        ),
    )

    response = ex.extract(_docs("s3://bucket/doc.docx"), _params())

    assert response["status"] == "succeeded"
    assert response["metadata"]["extraction"]["engine"] == "markitdown"
    assert response["metadata"]["extraction"]["format"] == "docx"
    assert written["ref_id"] == "ref-1"
    assert written["fmt"] == "docx"
    assert plugins.invoked == [
        ("marie/markitdown", "convert", {"path": "/tmp/doc.docx", "format": "docx"})
    ]
    assert ex.pipeline.executed is False  # OCR pipeline was bypassed
    assert ex.persisted  # bookkeeping preserved on the markitdown branch


def test_extract_gate_off_routes_to_ocr(monkeypatch):
    plugins = _StubPlugins({"markdown": "unused"})
    pipeline = _StubPipeline({"ocr": [{"lines": []}]})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline, markitdown_enabled=False)

    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/doc.docx", "docx")
    )
    monkeypatch.setattr(tee, "get_frames_from_file", lambda path, pages: ["frame"])

    response = ex.extract(_docs("s3://bucket/doc.docx"), _params())

    assert response["status"] == "succeeded"
    assert pipeline.executed is True  # fell to OCR
    assert plugins.invoked == []  # plugin never called when gate is off


def test_extract_pdf_low_yield_falls_back_to_ocr(monkeypatch):
    # Below-floor PDF markdown -> markitdown returns None internally -> OCR.
    plugins = _StubPlugins({"markdown": "x" * 10, "metadata": {"page_count": 1}})
    pipeline = _StubPipeline({"ocr": [{"lines": []}]})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline)

    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/scan.pdf", "pdf")
    )
    monkeypatch.setattr(tee, "get_frames_from_file", lambda path, pages: ["frame"])
    write_called = []
    monkeypatch.setattr(
        tee,
        "write_markitdown_artifact",
        lambda *a, **k: write_called.append(True),
    )

    response = ex.extract(_docs("s3://bucket/scan.pdf"), _params())

    assert response["status"] == "succeeded"
    assert plugins.invoked  # markitdown was attempted
    assert write_called == []  # no artifact written on fallback
    assert pipeline.executed is True  # OCR ran on the same download


def test_extract_non_pdf_empty_markdown_fails_loud(monkeypatch):
    plugins = _StubPlugins({"markdown": "   ", "metadata": {}})
    ex = _bare_executor(plugins=plugins, pipeline=_StubPipeline({}))

    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/doc.html", "html")
    )

    response = ex.extract(_docs("s3://bucket/doc.html"), _params())

    # The raise is caught by extract()'s error handler and surfaced as error.
    assert response["status"] == "error"


def test_extract_parse_mode_forces_ocr(monkeypatch):
    plugins = _StubPlugins({"markdown": "should not be used"})
    pipeline = _StubPipeline({"ocr": [{"lines": []}]})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline)

    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/doc.docx", "docx")
    )
    monkeypatch.setattr(tee, "get_frames_from_file", lambda path, pages: ["frame"])

    response = ex.extract(
        _docs("s3://bucket/doc.docx"),
        _params(run_params={"parse_mode": "ocr"}),
    )

    assert response["status"] == "succeeded"
    assert plugins.invoked == []
    assert pipeline.executed is True


def test_extract_asset_fetch_failure_raises(monkeypatch):
    # A transport failure (e.g. transient S3 read) must propagate as an
    # uncaught exception so the executor lifecycle records has_exception and the
    # scheduler retries the job -- NOT be flattened into a {status: "error"}
    # response that the job layer treats as a completed call.
    plugins = _StubPlugins({"markdown": "unused"})
    ex = _bare_executor(plugins=plugins, pipeline=_StubPipeline({"ocr": []}))

    def _boom(key):
        raise OSError("transient S3 read failure")

    monkeypatch.setattr(tee, "fetch_asset_to_temp", _boom)

    with pytest.raises(OSError, match="transient S3 read failure"):
        ex.extract(_docs("s3://bucket/doc.docx"), _params())

    assert plugins.invoked == []  # routing never reached


def test_extract_frame_parse_failure_raises(monkeypatch):
    # OCR pre-processing (frame rasterization) sits outside the inference try:
    # a parse failure raises uncaught rather than returning {status: "error"}.
    plugins = _StubPlugins({"markdown": "unused"})
    pipeline = _StubPipeline({"ocr": [{"lines": []}]})
    ex = _bare_executor(plugins=plugins, pipeline=pipeline, markitdown_enabled=False)

    monkeypatch.setattr(
        tee, "fetch_asset_to_temp", lambda key: ("/tmp/doc.docx", "docx")
    )

    def _boom(path, pages):
        raise ValueError("malformed document")

    monkeypatch.setattr(tee, "get_frames_from_file", _boom)

    with pytest.raises(ValueError, match="malformed document"):
        ex.extract(_docs("s3://bucket/doc.docx"), _params())

    assert pipeline.executed is False  # inference never reached
