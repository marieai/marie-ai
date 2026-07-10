"""Unit tests for the marie/markitdown plugin.

The plugin lives outside the ``marie`` package (it ships as a standalone daemon
package), so load ``main.py`` by path. markitdown is not installed in the repo
venv, so conversion is driven through a stubbed converter; a real-markitdown
test runs only when the library happens to be importable.
"""

import importlib.util
import zipfile
from pathlib import Path

import pytest
import yaml

PLUGIN_DIR = Path(__file__).resolve().parents[3] / "plugins" / "markitdown"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "marie_markitdown_plugin_under_test", PLUGIN_DIR / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()


class _StubResult:
    def __init__(self, text, title=None):
        self.text_content = text
        self.title = title


class _StubConverter:
    def __init__(self, text="# Heading\n\nbody", title=None):
        self._result = _StubResult(text, title)
        self.calls = []

    def convert(self, path):
        self.calls.append(path)
        return self._result


def _stub_factory(**kwargs):
    converter = _StubConverter(**kwargs)
    return lambda: converter, converter


@pytest.fixture
def docx_file(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_paragraph("Hello from a born-digital document.")
    document.save(path)
    return path


@pytest.fixture
def html_file(tmp_path):
    path = tmp_path / "sample.html"
    path.write_text("<html><body><h1>Title</h1><p>Paragraph.</p></body></html>")
    return path


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("a,b,c\n1,2,3\n")
    return path


# --- convert_document wiring -------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["docx_file", "html_file", "csv_file"])
def test_convert_document_returns_markdown_and_metadata(request, fixture_name):
    path = request.getfixturevalue(fixture_name)
    factory, converter = _stub_factory(text="# Doc\n\ncontent")
    result = plugin.convert_document(str(path), converter_factory=factory)
    assert result["markdown"] == "# Doc\n\ncontent"
    assert isinstance(result["metadata"], dict)
    assert converter.calls == [str(path)]


def test_convert_document_missing_path_raises():
    with pytest.raises(ValueError):
        plugin.convert_document("", converter_factory=lambda: _StubConverter())


def test_convert_document_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plugin.convert_document(
            str(tmp_path / "nope.docx"), converter_factory=lambda: _StubConverter()
        )


def test_metadata_title_from_result(html_file):
    factory, _ = _stub_factory(text="body", title="My Title")
    result = plugin.convert_document(str(html_file), converter_factory=factory)
    assert result["metadata"]["title"] == "My Title"


def test_metadata_page_count_only_for_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(plugin, "_pdf_page_count", lambda path: 7)
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    factory, _ = _stub_factory(text="pdf text")
    result = plugin.convert_document(str(pdf), fmt="pdf", converter_factory=factory)
    assert result["metadata"]["page_count"] == 7

    html = tmp_path / "sample.html"
    html.write_text("<p>x</p>")
    factory, _ = _stub_factory(text="x")
    result = plugin.convert_document(str(html), converter_factory=factory)
    assert "page_count" not in result["metadata"]


# --- input extraction --------------------------------------------------------


def test_extract_input_flat():
    assert plugin._extract_input({"path": "/x", "format": "pdf"}) == {
        "path": "/x",
        "format": "pdf",
    }


def test_extract_input_tool_parameters_wrapper():
    payload = {"tool_parameters": {"path": "/y"}, "user_id": "u", "credentials": {}}
    assert plugin._extract_input(payload) == {"path": "/y"}


def test_extract_input_non_dict():
    assert plugin._extract_input(None) == {}


# --- protocol dispatch -------------------------------------------------------


def test_dispatch_convert_emits_stream_then_end(html_file):
    factory, _ = _stub_factory(text="# md\n\ntext")
    request = {
        "session_id": "s1",
        "event": "request",
        "data": {"path": str(html_file), "format": "html"},
    }
    events = plugin.dispatch_request(request, converter_factory=factory)
    assert [e["data"]["type"] for e in events] == ["stream", "end"]
    stream = events[0]
    assert stream["session_id"] == "s1"
    assert stream["event"] == "session"
    assert stream["data"]["data"]["markdown"] == "# md\n\ntext"
    assert "metadata" in stream["data"]["data"]
    assert events[1]["data"]["data"] == {}


def test_dispatch_unknown_action_emits_error():
    request = {"session_id": "s2", "event": "request", "data": {"action": "delete"}}
    events = plugin.dispatch_request(request)
    assert len(events) == 1
    assert events[0]["data"]["type"] == "error"
    assert "delete" in events[0]["data"]["data"]["message"]


def test_dispatch_conversion_failure_emits_error(tmp_path):
    request = {
        "session_id": "s3",
        "event": "request",
        "data": {"path": str(tmp_path / "missing.pdf")},
    }
    events = plugin.dispatch_request(request)
    assert len(events) == 1
    assert events[0]["data"]["type"] == "error"


# --- packaging & manifest ----------------------------------------------------


def test_manifest_shape():
    manifest = yaml.safe_load((PLUGIN_DIR / "marie-extension.yaml").read_text())
    assert manifest["kind"] == "ExtensionPackage"
    assert manifest["metadata"]["id"] == "ext.marie.markitdown"
    assert manifest["runtime"]["type"] == "python_source"
    assert manifest["runtime"]["entrypoint"] == "main"


def test_requirements_declare_markitdown():
    reqs = (PLUGIN_DIR / "requirements.txt").read_text()
    assert "markitdown" in reqs


def test_package_script_builds_decodable_zip(tmp_path):
    import subprocess

    out = subprocess.run(
        ["bash", str(PLUGIN_DIR / "scripts" / "package.sh"), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    archives = list(tmp_path.glob("marie-markitdown_*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as zf:
        names = zf.namelist()
    manifests = [n for n in names if n.endswith("marie-extension.yaml")]
    assert manifests == ["marie-extension.yaml"]
    assert "main.py" in names
    assert "requirements.txt" in names


# --- real markitdown (only if installed) -------------------------------------


def test_real_markitdown_converts_html(html_file):
    pytest.importorskip("markitdown")
    result = plugin.convert_document(str(html_file), fmt="html")
    assert result["markdown"].strip()
    assert isinstance(result["metadata"], dict)
