from __future__ import annotations

import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from marie.agent.tools.base import ToolOutput
from marie.constants import DEFAULT_DAEMON_ADDR
from marie.plugins.embedded import EmbeddedPlugins, inspect_archive

_PACKAGE = "marie/markitdown"
_PACKAGE_ID = "ext.marie.markitdown"


def _build_plugin_zip(tmp_path: Path, package_id: str = _PACKAGE_ID) -> str:
    manifest = (
        "apiVersion: marie.ai/v1alpha1\n"
        "kind: ExtensionPackage\n"
        "metadata:\n"
        f"  id: {package_id}\n"
        "  author: marie\n"
        "  name: markitdown\n"
        "  version: 0.1.0\n"
        "runtime:\n"
        "  type: python_source\n"
        "  language: python\n"
        "  version: \"3.12\"\n"
        "  entrypoint: main\n"
    )
    zip_path = tmp_path / "marie-markitdown_0.1.0.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("marie-extension.yaml", manifest)
        archive.writestr("main.py", "def main():\n    pass\n")
        archive.writestr("requirements.txt", "markitdown\n")
    return str(zip_path)


def _config(zip_path: str) -> list[dict]:
    return [
        {
            "package": _PACKAGE,
            "path": zip_path,
            "actions": ["convert"],
            "timeout_s": 120,
        }
    ]


def _tool_output(frames: list[dict]) -> ToolOutput:
    return ToolOutput(
        content=json.dumps(frames),
        tool_name="markitdown",
        raw_input={},
        raw_output=frames,
    )


class _FakeClient:
    def __init__(self, url="http://stub", *, invoke_result=None, invoke_error=None):
        self.url = url
        self._invoke_result = invoke_result
        self._invoke_error = invoke_error
        self.closed = False
        self.invocations: list[tuple] = []

    def invoke(self, spec, payload):
        self.invocations.append((spec, payload))
        if self._invoke_error is not None:
            raise self._invoke_error
        return self._invoke_result

    def close(self):
        self.closed = True


def _factory(clients):
    stream = iter(clients)
    calls: list[float] = []

    def factory(timeout_s):
        calls.append(timeout_s)
        return next(stream)

    factory.calls = calls
    return factory


_CANNED_INSTALL = {
    "install": {"packageRef": _PACKAGE_ID, "digest": "sha256:canned", "state": "ready"},
    "state": "ready",
}


def _started(zip_path, clients, *, install_response=None):
    """EmbeddedPlugins with the install HTTP stubbed out (no real daemon)."""
    factory = _factory(clients)
    plugins = EmbeddedPlugins(_config(zip_path), "extract_executor", client_factory=factory)
    plugins._post_install = lambda client, envelope, archive, timeout_s: (
        install_response or _CANNED_INSTALL
    )
    return plugins, factory


def test_config_parsing(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    plugins = EmbeddedPlugins(_config(zip_path), "extract_executor")
    assert plugins.configured_packages == [_PACKAGE]
    entry = plugins._entries[0]
    assert entry.package == _PACKAGE
    assert entry.path == zip_path
    assert entry.actions == ["convert"]
    assert entry.timeout_s == 120


def test_config_parsing_defaults_timeout(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    plugins = EmbeddedPlugins([{"package": _PACKAGE, "path": zip_path}], "x")
    entry = plugins._entries[0]
    assert entry.timeout_s == 120
    assert entry.actions == []


def test_config_entry_requires_package_and_path():
    with pytest.raises(ValueError, match="package.*path"):
        EmbeddedPlugins([{"package": _PACKAGE}], "x")


def test_empty_config_is_lazy_and_ensure_started_raises():
    plugins = EmbeddedPlugins([], "x")
    assert plugins.configured_packages == []
    with pytest.raises(RuntimeError, match="no plugins configured"):
        plugins.ensure_started()


def test_lazy_start_no_client_until_ensure_started(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    factory = _factory([_FakeClient()])
    plugins = EmbeddedPlugins(_config(zip_path), "x", client_factory=factory)
    assert factory.calls == []
    assert plugins._client is None


def test_install_flow_posts_raw_bytes_and_envelope_claims(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    expected_ref, expected_digest = inspect_archive(zip_path)
    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (http.server API)
            length = int(self.headers["Content-Length"])
            captured["path"] = self.path
            captured["content_type"] = self.headers["Content-Type"]
            captured["envelope"] = json.loads(self.headers["X-Marie-Envelope"])
            captured["body"] = self.rfile.read(length)
            body = json.dumps(
                {
                    "install": {
                        "packageRef": expected_ref,
                        "digest": expected_digest,
                        "state": "ready",
                    },
                    "state": "ready",
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence the stub server
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        client = _FakeClient(url=url)
        plugins = EmbeddedPlugins(
            _config(zip_path), "extract_executor", client_factory=_factory([client])
        )
        plugins.ensure_started()
    finally:
        server.shutdown()

    assert captured["path"] == "/v1/plugins/install"
    assert captured["content_type"] == "application/zip"
    assert captured["body"] == Path(zip_path).read_bytes()
    assert captured["envelope"]["packageRef"] == expected_ref
    assert captured["envelope"]["packageDigest"] == expected_digest
    assert captured["envelope"]["signature"]["algorithm"] == "hmac-sha256"

    spec = plugins._specs[_PACKAGE]
    assert spec.package_ref == expected_ref
    assert spec.package_digest == expected_digest


def test_spec_identity_fields_deterministic(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    plugins, _ = _started(zip_path, [_FakeClient()])
    plugins.ensure_started()
    assert (
        plugins._install_ids[_PACKAGE]
        == f"extract_executor@{DEFAULT_DAEMON_ADDR}/{_PACKAGE}"
    )
    spec = plugins._specs[_PACKAGE]
    assert spec.install_id == f"extract_executor@{DEFAULT_DAEMON_ADDR}/{_PACKAGE}"
    assert spec.package_ref == _PACKAGE_ID


def test_invoke_delegates_and_extracts_stream(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    frames = [
        {"type": "stream", "data": {"markdown": "# Hi", "metadata": {"page_count": 2}}},
        {"type": "end"},
    ]
    client = _FakeClient(invoke_result=_tool_output(frames))
    plugins, factory = _started(zip_path, [client])

    result = plugins.invoke(_PACKAGE, "convert", {"path": "/x.pdf", "format": "pdf"})

    assert result == {"markdown": "# Hi", "metadata": {"page_count": 2}}
    assert factory.calls == [120]  # started once
    spec, payload = client.invocations[-1]
    assert payload["action"] == "convert"
    assert payload["path"] == "/x.pdf"
    assert payload["format"] == "pdf"
    assert spec.package_ref == _PACKAGE_ID


def test_invoke_raises_on_error_frame(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    frames = [{"type": "error", "data": {"message": "conversion failed"}}]
    client = _FakeClient(invoke_result=_tool_output(frames))
    plugins, _ = _started(zip_path, [client])

    with pytest.raises(RuntimeError, match="conversion failed"):
        plugins.invoke(_PACKAGE, "convert", {"path": "/x.pdf"})


def test_invoke_raises_when_no_stream_result(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    client = _FakeClient(invoke_result=_tool_output([{"type": "end"}]))
    plugins, _ = _started(zip_path, [client])

    with pytest.raises(RuntimeError, match="no stream result"):
        plugins.invoke(_PACKAGE, "convert", {"path": "/x.pdf"})


def test_close_terminates_child(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    client = _FakeClient()
    plugins, _ = _started(zip_path, [client])
    plugins.ensure_started()
    assert client.closed is False

    plugins.close()
    assert client.closed is True
    assert plugins._client is None


def test_crash_respawns_once_then_raises(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    dead_first = _FakeClient(invoke_error=RuntimeError("dead"))
    dead_second = _FakeClient(invoke_error=RuntimeError("still dead"))
    plugins, factory = _started(zip_path, [dead_first, dead_second])

    with pytest.raises(RuntimeError, match="still dead"):
        plugins.invoke(_PACKAGE, "convert", {"path": "/x.pdf"})

    assert len(factory.calls) == 2  # respawned exactly once
    assert dead_first.closed is True


def test_crash_respawns_once_then_succeeds(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    frames = [{"type": "stream", "data": {"markdown": "ok", "metadata": {}}}, {"type": "end"}]
    dead = _FakeClient(invoke_error=RuntimeError("dead"))
    healthy = _FakeClient(invoke_result=_tool_output(frames))
    plugins, factory = _started(zip_path, [dead, healthy])

    result = plugins.invoke(_PACKAGE, "convert", {"path": "/x.pdf"})

    assert result == {"markdown": "ok", "metadata": {}}
    assert len(factory.calls) == 2
    assert dead.closed is True


def test_inspect_archive_matches_daemon_algorithm(tmp_path):
    # Digest is stable and prefixed; ref is the manifest metadata.id. (Byte-for-byte
    # parity with the Go daemon's plugin_manager.Inspect is proven in Task 6's E2E.)
    zip_path = _build_plugin_zip(tmp_path)
    ref, digest = inspect_archive(zip_path)
    assert ref == _PACKAGE_ID
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
