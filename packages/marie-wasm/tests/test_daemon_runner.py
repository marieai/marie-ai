"""Tests for marie.wasm.daemon_runner — running the built-in http-request node
as a daemon-managed plugin (load_config, run_execute, and the stdio protocol).
"""

import http.server
import io
import json
import shutil
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("wasmtime")
pytest.importorskip("wasmtime.component")

from marie.wasm import BUILTIN_NODES_DIR, daemon_runner  # noqa: E402

FIXTURE_WASM = BUILTIN_NODES_DIR / "http-request.wasm"


def _start_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


def _make_plugin_dir(tmp_path: Path) -> Path:
    d = tmp_path / "plugin"
    d.mkdir()
    shutil.copy(FIXTURE_WASM, d / "node.wasm")
    (d / "marie-extension.yaml").write_text(
        "apiVersion: marie.ai/v1alpha1\n"
        "kind: ExtensionPackage\n"
        "metadata:\n  id: ext.test.http-wasm\n  name: http-wasm\n  version: 0.0.1\n"
        "runtime:\n  type: python_source\n  language: python\n  version: '3.12'\n"
        "  entrypoint: marie.wasm.daemon_runner\n  engine: python-wasmtime\n"
        "  module: node.wasm\n  permissions: http-request\n"
    )
    return d


@pytest.mark.skipif(not FIXTURE_WASM.exists(), reason="built-in node missing")
def test_load_config(tmp_path):
    d = _make_plugin_dir(tmp_path)
    cfg = daemon_runner.load_config(str(d))
    assert cfg.module_path.endswith("node.wasm")
    assert cfg.permissions.allow_http is True  # http-request preset


@pytest.mark.skipif(not FIXTURE_WASM.exists(), reason="built-in node missing")
def test_run_execute_against_local_http(tmp_path):
    srv, url = _start_server()
    try:
        d = _make_plugin_dir(tmp_path)
        cfg = daemon_runner.load_config(str(d))
        engine, component = daemon_runner.load_component(cfg)
        payload = {
            "input": [],
            "env": json.dumps({"method": "GET", "url": url, "headers": {}}),
            "ctx": {
                "workflow_id": "wf",
                "execution_id": "ex",
                "node_id": "n1",
                "run_index": 0,
            },
        }
        result = daemon_runner.run_execute(engine, component, cfg, payload)
        assert "items" in result, result
        out = json.loads(result["items"][0]["json"])
        assert out["status"] == 200
        assert out["success"] is True
    finally:
        srv.shutdown()


@pytest.mark.skipif(not FIXTURE_WASM.exists(), reason="built-in node missing")
def test_stdio_protocol_emits_heartbeat_stream_end(tmp_path, monkeypatch):
    srv, url = _start_server()
    try:
        d = _make_plugin_dir(tmp_path)
        monkeypatch.chdir(d)  # main() reads the manifest from cwd

        request = {
            "session_id": "s1",
            "event": "request",
            "conversation_id": None,
            "message_id": None,
            "app_id": None,
            "endpoint_id": None,
            "context": {},
            "data": {
                "input": [],
                "env": json.dumps({"method": "GET", "url": url, "headers": {}}),
                "ctx": {
                    "workflow_id": "wf",
                    "execution_id": "ex",
                    "node_id": "n1",
                    "run_index": 0,
                },
            },
        }
        stdin = io.StringIO(json.dumps(request) + "\n")

        class CollectingOut:
            def __init__(self):
                self.lines = []

            def write(self, s):
                if s.strip():
                    self.lines.append(s)

            def flush(self):
                pass

        out = CollectingOut()
        t = threading.Thread(target=daemon_runner.main, args=(stdin, out), daemon=True)
        t.start()
        t.join(timeout=30)

        events = [json.loads(ln) for ln in out.lines]
        kinds = [
            (
                e.get("event"),
                (
                    (e.get("data") or {}).get("type")
                    if isinstance(e.get("data"), dict)
                    else None
                ),
            )
            for e in events
        ]
        assert ("heartbeat", None) in kinds, kinds
        stream = [
            e
            for e in events
            if e.get("event") == "session" and e["data"]["type"] == "stream"
        ]
        end = [
            e
            for e in events
            if e.get("event") == "session" and e["data"]["type"] == "end"
        ]
        assert stream and end, kinds
        out_json = json.loads(stream[0]["data"]["data"]["json"])
        assert out_json["status"] == 200
    finally:
        srv.shutdown()
