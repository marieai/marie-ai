import asyncio
import json
import socket
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen

from docarray import DocList
from docarray.documents import TextDoc

from marie.executor.extensions.plugin_daemon_executor import (
    MariePluginDaemonExecutor,
    discover_daemon,
)


class HealthHandler(BaseHTTPRequestHandler):
    stub_payloads: list[dict] = []

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return

        body = json.dumps(
            {
                "ok": True,
                "ready": True,
                "version": "0.1.0-test",
                "mode": "decode_only",
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/runtime/stub-invocations":
            self.send_response(404)
            self.end_headers()
            return

        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.__class__.stub_payloads.append(json.loads(raw.decode("utf-8")))
        body = "\n".join(
            [
                'data: {"type":"text","payload":"hello","sequence":0}',
                'data: {"type":"structured_object","payload":{"ok":true},"sequence":1,"final":true}',
                "data: [DONE]",
            ]
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def test_explicit_missing_binary_reports_unavailable(tmp_path):
    discovery = discover_daemon(
        daemon_url=None,
        daemon_bin=str(tmp_path / "missing-daemon"),
        daemon_addr="127.0.0.1:8099",
        env={"PATH": ""},
    )

    assert discovery.mode == "unavailable"
    assert discovery.source == "explicit_binary"
    assert "not executable" in (discovery.message or "")


def test_sidecar_status_reports_health():
    HealthHandler.stub_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        url = f"http://127.0.0.1:{server.server_port}"
        executor = MariePluginDaemonExecutor(daemon_url=url, start=False, env={"PATH": ""})
        status = executor.status_payload()
    finally:
        server.shutdown()
        server.server_close()

    assert status["mode"] == "sidecar_proxy"
    assert status["runtime_execution"] == "decode_stub_only"
    assert status["daemon"]["ready"] is True
    assert status["daemon"]["version"] == "0.1.0-test"
    assert status["process"]["pid"] is None


def test_stub_invocation_forwards_signed_envelope_to_daemon():
    HealthHandler.stub_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        url = f"http://127.0.0.1:{server.server_port}"
        executor = MariePluginDaemonExecutor(daemon_url=url, start=False, env={"PATH": ""})
        docs = asyncio.run(
            executor.stub_invocation(
                DocList[TextDoc](
                    [
                        TextDoc(
                            text=json.dumps(
                                {
                                    "requestId": "request-1",
                                    "signature": {"keyId": "test", "value": "signed"},
                                }
                            )
                        )
                    ]
                ),
                parameters={"job_id": "request-1"},
            )
        )
    finally:
        server.shutdown()
        server.server_close()

    frames = [json.loads(doc.text) for doc in docs]
    assert HealthHandler.stub_payloads == [
        {"requestId": "request-1", "signature": {"keyId": "test", "value": "signed"}}
    ]
    assert frames[0]["requestId"] == "request-1"
    assert frames[0]["type"] == "text"
    assert frames[0]["payload"] == "hello"
    assert frames[1]["type"] == "structured_object"
    assert frames[1]["payload"] == {"ok": True}
    assert frames[1]["final"] is True


def test_stub_invocation_rejects_invalid_envelope():
    executor = MariePluginDaemonExecutor(daemon_url="http://127.0.0.1:1", start=False, env={"PATH": ""})

    docs = asyncio.run(
        executor.stub_invocation(
            DocList[TextDoc]([TextDoc(text="{not-json")]),
            parameters={"job_id": "request-2"},
        )
    )

    frame = json.loads(docs[0].text)
    assert frame["requestId"] == "request-2"
    assert frame["type"] == "error"
    assert frame["error"]["code"] == "invalid_envelope"


def test_binary_child_starts_and_shutdown(tmp_path):
    port = free_port()
    script = tmp_path / "marie-plugin-daemon"
    script.write_text(fake_daemon_script(), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    executor = MariePluginDaemonExecutor(
        daemon_bin=str(script),
        daemon_addr=f"127.0.0.1:{port}",
        startup_timeout_s=5,
        health_timeout_s=1,
        env={"PATH": ""},
    )
    try:
        status = executor.status_payload()
        assert status["mode"] == "binary_child"
        assert status["source"] == "explicit_binary"
        assert status["daemon"]["ready"] is True
        assert status["process"]["running"] is True
        assert status["process"]["binary"] == str(script)
    finally:
        executor.shutdown()

    assert executor.status_payload()["process"]["running"] is False
    assert executor.status_payload()["lifecycle"]["draining"] is True


def test_binary_child_restarts_after_crash(tmp_path):
    port = free_port()
    script = tmp_path / "marie-plugin-daemon"
    script.write_text(fake_daemon_script(can_exit=True), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    executor = MariePluginDaemonExecutor(
        daemon_bin=str(script),
        daemon_addr=f"127.0.0.1:{port}",
        startup_timeout_s=5,
        health_timeout_s=1,
        restart_limit=1,
        restart_backoff_s=0,
        env={"PATH": ""},
    )
    try:
        first_pid = executor.status_payload()["process"]["pid"]
        with urlopen(f"http://127.0.0.1:{port}/exit", timeout=1) as response:
            assert response.status == 200
        executor._child.wait(timeout=5)

        status = executor.status_payload()
        assert status["process"]["running"] is True
        assert status["process"]["pid"] != first_pid
        assert status["lifecycle"]["restart_attempts"] == 1
        assert status["daemon"]["ready"] is True
    finally:
        executor.shutdown()


def test_binary_child_restart_limit_is_bounded(tmp_path):
    script = tmp_path / "marie-plugin-daemon"
    script.write_text(fake_exiting_daemon_script(), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    executor = MariePluginDaemonExecutor(
        daemon_bin=str(script),
        daemon_addr=f"127.0.0.1:{free_port()}",
        startup_timeout_s=0.2,
        health_timeout_s=0.1,
        restart_limit=1,
        restart_backoff_s=0,
        env={"PATH": ""},
    )
    try:
        status = executor.status_payload()
        assert status["process"]["running"] is False
        assert status["lifecycle"]["restart_attempts"] == 1

        next_status = executor.status_payload()
        assert next_status["process"]["running"] is False
        assert next_status["lifecycle"]["restart_attempts"] == 1
    finally:
        executor.shutdown()


def free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def fake_daemon_script(can_exit: bool = False) -> str:
    exit_handler = ""
    if can_exit:
        exit_handler = """
        if self.path == "/exit":
            self.send_response(200)
            self.end_headers()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
"""

    return f"""#!{sys.executable}
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
{exit_handler}
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({{"ok": True, "ready": True, "version": "child-test", "mode": "decode_only"}}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class Server(ThreadingHTTPServer):
    allow_reuse_address = True


parser = argparse.ArgumentParser()
parser.add_argument("--addr", default="127.0.0.1:8099")
args = parser.parse_args()
host, port = args.addr.rsplit(":", 1)
Server((host, int(port)), Handler).serve_forever()
"""


def fake_exiting_daemon_script() -> str:
    return f"""#!{sys.executable}
import sys

sys.exit(7)
"""
