import asyncio
import json
import socket
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen

import pytest
from docarray import DocList
from docarray.documents import TextDoc

from marie.executor.extensions.plugin_daemon_executor import MariePluginDaemonExecutor
from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor
from marie.plugin_daemon import discover_daemon


class HealthHandler(BaseHTTPRequestHandler):
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


@pytest.mark.parametrize(
    "setting",
    ["MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID", "MARIE_PLUGIN_DAEMON_SIGNING_SECRET"],
)
def test_partial_signing_configuration_is_rejected(setting: str) -> None:
    with pytest.raises(ValueError, match="configured together"):
        MariePluginDaemonExecutor(
            daemon_url="http://127.0.0.1:1", start=False, env={setting: "test-value"}
        )


def test_sidecar_status_reports_health():
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
    assert status["daemon"]["ready"] is True
    assert status["daemon"]["version"] == "0.1.0-test"
    assert status["process"]["pid"] is None


def test_connector_invocation_rejects_missing_plugin_identity():
    executor = MariePluginDaemonExecutor(daemon_url="http://127.0.0.1:1", start=False, env={"PATH": ""})

    docs = asyncio.run(
        executor.connector_invoke(
            DocList[TextDoc]([]),
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

        async def concurrent_status() -> list[dict]:
            results = await asyncio.gather(*(executor.status() for _ in range(4)))
            return [json.loads(docs[0].text) for docs in results]

        statuses = asyncio.run(concurrent_status())
        restarted_pids = {status["process"]["pid"] for status in statuses}
        assert len(restarted_pids) == 1
        assert first_pid not in restarted_pids
        for status in statuses:
            assert status["process"]["running"] is True
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


# ── W2: CONNECTOR node /execute handler ─────────────────────────────────────


class DispatchHandler(BaseHTTPRequestHandler):
    """Fake daemon: records the posted envelope, streams one structured frame."""

    envelopes: list[dict] = []

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"ok": True, "ready": True, "version": "0.1.0-test"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/dispatch/invoke":
            self.send_response(404)
            self.end_headers()
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.__class__.envelopes.append(json.loads(raw.decode("utf-8")))
        body = "\n".join(
            [
                'data: {"type":"structured_object","payload":{"ok":true},"sequence":0,"final":true}',
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


def _run_connector_invoke(parameters: dict):
    DispatchHandler.envelopes = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), DispatchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        executor = MariePluginDaemonExecutor(daemon_url=url, start=False, env={"PATH": ""})
        out = asyncio.run(
            executor.connector_invoke(DocList[TextDoc]([]), parameters=parameters)
        )
    finally:
        server.shutdown()
        server.server_close()
    return out, DispatchHandler.envelopes


_PLUGIN = {
    "tool_ref": "web_reader",
    "package_ref": "ext.test.reader",
    "package_digest": "sha256:abc",
    "package_trust_level": "community",
    "install_id": "i1",
    "provider_id": "p1",
    "package_id": "pk1",
    "action_type": "tool",
}


@pytest.mark.parametrize("cancel_slow", [False, True])
def test_slow_connector_response_keeps_event_loop_and_other_requests_running(
    cancel_slow: bool,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class DelayedDispatchHandler(DispatchHandler):
        def do_POST(self) -> None:
            envelope = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = b'data: {"type":"structured_object","payload":{"ok":true}}\n\n'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.flush()
            if envelope["payload"].get("slow"):
                started.set()
                release.wait(3)
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), DelayedDispatchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    executor = MariePluginDaemonExecutor(
        daemon_url=f"http://127.0.0.1:{server.server_port}",
        start=False,
        env={"PATH": ""},
    )

    async def run() -> None:
        slow = asyncio.create_task(
            executor.connector_invoke(
                DocList[TextDoc]([]), parameters={"plugin": dict(_PLUGIN), "slow": True}
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 2), "request never arrived"
            assert (
                not slow.done()
            ), "blocking response read prevented event-loop progress"
            if cancel_slow:
                slow.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(slow, timeout=1)
            fast = await asyncio.wait_for(
                executor.connector_invoke(
                    DocList[TextDoc]([]), parameters={"plugin": dict(_PLUGIN)}
                ),
                timeout=1,
            )
            assert any('"ok":true' in doc.text for doc in fast)
            if not cancel_slow:
                assert (
                    not slow.done()
                ), "slow request should still be waiting for its body"
        finally:
            release.set()
            if not slow.cancelled():
                slow_result = await slow
                assert any('"ok":true' in doc.text for doc in slow_result)

    try:
        asyncio.run(run())
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        executor.shutdown()


def test_connector_invoke_reads_identity_from_params():
    out, envelopes = _run_connector_invoke(
        {
            "resource": "pages",
            "operation": "web_reader",
            "url": "https://example.com",
            "plugin": dict(_PLUGIN),
            "credential_requirements": [],
            "organization_id": "org-1",
            "workspace_id": "ws-1",
            "user_id": "u1",
        }
    )
    # plugin output streamed back as docs
    assert any('"ok":true' in d.text for d in out), [d.text for d in out]
    # identity came from params.plugin (not the endpoint), and tenant from context
    assert len(envelopes) == 1
    env = envelopes[0]
    assert env["packageRef"] == "ext.test.reader"
    assert env["packageDigest"] == "sha256:abc"
    assert env["actionId"] == "tools/web_reader"
    assert env["organizationId"] == "org-1"
    assert env["workspaceId"] == "ws-1"


def test_scheduled_connector_preserves_settings_through_distributor(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CONNECTOR_TEST_API_KEY", "test-credential")
    source = tmp_path / "source.json"
    source.write_text("{}")
    job = JobInfo(
        status=JobStatus.PENDING,
        entrypoint="plugin_daemon_executor://execute",
        metadata={
            "metadata": {
                "uri": source.as_uri(),
                "op_params": {
                    "plugin": dict(_PLUGIN),
                    "credential_requirements": [
                        {
                            "name": "api_key",
                            "secret_ref": "env:CONNECTOR_TEST_API_KEY",
                            "required": True,
                        }
                    ],
                    "resource": "pages",
                    "operation": "web_reader",
                    "url": "https://example.com/scheduled",
                    "include_images": False,
                    "limit": 0,
                    "configuration": {"format": "markdown"},
                    "organization_id": "org-scheduled",
                    "workspace_id": "ws-scheduled",
                    "user_id": "user-scheduled",
                    "job_id": "operator-job-id",
                },
            },
            "dag_id": "dag-scheduled",
            "run_owner": "scheduler-1",
            "run_attempt_id": "attempt-1",
        },
    )
    parameters, _ = asyncio.run(
        GatewayJobDistributor()._build_payload("job-scheduled", job)
    )

    out, envelopes = _run_connector_invoke(parameters)

    assert any('"ok":true' in doc.text for doc in out), [doc.text for doc in out]
    assert len(envelopes) == 1
    envelope = envelopes[0]
    assert envelope["requestId"] == "job-scheduled"
    assert envelope["packageRef"] == "ext.test.reader"
    assert envelope["packageDigest"] == "sha256:abc"
    assert envelope["actionId"] == "tools/web_reader"
    assert envelope["organizationId"] == "org-scheduled"
    assert envelope["workspaceId"] == "ws-scheduled"
    assert envelope["userId"] == "user-scheduled"
    assert envelope["payload"] == {
        "resource": "pages",
        "operation": "web_reader",
        "url": "https://example.com/scheduled",
        "include_images": False,
        "limit": 0,
        "configuration": {"format": "markdown"},
        "credentials": {"api_key": "test-credential"},
        "user_id": "user-scheduled",
    }


def test_scheduled_connector_prefers_nested_settings_and_execution_context() -> None:
    out, envelopes = _run_connector_invoke(
        {
            "job_id": "job-context",
            "organization_id": "org-context",
            "workspace_id": "ws-context",
            "user_id": "user-context",
            "plugin": {**_PLUGIN, "package_ref": "ext.test.stale"},
            "url": "https://example.com/stale",
            "payload": {
                "op_params": {
                    "plugin": dict(_PLUGIN),
                    "url": "https://example.com/scheduled",
                    "organization_id": "org-operator",
                    "workspace_id": "ws-operator",
                    "user_id": "user-operator",
                }
            },
        }
    )

    assert any('"ok":true' in doc.text for doc in out)
    assert len(envelopes) == 1
    envelope = envelopes[0]
    assert envelope["packageRef"] == "ext.test.reader"
    assert envelope["requestId"] == "job-context"
    assert envelope["organizationId"] == "org-context"
    assert envelope["workspaceId"] == "ws-context"
    assert envelope["userId"] == "user-context"
    assert envelope["payload"] == {
        "url": "https://example.com/scheduled",
        "credentials": {},
        "user_id": "user-context",
    }


def test_connector_invoke_nil_uuid_tenant_fallback():
    out, envelopes = _run_connector_invoke(
        {
            "operation": "web_reader",
            "plugin": dict(_PLUGIN),
            "credential_requirements": [],
        }
    )
    assert any('"ok":true' in d.text for d in out)
    env = envelopes[0]
    assert env["organizationId"] == "00000000-0000-0000-0000-000000000000"
    assert env["workspaceId"] == "00000000-0000-0000-0000-000000000000"


def test_connector_invoke_resolves_credentials_into_payload(monkeypatch):
    # W2 Task 3: a credential requirement with an env: secret_ref resolves in
    # marie-ai (CredentialResolver) and is injected into the daemon payload.
    monkeypatch.setenv("READER_API_KEY", "s3cr3t")
    out, envelopes = _run_connector_invoke(
        {
            "operation": "web_reader",
            "plugin": dict(_PLUGIN),
            "credential_requirements": [
                {"name": "api_key", "secret_ref": "env:READER_API_KEY", "required": True}
            ],
            "organization_id": "org-1",
            "workspace_id": "ws-1",
        }
    )
    assert any('"ok":true' in d.text for d in out)
    env = envelopes[0]
    # build_invocation_envelope forwards the op payload (incl. resolved creds)
    assert env["payload"]["credentials"] == {"api_key": "s3cr3t"}
