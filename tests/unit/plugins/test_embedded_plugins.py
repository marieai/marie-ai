from __future__ import annotations

import asyncio
import json
import os
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from marie.agent.tools.base import ToolOutput
from marie.constants import DEFAULT_DAEMON_ADDR
from marie.plugins.embedded import (
    EmbeddedPlugins,
    PluginInvocationResult,
    inspect_archive,
)

_PACKAGE = "marie/document-extraction"
_PACKAGE_ID = "ext.marie.document-extraction"


def _build_plugin_zip(tmp_path: Path, package_id: str = _PACKAGE_ID) -> str:
    manifest = (
        "apiVersion: marie.ai/v1alpha1\n"
        "kind: ExtensionPackage\n"
        "metadata:\n"
        f"  id: {package_id}\n"
        "  author: marie\n"
        "  name: document-extraction\n"
        "  version: 0.1.0\n"
        "runtime:\n"
        "  type: python_source\n"
        "  language: python\n"
        "  version: \"3.12\"\n"
        "  entrypoint: main\n"
    )
    zip_path = tmp_path / "marie-plugin-document-extraction_0.1.0.zip"
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
            "actions": ["extract"],
            "timeout_s": 120,
        }
    ]


def _tool_output(frames: list[dict]) -> ToolOutput:
    return ToolOutput(
        content=json.dumps(frames),
        tool_name="document-extraction",
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
        self.cancelled_requests: list[str] = []

    def invoke(self, spec, payload, **kwargs):
        self.invocations.append((spec, payload, kwargs))
        if self._invoke_error is not None:
            raise self._invoke_error
        return self._invoke_result

    def cancel(self, request_id):
        self.cancelled_requests.append(request_id)

    def close(self):
        self.closed = True


class _BlockingClient(_FakeClient):
    def __init__(self, invoke_result):
        super().__init__(invoke_result=invoke_result)
        self.started = threading.Event()
        self.release = threading.Event()

    def invoke(self, spec, payload, **kwargs):
        self.invocations.append((spec, payload, kwargs))
        self.started.set()
        self.release.wait(timeout=5)
        if self.cancelled_requests:
            raise RuntimeError("cancelled")
        return self._invoke_result

    def cancel(self, request_id):
        super().cancel(request_id)
        self.release.set()


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
    plugins = EmbeddedPlugins(
        _config(zip_path), "extract_executor", client_factory=factory
    )
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
    assert entry.actions == ["extract"]
    assert entry.timeout_s == 120


def test_config_parsing_defaults_timeout(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    plugins = EmbeddedPlugins([{"package": _PACKAGE, "path": zip_path}], "x")
    entry = plugins._entries[0]
    assert entry.timeout_s == 120
    assert entry.actions == []
    assert entry.runtime_policy["timeoutMs"] == 120_000


def test_config_parses_credentials_bindings_and_runtime_policy(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    config = _config(zip_path)
    config[0].update(
        {
            "provider_id": "provider/repair",
            "credentials": [
                {
                    "name": "llm_api_key",
                    "secret_ref": "env:REPAIR_API_KEY",
                    "required": True,
                    "binding_id": "binding/api-key",
                }
            ],
            "credential_binding_ids": ["binding/model-profile"],
            "runtime_policy": {
                "maxConcurrent": 2,
                "maxMemoryBytes": 1_073_741_824,
                "networkPolicy": "internal_only",
            },
        }
    )
    plugins = EmbeddedPlugins(config, "agent_executor")
    entry = plugins._entries[0]

    assert entry.provider_id == "provider/repair"
    assert entry.credential_requirements[0].name == "llm_api_key"
    assert entry.credential_requirements[0].secret_ref == "env:REPAIR_API_KEY"
    assert entry.credential_binding_ids == [
        "binding/api-key",
        "binding/model-profile",
    ]
    assert entry.runtime_policy == {
        "timeoutMs": 120_000,
        "maxConcurrent": 2,
        "maxMemoryBytes": 1_073_741_824,
        "networkPolicy": "internal_only",
    }


def test_config_rejects_invalid_runtime_policy(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    config = _config(zip_path)
    config[0]["runtime_policy"] = {"networkPolicy": "unrestricted"}

    with pytest.raises(ValueError, match="networkPolicy"):
        EmbeddedPlugins(config, "x")


def test_install_hydrates_credentials_and_provider_identity(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    config = _config(zip_path)
    config[0].update(
        {
            "provider_id": "provider/repair",
            "credentials": [
                {
                    "name": "llm_api_key",
                    "secret_ref": None,
                    "required": False,
                }
            ],
            "credential_binding_ids": ["binding/api-key"],
        }
    )
    client = _FakeClient()
    plugins = EmbeddedPlugins(config, "agent_executor", client_factory=_factory([client]))
    plugins._post_install = lambda client, envelope, archive, timeout_s: _CANNED_INSTALL

    plugins.ensure_started()

    spec = plugins._specs[_PACKAGE]
    assert spec.provider_id == "provider/repair"
    assert spec.credential_requirements[0].name == "llm_api_key"
    assert spec.credential_binding_ids == ["binding/api-key"]


def test_config_entry_requires_package_and_path():
    with pytest.raises(ValueError, match="package.*path"):
        EmbeddedPlugins([{"package": _PACKAGE}], "x")


def test_empty_config_is_lazy_and_ensure_started_raises():
    plugins = EmbeddedPlugins([], "x")
    assert plugins.configured_packages == []
    with pytest.raises(RuntimeError, match="no plugins configured"):
        plugins.ensure_started()


def test_daemon_client_discovery_inherits_process_environment(
    monkeypatch, tmp_path
):
    from marie.plugins import daemon_client

    binary = tmp_path / 'marie-plugin-daemon'
    binary.write_text('#!/bin/sh\n')
    binary.chmod(0o755)
    monkeypatch.setenv('MARIE_PLUGIN_DAEMON_BIN', str(binary))
    captured = {}

    def discover(daemon_url, daemon_bin, daemon_addr, env):
        captured['env'] = env
        values = os.environ if env is None else env
        captured['binary'] = values.get('MARIE_PLUGIN_DAEMON_BIN')
        return type(
            'Discovery',
            (),
            {'mode': 'unavailable', 'message': 'stop', 'binary': None, 'url': None},
        )()

    monkeypatch.setattr(daemon_client, 'discover_daemon', discover)

    with pytest.raises(RuntimeError, match='stop'):
        daemon_client.PluginDaemonClient(
            organization_id='org', workspace_id='workspace'
        )
    assert captured['env'] is None
    assert captured['binary'] == str(binary)


def test_lazy_start_no_client_until_ensure_started(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    factory = _factory([_FakeClient()])
    plugins = EmbeddedPlugins(_config(zip_path), "x", client_factory=factory)
    assert factory.calls == []
    assert plugins._client is None


def test_unknown_package_and_action_are_rejected_before_daemon_start(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    factory = _factory([_FakeClient()])
    plugins = EmbeddedPlugins(_config(zip_path), "x", client_factory=factory)

    with pytest.raises(ValueError, match="package is not configured"):
        plugins.invoke("marie/unknown", "extract", {})
    with pytest.raises(ValueError, match="action is not configured"):
        plugins.invoke(_PACKAGE, "repair", {})

    assert factory.calls == []


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

    result = plugins.invoke(_PACKAGE, "extract", {"path": "/x.pdf", "format": "pdf"})

    assert result == {"markdown": "# Hi", "metadata": {"page_count": 2}}
    assert factory.calls == [120]  # started once
    spec, payload, invocation = client.invocations[-1]
    assert payload["action"] == "extract"
    assert payload["path"] == "/x.pdf"
    assert payload["format"] == "pdf"
    assert payload["execution"]["request_id"] == invocation["request_id"]
    assert payload["execution"]["trace_id"] == invocation["trace_id"]
    assert invocation["action_id"] == "actions/extract"
    assert invocation["action_type"] == "stub"
    assert invocation["runtime_policy"]["timeoutMs"] == 120_000
    assert spec.package_ref == _PACKAGE_ID


def test_invoke_result_preserves_all_frames_and_execution_metadata(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    frames = [
        {"type": "stream", "data": {"progress": 0.5}},
        {"type": "stream", "data": {"outcome": "success"}},
        {"type": "end", "data": {"usage": {"tokens": 12}}},
    ]
    client = _FakeClient(invoke_result=_tool_output(frames))
    plugins, _ = _started(zip_path, [client])

    result = plugins.invoke_result(
        _PACKAGE,
        "extract",
        {"path": "/x.pdf"},
        execution_metadata={"dag_id": "dag-1", "task_id": "task-2", "attempt": 3},
        request_id="request-4",
        trace_id="trace-5",
    )

    assert isinstance(result, PluginInvocationResult)
    assert result.result == {"outcome": "success"}
    assert result.frames == tuple(frames)
    assert result.request_id == "request-4"
    assert result.trace_id == "trace-5"
    _, payload, invocation = client.invocations[-1]
    assert payload["execution"] == {
        "dag_id": "dag-1",
        "task_id": "task-2",
        "attempt": 3,
        "request_id": "request-4",
        "trace_id": "trace-5",
    }
    assert invocation["request_id"] == "request-4"
    assert invocation["trace_id"] == "trace-5"


def test_invoke_raises_on_error_frame(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    frames = [{"type": "error", "data": {"message": "conversion failed"}}]
    client = _FakeClient(invoke_result=_tool_output(frames))
    plugins, factory = _started(zip_path, [client])

    with pytest.raises(RuntimeError, match="conversion failed"):
        plugins.invoke(_PACKAGE, "extract", {"path": "/x.pdf"})
    assert len(factory.calls) == 1


def test_retryable_error_frame_respawns_once(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    failed = _FakeClient(
        invoke_result=_tool_output(
            [
                {
                    "type": "error",
                    "data": {"message": "runtime stopped", "retryable": True},
                }
            ]
        )
    )
    healthy = _FakeClient(
        invoke_result=_tool_output(
            [{"type": "stream", "data": {"outcome": "success"}}, {"type": "end"}]
        )
    )
    plugins, factory = _started(zip_path, [failed, healthy])

    assert plugins.invoke(_PACKAGE, "extract", {}) == {"outcome": "success"}
    assert len(factory.calls) == 2
    assert failed.closed is True


def test_respawn_retry_reuses_request_identity_and_payload(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    committed_effects: list[str] = []

    class CommitThenCrash(_FakeClient):
        def invoke(self, spec, payload, **kwargs):
            self.invocations.append((spec, payload, kwargs))
            operation_id = payload["operation_id"]
            if operation_id not in committed_effects:
                committed_effects.append(operation_id)
            raise RuntimeError("runtime stopped after commit")

    class IdempotentRetry(_FakeClient):
        def invoke(self, spec, payload, **kwargs):
            self.invocations.append((spec, payload, kwargs))
            operation_id = payload["operation_id"]
            if operation_id not in committed_effects:
                committed_effects.append(operation_id)
            return _tool_output(
                [{"type": "stream", "data": {"outcome": "success"}}]
            )

    failed = CommitThenCrash()
    healthy = IdempotentRetry()
    plugins, _ = _started(zip_path, [failed, healthy])

    plugins.invoke_result(
        _PACKAGE,
        "extract",
        {"operation_id": "effect-1"},
        request_id="request-1",
        trace_id="trace-1",
    )

    _, first_payload, first_invocation = failed.invocations[0]
    _, second_payload, second_invocation = healthy.invocations[0]
    assert first_payload == second_payload
    assert first_invocation["request_id"] == second_invocation["request_id"]
    assert first_invocation["trace_id"] == second_invocation["trace_id"]
    assert first_invocation["action_id"] == second_invocation["action_id"]
    assert committed_effects == ["effect-1"]


@pytest.mark.asyncio
async def test_invoke_async_does_not_block_event_loop(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    client = _BlockingClient(
        _tool_output([{"type": "stream", "data": {"outcome": "success"}}])
    )
    plugins, _ = _started(zip_path, [client])

    invocation = asyncio.create_task(plugins.invoke_async(_PACKAGE, "extract", {}))
    assert await asyncio.to_thread(client.started.wait, 1)
    await asyncio.sleep(0)
    assert invocation.done() is False

    client.release.set()
    result = await invocation
    assert result.result == {"outcome": "success"}


@pytest.mark.asyncio
async def test_invoke_async_cancellation_does_not_respawn(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    client = _BlockingClient(
        _tool_output([{"type": "stream", "data": {"outcome": "success"}}])
    )
    plugins, factory = _started(zip_path, [client])

    invocation = asyncio.create_task(
        plugins.invoke_async(
            _PACKAGE,
            "extract",
            {},
            request_id="request-cancel",
        )
    )
    assert await asyncio.to_thread(client.started.wait, 1)
    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation
    await asyncio.to_thread(client.release.wait, 1)

    assert client.cancelled_requests == ["request-cancel"]
    assert factory.calls == [120]


def test_invoke_raises_when_no_stream_result(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    client = _FakeClient(invoke_result=_tool_output([{"type": "end"}]))
    plugins, _ = _started(zip_path, [client])

    with pytest.raises(RuntimeError, match="no stream result"):
        plugins.invoke(_PACKAGE, "extract", {"path": "/x.pdf"})


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
        plugins.invoke(_PACKAGE, "extract", {"path": "/x.pdf"})

    assert len(factory.calls) == 2  # respawned exactly once
    assert dead_first.closed is True


def test_crash_respawns_once_then_succeeds(tmp_path):
    zip_path = _build_plugin_zip(tmp_path)
    frames = [
        {"type": "stream", "data": {"markdown": "ok", "metadata": {}}},
        {"type": "end"},
    ]
    dead = _FakeClient(invoke_error=RuntimeError("dead"))
    healthy = _FakeClient(invoke_result=_tool_output(frames))
    plugins, factory = _started(zip_path, [dead, healthy])

    result = plugins.invoke(_PACKAGE, "extract", {"path": "/x.pdf"})

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
