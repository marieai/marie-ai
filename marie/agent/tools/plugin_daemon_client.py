"""Client for invoking installed plugin tools via the marie plugin daemon.

Builds the marie runtime envelope and dispatches it to the daemon's real
``POST /v1/dispatch/invoke`` endpoint. The daemon protocol (discovery, envelope
build/sign, SSE frame parsing) lives in the shared ``marie.plugin_daemon``
package.

Without a signing key the envelope is sent UNSIGNED and the daemon must run
with the dev-verifier bypass (``MARIE_PLUGIN_DAEMON_DEV_INSECURE``). The
daemon's runtime policy is still enforced (package/action/mode claims), so the
envelope must be complete.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from opentelemetry import trace as trace_api
from opentelemetry.trace import StatusCode

from marie.agent.tools.base import ToolOutput
from marie.agent.tools.plugin_tool import PluginToolSpec
from marie.constants import DEFAULT_DAEMON_ADDR
from marie.instrumentation import start_span
from marie.logging_core.logger import MarieLogger
from marie.plugin_daemon import (
    build_invocation_envelope,
    discover_daemon,
    parse_daemon_frames,
    sign_envelope,
)
from marie.secret_store import CredentialResolver

logger = MarieLogger("marie.agent.tools.plugin_daemon")


class PluginDaemonClient:
    """Dispatches plugin-tool invocations to the marie plugin daemon.

    Transport is chosen by the execution environment:
      - ``base_url`` set → REMOTE (dispatch to that URL).
      - else ``discover_daemon`` → ``sidecar_proxy`` (external URL) or, with
        ``spawn_local=True``, ``binary_child`` (spawn and own the daemon child).
    """

    def __init__(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        user_id: str | None = None,
        signing_key_id: str | None = None,
        signing_secret: str | None = None,
        credential_resolver: CredentialResolver | None = None,
        base_url: str | None = None,
        daemon_bin: str | None = None,
        daemon_addr: str = DEFAULT_DAEMON_ADDR,
        spawn_local: bool = False,
        timeout_s: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.signing_key_id = signing_key_id
        self.signing_secret = signing_secret
        # Credentials are resolved in marie-ai (env by default); never fetched from
        # another system.
        self.credential_resolver = credential_resolver or CredentialResolver()
        self.timeout_s = timeout_s
        self._child: subprocess.Popen[bytes] | None = None

        if base_url:
            self._url = base_url.rstrip("/")
            return

        discovery = discover_daemon(None, daemon_bin, daemon_addr, env)
        if discovery.mode == "sidecar_proxy" and discovery.url:
            self._url = discovery.url.rstrip("/")
        elif discovery.mode == "binary_child" and discovery.binary:
            if not spawn_local:
                raise RuntimeError(
                    "marie-plugin-daemon resolved to a local binary; pass "
                    "spawn_local=True to spawn and own it"
                )
            self._child = subprocess.Popen(
                [str(discovery.binary), "--addr", daemon_addr]
            )
            self._url = f"http://{daemon_addr}"
            self._wait_ready()
        else:
            raise RuntimeError(
                discovery.message or "marie-plugin-daemon is not configured"
            )

    @property
    def url(self) -> str:
        """Base URL of the daemon this client dispatches to (owns, if spawned)."""
        return self._url

    def _wait_ready(self, attempts: int = 50, interval_s: float = 0.1) -> None:
        for _ in range(attempts):
            if self._child is not None and self._child.poll() is not None:
                raise RuntimeError(
                    f"marie-plugin-daemon exited during startup "
                    f"(code {self._child.returncode})"
                )
            try:
                with urlopen(f"{self._url}/health", timeout=2) as response:
                    if getattr(response, "status", response.getcode()) == 200:
                        return
            except (OSError, URLError):
                pass
            time.sleep(interval_s)
        raise RuntimeError("marie-plugin-daemon did not become ready before timeout")

    def invoke(self, spec: PluginToolSpec, payload: dict[str, Any]) -> ToolOutput:
        # The dify_plugin ToolInvokeRequest requires user_id; inject it from the
        # execution context (the daemon forwards `payload` opaquely to the plugin).
        # Credentials are resolved in marie-ai from the spec's requirements.
        credentials = self.credential_resolver.resolve(spec.credential_requirements)
        payload = {**payload, "user_id": self.user_id or "", "credentials": credentials}

        # OTel span for the daemon-dispatch hop (a child of the agent's tool span
        # when called via safe_call). Visible in the standard tracing pipeline.
        tracer = trace_api.get_tracer("marie.agent.tools.plugin_daemon")
        span = start_span(tracer, f"plugin.invoke:{spec.tool_name}", span_kind="tool")
        span.set_attribute("plugin.provider", spec.provider_ref or "")
        span.set_attribute("plugin.tool", spec.tool_ref or "")
        span.set_attribute("plugin.package_ref", spec.package_ref or "")
        span.set_attribute("plugin.organization_id", self.organization_id)
        span.set_attribute("plugin.daemon_url", self._url)
        span.set_input(
            {
                "provider": spec.provider_ref,
                "tool": spec.tool_ref,
                "tool_parameters": payload.get("tool_parameters"),
            }
        )
        started = time.perf_counter()
        try:
            envelope = build_invocation_envelope(
                spec,
                payload=payload,
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                user_id=self.user_id,
                timeout_ms=int(self.timeout_s * 1000),
            )
            if self.signing_key_id and self.signing_secret:
                envelope = sign_envelope(
                    envelope, key_id=self.signing_key_id, secret=self.signing_secret
                )
            request_id = envelope["requestId"]
            span.set_attribute("plugin.request_id", request_id)
            logger.info(
                f"plugin invoke start tool={spec.tool_ref} provider={spec.provider_ref} "
                f"package={spec.package_ref} request_id={request_id} url={self._url}"
            )
            body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
            request = Request(
                f"{self._url}/v1/dispatch/invoke",
                data=body,
                headers={
                    "Accept": "text/event-stream, application/x-ndjson, application/json",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    raw = response.read().decode("utf-8")
            except HTTPError as error:
                detail = ""
                try:
                    detail = error.read().decode("utf-8", "replace")
                except Exception:
                    pass
                raise RuntimeError(
                    f"daemon dispatch returned HTTP {error.code}: {detail}".rstrip(": ")
                ) from error
            except (OSError, URLError, TimeoutError) as error:
                raise RuntimeError(f"daemon dispatch failed: {error}") from error

            frames = parse_daemon_frames(raw, request_id)
            duration_ms = (time.perf_counter() - started) * 1000
            span.set_attribute("plugin.frame_count", len(frames))
            span.set_attribute("plugin.duration_ms", round(duration_ms, 1))
            span.set_output({"frame_count": len(frames)})
            span.set_status(StatusCode.OK)
            logger.info(
                f"plugin invoke ok tool={spec.tool_ref} frames={len(frames)} "
                f"duration_ms={duration_ms:.1f} request_id={request_id}"
            )
            return ToolOutput(
                content=json.dumps(frames, separators=(",", ":")),
                tool_name=spec.tool_name or "plugin_tool",
                raw_input=payload,
                raw_output=frames,
            )
        except Exception as error:
            span.set_status(StatusCode.ERROR, str(error)[:200])
            logger.error(f"plugin invoke failed tool={spec.tool_ref}: {error}")
            raise
        finally:
            span.end()

    def close(self) -> None:
        if self._child is not None and self._child.poll() is None:
            self._child.terminate()
            try:
                self._child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._child.kill()
        self._child = None

    def __enter__(self) -> "PluginDaemonClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
