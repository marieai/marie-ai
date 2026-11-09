"""Client for invoking installed plugin actions via the marie plugin daemon.

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
import logging
import subprocess
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from marie.instrumentation import start_span
from opentelemetry import trace as trace_api
from opentelemetry.trace import StatusCode

from marie.constants import DEFAULT_DAEMON_ADDR
from marie.plugin_daemon import (
    PluginInvocationSpec,
    build_invocation_envelope,
    discover_daemon,
    parse_daemon_frames,
    sign_envelope,
)
from marie.secret_store import CredentialResolver

logger = logging.getLogger("marie.plugin_daemon")


class PluginDaemonHTTPError(RuntimeError):
    """HTTP rejection from the daemon, retaining its status for worker frames."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(
            f'daemon dispatch returned HTTP {status_code}: {detail}'.rstrip(': ')
        )


class PluginDaemonClient:
    """Dispatches plugin actions and returns the daemon's runtime frames.

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
        max_response_bytes: int = 5 * 1024 * 1024,
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
        if max_response_bytes <= 0:
            raise ValueError('max_response_bytes must be positive')
        self.max_response_bytes = max_response_bytes
        self._child: subprocess.Popen[bytes] | None = None
        self._active_responses: dict[str, Any] = {}
        self._cancelled_requests: set[str] = set()
        self._response_lock = threading.Lock()

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

    def invoke(
        self,
        spec: PluginInvocationSpec,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
        runtime_policy: dict[str, Any] | None = None,
        redact_output: bool = False,
    ) -> list[dict[str, Any]]:
        credentials = self.credential_resolver.resolve(spec.credential_requirements)
        payload = {**payload, "user_id": self.user_id or "", "credentials": credentials}

        tracer = trace_api.get_tracer("marie.plugin_daemon")
        span = start_span(tracer, f"plugin.invoke:{spec.action_id}", span_kind="chain")
        span.set_attribute("plugin.provider", spec.provider_ref or "")
        span.set_attribute("plugin.action_type", spec.action_type)
        span.set_attribute("plugin.action_id", spec.action_id)
        span.set_attribute("plugin.package_ref", spec.package_ref or "")
        span.set_attribute("plugin.organization_id", self.organization_id)
        span.set_attribute("plugin.daemon_url", self._url)
        started = time.perf_counter()
        try:
            timeout_ms = int(
                (runtime_policy or {}).get("timeoutMs", self.timeout_s * 1000)
            )
            envelope = build_invocation_envelope(
                spec,
                payload=payload,
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                user_id=self.user_id,
                request_id=request_id,
                trace_id=trace_id,
                timeout_ms=timeout_ms,
                runtime_policy=runtime_policy,
            )
            if self.signing_key_id and self.signing_secret:
                envelope = sign_envelope(
                    envelope, key_id=self.signing_key_id, secret=self.signing_secret
                )
            request_id = envelope["requestId"]
            with self._response_lock:
                if request_id in self._cancelled_requests:
                    raise RuntimeError("plugin invocation cancelled")
            span.set_attribute("plugin.request_id", request_id)
            logger.info(
                f"plugin invoke start action={spec.action_id} provider={spec.provider_ref} "
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
                response = urlopen(request, timeout=timeout_ms / 1000)
                with self._response_lock:
                    self._active_responses[request_id] = response
                    cancelled = request_id in self._cancelled_requests
                try:
                    if cancelled:
                        raise RuntimeError("plugin invocation cancelled")
                    raw_bytes = response.read(self.max_response_bytes + 1)
                    with self._response_lock:
                        if request_id in self._cancelled_requests:
                            raise RuntimeError('plugin invocation cancelled')
                    if len(raw_bytes) > self.max_response_bytes:
                        raise RuntimeError(
                            f'plugin response exceeds {self.max_response_bytes} bytes'
                        )
                    raw = raw_bytes.decode("utf-8")
                finally:
                    with self._response_lock:
                        self._active_responses.pop(request_id, None)
                        self._cancelled_requests.discard(request_id)
                    response.close()
            except HTTPError as error:
                detail = ""
                try:
                    detail = error.read(self.max_response_bytes).decode(
                        "utf-8", "replace"
                    )
                except OSError:
                    pass
                finally:
                    error.close()
                if redact_output:
                    detail = _redact_values(detail, list(credentials.values()))
                raise PluginDaemonHTTPError(error.code, detail) from error
            except (OSError, URLError, TimeoutError) as error:
                raise RuntimeError(f"daemon dispatch failed: {error}") from error

            frames = parse_daemon_frames(raw, request_id)
            if redact_output:
                frames = _redact_values(frames, list(credentials.values()))
            duration_ms = (time.perf_counter() - started) * 1000
            span.set_attribute("plugin.frame_count", len(frames))
            span.set_attribute("plugin.duration_ms", round(duration_ms, 1))
            span.set_output({"frame_count": len(frames)})
            span.set_status(StatusCode.OK)
            logger.info(
                f"plugin invoke ok action={spec.action_id} frames={len(frames)} "
                f"duration_ms={duration_ms:.1f} request_id={request_id}"
            )
            return frames
        except Exception as error:
            span.set_status(StatusCode.ERROR, str(error)[:200])
            logger.error(f"plugin invoke failed action={spec.action_id}: {error}")
            raise
        finally:
            span.end()

    def cancel(self, request_id: str) -> None:
        """Close an active daemon response so its HTTP context is cancelled."""
        with self._response_lock:
            self._cancelled_requests.add(request_id)
            response = self._active_responses.get(request_id)
        if response is not None:
            try:
                response.close()
            except Exception as error:
                logger.warning(f"failed to close plugin request {request_id}: {error}")

    def close(self) -> None:
        with self._response_lock:
            responses = list(self._active_responses.values())
            self._active_responses.clear()
            self._cancelled_requests.clear()
        for response in responses:
            try:
                response.close()
            except Exception as error:
                logger.warning(f"failed to close plugin response: {error}")
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


def _redact_values(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, '[REDACTED]')
        return value
    if isinstance(value, list):
        return [_redact_values(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            _redact_values(key, secrets): _redact_values(item, secrets)
            for key, item in value.items()
        }
    return value
