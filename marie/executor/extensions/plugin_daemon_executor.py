import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from docarray import DocList
from docarray.documents import TextDoc

from marie import requests
from marie.constants import DEFAULT_DAEMON_ADDR, DEFAULT_TENANT_UUID
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger
from marie.plugin_daemon import (
    build_invocation_envelope,
    discover_daemon,
    parse_daemon_frames,
    runtime_error_frame,
)
from marie.plugin_daemon.frames import as_text, first_text, now_utc
from marie.secret_store import CredentialRequirement, CredentialResolver


class MariePluginDaemonExecutor(MarieExecutor):
    def __init__(
        self,
        daemon_url: str | None = None,
        daemon_bin: str | None = None,
        daemon_addr: str = DEFAULT_DAEMON_ADDR,
        start: bool = True,
        startup_timeout_s: float = 5.0,
        health_timeout_s: float = 2.0,
        invoke_timeout_s: float = 30.0,
        shutdown_timeout_s: float = 5.0,
        restart_limit: int = 3,
        restart_backoff_s: float = 1.0,
        env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.daemon_addr = daemon_addr
        self.startup_timeout_s = startup_timeout_s
        self.health_timeout_s = health_timeout_s
        self.invoke_timeout_s = invoke_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self.restart_limit = max(0, restart_limit)
        self.restart_backoff_s = max(0.0, restart_backoff_s)
        self._env = dict(os.environ if env is None else env)
        self._start_enabled = start
        self._child: subprocess.Popen[bytes] | None = None
        self._started_at: datetime | None = None
        self._last_health: dict[str, Any] | None = None
        self._last_exit_code: int | None = None
        self._restart_attempts = 0
        self._next_restart_at: datetime | None = None
        self._draining = False
        self.discovery = discover_daemon(daemon_url, daemon_bin, daemon_addr, self._env)

        if start and self.discovery.mode == "binary_child":
            self._start_child()

    def shutdown(self) -> None:
        self._draining = True
        child = self._child
        if child is None or child.poll() is not None:
            self._last_exit_code = child.returncode if child else self._last_exit_code
            return

        child.terminate()
        try:
            child.wait(timeout=self.shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=1)
        self._last_exit_code = child.returncode

    @requests(on="/status")
    async def status(self, **kwargs: Any) -> DocList[TextDoc]:
        return DocList[TextDoc](
            [TextDoc(text=json.dumps(self.status_payload(), separators=(",", ":")))]
        )

    @requests(on="/v1/runtime/stub-invocations")
    async def stub_invocation(
        self,
        docs: DocList[TextDoc],
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DocList[TextDoc]:
        request_id = str((parameters or {}).get("job_id") or "")
        if not docs:
            frame = runtime_error_frame(
                request_id, "Runtime invocation envelope is empty", "invalid_envelope"
            )
            return frames_to_docs([frame])

        try:
            envelope = json.loads(docs[0].text or "{}")
        except json.JSONDecodeError:
            frame = runtime_error_frame(
                request_id,
                "Runtime invocation envelope is not valid JSON",
                "invalid_envelope",
            )
            return frames_to_docs([frame])

        if not isinstance(envelope, dict):
            frame = runtime_error_frame(
                request_id,
                "Runtime invocation envelope must be a JSON object",
                "invalid_envelope",
            )
            return frames_to_docs([frame])

        request_id = (
            first_text(
                request_id,
                as_text(envelope.get("requestId")),
                as_text(envelope.get("request_id")),
            )
            or ""
        )
        return frames_to_docs(self._post_stub_invocation(envelope, request_id))

    @requests(on="/execute")
    async def connector_invoke(
        self,
        docs: DocList[TextDoc],
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DocList[TextDoc]:
        """Dispatch a CONNECTOR node to the plugin daemon and return its frames.

        Routed here by the fixed endpoint ``plugin_daemon_executor://execute``.
        All plugin identity + credentials ride in ``parameters`` (the endpoint
        carries none); tenant claims come from the execution context, falling
        back to the nil UUID when absent (W2 scope; install + invoke must share
        the same fallback tenant or the daemon returns ``plugin_not_installed``).
        """
        # Lazy import: marie.agent pulls in heavy dependencies (and itself
        # imports marie.executor); keep it out of this module's import path.
        from marie.agent.tools.plugin_tool import PluginToolSpec

        params = parameters or {}
        request_id = str(params.get("job_id") or "")
        plugin = params.get("plugin") or {}
        try:
            spec = PluginToolSpec(
                tool_ref=plugin["tool_ref"],
                tool_name=plugin["tool_ref"],
                package_ref=plugin["package_ref"],
                package_digest=plugin["package_digest"],
                package_trust_level=plugin.get("package_trust_level", "community"),
                install_id=plugin.get("install_id"),
                provider_id=plugin.get("provider_id"),
                package_id=plugin.get("package_id"),
            )
        except KeyError as error:
            return frames_to_docs(
                [
                    runtime_error_frame(
                        request_id,
                        f"connector node missing plugin identity field: {error}",
                        "invalid_envelope",
                    )
                ]
            )

        requirements = [
            CredentialRequirement(**req)
            for req in params.get("credential_requirements", [])
        ]
        credentials = CredentialResolver().resolve(requirements)

        organization_id = str(params.get("organization_id") or DEFAULT_TENANT_UUID)
        workspace_id = str(params.get("workspace_id") or DEFAULT_TENANT_UUID)
        user_id = str(params.get("user_id") or "")

        control_keys = {
            "plugin",
            "credential_requirements",
            "organization_id",
            "workspace_id",
            "user_id",
            "job_id",
            "payload",
        }
        op_payload = {k: v for k, v in params.items() if k not in control_keys}
        payload = {**op_payload, "credentials": credentials, "user_id": user_id}

        envelope = build_invocation_envelope(
            spec,
            payload=payload,
            organization_id=organization_id,
            workspace_id=workspace_id,
            user_id=user_id,
            action_type=plugin.get("action_type", "tool"),
            request_id=request_id or None,
        )
        return frames_to_docs(self._invoke_daemon(envelope, request_id))

    def status_payload(self) -> dict[str, Any]:
        self._ensure_child()
        health = self._probe_health()
        child = self._child
        return {
            "executor": "MariePluginDaemonExecutor",
            "runtime_execution": "decode_stub_only",
            "mode": self.discovery.mode,
            "source": self.discovery.source,
            "daemon": health,
            "process": {
                "pid": child.pid if child else None,
                "running": child is not None and child.poll() is None,
                "returncode": child.poll() if child else None,
                "started_at": (
                    self._started_at.isoformat() if self._started_at else None
                ),
                "binary": self.discovery.binary,
                "addr": (
                    self.daemon_addr if self.discovery.mode == "binary_child" else None
                ),
            },
            "lifecycle": {
                "draining": self._draining,
                "restart_enabled": self._start_enabled
                and self.discovery.mode == "binary_child",
                "restart_attempts": self._restart_attempts,
                "restart_limit": self.restart_limit,
                "restart_backoff_s": self.restart_backoff_s,
                "next_restart_at": (
                    self._next_restart_at.isoformat() if self._next_restart_at else None
                ),
                "last_exit_code": self._last_exit_code,
            },
        }

    def _start_child(self) -> None:
        binary = self.discovery.binary
        if not binary:
            return

        self._last_exit_code = None
        self._child = subprocess.Popen([binary, "--addr", self.daemon_addr])
        self._started_at = now_utc()
        self._next_restart_at = None

        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._child.poll() is not None:
                self._last_exit_code = self._child.returncode
                self.logger.warning(
                    "marie-plugin-daemon exited during startup with code %s",
                    self._child.returncode,
                )
                return
            health = self._probe_health()
            if health["ready"]:
                return
            time.sleep(0.1)

        self.logger.warning(
            "marie-plugin-daemon did not become ready before startup timeout"
        )

    def _ensure_child(self) -> None:
        if (
            not self._start_enabled
            or self.discovery.mode != "binary_child"
            or self._draining
        ):
            return

        child = self._child
        if child is not None and child.poll() is None:
            return

        self._last_exit_code = child.returncode if child else self._last_exit_code
        if self._restart_attempts >= self.restart_limit:
            return

        now = now_utc()
        if self._next_restart_at is not None and now < self._next_restart_at:
            return

        self._restart_attempts += 1
        self._next_restart_at = None
        self.logger.warning(
            "restarting marie-plugin-daemon child after exit code %s (%s/%s)",
            self._last_exit_code,
            self._restart_attempts,
            self.restart_limit,
        )
        self._start_child()
        if self._child is not None and self._child.poll() is not None:
            self._next_restart_at = datetime.fromtimestamp(
                time.time() + self.restart_backoff_s,
                timezone.utc,
            )

    def _probe_health(self) -> dict[str, Any]:
        if self.discovery.mode == "unavailable":
            return unavailable_health(
                self.discovery.message or "marie-plugin-daemon is not configured"
            )

        url = self.discovery.url
        if not url:
            return unavailable_health("marie-plugin-daemon URL is not configured")

        request = Request(
            f"{url.rstrip('/')}/health", headers={"Accept": "application/json"}
        )
        checked_at = now_utc()
        try:
            with urlopen(request, timeout=self.health_timeout_s) as response:
                body = json.loads(response.read().decode("utf-8") or "{}")
        except HTTPError as error:
            return unavailable_health(
                f"daemon health returned HTTP {error.code}", url, checked_at
            )
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
            return unavailable_health(str(error), url, checked_at)

        ready = bool(body.get("ready", body.get("ok", False)))
        health = {
            "configured": True,
            "reachable": True,
            "ready": ready,
            "status": "ready" if ready else "unavailable",
            "url": url,
            "version": body.get("version"),
            "mode": body.get("mode"),
            "message": body.get("message"),
            "checked_at": checked_at.isoformat(),
        }
        self._last_health = health
        return health

    def _post_stub_invocation(
        self, envelope: dict[str, Any], request_id: str
    ) -> list[dict[str, Any]]:
        self._ensure_child()
        if self.discovery.mode == "unavailable":
            return [
                runtime_error_frame(
                    request_id,
                    self.discovery.message or "marie-plugin-daemon is not configured",
                )
            ]

        url = self.discovery.url
        if not url:
            return [
                runtime_error_frame(
                    request_id, "marie-plugin-daemon URL is not configured"
                )
            ]

        body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        request = Request(
            f"{url.rstrip('/')}/v1/runtime/stub-invocations",
            data=body,
            headers={
                "Accept": "text/event-stream, application/x-ndjson, application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.invoke_timeout_s) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            return [
                runtime_error_frame(
                    request_id,
                    f"daemon stub invocation returned HTTP {error.code}",
                    "runtime_http_error",
                )
            ]
        except (OSError, URLError, TimeoutError) as error:
            return [runtime_error_frame(request_id, str(error))]

        return parse_daemon_frames(raw, request_id)

    def _invoke_daemon(
        self, envelope: dict[str, Any], request_id: str
    ) -> list[dict[str, Any]]:
        """POST a built envelope to the daemon's real ``/v1/dispatch/invoke`` and
        return parsed frames. Mirrors ``_post_stub_invocation`` but the real path."""
        self._ensure_child()
        if self.discovery.mode == "unavailable":
            return [
                runtime_error_frame(
                    request_id,
                    self.discovery.message or "marie-plugin-daemon is not configured",
                )
            ]

        url = self.discovery.url
        if not url:
            return [
                runtime_error_frame(
                    request_id, "marie-plugin-daemon URL is not configured"
                )
            ]

        body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        request = Request(
            f"{url.rstrip('/')}/v1/dispatch/invoke",
            data=body,
            headers={
                "Accept": "text/event-stream, application/x-ndjson, application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.invoke_timeout_s) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            return [
                runtime_error_frame(
                    request_id,
                    f"daemon invoke returned HTTP {error.code}",
                    "runtime_http_error",
                )
            ]
        except (OSError, URLError, TimeoutError) as error:
            return [runtime_error_frame(request_id, str(error))]

        return parse_daemon_frames(raw, request_id)


def frames_to_docs(frames: list[dict[str, Any]]) -> DocList[TextDoc]:
    return DocList[TextDoc](
        [TextDoc(text=json.dumps(frame, separators=(",", ":"))) for frame in frames]
    )


def unavailable_health(
    message: str, url: str | None = None, checked_at: datetime | None = None
) -> dict[str, Any]:
    return {
        "configured": url is not None,
        "reachable": False,
        "ready": False,
        "status": "unavailable",
        "url": url,
        "version": None,
        "mode": None,
        "message": message,
        "checked_at": (checked_at or now_utc()).isoformat(),
    }
