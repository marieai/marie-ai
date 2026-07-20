"""Executor-owned plugin bootstrap over the marie plugin daemon.

An executor declares a list of plugin packages under its YAML ``with.plugins``;
``EmbeddedPlugins`` owns a single :class:`PluginDaemonClient` (which spawns and
owns THE daemon child), installs each declared package into it, and dispatches
provider-neutral invocations. The daemon is started lazily on the first invoke —
construction only parses config, so importing an executor never spawns a daemon.

The signing key is a hardcoded loopback pair (spec §8.2): the same key-id/secret
is handed to the spawned daemon child through its environment so it verifies the
envelopes this process signs. This is a loopback integrity check, not a secret;
revisit before promoting to a shared daemon.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import posixpath
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import yaml

from marie.constants import DEFAULT_DAEMON_ADDR, DEFAULT_TENANT_UUID
from marie.logging_core.logger import MarieLogger
from marie.plugin_daemon import (
    DEFAULT_RUNTIME_POLICY,
    build_invocation_envelope,
    sign_envelope,
)
from marie.plugins.agent_tool import PluginToolSpec
from marie.plugins.daemon_client import PluginDaemonClient
from marie.secret_store import CredentialRequirement

# Loopback integrity key (spec §8.2). Handed to the spawned daemon child via env
# so it verifies the envelopes this process signs. NOT a secret; revisit before
# promoting EmbeddedPlugins to a shared daemon.
_SIGNING_KEY_ID = "marie-executor-embedded"
_SIGNING_SECRET = "marie-executor-embedded-loopback-secret"

_DEFAULT_TIMEOUT_S = 120
_CANCEL_GRACE_S = 1.0

ClientFactory = Callable[[float], PluginDaemonClient]


class PluginInvocationError(RuntimeError):
    """A classified error frame returned by the plugin runtime."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class PluginInvocationResult:
    """Final plugin result with the complete structured response stream."""

    result: dict[str, Any]
    frames: tuple[dict[str, Any], ...]
    request_id: str
    trace_id: str


class _PluginEntry:
    """One parsed ``with.plugins`` list entry."""

    __slots__ = (
        "package",
        "path",
        "actions",
        "timeout_s",
        "credential_requirements",
        "credential_binding_ids",
        "provider_id",
        "runtime_policy",
    )

    def __init__(
        self,
        package: str,
        path: str,
        actions: list[str],
        timeout_s: float,
        credential_requirements: list[CredentialRequirement],
        credential_binding_ids: list[str],
        provider_id: str,
        runtime_policy: dict[str, Any],
    ) -> None:
        self.package = package
        self.path = path
        self.actions = actions
        self.timeout_s = timeout_s
        self.credential_requirements = credential_requirements
        self.credential_binding_ids = credential_binding_ids
        self.provider_id = provider_id
        self.runtime_policy = runtime_policy

    @classmethod
    def from_config(cls, entry: dict[str, Any]) -> "_PluginEntry":
        package = (entry.get("package") or "").strip()
        path = (entry.get("path") or "").strip()
        if not package or not path:
            raise ValueError(
                "plugin config entry requires both 'package' and 'path' "
                f"(got {entry!r})"
            )

        actions = _string_list(entry.get("actions", []), "actions")
        if len(actions) != len(set(actions)):
            raise ValueError(f"plugin {package!r} contains duplicate actions")

        raw_policy = entry.get("runtime_policy", entry.get("runtimePolicy", {}))
        if not isinstance(raw_policy, dict):
            raise ValueError(f"plugin {package!r} runtime_policy must be a mapping")
        runtime_policy, timeout_s = _runtime_policy(entry, raw_policy, package)

        raw_credentials = entry.get("credentials", [])
        if not isinstance(raw_credentials, list):
            raise ValueError(f"plugin {package!r} credentials must be a list")
        requirements: list[CredentialRequirement] = []
        embedded_binding_ids: list[str] = []
        for raw_requirement in raw_credentials:
            if not isinstance(raw_requirement, dict):
                raise ValueError(
                    f"plugin {package!r} credential requirements must be mappings"
                )
            requirement = dict(raw_requirement)
            binding_id = requirement.pop(
                "binding_id", requirement.pop("bindingId", None)
            )
            if binding_id is not None:
                embedded_binding_ids.extend(
                    _string_list([binding_id], "credential binding IDs")
                )
            requirements.append(CredentialRequirement.model_validate(requirement))

        configured_binding_ids = entry.get(
            "credential_binding_ids", entry.get("credentialBindingIds", [])
        )
        binding_ids = embedded_binding_ids + _string_list(
            configured_binding_ids, "credential binding IDs"
        )
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError(
                f"plugin {package!r} contains duplicate credential binding IDs"
            )

        provider_id = str(
            entry.get("provider_id", entry.get("providerId", package)) or ""
        ).strip()
        if binding_ids and not provider_id:
            raise ValueError(
                f"plugin {package!r} requires provider_id when credentials are bound"
            )

        return cls(
            package=package,
            path=path,
            actions=actions,
            timeout_s=timeout_s,
            credential_requirements=requirements,
            credential_binding_ids=binding_ids,
            provider_id=provider_id,
            runtime_policy=runtime_policy,
        )


class EmbeddedPlugins:
    """Owns one plugin daemon child and the packages installed into it.

    Executor-agnostic: any executor can hold one, feed it its ``with.plugins``
    config, and call :meth:`invoke`. The daemon is spawned and packages installed
    lazily on the first :meth:`ensure_started` / :meth:`invoke`.
    """

    def __init__(
        self,
        plugins_config: list[dict[str, Any]] | None,
        executor_identity: str,
        *,
        daemon_addr: str = DEFAULT_DAEMON_ADDR,
        organization_id: str = DEFAULT_TENANT_UUID,
        workspace_id: str = DEFAULT_TENANT_UUID,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._entries = [_PluginEntry.from_config(e) for e in (plugins_config or [])]
        self._entries_by_package = {entry.package: entry for entry in self._entries}
        if len(self._entries_by_package) != len(self._entries):
            raise ValueError("plugin configuration contains duplicate packages")
        self._executor_identity = (
            executor_identity or "executor"
        ).strip() or "executor"
        self._daemon_addr = daemon_addr
        self._organization_id = organization_id
        self._workspace_id = workspace_id
        self._client_factory = client_factory or self._default_client_factory
        self._client: PluginDaemonClient | None = None
        self._specs: dict[str, PluginToolSpec] = {}
        self._install_ids: dict[str, str] = {}
        self._runtime_generation = 0
        self.logger = MarieLogger(self.__class__.__name__)

    @property
    def configured_packages(self) -> list[str]:
        return [entry.package for entry in self._entries]

    @property
    def runtime_generation(self) -> int:
        """Monotonic generation incremented after each daemon bootstrap."""
        return self._runtime_generation

    def validate_action(self, package: str, action: str) -> None:
        """Raise when a package/action pair is not deployment-allowlisted."""
        self._require_action(package, action)

    def ensure_started(self) -> None:
        """Spawn the daemon child (if needed) and install every declared package."""
        if self._client is not None:
            return
        if not self._entries:
            raise RuntimeError("EmbeddedPlugins has no plugins configured")

        max_timeout = max(entry.timeout_s for entry in self._entries)
        client = self._client_factory(max_timeout)
        try:
            for entry in self._entries:
                self._install_entry(client, entry)
        except Exception:
            client.close()
            raise
        self._client = client
        self._runtime_generation += 1

    def invoke(
        self,
        package: str,
        action: str,
        payload: dict[str, Any],
        *,
        execution_metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke ``action`` on ``package`` with ``payload``; return its stream data.

        Raises on an error frame or a missing stream result (fail-loud). On an
        invocation failure the daemon child is respawned exactly once and the call
        retried; a second failure propagates.
        """
        return self.invoke_result(
            package,
            action,
            payload,
            execution_metadata=execution_metadata,
            request_id=request_id,
            trace_id=trace_id,
        ).result

    def invoke_result(
        self,
        package: str,
        action: str,
        payload: dict[str, Any],
        *,
        execution_metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        _cancelled: threading.Event | None = None,
    ) -> PluginInvocationResult:
        """Invoke a configured action and retain every structured result frame."""
        entry = self._require_action(package, action)
        resolved_request_id = request_id or str(uuid4())
        resolved_trace_id = trace_id or str(uuid4())
        execution = dict(execution_metadata or {})
        execution["request_id"] = resolved_request_id
        execution["trace_id"] = resolved_trace_id
        plugin_payload = {**payload, "action": action, "execution": execution}

        self._raise_if_cancelled(_cancelled)
        self.ensure_started()
        self._raise_if_cancelled(_cancelled)
        spec, client = self._require_spec_client(package)
        try:
            output = self._invoke_client(
                client,
                spec,
                entry,
                action,
                plugin_payload,
                resolved_request_id,
                resolved_trace_id,
            )
            return _result_from_output(
                output,
                request_id=resolved_request_id,
                trace_id=resolved_trace_id,
            )
        except PluginInvocationError as error:
            if not error.retryable:
                raise
            first_error = error
        except Exception as error:
            if _cancelled is not None and _cancelled.is_set():
                raise PluginInvocationError(
                    "plugin invocation cancelled", retryable=False
                ) from error
            first_error = error

        self._raise_if_cancelled(_cancelled)
        self.logger.warning(
            f"plugin invoke failed for {package}/{action}; respawning daemon "
            f"once: {first_error}"
        )
        self._respawn()
        self._raise_if_cancelled(_cancelled)
        spec, client = self._require_spec_client(package)
        output = self._invoke_client(
            client,
            spec,
            entry,
            action,
            plugin_payload,
            resolved_request_id,
            resolved_trace_id,
        )
        return _result_from_output(
            output,
            request_id=resolved_request_id,
            trace_id=resolved_trace_id,
        )

    async def invoke_async(
        self,
        package: str,
        action: str,
        payload: dict[str, Any],
        *,
        execution_metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> PluginInvocationResult:
        """Invoke without blocking the event loop and propagate caller cancellation."""
        resolved_request_id = request_id or str(uuid4())
        cancelled = threading.Event()
        worker = asyncio.create_task(
            asyncio.to_thread(
                self.invoke_result,
                package,
                action,
                payload,
                execution_metadata=execution_metadata,
                request_id=resolved_request_id,
                trace_id=trace_id,
                _cancelled=cancelled,
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled.set()
            client = self._client
            if client is not None:
                client.cancel(resolved_request_id)
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout=_CANCEL_GRACE_S)
            except TimeoutError:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass
            except (asyncio.CancelledError, Exception):
                pass
            raise

    def capabilities(self, package: str) -> dict[str, Any]:
        """Return the package's current input-aware capability snapshot."""
        return self.invoke(package, "capabilities", {})

    def close(self) -> None:
        """Terminate the daemon child and forget installed specs."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as error:  # pragma: no cover - defensive teardown
                self.logger.warning(f"error closing plugin daemon client: {error}")
        self._client = None
        self._specs = {}

    def __enter__(self) -> "EmbeddedPlugins":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _require_spec_client(
        self, package: str
    ) -> tuple[PluginToolSpec, PluginDaemonClient]:
        spec = self._specs.get(package)
        if spec is None or self._client is None:
            raise ValueError(f"plugin package is not installed: {package}")
        return spec, self._client

    def _require_action(self, package: str, action: str) -> _PluginEntry:
        entry = self._entries_by_package.get(package)
        if entry is None:
            raise ValueError(f"plugin package is not configured: {package}")
        if action not in entry.actions:
            raise ValueError(f"plugin action is not configured for {package}: {action}")
        return entry

    @staticmethod
    def _raise_if_cancelled(cancelled: threading.Event | None) -> None:
        if cancelled is not None and cancelled.is_set():
            raise PluginInvocationError("plugin invocation cancelled", retryable=False)

    @staticmethod
    def _invoke_client(
        client: PluginDaemonClient,
        spec: PluginToolSpec,
        entry: _PluginEntry,
        action: str,
        payload: dict[str, Any],
        request_id: str,
        trace_id: str,
    ) -> Any:
        return client.invoke(
            spec,
            payload,
            action_id=f"actions/{action}",
            action_type="stub",
            request_id=request_id,
            trace_id=trace_id,
            runtime_policy=entry.runtime_policy,
        )

    def _respawn(self) -> None:
        self.close()
        self.ensure_started()

    def _default_client_factory(self, timeout_s: float) -> PluginDaemonClient:
        # Hand the loopback key to the spawned daemon child via env so it verifies
        # the envelopes we sign. Popen inherits this process's environment.
        os.environ["MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID"] = _SIGNING_KEY_ID
        os.environ["MARIE_PLUGIN_DAEMON_SIGNING_SECRET"] = _SIGNING_SECRET
        return PluginDaemonClient(
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            signing_key_id=_SIGNING_KEY_ID,
            signing_secret=_SIGNING_SECRET,
            daemon_addr=self._daemon_addr,
            spawn_local=True,
            timeout_s=timeout_s,
        )

    def _install_entry(self, client: PluginDaemonClient, entry: _PluginEntry) -> None:
        package_ref, digest = inspect_archive(entry.path)
        archive = Path(entry.path).read_bytes()
        install_id = f"{self._executor_identity}@{self._daemon_addr}/{entry.package}"

        spec = PluginToolSpec(
            type="extension_tool",
            tool_name=_tool_name(entry.package),
            tool_ref=entry.package,
            install_id=install_id,
            package_id=install_id,
            package_ref=package_ref,
            package_digest=digest,
            provider_id=entry.provider_id,
            provider_ref=entry.package,
            package_trust_level="builtin",
            credential_requirements=entry.credential_requirements,
            credential_binding_ids=entry.credential_binding_ids,
        )
        envelope = build_invocation_envelope(
            spec,
            payload={},
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
            timeout_ms=int(entry.timeout_s * 1000),
        )
        envelope = sign_envelope(
            envelope, key_id=_SIGNING_KEY_ID, secret=_SIGNING_SECRET
        )

        response = self._post_install(client, envelope, archive, entry.timeout_s)

        # The daemon is authoritative for the installed identity; keep the spec's
        # deterministic install_id but hydrate ref/digest from the response.
        install = response.get("install") if isinstance(response, dict) else None
        install = install if isinstance(install, dict) else {}
        spec.package_ref = install.get("packageRef") or package_ref
        spec.package_digest = install.get("digest") or digest

        self._specs[entry.package] = spec
        self._install_ids[entry.package] = install_id
        self.logger.info(
            f"installed plugin {entry.package} ref={spec.package_ref} "
            f"digest={spec.package_digest} state={response.get('state')!r}"
        )

    def _post_install(
        self,
        client: PluginDaemonClient,
        envelope: dict[str, Any],
        archive: bytes,
        timeout_s: float,
    ) -> dict[str, Any]:
        request = Request(
            f"{client.url}/v1/plugins/install",
            data=archive,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/zip",
                "X-Marie-Envelope": json.dumps(envelope, separators=(",", ":")),
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = ""
            try:
                detail = error.read().decode("utf-8", "replace")
            except Exception:
                pass
            raise RuntimeError(
                f"plugin install returned HTTP {error.code}: {detail}".rstrip(": ")
            ) from error
        except (OSError, URLError, TimeoutError) as error:
            raise RuntimeError(f"plugin install failed: {error}") from error


def inspect_archive(zip_path: str) -> tuple[str, str]:
    """Compute ``(packageRef, digest)`` for a plugin zip, matching the daemon.

    Mirrors ``plugin_manager.Inspect`` / ``decoder.checksum``: files sorted by
    slash-cleaned path, directory entries dropped, digest is
    ``sha256(sum(path + \\x00 + data + \\x00))`` and ``packageRef`` is the
    manifest's ``metadata.id``. The signed install envelope's claims must equal
    what the daemon recomputes, so this must stay byte-identical to the Go side.
    """
    files: list[tuple[str, bytes]] = []
    manifest_data: bytes | None = None
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = posixpath.normpath(info.filename)
            data = archive.read(info)
            files.append((name, data))
            if posixpath.basename(name) == "marie-extension.yaml":
                manifest_data = data

    if manifest_data is None:
        raise ValueError(f"plugin archive has no marie-extension.yaml: {zip_path}")

    files.sort(key=lambda item: item[0])
    hasher = hashlib.sha256()
    for name, data in files:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(data)
        hasher.update(b"\x00")

    manifest = yaml.safe_load(manifest_data) or {}
    metadata = manifest.get("metadata") or {}
    package_ref = (metadata.get("id") or "").strip()
    if not package_ref:
        raise ValueError(f"plugin manifest missing metadata.id: {zip_path}")

    return package_ref, "sha256:" + hasher.hexdigest()


def _tool_name(package: str) -> str:
    return package.rsplit("/", 1)[-1] or package


def _result_from_output(
    output: Any, *, request_id: str, trace_id: str
) -> PluginInvocationResult:
    frames = getattr(output, "raw_output", None)
    if not isinstance(frames, list):
        content = getattr(output, "content", None)
        try:
            frames = json.loads(content) if content else []
        except (TypeError, ValueError):
            frames = []

    structured_frames = tuple(frame for frame in frames if isinstance(frame, dict))
    stream_data: dict[str, Any] | None = None
    for frame in structured_frames:
        frame_type = frame.get("type")
        if frame_type == "error":
            data = frame.get("data")
            retryable = isinstance(data, dict) and data.get("retryable") is True
            raise PluginInvocationError(_error_message(frame), retryable=retryable)
        if frame_type == "stream":
            data = frame.get("data")
            if isinstance(data, dict):
                stream_data = data

    if stream_data is None:
        raise PluginInvocationError("plugin returned no stream result", retryable=False)
    return PluginInvocationResult(
        result=stream_data,
        frames=structured_frames,
        request_id=request_id,
        trace_id=trace_id,
    )


def _error_message(frame: dict[str, Any]) -> str:
    data = frame.get("data")
    if isinstance(data, dict) and data.get("message"):
        return str(data["message"])
    error = frame.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return "plugin invocation error"


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"plugin {field_name} must be a list")
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"plugin {field_name} must contain non-empty strings")
        values.append(item.strip())
    return values


def _runtime_policy(
    entry: dict[str, Any], raw_policy: dict[str, Any], package: str
) -> tuple[dict[str, Any], float]:
    policy = dict(DEFAULT_RUNTIME_POLICY)
    aliases = {
        "timeout_ms": "timeoutMs",
        "max_concurrent": "maxConcurrent",
        "max_memory_bytes": "maxMemoryBytes",
        "network_policy": "networkPolicy",
    }
    normalized_policy = {
        aliases.get(key, key): value for key, value in raw_policy.items()
    }
    allowed_fields = {
        "timeoutMs",
        "maxConcurrent",
        "maxMemoryBytes",
        "networkPolicy",
    }
    unknown_fields = sorted(set(normalized_policy) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            f"plugin {package!r} runtime_policy contains unsupported fields: "
            f"{', '.join(unknown_fields)}"
        )
    policy.update(normalized_policy)

    timeout_value = entry.get("timeout_s", entry.get("timeoutS"))
    if timeout_value is None:
        timeout_ms = normalized_policy.get("timeoutMs", _DEFAULT_TIMEOUT_S * 1000)
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or timeout_ms <= 0
        ):
            raise ValueError(f"plugin {package!r} timeoutMs must be a positive integer")
        timeout_s = timeout_ms / 1000
        policy["timeoutMs"] = timeout_ms
    else:
        timeout_s = _positive_number(timeout_value, "timeout_s", package)
        policy["timeoutMs"] = int(timeout_s * 1000)

    for name in ("timeoutMs", "maxConcurrent", "maxMemoryBytes"):
        value = policy.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"plugin {package!r} {name} must be a positive integer")

    network_policy = policy.get("networkPolicy")
    if network_policy not in {"none", "manifest_declared", "internal_only"}:
        raise ValueError(
            f"plugin {package!r} networkPolicy must be none, "
            "manifest_declared, or internal_only"
        )
    return policy, timeout_s


def _positive_number(value: Any, field_name: str, package: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"plugin {package!r} {field_name} must be a positive number")
    return float(value)
