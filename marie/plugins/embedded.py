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

import hashlib
import json
import os
import posixpath
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from marie.constants import DEFAULT_DAEMON_ADDR, DEFAULT_TENANT_UUID
from marie.logging_core.logger import MarieLogger
from marie.plugin_daemon import build_invocation_envelope, sign_envelope
from marie.plugins.agent_tool import PluginToolSpec
from marie.plugins.daemon_client import PluginDaemonClient

# Loopback integrity key (spec §8.2). Handed to the spawned daemon child via env
# so it verifies the envelopes this process signs. NOT a secret; revisit before
# promoting EmbeddedPlugins to a shared daemon.
_SIGNING_KEY_ID = "marie-executor-embedded"
_SIGNING_SECRET = "marie-executor-embedded-loopback-secret"

_DEFAULT_TIMEOUT_S = 120

ClientFactory = Callable[[float], PluginDaemonClient]


class PluginInvocationError(RuntimeError):
    """A classified error frame returned by the plugin runtime."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class _PluginEntry:
    """One parsed ``with.plugins`` list entry."""

    __slots__ = ("package", "path", "actions", "timeout_s")

    def __init__(
        self,
        package: str,
        path: str,
        actions: list[str],
        timeout_s: float,
    ) -> None:
        self.package = package
        self.path = path
        self.actions = actions
        self.timeout_s = timeout_s

    @classmethod
    def from_config(cls, entry: dict[str, Any]) -> "_PluginEntry":
        package = (entry.get("package") or "").strip()
        path = (entry.get("path") or "").strip()
        if not package or not path:
            raise ValueError(
                "plugin config entry requires both 'package' and 'path' "
                f"(got {entry!r})"
            )
        actions = list(entry.get("actions") or [])
        timeout_s = entry.get("timeout_s") or _DEFAULT_TIMEOUT_S
        return cls(package=package, path=path, actions=actions, timeout_s=timeout_s)


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
        self, package: str, action: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke ``action`` on ``package`` with ``payload``; return its stream data.

        Raises on an error frame or a missing stream result (fail-loud). On an
        invocation failure the daemon child is respawned exactly once and the call
        retried; a second failure propagates.
        """
        self.ensure_started()
        plugin_payload = {**payload, "action": action}
        spec, client = self._require_spec_client(package)
        try:
            output = client.invoke(spec, plugin_payload)
            return _result_from_output(output)
        except PluginInvocationError as error:
            if not error.retryable:
                raise
            first_error = error
        except Exception as error:
            first_error = error
        self.logger.warning(
            f"plugin invoke failed for {package}/{action}; respawning daemon "
            f"once: {first_error}"
        )
        self._respawn()
        spec, client = self._require_spec_client(package)
        output = client.invoke(spec, plugin_payload)
        return _result_from_output(output)

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
            provider_ref=entry.package,
            package_trust_level="builtin",
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


def _result_from_output(output: Any) -> dict[str, Any]:
    frames = getattr(output, "raw_output", None)
    if not isinstance(frames, list):
        content = getattr(output, "content", None)
        try:
            frames = json.loads(content) if content else []
        except (TypeError, ValueError):
            frames = []

    stream_data: dict[str, Any] | None = None
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_type = frame.get("type")
        if frame_type == "error":
            data = frame.get("data")
            retryable = isinstance(data, dict) and data.get("retryable") is True
            raise PluginInvocationError(_error_message(frame), retryable=retryable)
        if frame_type == "stream" and stream_data is None:
            data = frame.get("data")
            if isinstance(data, dict):
                stream_data = data

    if stream_data is None:
        raise PluginInvocationError("plugin returned no stream result", retryable=False)
    return stream_data


def _error_message(frame: dict[str, Any]) -> str:
    data = frame.get("data")
    if isinstance(data, dict) and data.get("message"):
        return str(data["message"])
    error = frame.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return "plugin invocation error"
