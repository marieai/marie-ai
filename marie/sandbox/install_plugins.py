"""
Marie Sandbox — Wave-3 plugin / extension install entrypoint.

Resolves the blueprint's ``extension_package`` plugin refs and installs each
into the sandbox gateway's connector registry.

Integration contract — what this module assumes
------------------------------------------------
Confirmed to exist in marie_gateway.py:
  POST /api/connectors/deploy  (line ~1128) — deploys a connector bundle

Integration seam — what is NOT yet built:
  * A plugin marketplace / registry serving
    ``<registry_url>/<package_id>/<version>.plugin`` tarballs.
  * A dedicated plugin-install gateway endpoint (e.g. POST /api/plugins/install)
    backed by the plugin daemon executor infrastructure that exists in
    ``marie/executor/extensions/plugin_daemon_executor.py``.
  * Credential binding: ``installMode: install-from-registry`` refs that
    require ``credentialBindings`` are logged and skipped until Slice 6
    provides the per-sandbox Secret delivery mechanism.

The Job installs plugin refs via the confirmed connector-deploy endpoint.
Plugin packages that require sidecar daemon registration (beyond what
connector-deploy provides) are noted in the result's ``deferred`` list.

CLI::

    python -m marie.sandbox install-plugins \\
        --gateway-url http://sbx-test-server:51000 \\
        --api-key mas_... \\
        --registry-url https://plugins.example.com \\
        --plugin-refs '[{"packageId":"p1","version":"1.0.0"}]'

Public surface
--------------
``install_plugins(gateway_url, api_key, plugin_refs, registry_url)``
    Installs each plugin ref.  Returns a :class:`PluginInstallResult`.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from marie.excepts import BadConfigSource
from marie.logging_core.logger import MarieLogger

_logger = MarieLogger('marie.sandbox.install_plugins')

# ----------------------------------------------------------------- models ---


@dataclass
class PluginRef:
    """A single plugin dependency declared in a blueprint's extension_package."""

    package_id: str
    version: str
    install_mode: str = 'install-from-registry'
    credential_bindings: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PluginRef':
        return cls(
            package_id=data['packageId'],
            version=data.get('version', 'latest'),
            install_mode=data.get('installMode', 'install-from-registry'),
            credential_bindings=data.get('credentialBindings', []),
        )


@dataclass
class PluginInstallResult:
    """Summary of what the plugin install Job did."""

    installed: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors


# ----------------------------------------------------------------- helpers ---


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code} fetching {url}: {exc.reason}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Cannot reach {url}: {exc.reason}') from exc


def _http_post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_txt = exc.read().decode('utf-8', errors='replace')
        if exc.code == 409:
            _logger.info(
                f'POST {url} → 409 Conflict (already installed), treating as OK'
            )
            return {'success': True, 'conflict': True}
        raise RuntimeError(f'HTTP {exc.code} posting to {url}: {body_txt}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Cannot reach {url}: {exc.reason}') from exc


def _download_plugin(registry_url: str, ref: PluginRef) -> bytes:
    """Fetch ``<registry_url>/<package_id>/<version>.plugin`` bytes."""
    url = f'{registry_url.rstrip("/")}/{ref.package_id}/{ref.version}.plugin'
    _logger.info(f'Downloading plugin {ref.package_id}@{ref.version} from {url}')
    return _http_get(url)


def _deploy_plugin(
    gateway_url: str,
    api_key: str,
    ref: PluginRef,
    archive_bytes: bytes,
) -> None:
    """Push a ``.plugin`` tar.gz bundle to the gateway connector-deploy endpoint."""
    headers = _auth_headers(api_key)
    endpoint = f'{gateway_url.rstrip("/")}/api/connectors/deploy'

    # Extract all files from the plugin archive into the connector-deploy payload
    files: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:gz') as tf:
        for member in tf.getmembers():
            if member.isfile():
                fh = tf.extractfile(member)
                if fh:
                    files[member.name] = fh.read().decode('utf-8', errors='replace')

    payload = {
        'connector_id': ref.package_id,
        'files': files,
        'source_type': 'plugin',
        'overwrite': False,
    }
    _logger.info(f'Deploying plugin {ref.package_id!r} via {endpoint}')
    _http_post_json(endpoint, payload, headers)


def _has_deferred_credentials(ref: PluginRef) -> bool:
    return any(b.get('bindingMode') == 'deferred' for b in ref.credential_bindings)


# ----------------------------------------------------------------- public ---


def install_plugins(
    gateway_url: str,
    api_key: str,
    plugin_refs: list[dict[str, Any]],
    registry_url: str,
) -> PluginInstallResult:
    """Install plugin refs into the sandbox gateway.

    Args:
        gateway_url:   Base URL of the sandbox gateway HTTP service.
        api_key:       Admin API key from Wave-1 seed-defaults.
        plugin_refs:   List of plugin ref dicts, each with at minimum
                       ``packageId`` and ``version`` keys (matching the
                       blueprint's ``extension_package`` format).
        registry_url:  Base URL of the plugin marketplace registry.

    Returns:
        :class:`PluginInstallResult`.

    Raises:
        :class:`~marie.excepts.BadConfigSource`: if required args are empty.
    """
    if not gateway_url:
        raise BadConfigSource('gateway_url is required for plugin install')
    if not api_key:
        raise BadConfigSource('api_key is required for plugin install')
    if not registry_url:
        raise BadConfigSource('registry_url is required for plugin install')

    result = PluginInstallResult()

    if not plugin_refs:
        _logger.info('No plugin refs provided; Wave-3 is a no-op')
        return result

    _logger.info(f'Starting Wave-3 plugin install: {len(plugin_refs)} ref(s)')

    for raw_ref in plugin_refs:
        try:
            ref = PluginRef.from_dict(raw_ref)
        except (KeyError, TypeError) as exc:
            msg = f'Invalid plugin ref {raw_ref!r}: {exc}'
            result.errors.append(msg)
            _logger.error(msg)
            continue

        if _has_deferred_credentials(ref):
            _logger.info(
                f'Plugin {ref.package_id!r} has deferred credential bindings — '
                'marking as deferred (user must bind via sandbox UI after launch)'
            )
            result.deferred.append(f'{ref.package_id}@{ref.version}')

        try:
            archive_bytes = _download_plugin(registry_url, ref)
            _deploy_plugin(gateway_url, api_key, ref, archive_bytes)
            result.installed.append(f'{ref.package_id}@{ref.version}')
        except RuntimeError as exc:
            msg = f'Plugin {ref.package_id}@{ref.version} failed: {exc}'
            result.errors.append(msg)
            _logger.error(msg)

    _logger.info(
        f'Plugin install complete: installed={result.installed} '
        f'deferred={result.deferred} errors={result.errors}'
    )
    return result


# -------------------------------------------------------------------- CLI ---


def _config_from_env() -> dict[str, str]:
    return {
        'gateway_url': os.getenv('SANDBOX_GATEWAY_URL', ''),
        'api_key': os.getenv('SANDBOX_ADMIN_API_KEY', ''),
        'registry_url': os.getenv('SANDBOX_PLUGIN_REGISTRY_URL', ''),
        'plugin_refs_json': os.getenv('SANDBOX_PLUGIN_REFS', '[]'),
    }


def _cli_install_plugins(args: 'argparse.Namespace') -> None:  # noqa: F821
    try:
        plugin_refs = json.loads(args.plugin_refs)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'--plugin-refs must be valid JSON: {exc}') from exc

    result = install_plugins(
        gateway_url=args.gateway_url,
        api_key=args.api_key,
        plugin_refs=plugin_refs,
        registry_url=args.registry_url,
    )
    print(json.dumps(result.__dict__))
    if not result.success:
        raise SystemExit(1)


def build_parser() -> 'argparse.ArgumentParser':  # noqa: F821
    """Return the argument parser for the install-plugins CLI sub-command."""
    import argparse

    env = _config_from_env()
    p = argparse.ArgumentParser(
        prog='python -m marie.sandbox install-plugins',
        description=(
            'Wave-3 plugin install: resolve and install blueprint extension_package '
            'plugin refs into the sandbox gateway.'
        ),
    )
    p.add_argument(
        '--gateway-url',
        default=env['gateway_url'],
        help='Base URL of the sandbox gateway HTTP service (env: SANDBOX_GATEWAY_URL)',
    )
    p.add_argument(
        '--api-key',
        default=env['api_key'],
        help='Admin API key from Wave-1 seed (env: SANDBOX_ADMIN_API_KEY)',
    )
    p.add_argument(
        '--registry-url',
        default=env['registry_url'],
        help='Plugin marketplace registry base URL (env: SANDBOX_PLUGIN_REGISTRY_URL)',
    )
    p.add_argument(
        '--plugin-refs',
        default=env['plugin_refs_json'],
        help=(
            'JSON array of plugin ref objects '
            '(env: SANDBOX_PLUGIN_REFS, e.g. '
            '\'[{"packageId":"p1","version":"1.0.0"}]\')'
        ),
    )
    return p
