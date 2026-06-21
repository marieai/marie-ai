"""
Marie Sandbox — Wave-2 blueprint install entrypoint.

Downloads a named blueprint archive from a registry and installs its
artifacts into a freshly seeded sandbox by calling the gateway's existing
HTTP endpoints.

Integration contract — what this module assumes
------------------------------------------------
Confirmed to exist in marie_gateway.py:
  POST /api/connectors/deploy  (line ~1128) — deploys a connector bundle
  POST /api/planners           (line ~1013) — registers a query planner

Integration seam — what is NOT yet built:
  * A blueprint package registry serving ``<registry_url>/<id>.blueprint``
    tarballs.  The Job will fail with a connection error until a registry is
    deployed and reachable from the sandbox namespace.
  * A dedicated ``POST /api/blueprints/import`` gateway endpoint.  The full
    blueprint importer (prompts, RAG indexes, workflows, agents, webapps,
    sample data) lives in marie-studio (TypeScript), not marie-ai.
  * This module only handles the connector and query-planner artifact kinds
    because those are the only gateway endpoints that exist today.

Until the full importer is wired, the Job runs and marks wave-2 complete for
the connector/planner subset.  All other artifact kinds are logged and skipped.

CLI::

    python -m marie.sandbox install-blueprint \\
        --gateway-url http://sbx-test-server:51000 \\
        --api-key mas_... \\
        --blueprint-id ner-vlm-ocr-entity-extraction \\
        --registry-url https://blueprints.example.com

Public surface
--------------
``install_blueprint(gateway_url, api_key, blueprint_id, registry_url)``
    Downloads and installs the blueprint.  Returns a :class:`BlueprintInstallResult`.
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

_logger = MarieLogger('marie.sandbox.install_blueprint')

# ----------------------------------------------------------------- models ---


@dataclass
class BlueprintInstallResult:
    """Summary of what the blueprint install Job did."""

    blueprint_id: str
    connectors_installed: list[str] = field(default_factory=list)
    planners_registered: list[str] = field(default_factory=list)
    skipped_kinds: list[str] = field(default_factory=list)
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
    """Download *url* and return the raw bytes."""
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
    """POST *payload* as JSON to *url*, return parsed response JSON."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_txt = exc.read().decode('utf-8', errors='replace')
        if exc.code == 409:
            # Conflict — already installed; treat as success
            _logger.info(
                f'POST {url} → 409 Conflict (already installed), treating as OK'
            )
            return {'success': True, 'conflict': True, 'body': body_txt}
        raise RuntimeError(f'HTTP {exc.code} posting to {url}: {body_txt}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Cannot reach {url}: {exc.reason}') from exc


def _download_blueprint(registry_url: str, blueprint_id: str) -> bytes:
    """Fetch ``<registry_url>/<blueprint_id>.blueprint`` and return the bytes."""
    url = f'{registry_url.rstrip("/")}/{blueprint_id}.blueprint'
    _logger.info(f'Downloading blueprint from {url}')
    return _http_get(url)


def _extract_manifest(archive_bytes: bytes) -> dict[str, Any]:
    """Extract and parse ``blueprint.yaml`` from a ``.blueprint`` tar.gz archive."""
    import yaml  # type: ignore[import]

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:gz') as tf:
        try:
            member = tf.getmember('blueprint.yaml')
        except KeyError as exc:
            raise ValueError('blueprint.yaml not found at archive root') from exc
        fh = tf.extractfile(member)
        if fh is None:
            raise ValueError('blueprint.yaml is not a regular file')
        return yaml.safe_load(fh.read())


# ----------------------------------------------------------------- install ---


def _install_connectors(
    gateway_url: str,
    api_key: str,
    archive_bytes: bytes,
    manifest: dict[str, Any],
) -> list[str]:
    """Deploy connector artifacts from the blueprint to the gateway."""
    installed: list[str] = []
    artifacts = manifest.get('artifacts', [])
    connector_artifacts = [a for a in artifacts if a.get('kind') == 'connector']

    if not connector_artifacts:
        return installed

    headers = _auth_headers(api_key)
    endpoint = f'{gateway_url.rstrip("/")}/api/connectors/deploy'

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:gz') as tf:
        for artifact in connector_artifacts:
            ref = artifact.get('ref', '')
            path = artifact.get('path', '')
            if not path:
                _logger.warning(f'Connector artifact {ref!r} has no path, skipping')
                continue

            # Collect all files under the connector directory into a bundle
            files: dict[str, str] = {}
            for member in tf.getmembers():
                if member.name.startswith(path) and member.isfile():
                    rel = member.name[len(path) :].lstrip('/')
                    fh = tf.extractfile(member)
                    if fh:
                        files[rel] = fh.read().decode('utf-8', errors='replace')

            if not files:
                _logger.warning(
                    f'No files found under path {path!r} for connector {ref!r}'
                )
                continue

            payload = {
                'connector_id': ref,
                'files': files,
                'source_type': 'blueprint',
                'overwrite': False,
            }
            _logger.info(f'Deploying connector {ref!r} to {endpoint}')
            _http_post_json(endpoint, payload, headers)
            installed.append(ref)

    return installed


def _install_planners(
    gateway_url: str,
    api_key: str,
    archive_bytes: bytes,
    manifest: dict[str, Any],
) -> list[str]:
    """Register query-plan artifacts from the blueprint with the gateway."""
    registered: list[str] = []
    artifacts = manifest.get('artifacts', [])
    plan_artifacts = [a for a in artifacts if a.get('kind') == 'query_plan']

    if not plan_artifacts:
        return registered

    headers = _auth_headers(api_key)
    endpoint = f'{gateway_url.rstrip("/")}/api/planners'

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:gz') as tf:
        for artifact in plan_artifacts:
            ref = artifact.get('ref', '')
            path = artifact.get('path', '')
            create = artifact.get('create', {})

            plan_def: dict[str, Any] = {}
            if path:
                import yaml  # type: ignore[import]

                try:
                    member = tf.getmember(path)
                    fh = tf.extractfile(member)
                    if fh:
                        plan_def = yaml.safe_load(fh.read()) or {}
                except KeyError:
                    _logger.warning(f'Query plan path {path!r} not found in archive')
            if create:
                plan_def.update(create)

            payload = {
                'name': ref,
                'plan': plan_def,
                'description': f'Installed from blueprint {manifest.get("id", "")}',
                'version': manifest.get('version', '1.0.0'),
                'tags': ['blueprint'],
                'category': 'blueprint',
            }
            _logger.info(f'Registering query plan {ref!r} to {endpoint}')
            _http_post_json(endpoint, payload, headers)
            registered.append(ref)

    return registered


# ----------------------------------------------------------------- public ---

_HANDLED_KINDS = {'connector', 'query_plan'}

_SKIPPABLE_KINDS = {
    'prompt_package',
    'script_package',
    'prefab',
    'skill',
    'agent',
    'webapp_package',
    'tenant_registry',
    'rag_index',
    'rag_source',
    'sample_data',
}


def install_blueprint(
    gateway_url: str,
    api_key: str,
    blueprint_id: str,
    registry_url: str,
) -> BlueprintInstallResult:
    """Download and install a blueprint into the sandbox gateway.

    Args:
        gateway_url:   Base URL of the sandbox gateway HTTP service,
                       e.g. ``http://sbx-test-server:51000``.
        api_key:       Admin API key produced by Wave-1 seed-defaults.
        blueprint_id:  Blueprint identifier, e.g.
                       ``ner-vlm-ocr-entity-extraction``.
        registry_url:  Base URL of the blueprint registry.  The Job GETs
                       ``<registry_url>/<blueprint_id>.blueprint``.

    Returns:
        :class:`BlueprintInstallResult` — what was installed and what was skipped.

    Raises:
        :class:`~marie.excepts.BadConfigSource`: if required args are empty.
        ``RuntimeError``: on download or gateway API failure.
    """
    if not gateway_url:
        raise BadConfigSource('gateway_url is required for blueprint install')
    if not api_key:
        raise BadConfigSource('api_key is required for blueprint install')
    if not blueprint_id:
        raise BadConfigSource('blueprint_id is required for blueprint install')
    if not registry_url:
        raise BadConfigSource('registry_url is required for blueprint install')

    _logger.info(f'Starting Wave-2 blueprint install: blueprint_id={blueprint_id!r}')

    result = BlueprintInstallResult(blueprint_id=blueprint_id)

    # Download archive
    try:
        archive_bytes = _download_blueprint(registry_url, blueprint_id)
    except RuntimeError as exc:
        result.errors.append(f'Download failed: {exc}')
        _logger.error(str(exc))
        return result

    # Parse manifest
    try:
        manifest = _extract_manifest(archive_bytes)
    except (ValueError, Exception) as exc:
        result.errors.append(f'Manifest parse failed: {exc}')
        _logger.error(str(exc))
        return result

    # Identify artifact kinds present but not handled
    all_kinds = {a.get('kind', '') for a in manifest.get('artifacts', [])}
    result.skipped_kinds = sorted(all_kinds & _SKIPPABLE_KINDS)
    if result.skipped_kinds:
        _logger.warning(
            f'Skipping artifact kinds not yet supported by marie-ai gateway: '
            f'{result.skipped_kinds}.  '
            f'These require the full blueprint importer in marie-studio.'
        )

    # Install connectors
    try:
        result.connectors_installed = _install_connectors(
            gateway_url, api_key, archive_bytes, manifest
        )
    except RuntimeError as exc:
        result.errors.append(f'Connector install failed: {exc}')
        _logger.error(str(exc))

    # Install query planners
    try:
        result.planners_registered = _install_planners(
            gateway_url, api_key, archive_bytes, manifest
        )
    except RuntimeError as exc:
        result.errors.append(f'Planner registration failed: {exc}')
        _logger.error(str(exc))

    _logger.info(
        f'Blueprint install complete: connectors={result.connectors_installed} '
        f'planners={result.planners_registered} '
        f'skipped={result.skipped_kinds} '
        f'errors={result.errors}'
    )
    return result


# -------------------------------------------------------------------- CLI ---


def _config_from_env() -> dict[str, str]:
    return {
        'gateway_url': os.getenv('SANDBOX_GATEWAY_URL', ''),
        'api_key': os.getenv('SANDBOX_ADMIN_API_KEY', ''),
        'blueprint_id': os.getenv('SANDBOX_BLUEPRINT_ID', ''),
        'registry_url': os.getenv('SANDBOX_BLUEPRINT_REGISTRY_URL', ''),
    }


def _cli_install_blueprint(args: 'argparse.Namespace') -> None:  # noqa: F821
    result = install_blueprint(
        gateway_url=args.gateway_url,
        api_key=args.api_key,
        blueprint_id=args.blueprint_id,
        registry_url=args.registry_url,
    )
    print(json.dumps(result.__dict__))
    if not result.success:
        raise SystemExit(1)


def build_parser() -> 'argparse.ArgumentParser':  # noqa: F821
    """Return the argument parser for the install-blueprint CLI sub-command."""
    import argparse

    env = _config_from_env()
    p = argparse.ArgumentParser(
        prog='python -m marie.sandbox install-blueprint',
        description=(
            'Wave-2 blueprint install: download a blueprint archive from the '
            'registry and deploy its connector + query-plan artifacts to the '
            'sandbox gateway.'
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
        '--blueprint-id',
        default=env['blueprint_id'],
        help='Blueprint identifier (env: SANDBOX_BLUEPRINT_ID)',
    )
    p.add_argument(
        '--registry-url',
        default=env['registry_url'],
        help='Blueprint registry base URL (env: SANDBOX_BLUEPRINT_REGISTRY_URL)',
    )
    return p
