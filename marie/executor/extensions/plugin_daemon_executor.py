import asyncio
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from docarray import DocList
from docarray.documents import TextDoc
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as SchemaValidationError
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable

from marie.constants import DEFAULT_DAEMON_ADDR, DEFAULT_TENANT_UUID
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger
from marie.plugin_daemon import (
    PluginInvocationSpec,
    build_invocation_envelope,
    discover_daemon,
    runtime_error_frame,
    sign_envelope,
)
from marie.plugin_daemon.envelope import verify_envelope
from marie.plugin_daemon.frames import now_utc
from marie.plugins.daemon_client import PluginDaemonClient, PluginDaemonHTTPError
from marie.runtime import requests


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
        self._signing_key_id = self._env.get("MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID", "")
        self._signing_secret = self._env.get("MARIE_PLUGIN_DAEMON_SIGNING_SECRET", "")
        if bool(self._signing_key_id) != bool(self._signing_secret):
            raise ValueError(
                "MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID and "
                "MARIE_PLUGIN_DAEMON_SIGNING_SECRET must be configured together"
            )
        self._start_enabled = start
        self._child: subprocess.Popen[bytes] | None = None
        self._started_at: datetime | None = None
        self._last_health: dict[str, Any] | None = None
        self._last_exit_code: int | None = None
        self._restart_attempts = 0
        self._next_restart_at: datetime | None = None
        self._draining = False
        self._lifecycle_lock = threading.Lock()
        self._datasource_bucket = (kwargs.get('storage', {}).get('s3') or {}).get(
            'bucket_name'
        )
        self._datasource_pool = None
        self._datasource_pool_lock = asyncio.Lock()
        self.discovery = discover_daemon(daemon_url, daemon_bin, daemon_addr, self._env)

        if start and self.discovery.mode == "binary_child":
            self._start_child()

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            self._draining = True
            child = self._child
            if child is None or child.poll() is not None:
                self._last_exit_code = (
                    child.returncode if child else self._last_exit_code
                )
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
        status = await asyncio.to_thread(self.status_payload)
        return DocList[TextDoc](
            [TextDoc(text=json.dumps(status, separators=(",", ":")))]
        )

    @requests(on="/execute")
    async def invoke_plugin(
        self,
        docs: DocList[TextDoc],
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DocList[TextDoc]:
        """Dispatch an admitted plugin action and return its runtime frames.

        Routed here by the fixed endpoint ``plugin_daemon_executor://execute``.
        Scheduled settings come from ``payload.op_params``; direct requests use
        top-level parameters. Legacy tools retain the nil-tenant fallback;
        datasources require explicit organization, workspace and source identity.
        """
        execution_params = parameters or {}
        request_id = str(execution_params.get("job_id") or "")
        scheduled_payload = execution_params.get("payload")
        op_params = (
            scheduled_payload.get("op_params")
            if isinstance(scheduled_payload, dict)
            else None
        )
        params = op_params if isinstance(op_params, dict) else execution_params
        if params.get('operation') == 'kb_datasource_sync':
            return await self._sync_datasource(execution_params)
        if 'invocation_envelope' in params:
            return frames_to_docs(
                await self._invoke_datasource(params['invocation_envelope'], request_id)
            )
        try:
            plugin = params.get('plugin') or {}
            if not isinstance(plugin, dict):
                raise ValueError('plugin identity must be an object')
            action_type = plugin.get('action_type', 'tool')
            action_id = plugin.get('action_id')
            if action_type == 'tool' and action_id is None and plugin.get('tool_ref'):
                action_id = f"tools/{plugin['tool_ref']}"
            spec = PluginInvocationSpec.model_validate(
                {
                    **plugin,
                    'action_type': action_type,
                    'action_id': action_id,
                    'credential_requirements': params.get(
                        'credential_requirements', []
                    ),
                }
            )
            if not spec.package_ref or not spec.package_digest:
                raise ValueError(
                    'plugin invocation requires package_ref and package_digest'
                )
            organization_id = execution_params.get('organization_id') or params.get(
                'organization_id'
            )
            workspace_id = execution_params.get('workspace_id') or params.get(
                'workspace_id'
            )
            if spec.action_type == 'datasource':
                raise ValueError(
                    'Datasource invocation requires signed saved-binding admission'
                )
            if params.get('type') == 'datasource':
                raise ValueError(
                    'Datasource payload requires signed saved-binding admission'
                )
            runtime_policy = params.get('runtime_policy')
            if runtime_policy is not None and not isinstance(runtime_policy, dict):
                raise ValueError('runtime_policy must be an object')
        except (ValueError, TypeError) as error:
            return frames_to_docs(
                [
                    runtime_error_frame(
                        request_id,
                        f"invalid plugin invocation: {error}",
                        "invalid_envelope",
                    )
                ]
            )

        organization_id = str(organization_id or DEFAULT_TENANT_UUID)
        workspace_id = str(workspace_id or DEFAULT_TENANT_UUID)
        user_id = str(execution_params.get("user_id") or params.get("user_id") or "")

        control_keys = {
            "plugin",
            "credential_requirements",
            "organization_id",
            "workspace_id",
            "user_id",
            "job_id",
            "payload",
            "runtime_policy",
        }
        op_payload = {k: v for k, v in params.items() if k not in control_keys}
        frames = await asyncio.to_thread(
            self._invoke_daemon,
            spec,
            op_payload,
            request_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            user_id=user_id,
            runtime_policy=runtime_policy,
        )
        return frames_to_docs(frames)

    async def _sync_datasource(self, parameters: dict[str, Any]) -> DocList[TextDoc]:
        from marie.kb.datasource import admit_datasource_items
        from marie.plugins.datasource import load_datasource_binding
        from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool

        if not self._datasource_bucket:
            raise ValueError(
                'Datasource sync requires the existing storage.s3.bucket_name'
            )
        if self._datasource_pool is None:
            async with self._datasource_pool_lock:
                if self._datasource_pool is None:
                    storage = getattr(self, 'storage_conf', None)
                    if not storage:
                        raise ValueError(
                            'Datasource sync requires the existing storage.psql configuration'
                        )
                    pool = AsyncPostgresConnectionPool()
                    await pool.initialize(storage, autocommit=True)
                    self._datasource_pool = pool
        root = Path(
            self._env.get('MARIE_EXTENSION_PACKAGE_STORE')
            or str(
                Path(self._env.get('MARIE_DEFAULT_MOUNT', '/mnt/data/marie-ai'))
                / 'plugin-packages'
            )
        )
        binding = await load_datasource_binding(self._datasource_pool, parameters, root)
        binding['bucket_name'] = self._datasource_bucket
        spec = PluginInvocationSpec(
            action_type='datasource',
            action_id=binding['datasource_ref'],
            install_id=str(binding['extension_install_id']),
            package_id=str(binding['package_id']),
            package_ref=binding['package_ref'],
            package_digest=binding['package_digest'],
            package_trust_level=binding['trust_level'],
            provider_id=str(binding['provider_id']) if binding['provider_id'] else None,
            credential_binding_ids=[
                str(row['binding_id'])
                for row in binding['credentials']
                if row['binding_id']
            ],
            credential_requirements=[
                {
                    'name': row['name'],
                    'required': row['required'],
                    'secret_ref': row['secret_ref'],
                }
                for row in binding['credentials']
            ],
        )
        envelope = build_invocation_envelope(
            spec,
            organization_id=str(binding['organization_id']),
            workspace_id=str(binding['workspace_id']),
            user_id=binding['user_id'],
            request_id=parameters['job_id'],
            payload={
                'type': 'datasource',
                'datasource': binding['datasource_ref'],
                'source_id': str(binding['source_id']),
                'index_id': str(binding['index_id']),
                'tenant_id': str(binding['tenant_id']),
                'configuration': binding['dataset_inputs'],
            },
            timeout_ms=(binding['manifest_snapshot'].get('runtimePolicy') or {}).get(
                'timeoutMs', int(self.invoke_timeout_s * 1000)
            ),
            runtime_policy=binding['manifest_snapshot'].get('runtimePolicy'),
        )
        envelope['input'] = {
            'credential_requirements': spec.credential_requirements,
            'output_schema': binding['output_schema'],
        }
        envelope = sign_envelope(
            envelope, key_id=self._signing_key_id, secret=self._signing_secret
        )
        frames = await self._invoke_datasource(envelope, parameters['job_id'])
        if any(frame.get('type') == 'error' for frame in frames):
            raise ValueError(
                'Datasource execution failed; inspect the plugin worker logs'
            )
        records = next(
            frame['data']['records']
            for frame in frames
            if frame.get('type') == 'stream'
        )
        count = await admit_datasource_items(self._datasource_pool, binding, records)
        return frames_to_docs(
            [{'type': 'stream', 'data': {'records_admitted': count}}, {'type': 'end'}]
        )

    async def _invoke_datasource(
        self, envelope: dict[str, Any], request_id: str
    ) -> list[dict[str, Any]]:
        try:
            if not isinstance(envelope, dict):
                raise ValueError('Plugin admission must be an object')
            verify_envelope(
                envelope, key_id=self._signing_key_id, secret=self._signing_secret
            )
            payload = envelope['payload']
            claims = envelope['input']
            scope = [
                envelope.get('organizationId'),
                envelope.get('workspaceId'),
                envelope.get('installId'),
                envelope.get('userId'),
                envelope.get('packageId'),
                envelope.get('packageRef'),
                envelope.get('packageDigest'),
            ]
            scope += [
                payload.get(key) for key in ('source_id', 'index_id', 'tenant_id')
            ]
            if not all(
                isinstance(value, str)
                and value.strip()
                and value.strip() != DEFAULT_TENANT_UUID
                for value in scope
            ):
                raise ValueError(
                    'Datasource admission requires a complete binding scope'
                )
            if (
                envelope.get('actionType') != 'datasource'
                or payload.get('type') != 'datasource'
                or payload.get('datasource') != envelope.get('actionId')
            ):
                raise ValueError('Datasource admission action mismatch')
            if 'credentials' in payload:
                raise ValueError(
                    'Datasource admission must carry references, not credentials'
                )
            spec = PluginInvocationSpec(
                action_type='datasource',
                action_id=envelope['actionId'],
                install_id=envelope['installId'],
                package_id=envelope['packageId'],
                package_ref=envelope['packageRef'],
                package_digest=envelope['packageDigest'],
                package_trust_level=envelope['packageTrustLevel'],
                provider_id=envelope.get('providerId'),
                credential_binding_ids=envelope['credentialBindingIds'],
                credential_requirements=claims['credential_requirements'],
            )
            request_id = envelope['requestId']
            schema = claims['output_schema']
            if not isinstance(schema, dict):
                raise ValueError('Datasource admission requires an output schema')
            validator_type = validator_for(schema)
            validator_type.check_schema(schema)
            validator = validator_type(schema, registry=Registry())
        except (ValueError, TypeError, KeyError, AttributeError, SchemaError):
            return [
                runtime_error_frame(
                    request_id,
                    'Invalid or expired datasource admission',
                    'invalid_envelope',
                )
            ]
        try:
            frames = await asyncio.to_thread(
                self._invoke_daemon,
                spec,
                payload,
                request_id,
                organization_id=envelope['organizationId'],
                workspace_id=envelope['workspaceId'],
                user_id=envelope['userId'],
                runtime_policy=envelope['runtimePolicy'],
                redact_output=True,
            )
        except (ValueError, TypeError):
            return [
                runtime_error_frame(
                    request_id,
                    'Datasource credentials are unavailable in this executor',
                    'credential_unavailable',
                )
            ]
        if any(frame.get('type') == 'error' for frame in frames):
            return frames
        try:
            results = [
                frame['data'] for frame in frames if frame.get('type') == 'stream'
            ]
            if len(results) != 1 or not any(
                frame.get('type') == 'end' for frame in frames
            ):
                raise ValueError('Incomplete datasource result')
            result = results[0]
            records = result['records']
            if not isinstance(records, list) or set(result) - {'records'}:
                raise ValueError('Invalid datasource result')
            for record in records:
                validator.validate(record)
            if (
                len(
                    json.dumps(result, ensure_ascii=False, allow_nan=False).encode(
                        'utf-8'
                    )
                )
                > 1024 * 1024
            ):
                raise ValueError('Datasource output exceeds its transport bound')
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            SchemaError,
            SchemaValidationError,
            Unresolvable,
        ):
            return [
                runtime_error_frame(
                    request_id, 'Invalid datasource output', 'invalid_datasource_output'
                )
            ]
        return frames

    def status_payload(self) -> dict[str, Any]:
        self._ensure_child()
        health = self._probe_health()
        child = self._child
        return {
            "executor": "MariePluginDaemonExecutor",
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
        self._child = subprocess.Popen(
            [binary, "--addr", self.daemon_addr], env=self._env
        )
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
        with self._lifecycle_lock:
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

    def _invoke_daemon(
        self,
        spec: PluginInvocationSpec,
        payload: dict[str, Any],
        request_id: str,
        *,
        organization_id: str,
        workspace_id: str,
        user_id: str,
        runtime_policy: dict[str, Any] | None,
        redact_output: bool = False,
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

        try:
            with PluginDaemonClient(
                base_url=url,
                organization_id=organization_id,
                workspace_id=workspace_id,
                user_id=user_id,
                signing_key_id=self._signing_key_id,
                signing_secret=self._signing_secret,
                timeout_s=self.invoke_timeout_s,
            ) as client:
                return client.invoke(
                    spec,
                    payload,
                    request_id=request_id or None,
                    runtime_policy=runtime_policy,
                    redact_output=redact_output,
                )
        except PluginDaemonHTTPError as error:
            return [
                runtime_error_frame(
                    request_id,
                    f"daemon invoke returned HTTP {error.status_code}",
                    "runtime_http_error",
                )
            ]
        except RuntimeError as error:
            return [runtime_error_frame(request_id, str(error))]


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
