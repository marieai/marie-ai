"""FastAPI route registration for blueprint-import and plugin-install endpoints.

Call :func:`register_blueprint_routes` from the gateway's
``_extend_rest_function`` hook to mount:

    POST /api/v1/blueprints/import
    POST /api/v1/plugins/install

Both endpoints require ``Authorization: Bearer <token>`` validated by
:class:`~marie.auth.auth_bearer.TokenBearer`, matching the pattern used by
``/api/v1/invoke``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from marie.logging_core.logger import MarieLogger
from marie.sandbox.blueprints.registry import BlueprintRegistry
from marie.sandbox.blueprints.service import BlueprintImportService, install_plugin

if TYPE_CHECKING:
    from fastapi import FastAPI

_logger = MarieLogger('marie.sandbox.blueprints.gateway_routes')

_DEFAULT_REGISTRY = BlueprintRegistry()
_DEFAULT_SERVICE = BlueprintImportService()


def register_blueprint_routes(
    app: 'FastAPI',
    registry: BlueprintRegistry | None = None,
    service: BlueprintImportService | None = None,
) -> None:
    """Mount blueprint-import and plugin-install routes onto *app*.

    Args:
        app:      The FastAPI application instance.
        registry: Optional :class:`BlueprintRegistry` override (useful in tests).
        service:  Optional :class:`BlueprintImportService` override (useful in tests).
    """
    from fastapi import Depends, Request
    from fastapi.responses import JSONResponse

    from marie.auth.auth_bearer import TokenBearer

    _registry = registry or _DEFAULT_REGISTRY
    _service = service or _DEFAULT_SERVICE

    @app.post(
        '/api/v1/blueprints/import',
        summary='Import a blueprint into this gateway (sandbox seeding)',
        tags=['Sandbox'],
        dependencies=[Depends(TokenBearer())],
    )
    async def import_blueprint(
        request: Request,
        _token: str = Depends(TokenBearer()),
    ) -> JSONResponse:
        """Install all artifacts from a blueprint into this gateway's registries.

        Request body::

            { "blueprintId": "<id>" }

        The Studio seam (sandbox-seeding.service.ts) calls this endpoint after
        Argo reports Synced and the Wave-1 seed-defaults Job has completed.

        Returns HTTP 200 for all outcomes (completed / partial / deferred).
        Returns HTTP 400 for malformed requests and HTTP 404 when the blueprint
        is not found in the local registry.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={'error': 'invalid JSON body'})

        blueprint_id = body.get('blueprintId') or body.get('blueprint_id')
        if not blueprint_id or not isinstance(blueprint_id, str):
            return JSONResponse(
                status_code=400,
                content={'error': "missing or invalid 'blueprintId' field"},
            )

        manifest = _registry.lookup(blueprint_id)
        if manifest is None:
            _logger.warning(f'Blueprint not found: {blueprint_id!r}')
            return JSONResponse(
                status_code=404,
                content={
                    'error': f'Blueprint {blueprint_id!r} not found in local registry',
                    'hint': (
                        'Set MARIE_BLUEPRINTS_DIR to a directory containing '
                        f'{blueprint_id}.yaml, or add it to marie/sandbox/blueprints/builtin/'
                    ),
                },
            )

        _logger.info(f'Importing blueprint {blueprint_id!r}')
        result = _service.import_blueprint(blueprint_id, manifest)
        return JSONResponse(status_code=200, content=result.model_dump())

    @app.post(
        '/api/v1/plugins/install',
        summary='Install a plugin/extension into this gateway (sandbox seeding)',
        tags=['Sandbox'],
        dependencies=[Depends(TokenBearer())],
    )
    async def install_plugin_endpoint(
        request: Request,
        _token: str = Depends(TokenBearer()),
    ) -> JSONResponse:
        """Install a plugin into this gateway.

        Request body::

            { "packageId": "<id>", "version": "<semver>" }

        The Studio seam calls this endpoint once per plugin ref in the Snapshot's
        ``pluginRefs`` array, after blueprint import completes.

        Current reality: plugin daemon install is not yet implemented (pending
        dify-parity slices 03/05).  The endpoint always returns HTTP 200; the
        ``status`` field in the body reports the true outcome.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={'error': 'invalid JSON body'})

        package_id = body.get('packageId') or body.get('package_id')
        version = body.get('version')

        if not package_id or not isinstance(package_id, str):
            return JSONResponse(
                status_code=400,
                content={'error': "missing or invalid 'packageId' field"},
            )
        if not version or not isinstance(version, str):
            return JSONResponse(
                status_code=400,
                content={'error': "missing or invalid 'version' field"},
            )

        _logger.info(f'Plugin install request: {package_id!r}@{version}')
        result = install_plugin(package_id, version)
        return JSONResponse(status_code=200, content=result)
