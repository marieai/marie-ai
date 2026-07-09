"""FastAPI route registration for the blueprint-import endpoint.

Call :func:`register_blueprint_routes` from the gateway's
``_extend_rest_function`` hook to mount:

    POST /api/v1/blueprints/import

The endpoint requires ``Authorization: Bearer <token>`` validated by
:class:`~marie.auth.auth_bearer.TokenBearer`, matching the pattern used by
``/api/v1/invoke``.

Plugin installation is NOT a gateway concern: plugins are installed and managed
system-wide by the daemon-plugin system via Studio (the Extensions package
store).  The gateway only *invokes* installed plugins (plugin_daemon_executor).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from marie.logging_core.logger import MarieLogger
from marie.sandbox.blueprints.registry import BlueprintRegistry
from marie.sandbox.blueprints.service import BlueprintImportService

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
