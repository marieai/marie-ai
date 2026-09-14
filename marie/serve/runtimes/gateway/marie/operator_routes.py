"""Read-only fabric-scoped operator diagnostics."""

from typing import Callable
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from marie.engine.llm_queue import registry

from marie.auth.api_key_manager import APIKeyManager
from marie.auth.auth_bearer import TokenBearer


def add_runtime_routes(app: FastAPI, effective_fabric: Callable[[], str]) -> None:
    async def authorize(
        fabric_group_id: str = Query(min_length=1, max_length=128),
        tenant_id: str | None = None,
        token: str = Depends(TokenBearer()),
    ) -> str:
        if (
            fabric_group_id != effective_fabric()
            or not APIKeyManager.can_observe_runtime(token, fabric_group_id, tenant_id)
        ):
            raise HTTPException(
                status_code=403, detail='runtime_observability_forbidden'
            )
        return fabric_group_id

    async def read(fabric_group_id: str, limit: int, *, debug: bool = False):
        try:
            snapshot = await registry.read_runtime_snapshot(
                fabric_group_id=fabric_group_id, limit=limit
            )
        except registry.SnapshotUnavailable:
            return JSONResponse(
                status_code=503,
                content={
                    'status': 'error',
                    'category': 'runtime_unavailable',
                    'correlation_id': uuid4().hex,
                },
            )
        result = (
            {'llm_dispatch': snapshot, 'scheduler_details_available': False}
            if debug
            else snapshot
        )
        return {'status': 'OK', 'result': result}

    @app.get('/api/llm-dispatch/runtime')
    async def runtime(
        fabric_group_id: str = Depends(authorize),
        limit: int = Query(default=50, ge=1, le=250),
    ):
        return await read(fabric_group_id, limit)

    @app.get('/api/debug')
    async def debug(fabric_group_id: str = Depends(authorize)):
        return await read(fabric_group_id, 50, debug=True)
