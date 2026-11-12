"""Authenticated catalog commands backed by the gateway's PostgreSQL pool."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from psycopg import OperationalError
from psycopg.errors import (
    ForeignKeyViolation,
    UndefinedColumn,
    UndefinedTable,
    UniqueViolation,
)
from pydantic import ValidationError

from marie.auth.auth_bearer import TokenBearer
from marie.kb.catalog import CatalogError, KbCatalog
from marie.kb.catalog_models import COMMANDS
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool


def register_kb_catalog_routes(
    app: FastAPI, get_pool: Callable[[], AsyncPostgresConnectionPool | None]
) -> None:
    """Mount explicit KB operations without requiring an embedding executor."""

    def error(message: str, status: int, code: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={'status': 'error', 'code': code, 'message': message},
        )

    @app.post(
        '/api/v1/kb/catalog/{operation}',
        tags=['KB'],
        dependencies=[Depends(TokenBearer())],
    )
    async def catalog(operation: str, request: Request) -> Any:
        if operation not in COMMANDS:
            return error('Unknown KB catalog operation', 404, 'not_found')
        raw = await request.body()
        if len(raw) > 8 * 1024 * 1024:
            return error('KB catalog request is too large', 400, 'invalid_request')
        try:
            body = json.loads(raw)
            command = COMMANDS[operation].model_validate(body)
        except (ValueError, ValidationError):
            return error('Invalid KB catalog request', 400, 'invalid_request')
        pool = get_pool()
        if pool is None or not pool.is_initialized:
            return error('KB catalog database is not available', 503, 'unavailable')
        try:
            result = await KbCatalog(pool).call(
                operation, command.model_dump(mode='json', exclude_unset=True)
            )
            return {'status': 'ok', 'result': result}
        except CatalogError as failure:
            code = {
                400: 'invalid_request',
                403: 'forbidden',
                404: 'not_found',
                409: 'conflict',
            }.get(failure.status_code, 'unavailable')
            return error(str(failure), failure.status_code, code)
        except (UniqueViolation, ForeignKeyViolation):
            return error(
                'Knowledge base identity or ownership conflicts with existing state',
                409,
                'conflict',
            )
        except OperationalError:
            return error('KB catalog database is not available', 503, 'unavailable')
        except (UndefinedTable, UndefinedColumn):
            return error('KB catalog schema is not available', 503, 'unavailable')
