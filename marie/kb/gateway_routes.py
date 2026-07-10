"""FastAPI route registration for the knowledge-base gateway API extension.

Call :func:`register_kb_routes` from the gateway's ``_extend_rest_function``
hook to mount:

    POST /api/v1/kb/search
    POST /api/v1/kb/hybrid_search
    POST /api/v1/kb/index_stats
    POST /api/v1/kb/source_stats
    POST /api/v1/kb/delete_source
    POST /api/v1/kb/delete_index

Endpoints require ``Authorization: Bearer <token>`` validated by
:class:`~marie.auth.auth_bearer.TokenBearer`, matching the pattern used by
``/api/v1/invoke`` and ``/api/v1/blueprints/import``.

Architecture: the gateway registers the routes and validates requests; the
KB executor *executes* them (model, vector store, SQL — all its
dependencies live executor-side). Each route forwards to the executor
deployment declared by the config ``executor`` key — deployment names are
config data, never literals in code — and the executor endpoint returns a
plain dict payload (no doc schemas cross this boundary).
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from marie.logging_core.logger import MarieLogger
from marie.proto import jina_pb2
from marie.types_core.request.data import DataRequest

_logger = MarieLogger('marie.kb.gateway_routes').logger

#: action name → (executor endpoint, required body fields)
_KB_ACTIONS: Dict[str, tuple[str, tuple[str, ...]]] = {
    "search": ("/search", ("query",)),
    "hybrid_search": ("/hybrid_search", ("query",)),
    "index_stats": ("/index_stats", ("index_name",)),
    "source_stats": ("/source_stats", ("source_id",)),
    "delete_source": ("/delete_source", ("source_id",)),
    "delete_index": ("/delete_index", ("index_name",)),
}


def _extract_result(parameters: Any) -> Optional[Dict[str, Any]]:
    """Pull a dict-returning executor endpoint's payload out of the response.

    Such payloads land under ``parameters["__results__"][<executor_name>]``;
    read it back without depending on the executor's runtime name.
    """
    params = dict(parameters or {})
    results = params.get("__results__") or {}
    if isinstance(results, dict):
        for payload in results.values():
            if isinstance(payload, dict):
                return payload
    return None


def register_kb_routes(
    app: FastAPI,
    kb_config: Optional[Dict[str, Any]],
    get_streamer: Callable[[], Any],
) -> None:
    """Mount the KB API extension routes onto *app*.

    Args:
        app:          The FastAPI application instance.
        kb_config:    The gateway's ``with.kb`` config block; must declare
                      ``executor`` (the KB executor deployment name).
                      ``None``/invalid mounts the routes as clear
                      "not configured" errors.
        get_streamer: Lazy accessor for the gateway streamer (registration
                      happens before the streamer exists).
    """
    from marie.auth.auth_bearer import TokenBearer

    executor = (kb_config or {}).get("executor")
    if executor:
        _logger.info(f"KB gateway routes configured (executor={executor})")
    else:
        _logger.warning(
            "KB gateway routes not configured "
            "(missing 'kb.executor' in gateway config)"
        )

    def _error(message: str, status_code: int = 400) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"status": "error", "message": message},
        )

    async def _forward(action: str, body: Dict[str, Any]):
        """Validate *body* and forward the action to the KB executor."""
        if not executor:
            return _error(
                "kb service not configured "
                "(missing 'kb.executor' in gateway config)",
                status_code=503,
            )

        endpoint, required = _KB_ACTIONS[action]
        for field in required:
            if not body.get(field):
                return _error(f"kb {action} requires '{field}'")

        req = DataRequest()
        req.header.exec_endpoint = endpoint
        req.header.target_executor = executor
        # The executor lifecycle wrapper hard-requires a job_id for monitored
        # endpoints; these interactive calls are not scheduler jobs.
        req.parameters = {**body, "job_id": f"kb-{uuid.uuid4().hex}"}

        try:
            response = await get_streamer().process_single_data(request=req)
        except Exception as e:
            _logger.error(f"kb {action} failed : {e}")
            return _error(f"kb {action} failed : {e}", status_code=502)

        status = getattr(response, "status", None)
        if status is not None and status.code != jina_pb2.StatusProto.SUCCESS:
            desc = status.description or f"kb {action} failed"
            _logger.error(f"kb {action} failed : {desc}")
            return _error(f"kb {action} failed : {desc}", status_code=502)

        payload = _extract_result(response.parameters)
        if payload is None:
            return _error(
                f"kb {action} returned no payload from executor",
                status_code=502,
            )

        # Search-style payloads carry the row list under "results"; surface
        # that list as the result to keep the client contract flat.
        result = payload.get("results", payload)
        return {"status": "ok", "result": result}

    async def _body(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _mount(action: str, summary: str) -> None:
        @app.post(
            f'/api/v1/kb/{action}',
            summary=summary,
            tags=['KB'],
            dependencies=[Depends(TokenBearer())],
        )
        async def kb_route(request: Request, _action: str = action):
            return await _forward(_action, await _body(request))

    _mount("search", "Semantic search over a knowledge base index")
    _mount(
        "hybrid_search",
        "Hybrid (vector + full-text, RRF) search over a knowledge base index",
    )
    _mount("index_stats", "Statistics for a knowledge base index")
    _mount("source_stats", "Statistics for a knowledge base source")
    _mount("delete_source", "Delete all vectors for a knowledge base source")
    _mount("delete_index", "Delete all vectors for a knowledge base index")
