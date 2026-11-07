"""Tests for the KB gateway API extension routes (marie/kb/gateway_routes.py).

The routes are thin forwarders: validate the request, forward to the KB
executor deployment declared in config, unwrap the dict payload. All
vector-store/model dependencies live executor-side.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marie.auth.api_key_manager import APIKeyManager
from marie.kb.gateway_routes import _extract_result, register_kb_routes
from marie.proto import jina_pb2

AUTH = {"Authorization": "Bearer test-token"}
TENANT = '10000000-0000-4000-8000-000000000001'
SCOPE = {'tenant_id': TENANT, 'index_name': 'idx1'}

KB_PATHS = [
    "/api/v1/kb/search",
    "/api/v1/kb/hybrid_search",
    "/api/v1/kb/index_stats",
    "/api/v1/kb/source_stats",
    "/api/v1/kb/delete_source",
    "/api/v1/kb/delete_index",
    '/api/v1/kb/delete_item',
    '/api/v1/kb/item_status',
]


@pytest.fixture(autouse=True)
def _valid_token(monkeypatch):
    monkeypatch.setattr(APIKeyManager, "is_valid", staticmethod(lambda token: True))


def _streamer_returning(payload, status=None):
    """A stub streamer whose single-data response wraps *payload* in
    the __results__ envelope dict-returning executor endpoints produce."""
    response = MagicMock()
    response.status = status
    response.parameters = {"__results__": {"kb_executor/rep-0": payload}}
    streamer = MagicMock()
    streamer.process_single_data = AsyncMock(return_value=response)
    return streamer


def _client(streamer=None, kb_config={"executor": "kb_executor"}) -> TestClient:
    app = FastAPI()
    register_kb_routes(app, kb_config, lambda: streamer)
    return TestClient(app)


def test_all_routes_mounted_in_openapi():
    app = FastAPI()
    register_kb_routes(app, {"executor": "kb_executor"}, lambda: None)
    paths = app.openapi()["paths"]
    for p in KB_PATHS:
        assert p in paths


def test_unconfigured_returns_503():
    client = _client(streamer=None, kb_config=None)
    r = client.post("/api/v1/kb/search", json={**SCOPE, "query": "x"}, headers=AUTH)
    assert r.status_code == 503
    assert r.json()["status"] == "error"


def test_missing_auth_rejected():
    client = _client(streamer=_streamer_returning({"results": []}))
    r = client.post("/api/v1/kb/search", json={**SCOPE, "query": "x"})
    assert r.status_code == 401


def test_search_requires_query():
    client = _client(streamer=_streamer_returning({"results": []}))
    r = client.post("/api/v1/kb/search", json={}, headers=AUTH)
    assert r.status_code == 400
    assert "requires 'query'" in r.json()["message"]


def test_index_stats_requires_index_name():
    client = _client(streamer=_streamer_returning({}))
    r = client.post("/api/v1/kb/index_stats", json={}, headers=AUTH)
    assert r.status_code == 400


def test_delete_source_requires_source_id():
    client = _client(streamer=_streamer_returning({}))
    r = client.post("/api/v1/kb/delete_source", json={}, headers=AUTH)
    assert r.status_code == 400


def test_source_delete_cannot_omit_its_tenant_and_index_scope():
    client = _client(streamer=_streamer_returning({'deleted_count': 0}))
    result = client.post(
        '/api/v1/kb/delete_source', json={'source_id': 'source'}, headers=AUTH
    )
    assert result.status_code == 400


def test_exact_item_delete_forwards_all_identity_fields():
    streamer = _streamer_returning({'item_id': 'item', 'state': 'removed'})
    result = _client(streamer=streamer).post(
        '/api/v1/kb/delete_item',
        json={
            **SCOPE,
            'tenant_id': TENANT,
            'index_id': 'index',
            'index_name': 'index',
            'source_id': 'source',
            'item_id': 'item',
        },
        headers=AUTH,
    )
    assert result.status_code == 200
    request = streamer.process_single_data.await_args.kwargs['request']
    assert request.header.exec_endpoint == '/delete_item'
    assert dict(request.parameters)['item_id'] == 'item'


def test_search_forwards_to_configured_executor_and_unwraps_results():
    rows = [{"id": "n1", "content": "hello chunk", "score": 0.9}]
    streamer = _streamer_returning({"results": rows, "count": 1})
    client = _client(streamer=streamer)

    r = client.post(
        "/api/v1/kb/search",
        json={**SCOPE, "query": "hello", "index_name": "idx1", "top_k": 3},
        headers=AUTH,
    )

    assert r.status_code == 200
    assert r.json() == {"status": "ok", "result": rows}

    request = streamer.process_single_data.await_args.kwargs["request"]
    assert request.header.exec_endpoint == "/search"
    assert request.header.target_executor == "kb_executor"
    params = dict(request.parameters)
    assert params["query"] == "hello"
    assert params["index_name"] == "idx1"
    assert params["top_k"] == 3
    assert params["job_id"].startswith("kb-")


def test_hybrid_search_forwards_alpha():
    streamer = _streamer_returning({"results": [], "count": 0})
    client = _client(streamer=streamer)
    r = client.post(
        "/api/v1/kb/hybrid_search",
        json={**SCOPE, "query": "hello", "alpha": 0.7},
        headers=AUTH,
    )
    assert r.status_code == 200
    request = streamer.process_single_data.await_args.kwargs["request"]
    assert request.header.exec_endpoint == "/hybrid_search"
    assert dict(request.parameters)["alpha"] == 0.7


def test_stats_payload_passed_through_whole():
    payload = {"index_name": "idx1", "count": 5, "sources": 2}
    streamer = _streamer_returning(payload)
    client = _client(streamer=streamer)
    r = client.post(
        "/api/v1/kb/index_stats", json={**SCOPE, "index_name": "idx1"}, headers=AUTH
    )
    assert r.status_code == 200
    assert r.json()["result"] == payload


def test_executor_error_status_surfaces_as_502_not_ok():
    status = MagicMock()
    status.code = jina_pb2.StatusProto.ERROR
    status.description = "deployment unavailable"
    streamer = _streamer_returning({}, status=status)
    client = _client(streamer=streamer)

    r = client.post("/api/v1/kb/search", json={**SCOPE, "query": "x"}, headers=AUTH)
    assert r.status_code == 502
    assert r.json()["status"] == "error"
    assert "deployment unavailable" in r.json()["message"]


def test_streamer_exception_surfaces_as_502():
    streamer = MagicMock()
    streamer.process_single_data = AsyncMock(side_effect=RuntimeError("boom"))
    client = _client(streamer=streamer)
    r = client.post("/api/v1/kb/search", json={**SCOPE, "query": "x"}, headers=AUTH)
    assert r.status_code == 502
    assert "boom" in r.json()["message"]


def test_missing_payload_surfaces_as_502():
    response = MagicMock()
    response.status = None
    response.parameters = {}
    streamer = MagicMock()
    streamer.process_single_data = AsyncMock(return_value=response)
    client = _client(streamer=streamer)
    r = client.post("/api/v1/kb/search", json={**SCOPE, "query": "x"}, headers=AUTH)
    assert r.status_code == 502
    assert "no payload" in r.json()["message"]


def test_extract_result_shapes():
    assert _extract_result({"__results__": {"dep/rep-0": {"results": [1]}}}) == {
        "results": [1]
    }
    assert _extract_result({}) is None
    assert _extract_result(None) is None


@pytest.mark.parametrize(
    'action', ['text_search', 'open_text', 'delete_text_item', 'delete_text_index']
)
def test_text_endpoints_require_auth_and_forward_exact_identity(action):
    fields = {
        **SCOPE,
        'index_id': '20000000-0000-4000-8000-000000000001',
        'source_id': 'source',
        'item_id': 'item',
        'section_id': 'section',
        'indexed_dag_id': 'attempt',
        'source_digest': 'digest',
        'query': 'coverage',
    }
    streamer = _streamer_returning({'outcome': 'ready'})
    client = _client(streamer=streamer)
    assert client.post(f'/api/v1/kb/{action}', json=fields).status_code == 401
    response = client.post(f'/api/v1/kb/{action}', json=fields, headers=AUTH)
    assert response.status_code == 200
    forwarded = streamer.process_single_data.call_args.kwargs['request']
    for key, value in fields.items():
        assert forwarded.parameters[key] == value
