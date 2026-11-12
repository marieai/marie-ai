from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from marie.auth.api_key_manager import APIKeyManager


def client(monkeypatch):
    from marie.kb.catalog_routes import register_kb_catalog_routes

    monkeypatch.setattr(
        APIKeyManager, 'is_valid', staticmethod(lambda token: token == 'catalog-token')
    )
    app = FastAPI()
    register_kb_catalog_routes(app, lambda: None)
    return TestClient(app)


def test_catalog_routes_require_auth_and_validate_scope_before_pool(monkeypatch):
    api = client(monkeypatch)
    path = '/api/v1/kb/catalog/get_index'
    assert api.post(path, json={}).status_code == 401
    auth = {'Authorization': 'Bearer catalog-token'}
    response = api.post(path, json={'tenant_id': 'invalid'}, headers=auth)
    assert response.status_code == 400
    assert response.json()['code'] == 'invalid_request'
    response = api.post(
        path,
        json={
            'tenant_id': str(uuid4()),
            'organization_id': str(uuid4()),
            'actor_id': str(uuid4()),
            'indexId': str(uuid4()),
        },
        headers=auth,
    )
    assert response.status_code == 503
    assert (
        api.post('/api/v1/kb/catalog/query_sql', json={}, headers=auth).status_code
        == 404
    )
