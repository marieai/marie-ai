from fastapi import FastAPI

from marie.sandbox.blueprints.gateway_routes import register_blueprint_routes


def test_blueprint_import_route_can_generate_openapi():
    app = FastAPI()
    register_blueprint_routes(app)

    schema = app.openapi()

    assert "/api/v1/blueprints/import" in schema["paths"]
