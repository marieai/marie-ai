import json
import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from openinference.semconv.trace import SpanAttributes
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import marie.helper
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (  # noqa: F401
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.gateway.request_handling import GatewayRequestHandler
from marie.serve.runtimes.servers import marie_gateway as gateway_module


@pytest_asyncio.fixture
async def invoke_app(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[
    tuple[FastAPI, gateway_module.MarieServerGateway, InMemorySpanExporter]
]:
    def init_server(self, **_kwargs) -> None:
        self.logger = MagicMock()
        self.meter = None
        self.runtime_args = SimpleNamespace()

    monkeypatch.setattr(gateway_module.CompositeServer, '__init__', init_server)
    monkeypatch.setattr(sys, 'excepthook', sys.excepthook)
    monkeypatch.setattr(gateway_module, 'install', MagicMock())
    monkeypatch.setattr(gateway_module, 'load_env_file', MagicMock())
    monkeypatch.setattr(gateway_module.ModelRegistry, 'configure', MagicMock())
    for name in (
        'PostgreSQLKV',
        'get_etcd_client',
        'convert_to_etcd_args',
        'DesiredStore',
        'StatusStore',
        'SemaphoreStore',
        'SlotCapacityManager',
        'JobManager',
        'PostgreSQLJobScheduler',
        'register_wasm_routes',
        'register_blueprint_routes',
        'register_kb_routes',
    ):
        monkeypatch.setattr(gateway_module, name, MagicMock())
    for name in ('stream', 'Call', 'dry_run'):
        monkeypatch.setattr(
            GatewayRequestHandler, name, getattr(GatewayRequestHandler, name)
        )
    monkeypatch.setattr(
        marie.helper, 'extend_rest_interface', marie.helper.extend_rest_interface
    )
    monkeypatch.setattr(
        gateway_module.APIKeyManager, 'is_valid', lambda token: token == 'test-auth'
    )
    config = dict.fromkeys(
        ('provider', 'hostname', 'port', 'username', 'password', 'database'), 'test'
    )
    gateway = gateway_module.MarieServerGateway(
        kv_store_kwargs=config, job_scheduler_kwargs=config
    )
    gateway.decode_request = AsyncMock(
        return_value=SimpleNamespace(parameters={'status': 'ok', 'job_id': 'job-1'})
    )
    app = marie.helper.extend_rest_interface(FastAPI())
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    yield app, gateway, exporter
    FastAPIInstrumentor.uninstrument_app(app)
    provider.shutdown()


@pytest.mark.asyncio
async def test_invoke_records_original_json_on_http_root(invoke_app) -> None:
    app, gateway, exporter = invoke_app
    payload = {
        'header': {'request_id': 'request-1'},
        'parameters': {
            'command': 'submit',
            'metadata': {'ref_id': 'document-1', 'pages': [1, 2]},
            'prompt': 'Full submitted content ' * 300,
        },
        'data': None,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        response = await client.post(
            '/api/v1/invoke',
            json=payload,
            headers={'Authorization': 'Bearer test-auth'},
        )

    assert response.status_code == 200
    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.parent is None)
    assert root.name == 'POST /api/v1/invoke'
    assert json.loads(root.attributes[SpanAttributes.INPUT_VALUE]) == payload
    assert root.attributes[SpanAttributes.INPUT_MIME_TYPE] == 'application/json'
    assert (
        gateway.decode_request.await_args.args[0].parameters['api_key'] == 'test-auth'
    )
    assert all(
        SpanAttributes.INPUT_VALUE not in span.attributes
        for span in spans
        if span is not root
    )


@pytest.mark.asyncio
async def test_invoke_redacts_submitted_credentials_without_mutating_request(
    invoke_app,
) -> None:
    app, gateway, exporter = invoke_app
    payload = {
        'header': {'Authorization': 'secret-header', 'request_id': 'request-1'},
        'parameters': {
            'api_key': 'caller-key',
            'metadata': {'Password': 'secret-password', 'ref_id': 'document-1'},
            'image': 'data:image/png;base64,secret-image',
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        response = await client.post(
            '/api/v1/invoke',
            json=payload,
            headers={'Authorization': 'Bearer test-auth'},
        )

    assert response.status_code == 200
    root = next(span for span in exporter.get_finished_spans() if span.parent is None)
    assert json.loads(root.attributes[SpanAttributes.INPUT_VALUE]) == {
        'header': {'request_id': 'request-1'},
        'parameters': {
            'metadata': {'ref_id': 'document-1'},
            'image': 'marie://unresolved-media',
        },
    }
    assert gateway.decode_request.await_args.args[0].parameters == payload['parameters']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('body', 'token', 'status'),
    [(b'not-json', 'test-auth', 400), (b'{"parameters": {}}', 'invalid', 401)],
)
async def test_rejected_invoke_does_not_record_input(
    invoke_app, body, token, status
) -> None:
    app, gateway, exporter = invoke_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        response = await client.post(
            '/api/v1/invoke',
            content=body,
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            },
        )

    assert response.status_code == status
    gateway.decode_request.assert_not_awaited()
    assert all(
        SpanAttributes.INPUT_VALUE not in span.attributes
        for span in exporter.get_finished_spans()
    )
