import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from marie.engine.llm_queue import registry

from marie.auth.api_key_manager import APIKeyManager
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
    GatewayLlmDispatchRuntime,
)


@pytest.mark.asyncio
async def test_operator_routes_authorize_before_any_snapshot_or_debug_read(monkeypatch):
    from marie.serve.runtimes.gateway.marie.operator_routes import add_runtime_routes

    app = FastAPI()
    reads = []

    async def snapshot(**kwargs):
        reads.append(kwargs)
        return {'fabric_group_id': kwargs['fabric_group_id']}

    monkeypatch.setattr(registry, 'read_runtime_snapshot', snapshot)
    monkeypatch.setattr(APIKeyManager, '_keys', {})
    for name, policy in [
        ('inference', {}),
        ('wrong', {'allowed_fabrics': ['b']}),
        ('operator', {'allowed_fabrics': ['a']}),
    ]:
        APIKeyManager.add_key(
            dict(
                name=name,
                api_key='mas_' + name[0] * 54,
                scopes=[] if name == 'inference' else ['runtime-observability'],
                **policy,
            )
        )
    add_runtime_routes(app, lambda: 'a')
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        for route in ['/api/llm-dispatch/runtime', '/api/debug']:
            for token, expected in [
                (None, 401),
                ('mas_' + 'i' * 54, 403),
                ('mas_' + 'w' * 54, 403),
            ]:
                headers = {'Authorization': 'Bearer ' + token} if token else {}
                response = await client.get(
                    route + '?fabric_group_id=a', headers=headers
                )
                assert response.status_code == expected
                assert reads == []
            headers = {'Authorization': 'Bearer mas_' + 'o' * 54}
            response = await client.get(
                route + '?fabric_group_id=a&tenant_id=org', headers=headers
            )
            assert response.status_code == 403 and reads == []
            response = await client.get(route + '?fabric_group_id=b', headers=headers)
            assert response.status_code == 403 and reads == []
            response = await client.get(route + '?fabric_group_id=a', headers=headers)
            assert response.status_code == 200
            assert reads.pop()['fabric_group_id'] == 'a'

        async def failing(**kwargs):
            raise registry.SnapshotUnavailable('PHI raw secret')

        monkeypatch.setattr(registry, 'read_runtime_snapshot', failing)
        response = await client.get(
            '/api/llm-dispatch/runtime?fabric_group_id=a', headers=headers
        )
        assert response.status_code == 503
        assert 'secret' not in response.text
        assert response.json()['correlation_id']


def test_runtime_fingerprint_ignores_age_but_retains_state():
    from marie.serve.runtimes.servers.marie_gateway import (
        _llm_dispatch_runtime_event_fingerprint as fingerprint,
    )

    first = {
        'live_requests': [
            {'request_id': 'a', 'submitted_at': 1, 'inflight_age_seconds': 5}
        ],
        'observed_at': 1,
    }
    second = {
        'live_requests': [
            {'request_id': 'a', 'submitted_at': 1, 'inflight_age_seconds': 10}
        ],
        'observed_at': 2,
    }
    assert fingerprint(first) == fingerprint(second)
    second['live_requests'][0]['submitted_at'] = 2
    assert fingerprint(first) != fingerprint(second)


@pytest.mark.asyncio
async def test_mounted_gateway_operator_auth_envelope_and_challenge(monkeypatch):
    from types import CodeType, FunctionType

    from marie.serve.runtimes.servers import marie_gateway

    # Install the actual gateway REST closure without starting external services.
    code = next(
        value
        for value in marie_gateway.MarieServerGateway.__init__.__code__.co_consts
        if isinstance(value, CodeType) and value.co_name == '_extend_rest_function'
    )
    gateway = SimpleNamespace(
        args={},
        llm_dispatch_runtime=SimpleNamespace(
            config=SimpleNamespace(fabric_group_id='a')
        ),
    )
    cell = (lambda: gateway).__closure__[0]
    install_routes = FunctionType(code, vars(marie_gateway), closure=(cell,))
    for name in (
        'register_wasm_routes',
        'register_blueprint_routes',
        'register_kb_routes',
    ):
        monkeypatch.setattr(marie_gateway, name, lambda *args: None)
    app = install_routes(FastAPI())
    monkeypatch.setattr(APIKeyManager, '_keys', {})
    APIKeyManager.add_key(dict(name='inference', api_key='mas_' + 'i' * 54))

    async def unexpected_read(**kwargs):
        pytest.fail('unauthorized operator read')

    monkeypatch.setattr(registry, 'read_runtime_snapshot', unexpected_read)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        for route in ('/api/llm-dispatch/runtime', '/api/debug'):
            response = await client.get(route + '?fabric_group_id=a')
            assert response.status_code == 401
            assert response.json() == {
                'status': 'error',
                'message': 'authentication_required',
            }
            assert response.headers.get('www-authenticate') == 'Bearer'
            response = await client.get(
                route + '?fabric_group_id=a',
                headers={'Authorization': 'Bearer mas_' + 'i' * 54},
            )
            assert response.status_code == 403
            assert response.json() == {
                'status': 'error',
                'message': 'runtime_observability_forbidden',
            }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'runtime_config,explicit,queued,expected',
    [
        ({'fabric_group_id': 'runtime'}, None, None, 'runtime'),
        (
            {
                'scheduler': {'fabric_group_id': 'scheduler'},
                'fabric_group_id': 'runtime',
            },
            None,
            'queue',
            'scheduler',
        ),
        ({'fabric_group_id': 'runtime'}, 'explicit', 'queue', 'explicit'),
        ({}, None, 'queue', 'queue'),
        ({}, None, None, 'default'),
    ],
)
async def test_v2_effective_fabric_reaches_scheduler_operator_and_events(
    monkeypatch, runtime_config, explicit, queued, expected
):
    from marie.engine.llm_queue.config import LlmQueueConfig

    from marie.serve.runtimes.gateway.marie import llm_dispatch_runtime as module
    from marie.serve.runtimes.gateway.marie.operator_routes import add_runtime_routes
    from marie.serve.runtimes.servers.marie_gateway import (
        _llm_dispatch_runtime_event_message,
    )

    selected = []
    monkeypatch.setattr(
        module,
        '_build_scheduler_config_source',
        lambda **kwargs: selected.append(kwargs['fabric_group_id']),
    )
    runtime = GatewayLlmDispatchRuntime(
        config=runtime_config,
        fabric_group_id=explicit,
        queue_config=LlmQueueConfig(enabled=True, fabric_group_id=queued),
    )
    assert runtime.config.queue_contract_version == 'v2'
    assert selected == [expected]
    assert runtime.config.fabric_group_id == expected
    event = _llm_dispatch_runtime_event_message(
        snapshot={'runtime_summary': {}, 'live_requests': [], 'dispatchers': []},
        queue_config=runtime.config,
    )
    assert event.payload['fabric_group_id'] == expected
    app = FastAPI()
    add_runtime_routes(app, lambda: runtime.config.fabric_group_id)
    monkeypatch.setattr(APIKeyManager, '_keys', {})
    token = 'mas_' + 'x' * 54
    APIKeyManager.add_key(
        dict(
            name='operator',
            api_key=token,
            scopes=['runtime-observability'],
            allowed_fabrics=[expected],
        )
    )

    async def snapshot(**kwargs):
        return {'fabric_group_id': kwargs['fabric_group_id']}

    monkeypatch.setattr(registry, 'read_runtime_snapshot', snapshot)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        response = await client.get(
            '/api/llm-dispatch/runtime',
            params={'fabric_group_id': expected},
            headers={'Authorization': 'Bearer ' + token},
        )
        assert response.status_code == 200
        response = await client.get(
            '/api/llm-dispatch/runtime?fabric_group_id=other',
            headers={'Authorization': 'Bearer ' + token},
        )
        assert response.status_code == 403


@pytest.mark.asyncio
async def test_v2_environment_fabric_reaches_runtime_and_operator(monkeypatch):
    from marie.serve.runtimes.gateway.marie.operator_routes import add_runtime_routes

    monkeypatch.setenv('LLM_QUEUE_CONTRACT_VERSION', 'v2')
    monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'env-fabric')
    runtime = GatewayLlmDispatchRuntime()
    assert runtime.config.fabric_group_id == 'env-fabric'
    app = FastAPI()
    add_runtime_routes(app, lambda: runtime.config.fabric_group_id)
    monkeypatch.setattr(APIKeyManager, '_keys', {})
    token = 'mas_' + 'x' * 54
    APIKeyManager.add_key(
        dict(
            name='operator',
            api_key=token,
            scopes=['runtime-observability'],
            allowed_fabrics=['env-fabric'],
        )
    )

    async def snapshot(**kwargs):
        return {'fabric_group_id': kwargs['fabric_group_id']}

    monkeypatch.setattr(registry, 'read_runtime_snapshot', snapshot)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        assert (
            await client.get(
                '/api/llm-dispatch/runtime?fabric_group_id=env-fabric',
                headers={'Authorization': 'Bearer ' + token},
            )
        ).status_code == 200


def test_v3_runtime_preserves_conflicting_fabric_rejection(monkeypatch):
    from marie.engine.llm_queue.config import LlmQueueConfig

    monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'env-fabric')
    with pytest.raises(ValueError, match='consistent fabric identity'):
        GatewayLlmDispatchRuntime(
            config={'fabric_group_id': 'runtime-fabric'},
            queue_config=LlmQueueConfig(
                enabled=True, queue_contract_version='v3', fabric_group_id='env-fabric'
            ),
        )
