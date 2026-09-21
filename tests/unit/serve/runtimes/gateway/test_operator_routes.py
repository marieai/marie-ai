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


@pytest.mark.asyncio
async def test_routing_policy_routes_require_admin_scope(monkeypatch):
    from marie.engine.llm_queue.admission_policy import AdmissionPolicy

    from marie.serve.runtimes.gateway.marie.operator_routes import add_runtime_routes

    policy = AdmissionPolicy.from_rows(
        'a',
        3,
        [
            {
                'pool_id': 'default',
                'enabled': True,
                'metadata': {
                    'admission': {
                        'schema_version': 1,
                        'priority': 1_000_000,
                        'accepting': True,
                        'match': {},
                    }
                },
            }
        ],
    )
    calls = []

    class Repository:
        def load_active_admission_policy(self, fabric_group_id):
            calls.append(('preview', fabric_group_id))
            return policy

        def activate_admission_policy(self, fabric_group_id, actor_id):
            calls.append(('activate', fabric_group_id, actor_id))
            return SimpleNamespace(
                fabric_group_id=fabric_group_id,
                generation=4,
                policy_digest='d' * 64,
                activated_by=actor_id,
            )

    app = FastAPI()
    repository = Repository()
    add_runtime_routes(app, lambda: 'a', lambda: repository)
    monkeypatch.setattr(APIKeyManager, '_keys', {})
    observer = 'mas_' + 'o' * 54
    admin = 'mas_' + 'a' * 54
    APIKeyManager.add_key(
        {
            'name': 'observer',
            'api_key': observer,
            'scopes': ['runtime-observability'],
            'allowed_fabrics': ['a'],
        }
    )
    APIKeyManager.add_key(
        {
            'name': 'routing-admin',
            'api_key': admin,
            'scopes': ['runtime-routing-admin'],
            'allowed_fabrics': ['a'],
        }
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        response = await client.post(
            '/api/llm-dispatch/policy/preview?fabric_group_id=a',
            headers={'Authorization': f'Bearer {observer}'},
            json={'facts': {'document.effective_page_count': 3}},
        )
        assert response.status_code == 403
        assert calls == []

        response = await client.post(
            '/api/llm-dispatch/policy/preview?fabric_group_id=a',
            headers={'Authorization': f'Bearer {admin}'},
            json={'facts': {'document.effective_page_count': 3}},
        )
        assert response.status_code == 200
        assert response.json()['result'] == {
            'pool_id': 'default',
            'priority': 1_000_000,
            'policy_generation': 3,
            'policy_digest': policy.policy_digest,
            'rule_digest': policy.rules[0].rule_digest,
            'matched_facts': [],
        }
        assert 'effective_page_count' not in response.text

        response = await client.post(
            '/api/llm-dispatch/policy/activate?fabric_group_id=a',
            headers={'Authorization': f'Bearer {admin}'},
        )
        assert response.status_code == 200
        assert response.json()['result']['generation'] == 4
        assert calls == [('preview', 'a'), ('activate', 'a', 'routing-admin')]


@pytest.mark.asyncio
async def test_runtime_joins_exact_fabric_database_diagnostics(monkeypatch):
    from marie.serve.runtimes.gateway.marie import operator_routes

    add_runtime_routes = operator_routes.add_runtime_routes
    monkeypatch.setattr(
        operator_routes.admission_routing_metrics,
        'snapshot',
        lambda _fabric: {'available': False},
    )

    class Repository:
        def load_routing_diagnostics(self, fabric_group_id, limit):
            assert (fabric_group_id, limit) == ('a', 25)
            return {
                'policy': {
                    'admission_mode': 'shadow',
                    'desired_generation': 5,
                    'desired_digest': 'b' * 64,
                },
                'routing': {
                    'projection_pending_count': 2,
                    'shadow': {'match_count': 8, 'disagreement_count': 1},
                    'matched': {'automatic': 9},
                    'rejected': {'routing_facts_missing': 2},
                },
                'pools': [
                    {
                        'pool_id': 'document-small',
                        'accepted': 11,
                        'completed': 6,
                        'drain_references': 5,
                    }
                ],
                'pool_count': 1,
                'pools_truncated': False,
            }

    async def snapshot(**kwargs):
        assert kwargs == {'fabric_group_id': 'a', 'limit': 25}
        return {
            'fabric_group_id': 'a',
            'policy': {
                'observed_generation': 4,
                'observed_digest': 'a' * 64,
            },
            'pools': [
                {
                    'pool_id': 'document-small',
                    'state_counts': {
                        'ready': 3,
                        'claimed': 1,
                        'running': 1,
                        'unknown': 0,
                    },
                    'drain_references': 5,
                }
            ],
        }

    app = FastAPI()
    add_runtime_routes(app, lambda: 'a', lambda: Repository())
    monkeypatch.setattr(registry, 'read_runtime_snapshot', snapshot)
    monkeypatch.setattr(APIKeyManager, '_keys', {})
    token = 'mas_' + 'o' * 54
    APIKeyManager.add_key(
        {
            'name': 'observer',
            'api_key': token,
            'scopes': ['runtime-observability'],
            'allowed_fabrics': ['a'],
        }
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        response = await client.get(
            '/api/llm-dispatch/runtime?fabric_group_id=a&limit=25',
            headers={'Authorization': f'Bearer {token}'},
        )

    assert response.status_code == 200
    result = response.json()['result']
    assert result['policy']['desired_generation'] == 5
    assert result['policy']['observed_generation'] == 4
    assert result['policy']['synchronized'] is False
    assert result['routing']['projection_pending_count'] == 2
    assert result['routing']['shadow']['disagreement_count'] == 1
    assert result['pools'][0]['accepted'] == 11
    assert result['pools'][0]['state_counts']['running'] == 1
    assert result['pools'][0]['drain_references'] == {
        'postgres': 5,
        'valkey': 5,
        'total': 10,
    }


@pytest.mark.asyncio
async def test_routing_resource_preflight_fails_closed_and_reports_references(
    monkeypatch,
):
    from marie.serve.runtimes.gateway.marie.operator_routes import add_runtime_routes

    class Repository:
        def routing_resource_references(self, **kwargs):
            assert kwargs == {
                'fabric_group_id': 'a',
                'resource_type': 'pool',
                'resource_id': 'document-small',
                'revision': None,
                'limit': 25,
            }
            return {'postgres': 3, 'samples': ['work-1'], 'truncated': True}

    monkeypatch.setattr(
        registry,
        'routing_resource_references',
        lambda **kwargs: {'valkey': 2, 'ready': 1, 'active': 1},
    )
    app = FastAPI()
    add_runtime_routes(app, lambda: 'a', lambda: Repository())
    monkeypatch.setattr(APIKeyManager, '_keys', {})
    observer = 'mas_' + 'o' * 54
    admin = 'mas_' + 'a' * 54
    APIKeyManager.add_key(
        {
            'name': 'observer',
            'api_key': observer,
            'scopes': ['runtime-observability'],
            'allowed_fabrics': ['a'],
        }
    )
    APIKeyManager.add_key(
        {
            'name': 'admin',
            'api_key': admin,
            'scopes': ['runtime-routing-admin'],
            'allowed_fabrics': ['a'],
        }
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        response = await client.post(
            '/api/llm-dispatch/resources/check?fabric_group_id=a',
            headers={'Authorization': f'Bearer {observer}'},
            json={'resource_type': 'pool', 'resource_id': 'document-small'},
        )
        assert response.status_code == 403
        response = await client.post(
            '/api/llm-dispatch/resources/check?fabric_group_id=a',
            headers={'Authorization': f'Bearer {admin}'},
            json={'resource_type': 'pool', 'resource_id': 'document-small'},
        )

    assert response.status_code == 409
    assert response.json()['detail'] == {
        'category': 'routing_resource_in_use',
        'references': {
            'postgres': 3,
            'samples': ['work-1'],
            'truncated': True,
            'valkey': 2,
            'ready': 1,
            'active': 1,
            'total': 5,
        },
    }


@pytest.mark.asyncio
async def test_routing_resource_preflight_never_treats_truncation_as_zero(monkeypatch):
    from marie.serve.runtimes.gateway.marie.operator_routes import add_runtime_routes

    repository = SimpleNamespace(
        routing_resource_references=lambda **_kwargs: {
            'postgres': 0,
            'samples': [],
            'truncated': False,
        }
    )
    monkeypatch.setattr(
        registry,
        'routing_resource_references',
        lambda **_kwargs: {
            'valkey': 0,
            'ready': 0,
            'active': 0,
            'truncated': True,
        },
    )
    app = FastAPI()
    add_runtime_routes(app, lambda: 'a', lambda: repository)
    monkeypatch.setattr(APIKeyManager, '_keys', {})
    token = 'mas_' + 'a' * 54
    APIKeyManager.add_key(
        {
            'name': 'admin',
            'api_key': token,
            'scopes': ['runtime-routing-admin'],
            'allowed_fabrics': ['a'],
        }
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        response = await client.post(
            '/api/llm-dispatch/resources/check?fabric_group_id=a',
            headers={'Authorization': f'Bearer {token}'},
            json={'resource_type': 'policy_generation', 'resource_id': '4'},
        )

    assert response.status_code == 503
    assert response.json()['detail'] == 'routing_reference_state_incomplete'


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
