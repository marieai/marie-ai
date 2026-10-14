from types import SimpleNamespace

import pytest
from marie.engine.llm_queue.config import LlmQueueRuntimeConfig

from marie.excepts import RuntimeFailToStart
from marie.serve.runtimes.gateway.marie import llm_dispatch_runtime as module
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.servers.marie_gateway import MarieServerGateway


def registered_policy():
    return {
        'endpoints': [
            {
                'endpoint_id': 'primary',
                'base_url': 'https://inference.example/v1',
                'credential_env': 'TEST_DISPATCH_KEY',
                'execution_limit': 2,
            }
        ],
        'lanes': [
            {
                'pool_id': 'documents',
                'endpoint_id': 'primary',
                'revision': 'r1',
                'execution_limit': 2,
            }
        ],
        'limits': {'max_execution_items': 2},
    }


@pytest.fixture
def runtime_env(monkeypatch):
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'true')
    monkeypatch.setenv('LLM_QUEUE_URL', 'redis://fixture:6379/0')
    monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'fixture-fabric')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('TEST_DISPATCH_KEY', raising=False)


@pytest.mark.asyncio
async def test_missing_registered_policy_refuses_enabled_start(runtime_env):
    runtime = GatewayLlmDispatchRuntime()
    with pytest.raises(RuntimeFailToStart):
        await runtime.start()
    assert runtime.health()['last_error'] == 'dispatch_start_failed'
    assert runtime.health()['running'] is False


@pytest.mark.asyncio
async def test_missing_named_credential_uses_common_failure_boundary(runtime_env):
    runtime = GatewayLlmDispatchRuntime(config={'llm_dispatch': registered_policy()})
    with pytest.raises(RuntimeFailToStart) as caught:
        await runtime.start()
    assert runtime.health()['last_error'] == 'dispatch_start_failed'
    assert 'inference.example' not in str(caught.value)
    assert 'TEST_DISPATCH_KEY' not in str(caught.value)


@pytest.mark.parametrize('environment_fabric', [None, 'conflicting-fabric'])
@pytest.mark.asyncio
async def test_disabled_initialization_needs_no_policy_fabric_or_database(
    monkeypatch, environment_fabric
):
    if environment_fabric is None:
        monkeypatch.delenv('LLM_QUEUE_FABRIC_GROUP_ID', raising=False)
    else:
        monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', environment_fabric)
    monkeypatch.setattr(
        module,
        '_build_scheduler_config_source',
        lambda **kwargs: pytest.fail('disabled runtime opened scheduler database'),
    )
    runtime = GatewayLlmDispatchRuntime(
        config={
            'fabric_group_id': 'unused-fabric',
            'llm_dispatch': {'policy_source': 'database'},
        },
        queue_config=LlmQueueRuntimeConfig(enabled=False),
    )
    await runtime.start()
    assert runtime.mode == 'direct-batch'
    assert runtime.health()['running'] is False
    await runtime.stop()


@pytest.mark.asyncio
async def test_enabled_fabric_mismatch_uses_common_failure_boundary(runtime_env):
    runtime = GatewayLlmDispatchRuntime(
        config={'fabric_group_id': 'conflicting-fabric'}
    )
    with pytest.raises(RuntimeFailToStart):
        await runtime.start()
    assert runtime.health()['last_error'] == 'dispatch_start_failed'


@pytest.mark.asyncio
async def test_background_projection_uses_started_enabled_runtime():
    store = object()
    projected = []
    runtime = SimpleNamespace(
        enabled=True, mode='queued-dispatch', _dispatcher=object(), _queue_client=store
    )

    async def start():
        pass

    runtime.start = start
    gateway = SimpleNamespace(
        llm_dispatch_runtime=runtime,
        logger=SimpleNamespace(info=lambda *args: None),
        job_scheduler=SimpleNamespace(start_llm_routing_projection=projected.append),
    )
    await MarieServerGateway._start_gateway_background_runtimes(gateway)
    assert projected == [store]


@pytest.mark.asyncio
async def test_default_enabled_runtime_owns_durable_start_and_stop(
    runtime_env, monkeypatch
):
    from marie.engine.llm_queue import request_dispatcher
    from marie.engine.llm_queue import store as store_module

    lifecycle = []

    class Store:
        def __init__(self, url, *, fabric_id, limits):
            assert url == 'redis://fixture:6379/0'
            assert fabric_id == 'fixture-fabric'
            assert limits.max_execution_items == 2
            lifecycle.append('store_open')

        def close(self):
            lifecycle.append('store_close')

    class Dispatcher:
        def __init__(self, **kwargs):
            assert kwargs['endpoints'][0].api_key == 'fixture-key'
            assert kwargs['lanes'][0].pool_id == 'documents'
            assert kwargs['policy'] == 'drr'
            assert kwargs['total_concurrent_dispatch'] == 2
            self.running = False

        async def start(self):
            lifecycle.append('dispatcher_start')
            self.running = True

        async def stop(self):
            lifecycle.append('dispatcher_stop')
            self.running = False

        def health(self):
            return {'running': self.running, 'mode': 'queued-dispatch'}

    monkeypatch.setenv('TEST_DISPATCH_KEY', 'fixture-key')
    monkeypatch.setattr(store_module, 'RequestStore', Store)
    monkeypatch.setattr(request_dispatcher, 'RequestDispatcher', Dispatcher)
    runtime = GatewayLlmDispatchRuntime(config={'llm_dispatch': registered_policy()})
    await runtime.start()
    await runtime.start()
    assert runtime.health()['running'] is True
    assert runtime.health()['pool_ids'] == ['documents']
    assert runtime.health()['scheduler_policy'] == 'drr'
    await runtime.stop()
    await runtime.stop()
    assert runtime.health()['running'] is False
    assert lifecycle == [
        'store_open',
        'dispatcher_start',
        'dispatcher_stop',
        'store_close',
    ]


@pytest.mark.parametrize('failure', ['storage', 'dispatcher'])
@pytest.mark.asyncio
async def test_start_failure_closes_partial_resources_without_publishing_runtime(
    runtime_env, monkeypatch, failure
):
    from marie.engine.llm_queue import request_dispatcher
    from marie.engine.llm_queue import store as store_module

    lifecycle = []

    class Store:
        def __init__(self, url, *, fabric_id, limits):
            if failure == 'storage':
                raise OSError('private-store-sentinel')

        def close(self):
            lifecycle.append('store_close')

    class Dispatcher:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            raise OSError('private-endpoint-sentinel')

        async def stop(self):
            lifecycle.append('dispatcher_stop')

    monkeypatch.setenv('TEST_DISPATCH_KEY', 'fixture-key')
    monkeypatch.setattr(store_module, 'RequestStore', Store)
    monkeypatch.setattr(request_dispatcher, 'RequestDispatcher', Dispatcher)
    runtime = GatewayLlmDispatchRuntime(config={'llm_dispatch': registered_policy()})
    with pytest.raises(RuntimeFailToStart) as caught:
        await runtime.start()
    assert 'sentinel' not in str(caught.value)
    assert runtime.health()['last_error'] == 'dispatch_start_failed'
    assert runtime.health()['running'] is False
    assert runtime._queue_client is None
    assert lifecycle == (
        [] if failure == 'storage' else ['dispatcher_stop', 'store_close']
    )


@pytest.mark.parametrize('enabled,started', [(False, True), (True, False)])
@pytest.mark.asyncio
async def test_background_projection_requires_enabled_and_started_runtime(
    enabled, started
):
    projected = []
    runtime = SimpleNamespace(
        enabled=enabled,
        mode='queued-dispatch',
        _dispatcher=object() if started else None,
        _queue_client=object(),
    )

    async def start():
        pass

    runtime.start = start
    gateway = SimpleNamespace(
        llm_dispatch_runtime=runtime,
        logger=SimpleNamespace(info=lambda *args: None),
        job_scheduler=SimpleNamespace(start_llm_routing_projection=projected.append),
    )
    await MarieServerGateway._start_gateway_background_runtimes(gateway)
    assert projected == []
