from dataclasses import replace
from traceback import format_exception
from types import SimpleNamespace

import pytest
from marie.engine.llm_queue.config import LlmQueueRuntimeConfig
from marie.engine.llm_queue.scheduler_config import DatabaseSchedulerConfigSource

from marie.excepts import RuntimeFailToStart
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
    GatewayLlmDispatchRuntime,
)
from tests.unit.serve.runtimes.gateway.test_dispatch_policy_postgres import (
    _memory_repository,
    _PolicyDatabase,
)
from tests.unit.serve.runtimes.gateway.test_llm_dispatch_runtime import _Logger


@pytest.fixture
def startup(monkeypatch):
    monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'default')
    monkeypatch.setenv('TEST_OPERATOR_KEY', 'private-provider-token')
    database = _PolicyDatabase()
    logger = _Logger()
    runtime = GatewayLlmDispatchRuntime(
        logger=logger,
        config={'llm_dispatch': {'policy_source': 'database'}},
        queue_config=LlmQueueRuntimeConfig(
            enabled=True,
            queue_url='redis://private-user:private-store-token@queue.example/0',
            fabric_group_id='default',
        ),
        scheduler_config_source=DatabaseSchedulerConfigSource(
            _memory_repository(database), 'default'
        ),
    )
    return database, runtime, logger


@pytest.mark.parametrize(
    'failure,code,action',
    [
        ('seed', 'dispatch_policy_not_configured', 'LLM_QUEUE_ENABLED=false'),
        ('capacity', 'dispatch_capacity_invalid', 'total_concurrent_dispatch'),
        ('pools', 'dispatch_pools_missing', 'llm_queue_pool'),
        ('policy', 'dispatch_policy_invalid', 'metadata.llm_dispatch'),
        ('credential', 'dispatch_credentials_missing', 'credential'),
        ('fabric', 'dispatch_fabric_invalid', 'LLM_QUEUE_FABRIC_GROUP_ID'),
        ('fabric_record', 'dispatch_fabric_not_configured', 'llm_queue_fabric_config'),
        ('policy_source', 'dispatch_policy_source_unavailable', 'policy_source'),
        ('store_config', 'dispatch_store_not_configured', 'LLM_QUEUE_URL'),
        ('database', 'dispatch_policy_unavailable', 'PostgreSQL'),
        ('generation', 'dispatch_policy_generation_invalid', 'generation'),
    ],
)
async def test_startup_failure_reports_prerequisite_and_action(
    startup, monkeypatch, failure, code, action
) -> None:
    database, runtime, logger = startup
    if failure == 'seed':
        database.data.update(
            policy='fifo', total_concurrent_dispatch=0, metadata={}, lanes=[]
        )
    elif failure == 'capacity':
        database.data['total_concurrent_dispatch'] = 0
    elif failure == 'pools':
        database.data['lanes'] = []
    elif failure == 'policy':
        database.data['metadata']['llm_dispatch']['schema_version'] = 99
    elif failure == 'credential':
        monkeypatch.delenv('TEST_OPERATOR_KEY')
    elif failure == 'fabric':
        monkeypatch.setenv('LLM_QUEUE_FABRIC_GROUP_ID', 'conflicting-fabric')
    elif failure == 'fabric_record':

        class MissingFabricCursor:
            def execute(self, *args):
                pass

            def fetchone(self):
                return None

            def close(self):
                pass

        runtime._scheduler_config_source.repository._get_connection = lambda: (
            SimpleNamespace(cursor=MissingFabricCursor, rollback=lambda: None)
        )
    elif failure == 'policy_source':
        runtime.runtime_config['llm_dispatch']['policy_source'] = (
            'https://private-token@policy.example'
        )
    elif failure == 'store_config':
        runtime.config = replace(runtime.config, queue_url=None)
    elif failure == 'database':

        def unavailable():
            raise OSError('postgresql://private-user:private-db-token@db.example')

        runtime._scheduler_config_source.repository._get_connection = unavailable
    elif failure == 'generation':
        database.active_generation = 1
        database.generations.append((1, 'wrong-digest', {}, 'operator', None))

    with pytest.raises(RuntimeFailToStart) as caught:
        await runtime.start()

    message = str(caught.value)
    assert code in message
    assert action in message
    assert 'scheduler-database-upgrades.md' in message
    assert message in logger.error_messages
    assert runtime.health()['last_error'] == 'dispatch_start_failed'
    assert runtime.health()['running'] is False
    output = '\n'.join(logger.error_messages) + ''.join(
        format_exception(caught.type, caught.value, caught.tb)
    )
    for secret in (
        'private-provider-token',
        'private-store-token',
        'private-db-token',
        'queue.example',
        'db.example',
        'inference.example',
        'TEST_OPERATOR_KEY',
        'policy.example',
        'private-token',
    ):
        assert secret not in output


@pytest.mark.parametrize('failure', ['store', 'store_client', 'dispatcher'])
async def test_dependency_failure_logs_safe_action_and_cleans_up(
    startup, monkeypatch, failure
) -> None:
    from marie.engine.llm_queue import request_dispatcher
    from marie.engine.llm_queue import store as store_module

    _, runtime, logger = startup
    lifecycle = []

    class Store:
        def __init__(self, *args, **kwargs):
            if failure == 'store':
                raise ConnectionError('redis://private:secret@queue.example/0')
            if failure == 'store_client':
                raise ModuleNotFoundError('private-module-secret', name='valkey')

        def close(self):
            lifecycle.append('store_close')

    class Dispatcher:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            raise RuntimeError('https://private:secret@inference.example/v1')

        async def stop(self):
            lifecycle.append('dispatcher_stop')

    monkeypatch.setattr(store_module, 'RequestStore', Store)
    monkeypatch.setattr(request_dispatcher, 'RequestDispatcher', Dispatcher)
    with pytest.raises(RuntimeFailToStart) as caught:
        await runtime.start()

    expected = {
        'store': 'dispatch_store_unavailable',
        'store_client': 'dispatch_store_client_missing',
        'dispatcher': 'dispatch_runtime_start_failed',
    }[failure]
    assert expected in str(caught.value)
    assert str(caught.value) in logger.error_messages
    output = '\n'.join(logger.error_messages) + ''.join(
        format_exception(caught.type, caught.value, caught.tb)
    )
    assert 'secret' not in output
    assert '.example' not in output
    assert runtime._queue_client is None
    assert runtime._dispatcher is None
    assert lifecycle == (
        ['dispatcher_stop', 'store_close'] if failure == 'dispatcher' else []
    )


async def test_unknown_failure_does_not_expose_exception_details(
    startup, monkeypatch
) -> None:
    _, runtime, logger = startup

    async def fail():
        raise RuntimeError('unknown-private-token')

    monkeypatch.setattr(runtime, '_start_queued_dispatch', fail)
    with pytest.raises(RuntimeFailToStart) as caught:
        await runtime.start()
    assert 'dispatch_start_failed' in str(caught.value)
    assert str(caught.value) in logger.error_messages
    assert 'unknown-private-token' not in ''.join(
        format_exception(caught.type, caught.value, caught.tb)
    )
