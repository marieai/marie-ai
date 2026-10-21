import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from marie.serve.runtimes.gateway.marie import dispatch_policy
from tests.unit.serve.runtimes.gateway.test_dispatch_policy import persisted
from tests.unit.serve.runtimes.gateway.test_dispatch_policy_postgres import (
    _memory_repository,
    _PolicyDatabase,
)


def scheduled(data=None, expected_generation=None):
    return {
        'schema_version': 1,
        'request_id': 'request-1',
        'requested_by': 'operator-1',
        'requested_at': '2026-10-21T12:00:00.000Z',
        'expected_generation': expected_generation,
        'policy': copy.deepcopy(data or {**persisted(), 'enabled': True}),
    }


def test_startup_request_validates_and_detaches_the_captured_policy():
    raw = scheduled()
    request = dispatch_policy.parse_startup_dispatch_policy('default', raw)
    raw['policy']['lanes'][0]['quantum'] = 99
    assert request.request_id == 'request-1'
    assert request.dispatch['lanes'][0]['quantum'] == 3
    assert request.data['lanes'][0]['quantum'] == 3


@pytest.mark.parametrize(
    'mutation', ['version', 'actor', 'generation', 'capacity', 'admission']
)
def test_invalid_startup_request_is_rejected(mutation):
    raw = scheduled()
    if mutation == 'version':
        raw['schema_version'] = 2
    elif mutation == 'actor':
        raw['requested_by'] = ''
    elif mutation == 'generation':
        raw['expected_generation'] = True
    elif mutation == 'capacity':
        raw['policy']['total_concurrent_dispatch'] = 0
    else:
        raw['policy']['lanes'][0]['metadata'].pop('admission')
    with pytest.raises(dispatch_policy.DispatchPolicyError) as error:
        dispatch_policy.parse_startup_dispatch_policy('default', raw)
    assert error.value.code == 'dispatch_startup_policy_invalid'


def test_startup_activates_captured_settings_and_keeps_newer_draft_edits():
    database = _PolicyDatabase()
    database.data['metadata']['llm_dispatch_startup'] = scheduled(database.data)
    repository = _memory_repository(database)
    request = repository.load_startup_dispatch_policy('default')
    database.data['lanes'][0]['quantum'] = 99
    check_store = Mock()

    activation = repository.activate_startup_policy('default', request, check_store)

    assert activation.generation == 1
    assert database.data['lanes'][0]['quantum'] == 99
    assert (
        repository.load_runtime_dispatch_policy('default')['lanes'][0]['quantum'] == 3
    )
    assert 'llm_dispatch_startup' not in database.data['metadata']
    assert check_store.call_count == 2


@pytest.mark.parametrize(
    'failure', ['changed_request', 'active_generation', 'postgres_work', 'store_work']
)
def test_startup_preflight_failure_preserves_active_generation_and_request(failure):
    database = _PolicyDatabase()
    repository = _memory_repository(database)
    first = repository.activate_admission_policy('default', 'operator-1')
    database.data['metadata']['llm_dispatch_startup'] = scheduled(
        database.data, first.generation
    )
    request = repository.load_startup_dispatch_policy('default')
    check_store = Mock()
    if failure == 'changed_request':
        database.data['metadata']['llm_dispatch_startup']['request_id'] = 'request-2'
    elif failure == 'active_generation':
        request = dispatch_policy.parse_startup_dispatch_policy(
            'default', scheduled(database.data, 5)
        )
    elif failure == 'postgres_work':
        database.routes_in_use = True
    else:
        check_store.side_effect = RuntimeError('store unavailable')
    with pytest.raises((dispatch_policy.DispatchPolicyError, RuntimeError)):
        repository.activate_startup_policy('default', request, check_store)
    assert database.active_generation == first.generation
    assert 'llm_dispatch_startup' in database.data['metadata']
    assert len(database.generations) == 1


def test_explicit_activation_cancels_a_scheduled_startup_request():
    database = _PolicyDatabase()
    database.data['metadata']['llm_dispatch_startup'] = scheduled(database.data)
    repository = _memory_repository(database)
    repository.activate_admission_policy('default', 'operator-1')
    assert 'llm_dispatch_startup' not in database.data['metadata']


def test_first_startup_prepares_and_activates_sql_configuration_without_a_ui_save():
    database = _PolicyDatabase()
    repository = _memory_repository(database)
    assert repository.load_startup_dispatch_policy('default') is None
    request = repository.load_startup_dispatch_policy('default', initialize=True)
    assert request.expected_generation is None
    assert request.requested_by == 'gateway-startup'
    assert database.active_generation is None
    assert (
        repository.load_startup_dispatch_policy('default', initialize=True).request_id
        == request.request_id
    )
    check_store = Mock()
    activation = repository.activate_startup_policy('default', request, check_store)
    assert activation.generation == 1
    assert repository.load_runtime_dispatch_policy('default')['policy_generation'] == 1
    assert check_store.call_count == 2


def test_initialization_does_not_replace_an_active_policy_with_unsaved_restart_intent():
    database = _PolicyDatabase()
    repository = _memory_repository(database)
    activation = repository.activate_admission_policy('default', 'operator-1')
    database.data['total_concurrent_dispatch'] = 0
    assert repository.load_startup_dispatch_policy('default', initialize=True) is None
    assert database.active_generation == activation.generation
    assert 'llm_dispatch_startup' not in database.data['metadata']


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [None, 'credentials', 'owner', 'work'])
async def test_gateway_startup_uses_scheduled_credentials_and_transfers_owner(
    monkeypatch, failure
):
    from marie.engine.llm_queue import request_dispatcher, store
    from marie.engine.llm_queue.config import LlmQueueRuntimeConfig
    from marie.engine.llm_queue.scheduler_config import DatabaseSchedulerConfigSource

    from marie.excepts import RuntimeFailToStart
    from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
        GatewayLlmDispatchRuntime,
    )

    database = _PolicyDatabase()
    repository = _memory_repository(database)
    first = repository.activate_admission_policy('default', 'operator-1')
    captured = copy.deepcopy(database.data)
    captured['metadata']['llm_dispatch']['endpoints'][0][
        'credential_env'
    ] = 'NEW_TEST_KEY'
    database.data['metadata']['llm_dispatch_startup'] = scheduled(
        captured, first.generation
    )
    monkeypatch.delenv('TEST_OPERATOR_KEY', raising=False)
    monkeypatch.delenv('NEW_TEST_KEY', raising=False)
    if failure != 'credentials':
        monkeypatch.setenv('NEW_TEST_KEY', 'test-only-value')
    events = []
    owner = object()

    class Store:
        def __init__(self, *args, **kwargs):
            events.append('opened')

        def acquire_owner(self, *args, **kwargs):
            return None if failure == 'owner' else owner

        def renew_owner(self, *args, **kwargs):
            events.append('renewed')

        def scan_routes(self, **kwargs):
            return 0, ['default']

        def disable_route(self, *args):
            events.append('disabled')

        def reset_idle_bindings(self, token):
            assert token is owner
            return SimpleNamespace(
                disposition='work_pending' if failure == 'work' else 'reset'
            )

        def release_owner(self, token):
            assert token is owner
            events.append('released')

        def close(self):
            events.append('closed')

    class Dispatcher:
        dispatcher_id = 'test-dispatcher'
        owner_lease_ms = 10_000

        def __init__(self, **kwargs):
            self.policy_generation = kwargs['policy_generation']
            self.policy_digest = kwargs['policy_digest']

        async def start(self, *, owner=None):
            assert owner is not None
            assert self.policy_generation == 2
            events.append('started')

        async def stop(self):
            events.append('stopped')

    monkeypatch.setattr(store, 'RequestStore', Store)
    monkeypatch.setattr(request_dispatcher, 'RequestDispatcher', Dispatcher)
    runtime = GatewayLlmDispatchRuntime(
        queue_config=LlmQueueRuntimeConfig(
            enabled=True, fabric_group_id='default', queue_url='redis://test'
        ),
        scheduler_config_source=DatabaseSchedulerConfigSource(repository, 'default'),
        config={'llm_dispatch': {'policy_source': 'database'}},
    )
    if failure:
        with pytest.raises(RuntimeFailToStart):
            await runtime.start()
        assert database.active_generation == first.generation
        assert 'llm_dispatch_startup' in database.data['metadata']
        assert 'started' not in events
        if failure == 'work':
            assert 'released' in events
    else:
        await runtime.start()
        assert database.active_generation == 2
        assert events.index('disabled') < events.index('started')
        assert 'released' not in events
        assert 'llm_dispatch_startup' not in database.data['metadata']
        await runtime.stop()
