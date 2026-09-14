from types import SimpleNamespace

import pytest
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.scheduler_config import DatabaseSchedulerConfigSource

from marie.excepts import RuntimeFailToStart
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.gateway.marie.llm_scheduler_config import (
    PostgresSchedulerConfigRepository,
)


def persisted():
    return {
        'policy': 'drr',
        'total_concurrent_dispatch': 4,
        'metadata': {
            'llm_dispatch': {
                'schema_version': 1,
                'endpoints': [
                    {
                        'endpoint_id': 'primary',
                        'base_url': 'https://inference.example/v1',
                        'credential_env': 'TEST_OPERATOR_KEY',
                        'execution_limit': 4,
                    }
                ],
                'limits': {'max_execution_items': 4},
            }
        },
        'lanes': [
            {
                'pool_id': 'default',
                'enabled': True,
                'max_concurrent': 2,
                'min_concurrent': 1,
                'quantum': 3,
                'metadata': {
                    'llm_dispatch': {
                        'schema_version': 1,
                        'endpoint_id': 'primary',
                        'revision': 'r1',
                        'execution_bytes': 67108864,
                    }
                },
            }
        ],
    }


def test_persisted_policy_uses_existing_pool_capacity_and_immutable_limits():
    from marie.serve.runtimes.gateway.marie.dispatch_policy import (
        persisted_dispatch_policy,
    )

    policy = persisted_dispatch_policy(persisted())
    assert policy['lanes'][0]['execution_limit'] == 2
    assert policy['limits']['max_execution_items'] == 4
    assert 'api_key' not in policy['endpoints'][0]


@pytest.mark.parametrize(
    'mutation', ['secret', 'unknown', 'version', 'ceiling', 'missing_capacity']
)
def test_persisted_policy_rejects_unsafe_ambiguous_metadata(mutation):
    from marie.serve.runtimes.gateway.marie.dispatch_policy import (
        persisted_dispatch_policy,
    )

    data = persisted()
    config = data['metadata']['llm_dispatch']
    if mutation == 'secret':
        config['endpoints'][0]['api_key'] = 'raw-secret'
    if mutation == 'unknown':
        config['endpoints'][0]['credential_env'] = '$(unsafe)'
    if mutation == 'version':
        config['schema_version'] = 2
    if mutation == 'ceiling':
        data['lanes'][0]['max_concurrent'] = 5
    if mutation == 'missing_capacity':
        data['lanes'][0]['max_concurrent'] = None
    with pytest.raises(ValueError):
        persisted_dispatch_policy(data)


@pytest.mark.asyncio
async def test_database_source_selected_at_v3_startup_without_yaml_fallback(
    monkeypatch,
):
    from marie.engine.llm_queue import request_dispatcher
    from marie.engine.llm_queue import store as store_module

    selected = []

    class Repository:
        def load_scheduler_config(self, fabric):
            selected.append(fabric)
            return persisted()

    source = DatabaseSchedulerConfigSource(Repository(), 'chosen')
    config = LlmQueueConfig(
        enabled=True,
        queue_contract_version='v3',
        fabric_group_id='chosen',
        queue_url='redis://test',
    )
    runtime = GatewayLlmDispatchRuntime(
        queue_config=config,
        scheduler_config_source=source,
        config={
            'llm_dispatch': {
                'policy_source': 'database',
                'endpoints': [{'api_key': 'must-not-read'}],
            }
        },
    )
    observed = {}

    class Store:
        def __init__(self, url, **kwargs):
            observed.update(kwargs)

        def close(self):
            pass

    class Dispatcher:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        async def start(self):
            pass

        async def stop(self):
            pass

    monkeypatch.setenv('TEST_OPERATOR_KEY', 'fixture-credential')
    monkeypatch.setattr(store_module, 'RequestStore', Store)
    monkeypatch.setattr(request_dispatcher, 'RequestDispatcher', Dispatcher)
    await runtime.start()
    assert selected == ['chosen']
    assert observed['limits'].max_execution_items == 4
    assert observed['lanes'][0].execution_limit == 2
    assert observed['endpoints'][0].api_key == 'fixture-credential'
    await runtime.stop()

    def fail(fabric):
        raise RuntimeError('database secret')

    source.repository.load_scheduler_config = fail
    with pytest.raises(RuntimeFailToStart) as caught:
        await runtime.start()
    assert 'secret' not in str(caught.value)


def test_postgres_startup_does_not_disclose_connection_configuration(monkeypatch):
    from marie.storage.database import postgres

    messages = []
    pool = SimpleNamespace(wait=lambda **kwargs: None)
    monkeypatch.setattr(postgres, 'ConnectionPool', lambda *args, **kwargs: pool)
    repository = PostgresSchedulerConfigRepository(
        {
            'hostname': 'localhost',
            'port': 5432,
            'username': 'fixture',
            'password': 'credential-sentinel',
            'database': 'fixture',
        },
        logger=SimpleNamespace(info=messages.append),
    )
    assert 'credential-sentinel' not in str(messages)

    def fail(*args, **kwargs):
        raise RuntimeError('raw-detail-sentinel')

    monkeypatch.setattr(postgres, 'ConnectionPool', fail)
    with pytest.raises(Exception) as caught:
        PostgresSchedulerConfigRepository(
            {
                'hostname': 'localhost',
                'port': 5432,
                'username': 'fixture',
                'password': 'credential-sentinel',
                'database': 'fixture',
            }
        )
    assert 'sentinel' not in str(caught.value)


@pytest.mark.parametrize(
    'endpoint',
    ['https://other.example/v1', 'https://inference.example/v1?credential=sentinel'],
)
@pytest.mark.asyncio
async def test_v2_database_lane_cannot_receive_runtime_global_key(
    monkeypatch, endpoint
):
    source = DatabaseSchedulerConfigSource(
        SimpleNamespace(
            load_scheduler_config=lambda _: {
                'policy': 'drr',
                'total_concurrent_dispatch': 1,
                'lanes': [{'pool_id': 'default', 'endpoint_url': endpoint}],
            }
        ),
        'chosen',
    )
    config = LlmQueueConfig(
        enabled=True,
        queue_contract_version='v2',
        fabric_group_id='chosen',
        queue_url='redis://test',
    )
    monkeypatch.setenv('OPENAI_API_KEY', 'runtime-fixture-key')
    monkeypatch.setenv('OPENAI_API_BASE', 'https://inference.example/v1')
    from marie.serve.runtimes.gateway.marie import llm_dispatch_runtime as module

    monkeypatch.setattr(
        module,
        'resolve_openai_base_url_from_env',
        lambda: 'https://inference.example/v1',
    )
    queue = SimpleNamespace(request_queue_depth=lambda _: 0, close=lambda: None)
    runtime = GatewayLlmDispatchRuntime(
        queue_config=config,
        scheduler_config_source=source,
        queue_client_factory=lambda _: queue,
        openai_client_factory=lambda *a: pytest.fail(
            'client constructed for unauthorized target'
        ),
    )
    with pytest.raises(RuntimeFailToStart) as caught:
        await runtime.start()
    assert 'sentinel' not in str(caught.value)


def test_persisted_fairness_and_total_reach_dispatch_policy():
    from marie.serve.runtimes.gateway.marie.dispatch_policy import (
        persisted_dispatch_policy,
    )

    data = persisted()
    data['lanes'][0]['max_burst_per_visit'] = 2
    policy = persisted_dispatch_policy(data)
    assert policy['policy'] == 'drr'
    assert policy['total_concurrent_dispatch'] == 4
    assert policy['lanes'][0]['quantum'] == 3
    assert policy['lanes'][0]['min_concurrent'] == 1
    assert policy['lanes'][0]['max_burst_per_visit'] == 2


@pytest.mark.parametrize(
    'name,value',
    [
        ('quantum', True),
        ('quantum', '3'),
        ('min_concurrent', 3),
        ('max_burst_per_visit', 0),
    ],
)
def test_persisted_fairness_is_strict(name, value):
    from marie.serve.runtimes.gateway.marie.dispatch_policy import (
        persisted_dispatch_policy,
    )

    data = persisted()
    data['lanes'][0][name] = value
    with pytest.raises(ValueError):
        persisted_dispatch_policy(data)


def _bounded_lane_repository(row_count):
    import re

    data = persisted()
    data['metadata']['llm_dispatch']['limits']['max_routes'] = 2000
    lane_metadata = data['lanes'][0]['metadata']
    transactions = []

    class Cursor:
        limit = None

        def execute(self, query, params=None):
            if 'FROM marie_scheduler.llm_queue_pool' in query:
                match = re.search(r'LIMIT\s+(\d+)', query)
                assert match, 'The repository must keep a finite lane read'
                self.limit = int(match.group(1))

        def fetchone(self):
            return ('drr', 4, data['metadata'])

        def fetchall(self):
            return [
                (f'pool-{index}', None, None, 1, 0, 2, None, True, lane_metadata)
                for index in range(min(row_count, self.limit))
            ]

    cursor = Cursor()
    connection = SimpleNamespace(
        cursor=lambda: cursor,
        commit=lambda: transactions.append('commit'),
        rollback=lambda: transactions.append('rollback'),
    )
    repository = object.__new__(PostgresSchedulerConfigRepository)
    repository.config_schema = 'marie_scheduler'
    repository._get_connection = lambda: connection
    repository._close_cursor = lambda value: transactions.append('cursor_closed')
    repository._close_connection = lambda value: transactions.append(
        'connection_returned'
    )
    return repository, cursor, transactions


@pytest.mark.parametrize('row_count', [1000, 1001, 1100])
def test_database_lane_ceiling_rejects_truncated_policy_despite_larger_store_limit(
    row_count,
):
    from marie.serve.runtimes.gateway.marie.dispatch_policy import (
        persisted_dispatch_policy,
    )

    repository, cursor, transactions = _bounded_lane_repository(row_count)
    if row_count == 1000:
        policy = persisted_dispatch_policy(repository.load_scheduler_config('fabric'))
        assert policy['limits']['max_routes'] == 2000
        assert len(policy['lanes']) == 1000
        assert transactions == ['commit', 'cursor_closed', 'connection_returned']
    else:
        with pytest.raises(ValueError, match='1000'):
            persisted_dispatch_policy(repository.load_scheduler_config('fabric'))
        assert transactions == ['rollback', 'cursor_closed', 'connection_returned']
    assert cursor.limit == 1001


@pytest.mark.asyncio
async def test_overflow_refresh_disables_whole_policy_without_publishing_subset():
    import asyncio

    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )
    from marie.engine.llm_queue.store import OwnerToken, StoreLimits

    from marie.serve.runtimes.gateway.marie.dispatch_policy import (
        persisted_dispatch_policy,
    )

    repository, _, _ = _bounded_lane_repository(1100)
    mutations = []
    limits = StoreLimits(max_routes=2000, max_execution_items=4)

    def unexpected_configuration(*args, **kwargs):
        pytest.fail('A truncated policy must not configure any lane or endpoint')

    store = SimpleNamespace(
        limits=limits,
        scan_routes=lambda **kwargs: (0, ['current', 'omitted-by-truncation']),
        disable_route=lambda owner, pool: mutations.append(('disable', pool)),
        configure_route=unexpected_configuration,
        configure_endpoint=unexpected_configuration,
    )

    def load():
        policy = persisted_dispatch_policy(repository.load_scheduler_config('fabric'))
        endpoint_values = dict(policy['endpoints'][0])
        endpoint_values.pop('credential_env')
        return dict(
            limits=StoreLimits(**policy['limits']),
            endpoints=[RegisteredEndpoint(**endpoint_values)],
            lanes=[DispatchLane(**lane) for lane in policy['lanes']],
            policy=policy['policy'],
            total_concurrent_dispatch=policy['total_concurrent_dispatch'],
        )

    original_lanes = [DispatchLane('current', 'primary', execution_limit=2)]
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'primary', 'https://inference.example/v1', execution_limit=4
            )
        ],
        lanes=original_lanes,
        policy_loader=load,
    )
    runtime.owner = OwnerToken('owner', 1)
    try:
        await runtime._refresh_policy()
        if runtime._refresh_future is not None:
            await asyncio.gather(
                asyncio.wrap_future(runtime._refresh_future), return_exceptions=True
            )
            await runtime._refresh_policy()
        assert runtime._policy_paused
        assert runtime._category == 'policy_refresh_unavailable'
        assert runtime.lanes == tuple(original_lanes)
        assert runtime._config_revision == 1
        assert mutations == [
            ('disable', 'current'),
            ('disable', 'omitted-by-truncation'),
        ]
    finally:
        runtime._store_workers.shutdown(wait=True)
        runtime._lease_worker.shutdown(wait=True)
        runtime._refresh_worker.shutdown(wait=True)
