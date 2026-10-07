import copy
import json
from types import SimpleNamespace

from marie.serve.runtimes.gateway.marie.dispatch_policy import (
    build_policy_generation_snapshot,
)
from tests.unit.serve.runtimes.gateway.test_dispatch_policy import persisted


def test_scheduler_observation_keeps_activated_rules_and_omits_credentials():
    from marie.serve.runtimes.gateway.marie.scheduler_observation import (
        scheduler_observation,
    )

    desired = persisted()
    desired['enabled'] = True
    rule = desired['lanes'][0]['metadata']['admission']
    rule['priority'] = 10
    rule['match'] = {
        'document.effective_page_count': {'gte': 1, 'lte': 4},
        'workload.mode': {'eq': 'interactive'},
    }
    fallback = copy.deepcopy(desired['lanes'][0])
    fallback['pool_id'] = 'fallback'
    fallback['min_concurrent'] = 0
    fallback['metadata']['admission'] = {
        'schema_version': 1,
        'priority': 1_000_000,
        'accepting': True,
        'match': {},
    }
    desired['lanes'].append(fallback)
    activated, _ = build_policy_generation_snapshot('a', 4, desired)
    desired['lanes'][0]['quantum'] = 99
    observed = scheduler_observation(activated, 4, limit=50)
    assert observed['generation'] == 4
    assert observed['policy'] == 'drr'
    assert observed['total_concurrent_dispatch'] == 4
    assert observed['pools'][0]['quantum'] == 3
    assert observed['pools'][0]['admission']['match'] == {
        'document.effective_page_count': {'gte': 1, 'lte': 4},
        'workload.mode': {'eq': 'interactive'},
    }
    replica = observed['endpoint_groups'][0]['replicas'][0]
    assert replica['address'] == 'https://inference.example/v1'
    assert replica['execution_limit'] == 4
    assert replica['execution_bytes'] == 67108864
    assert replica['call_timeout_seconds'] == 60
    assert 'credential' not in json.dumps(observed)
    assert 'TEST_OPERATOR_KEY' not in json.dumps(observed)


def test_scheduler_observation_bounds_pools_and_never_exposes_url_credentials():
    from marie.serve.runtimes.gateway.marie.scheduler_observation import (
        scheduler_observation,
    )

    activated, _ = build_policy_generation_snapshot('a', 4, persisted())
    activated['dispatch']['lanes'].append(
        dict(activated['dispatch']['lanes'][0], pool_id='other')
    )
    activated['dispatch']['endpoints'][0]['base_url'] = (
        'https://user:secret@host/v1?api_key=secret'
    )
    observed = scheduler_observation(activated, 4, limit=1)
    assert len(observed['pools']) == 1
    assert observed['pools_truncated'] is True
    assert observed['endpoint_groups'][0]['replicas'][0]['address'] is None
    assert 'secret' not in json.dumps(observed)


def test_routing_diagnostics_reads_activated_snapshot_and_checks_digest():
    import pytest

    from marie.serve.runtimes.gateway.marie.llm_scheduler_config import (
        PostgresSchedulerConfigRepository,
        _policy_digest,
    )

    activated, _ = build_policy_generation_snapshot('a', 4, persisted())
    digest = _policy_digest(activated)

    class Cursor:
        result = None

        def execute(self, query, params=None):
            if 'config.active_policy_generation' in query:
                assert 'generation.policy_snapshot' in query
                assert params == ('a',)
                self.result = (4, digest, activated)
            elif 'FROM marie_scheduler.llm_job_route' in query:
                self.result = (0, 0, 0)

        def fetchone(self):
            return self.result

        def fetchall(self):
            return []

        def close(self):
            pass

    repository = object.__new__(PostgresSchedulerConfigRepository)
    repository.config_schema = 'marie_scheduler'
    repository._get_connection = lambda: SimpleNamespace(
        cursor=Cursor, commit=lambda: None, rollback=lambda: None
    )
    repository._close_cursor = lambda cursor: cursor.close()
    repository._close_connection = lambda connection: None
    diagnostics = repository.load_routing_diagnostics('a')
    assert diagnostics['scheduler_config']['pools'][0]['quantum'] == 3
    assert diagnostics['scheduler_config']['generation'] == 4
    activated['dispatch']['lanes'][0]['quantum'] = 99
    with pytest.raises(ValueError, match='digest mismatch'):
        repository.load_routing_diagnostics('a')
