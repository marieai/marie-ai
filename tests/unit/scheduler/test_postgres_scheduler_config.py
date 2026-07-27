from pathlib import Path

import pytest

from marie.excepts import BadConfigSource
from marie.jaml import JAML
from marie.scheduler.postgres_scheduler_config import PostgreSQLSchedulerConfig

REPOSITORY_ROOT = Path(__file__).parents[3]


def test_scheduler_config_parses_runtime_defaults() -> None:
    config = PostgreSQLSchedulerConfig.from_dict({'queue_names': ['extract']})

    assert config.queue_names == frozenset({'extract'})
    assert config.job_event_worker_count == 8
    assert config.job_event_queue_size == 1024
    assert config.dispatch_confirmation_max_in_flight == 256
    assert config.priority_refresh_enabled is False
    assert config.priority_refresh_interval == 10
    assert config.max_concurrent_dags == 16
    assert config.run_ttl_seconds == 60
    assert config.run_lease_renewal_interval_seconds == 20


def test_scheduler_config_accepts_consumed_bundle_keys() -> None:
    config = PostgreSQLSchedulerConfig.from_dict(
        {
            'queue_names': ['extract'],
            'provider': 'postgresql',
            'hostname': 'postgres',
            'port': 5432,
            'username': 'marie',
            'password': 'secret',
            'database': 'marie',
            'schema': 'marie_scheduler',
            'application_name': 'gateway',
            'options': '-c timezone=UTC',
            'min_pool_size': 1,
            'max_pool_size': 5,
            'pool_acquire_timeout_seconds': 30,
            'pool_open_timeout_seconds': 10,
            'desired_state_worker_count': 16,
            'desired_state_max_pending': 128,
            'dag_manager': {
                'max_concurrent_dags': 4,
                'dag_cache_size': 5000,
                'frontier_batch_size': 1000,
            },
        }
    )

    assert config.max_concurrent_dags == 4


@pytest.mark.parametrize(
    'key',
    [
        'distributed_scheduler',
        'hard_sla_policy',
        'max_workers',
        'max_wokers',
        'scheduler_mode',
        'submission_queue_size',
    ],
)
def test_scheduler_config_rejects_unknown_top_level_keys(key: str) -> None:
    with pytest.raises(BadConfigSource, match=rf'Unknown scheduler.*{key}'):
        PostgreSQLSchedulerConfig.from_dict(
            {
                'queue_names': ['extract'],
                key: True,
            }
        )


@pytest.mark.parametrize(
    'key',
    ['cache_ttl_seconds', 'min_concurrent_dags', 'strategy'],
)
def test_scheduler_config_rejects_unknown_dag_manager_keys(key: str) -> None:
    with pytest.raises(BadConfigSource, match=rf'Unknown dag_manager.*{key}'):
        PostgreSQLSchedulerConfig.from_dict(
            {
                'queue_names': ['extract'],
                'dag_manager': {key: True},
            }
        )


def test_scheduler_config_preserves_heartbeat_migration_error() -> None:
    with pytest.raises(BadConfigSource, match='heartbeat.*no longer supported'):
        PostgreSQLSchedulerConfig.from_dict(
            {
                'queue_names': ['extract'],
                'heartbeat': {},
            }
        )


@pytest.mark.parametrize(
    'relative_path',
    [
        'config/service/marie-gateway-4.0.0.yml',
        'config/service/mock/marie-mock-scheduler-test.yml',
        'deploy/helm/charts/marie/charts/server/files/service/marie-gateway-4.0.0.yml',
    ],
)
def test_shipped_scheduler_config_uses_supported_keys(relative_path: str) -> None:
    with (REPOSITORY_ROOT / relative_path).open() as stream:
        document = JAML.load_no_tags(stream)

    PostgreSQLSchedulerConfig.from_dict(document['with']['job_scheduler_kwargs'])


def test_scheduler_config_can_enable_priority_refresh() -> None:
    config = PostgreSQLSchedulerConfig.from_dict(
        {
            'queue_names': ['extract'],
            'priority_refresh_enabled': True,
        }
    )

    assert config.priority_refresh_enabled is True


@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'priority_refresh_interval': 0}, 'priority_refresh_interval'),
        ({'job_event_worker_count': 0}, 'job_event_worker_count'),
        (
            {'dispatch_confirmation_max_in_flight': 0},
            'dispatch_confirmation_max_in_flight',
        ),
        (
            {'job_event_worker_count': 4, 'job_event_queue_size': 3},
            'job_event_queue_size',
        ),
        ({'dag_manager': {'max_concurrent_dags': 0}}, 'max_concurrent_dags'),
        (
            {'run_lease_renewal_interval_seconds': 0},
            'run_lease_renewal_interval_seconds',
        ),
        (
            {
                'run_ttl_seconds': 300,
                'run_lease_renewal_interval_seconds': 101,
            },
            'one-third',
        ),
    ],
)
def test_scheduler_config_rejects_invalid_runtime_values(
    override: dict,
    message: str,
) -> None:
    with pytest.raises(BadConfigSource, match=message):
        PostgreSQLSchedulerConfig.from_dict(
            {
                'queue_names': ['extract'],
                **override,
            }
        )


def test_scheduler_config_derives_renewal_interval_from_run_ttl() -> None:
    config = PostgreSQLSchedulerConfig.from_dict(
        {'queue_names': ['extract'], 'run_ttl_seconds': 30}
    )

    assert config.run_lease_renewal_interval_seconds == 10
