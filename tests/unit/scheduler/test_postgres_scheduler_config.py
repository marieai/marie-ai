import pytest

from marie.excepts import BadConfigSource
from marie.scheduler.postgres_scheduler_config import PostgreSQLSchedulerConfig


def test_scheduler_config_parses_runtime_defaults() -> None:
    config = PostgreSQLSchedulerConfig.from_dict({'queue_names': ['extract']})

    assert config.queue_names == frozenset({'extract'})
    assert config.max_workers == 5
    assert config.submission_queue_size == 1000
    assert config.job_event_worker_count == 8
    assert config.job_event_queue_size == 1024
    assert config.priority_refresh_enabled is False
    assert config.priority_refresh_interval == 10
    assert config.max_concurrent_dags == 16
    assert config.hard_sla_policy == 'track_only'
    assert config.run_ttl_seconds == 60
    assert config.run_lease_renewal_interval_seconds == 20


def test_scheduler_config_normalizes_unknown_hard_sla_policy() -> None:
    config = PostgreSQLSchedulerConfig.from_dict(
        {
            'queue_names': ['extract'],
            'hard_sla_policy': 'unknown',
        }
    )

    assert config.hard_sla_policy == 'track_only'
    assert config.invalid_hard_sla_policy == 'unknown'


def test_scheduler_config_can_enable_priority_refresh() -> None:
    config = PostgreSQLSchedulerConfig.from_dict(
        {
            'queue_names': ['extract'],
            'priority_refresh_enabled': True,
        }
    )

    assert config.priority_refresh_enabled is True


def test_scheduler_config_rejects_unimplemented_expire_policy() -> None:
    with pytest.raises(BadConfigSource, match="expire_unfinished.*not implemented"):
        PostgreSQLSchedulerConfig.from_dict(
            {
                'queue_names': ['extract'],
                'hard_sla_policy': 'expire_unfinished',
            }
        )


@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'priority_refresh_interval': 0}, 'priority_refresh_interval'),
        ({'submission_queue_size': 0}, 'submission_queue_size'),
        ({'job_event_worker_count': 0}, 'job_event_worker_count'),
        (
            {'job_event_worker_count': 4, 'job_event_queue_size': 3},
            'job_event_queue_size',
        ),
        ({'distributed_scheduler': False}, 'distributed_scheduler=false'),
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
