import pytest

from marie.excepts import BadConfigSource
from marie.scheduler.postgres_scheduler_config import PostgreSQLSchedulerConfig


def test_scheduler_config_parses_runtime_defaults() -> None:
    config = PostgreSQLSchedulerConfig.from_dict({'queue_names': ['extract']})

    assert config.queue_names == frozenset({'extract'})
    assert config.max_workers == 5
    assert config.submission_queue_size == 1000
    assert config.priority_refresh_interval == 10
    assert config.max_concurrent_dags == 16
    assert config.hard_sla_policy == 'track_only'


def test_scheduler_config_normalizes_unknown_hard_sla_policy() -> None:
    config = PostgreSQLSchedulerConfig.from_dict(
        {
            'queue_names': ['extract'],
            'hard_sla_policy': 'unknown',
        }
    )

    assert config.hard_sla_policy == 'track_only'
    assert config.invalid_hard_sla_policy == 'unknown'


@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'priority_refresh_interval': 0}, 'priority_refresh_interval'),
        ({'submission_queue_size': 0}, 'submission_queue_size'),
        ({'distributed_scheduler': False}, 'distributed_scheduler=false'),
        ({'dag_manager': {'max_concurrent_dags': 0}}, 'max_concurrent_dags'),
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
