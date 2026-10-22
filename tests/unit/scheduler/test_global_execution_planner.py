from typing import Any

import pytest

from marie.scheduler.global_execution_planner import GlobalPriorityExecutionPlanner
from marie.scheduler.models import WorkInfo


def job(job_id: str, data: dict[str, Any], *, priority: int = 0) -> WorkInfo:
    return WorkInfo.model_construct(
        id=job_id,
        name='extract',
        dag_id='dag-1',
        job_level=1,
        priority=priority,
        data=data,
    )


@pytest.mark.parametrize(
    'data',
    [
        pytest.param({'metadata': None}, id='null-metadata'),
        pytest.param({'metadata': 'invalid'}, id='string-metadata'),
        pytest.param({'metadata': []}, id='array-metadata'),
        pytest.param({'metadata': 42}, id='numeric-metadata'),
        pytest.param({'metadata': {'estimated_runtime': 'invalid'}}, id='invalid-runtime'),
        pytest.param({'metadata': {'estimated_runtime': {}}}, id='object-runtime'),
        pytest.param({'metadata': {'estimated_runtime': []}}, id='array-runtime'),
        pytest.param({'metadata': {'estimated_runtime': None}}, id='null-runtime'),
        pytest.param({'metadata': {}}, id='missing-runtime'),
        pytest.param({}, id='missing-metadata'),
    ],
)
def test_invalid_runtime_ranks_as_unknown_without_aborting_batch(
    data: dict[str, Any],
) -> None:
    jobs = [
        job('unknown-first', data),
        job('slow', {'metadata': {'estimated_runtime': 10}}),
        job('fast', {'metadata': {'estimated_runtime': 1}}),
        job('unknown-last', data),
    ]

    ranked = GlobalPriorityExecutionPlanner().plan(
        [('extract://default', wi) for wi in jobs], {'extract': 1}, {'dag-1'}
    )

    assert [wi.id for _, wi in ranked] == [
        'fast',
        'slow',
        'unknown-first',
        'unknown-last',
    ]


@pytest.mark.parametrize('estimate', [0, 2, 2.5, '2', '2.5'])
def test_numeric_runtime_estimates_still_rank_before_slower_jobs(estimate: Any) -> None:
    slow = job('slow', {'metadata': {'estimated_runtime': 5}})
    fast = job('fast', {'metadata': {'estimated_runtime': estimate}})

    ranked = GlobalPriorityExecutionPlanner().plan(
        [('extract://default', slow), ('extract://default', fast)],
        {'extract': 1},
        {'dag-1'},
    )

    assert [wi.id for _, wi in ranked] == ['fast', 'slow']


def test_invalid_runtime_does_not_override_manual_priority() -> None:
    fast = job('fast', {'metadata': {'estimated_runtime': 1}})
    urgent = job('urgent', {'metadata': {'estimated_runtime': []}}, priority=5)

    ranked = GlobalPriorityExecutionPlanner().plan(
        [('extract://default', fast), ('extract://default', urgent)],
        {'extract': 1},
        {'dag-1'},
    )

    assert [wi.id for _, wi in ranked] == ['urgent', 'fast']
