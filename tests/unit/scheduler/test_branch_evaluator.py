from typing import Any

import pytest

from marie.query_planner.base import Query, QueryPlan
from marie.query_planner.branching import SwitchQueryDefinition
from marie.scheduler.branch_evaluator import BranchEvaluationContext, BranchEvaluator
from marie.scheduler.models import WorkInfo


def switch_context(data: dict[str, Any]) -> BranchEvaluationContext:
    node = Query(task_id='switch', query_str='switch')
    return BranchEvaluationContext(
        work_info=WorkInfo.model_construct(name='extract', data=data),
        dag_plan=QueryPlan(nodes=[node]),
        branch_node=node,
    )


@pytest.mark.parametrize('default_case', [['fallback'], None])
@pytest.mark.parametrize(
    ('switch_field', 'data'),
    [
        pytest.param(
            '$.data.values[*]', {'values': ['invoice', 'contract']}, id='multiple-matches'
        ),
        pytest.param('$.data.value', {'value': ['invoice']}, id='array'),
        pytest.param('$.data.value', {'value': []}, id='empty-array'),
        pytest.param('$.data.value', {'value': {'type': 'invoice'}}, id='object'),
        pytest.param('$.data.value', {'value': {}}, id='empty-object'),
    ],
)
async def test_switch_unhashable_value_uses_default(
    switch_field: str, data: dict[str, Any], default_case: list[str] | None
) -> None:
    switch_def = SwitchQueryDefinition(
        switch_field=switch_field,
        cases={'invoice': ['invoice-node']},
        default_case=default_case,
    )

    result = await BranchEvaluator().evaluate_switch(switch_def, switch_context(data))

    assert result == default_case


@pytest.mark.parametrize('value', ['invoice', 42, True, None])
async def test_switch_matching_scalar_selects_case(value: Any) -> None:
    switch_def = SwitchQueryDefinition(
        switch_field='$.data.value',
        cases={value: ['matched-node']},
        default_case=['fallback'],
    )

    result = await BranchEvaluator().evaluate_switch(
        switch_def, switch_context({'value': value})
    )

    assert result == ['matched-node']


@pytest.mark.parametrize('data', [{'value': 'unknown'}, {}])
async def test_switch_unmatched_or_missing_value_uses_default(
    data: dict[str, Any],
) -> None:
    switch_def = SwitchQueryDefinition(
        switch_field='$.data.value',
        cases={'invoice': ['invoice-node']},
        default_case=['fallback'],
    )

    result = await BranchEvaluator().evaluate_switch(switch_def, switch_context(data))

    assert result == ['fallback']
