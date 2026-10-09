import asyncio
import threading
from typing import Any

import pytest

import marie.scheduler.branch_evaluator as branch_module
from marie.query_planner.base import Query, QueryPlan
from marie.query_planner.branching import (
    BranchPath,
    BranchQueryDefinition,
    PythonBranchQueryDefinition,
    SwitchQueryDefinition,
)
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


def route_on_prior_result(context: dict[str, Any]) -> str:
    return 'prior-result' if context['execution_results'].get('extract') else 'no-results'


@pytest.mark.parametrize(
    ('context_kwargs', 'expected_path'),
    [
        pytest.param({}, 'no-results', id='omitted'),
        pytest.param({'execution_results': None}, 'no-results', id='null'),
        pytest.param({'execution_results': {}}, 'no-results', id='empty'),
        pytest.param(
            {'execution_results': {'extract': {'status': 'completed'}}},
            'prior-result',
            id='populated',
        ),
    ],
)
async def test_python_branch_reads_normalized_execution_results(
    context_kwargs: dict[str, Any], expected_path: str
) -> None:
    node = Query(task_id='branch', query_str='branch')
    context = BranchEvaluationContext(
        work_info=WorkInfo.model_construct(name='extract', data={}),
        dag_plan=QueryPlan(nodes=[node]),
        branch_node=node,
        **context_kwargs,
    )
    branch_def = PythonBranchQueryDefinition(
        branch_function=f'{__name__}.route_on_prior_result',
        paths=[
            BranchPath(path_id='no-results', target_node_ids=['new-extract']),
            BranchPath(path_id='prior-result', target_node_ids=['use-prior-result']),
        ],
    )

    result = await BranchEvaluator().evaluate_branch(branch_def, context)

    assert result == [expected_path]


def blocking_condition(context: dict[str, Any]) -> bool:
    data = context['data']
    try:
        data['release'].wait(timeout=2)
        return True
    finally:
        data['finished'].set()


@pytest.mark.parametrize('include_later_path', [True, False])
async def test_condition_timeout_allows_later_or_default_path(
    include_later_path: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(branch_module, 'DEFAULT_CONDITION_TIMEOUT', 0.02, raising=False)
    release = threading.Event()
    finished = threading.Event()
    paths = [
        BranchPath(
            path_id='blocked',
            target_node_ids=['blocked-node'],
            condition_function=f'{__name__}.blocking_condition',
        )
    ]
    if include_later_path:
        paths.append(BranchPath(path_id='later', target_node_ids=['later-node']))
    branch_def = BranchQueryDefinition(paths=paths, default_path_id='fallback')
    context = switch_context({'release': release, 'finished': finished})
    evaluation = asyncio.create_task(BranchEvaluator().evaluate_branch(branch_def, context))

    try:
        done, _ = await asyncio.wait([evaluation], timeout=0.2)

        assert evaluation in done, 'A blocked condition held up branch evaluation'
        assert await evaluation == (['later'] if include_later_path else ['fallback'])
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.gather(evaluation, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 2)


def configured_condition(context: dict[str, Any]) -> bool:
    return context['data']['result']


@pytest.mark.parametrize('condition_result', [True, False])
async def test_condition_function_preserves_boolean_routing(condition_result: bool) -> None:
    branch_def = BranchQueryDefinition(
        paths=[
            BranchPath(
                path_id='matched',
                target_node_ids=['matched-node'],
                condition_function=f'{__name__}.configured_condition',
            )
        ],
        default_path_id='fallback',
    )

    result = await BranchEvaluator().evaluate_branch(
        branch_def, switch_context({'result': condition_result})
    )

    assert result == (['matched'] if condition_result else ['fallback'])
