from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from marie.query_planner.base import Query, QueryPlan, QueryType
from marie.query_planner.branching import (
    BranchPath,
    BranchQueryDefinition,
    SwitchQueryDefinition,
)
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.models import WorkInfo
from marie.scheduler.services.control_flow_execution_service import (
    ControlFlowExecutionOutcome,
    ControlFlowExecutionService,
)


def build_service() -> ControlFlowExecutionService:
    return ControlFlowExecutionService(
        repository=AsyncMock(),
        frontier=AsyncMock(),
        dag_service=AsyncMock(),
        status_update_lock=AsyncJobLock(),
        topology_cache=DagTopologyCache(),
        job_cache={},
        lease_owner='scheduler-1',
        run_ttl_seconds=60,
        gateway_instance_id='gateway-1',
        notify_callback=AsyncMock(return_value=True),
    )


def work_item(job_id: str) -> WorkInfo:
    return WorkInfo.model_construct(
        id=job_id,
        dag_id='dag-1',
        name='extract',
        data={'metadata': {'on': 'noop://default'}},
    )


async def test_process_node_returns_completed_after_durable_completion() -> None:
    service = build_service()
    item = work_item('noop')
    service.dag_service.active_dags = {'dag-1': object()}
    service._topology_cache = Mock(
        get_sorted_nodes_and_levels=Mock(
            return_value=([], {'noop': 0, 'downstream': 1})
        )
    )
    service._activate = AsyncMock(return_value=True)
    service._complete_attempt = AsyncMock(return_value=True)

    outcome = await service.process_node(item)

    assert outcome is ControlFlowExecutionOutcome.COMPLETED
    service.frontier.on_job_completed.assert_awaited_once_with('noop')
    service._notify_callback.assert_awaited_once_with()


async def test_process_node_returns_cleaned_up_when_dag_is_missing() -> None:
    service = build_service()
    service.dag_service.active_dags = {}
    service.dag_service.get_dag.return_value = None

    outcome = await service.process_node(work_item('noop'))

    assert outcome is ControlFlowExecutionOutcome.CLEANED_UP
    service.repository.release_lease.assert_awaited_once_with(job_ids=['noop'])
    service.frontier.release_lease_local.assert_awaited_once_with('noop')


async def test_process_node_returns_admission_refused() -> None:
    service = build_service()
    service.dag_service.active_dags = {}
    service.dag_service.get_dag.return_value = object()
    service.dag_service.admit_dag.return_value = False

    outcome = await service.process_node(work_item('noop'))

    assert outcome is ControlFlowExecutionOutcome.ADMISSION_REFUSED
    service.repository.release_lease.assert_awaited_once_with(job_ids=['noop'])


async def test_process_node_returns_activation_refused() -> None:
    service = build_service()
    service.dag_service.active_dags = {'dag-1': object()}
    service._activate = AsyncMock(return_value=False)

    outcome = await service.process_node(work_item('noop'))

    assert outcome is ControlFlowExecutionOutcome.ACTIVATION_REFUSED
    service.repository.release_lease.assert_awaited_once_with(job_ids=['noop'])


async def test_process_node_returns_completion_rejected() -> None:
    service = build_service()
    service.dag_service.active_dags = {'dag-1': object()}
    service._topology_cache = Mock(
        get_sorted_nodes_and_levels=Mock(return_value=([], {'noop': 0}))
    )
    service._activate = AsyncMock(return_value=True)
    service._complete_attempt = AsyncMock(return_value=False)

    outcome = await service.process_node(work_item('noop'))

    assert outcome is ControlFlowExecutionOutcome.COMPLETION_REJECTED


async def test_process_node_returns_failed_after_handled_error() -> None:
    service = build_service()
    service.dag_service.active_dags = {'dag-1': object()}
    service._activate = AsyncMock(side_effect=RuntimeError('activation failed'))

    outcome = await service.process_node(work_item('noop'))

    assert outcome is ControlFlowExecutionOutcome.FAILED
    service.repository.release_lease.assert_awaited_once_with(job_ids=['noop'])


async def test_process_node_preserves_progress_when_followup_fails() -> None:
    service = build_service()
    service.dag_service.active_dags = {'dag-1': object()}
    service._topology_cache = Mock(
        get_sorted_nodes_and_levels=Mock(
            return_value=([], {'noop': 0, 'downstream': 1})
        )
    )
    service._activate = AsyncMock(return_value=True)
    service._complete_attempt = AsyncMock(return_value=True)
    service.frontier.on_job_completed.side_effect = RuntimeError('frontier failed')

    outcome = await service.process_node(work_item('noop'))

    assert outcome is ControlFlowExecutionOutcome.COMPLETED_WITH_ERROR
    assert outcome.made_progress is True
    service.repository.release_lease.assert_awaited_once_with(job_ids=['noop'])


async def test_branch_marks_selected_target_and_skips_only_inactive_closure() -> None:
    service = build_service()
    service._branch_evaluator = SimpleNamespace(
        evaluate_branch=AsyncMock(return_value=['selected-path'])
    )
    service.repository.mark_jobs_as_skipped.return_value = {'inactive'}
    plan = QueryPlan(
        nodes=[
            Query(
                task_id='branch',
                query_str='branch',
                node_type=QueryType.BRANCH,
                definition=BranchQueryDefinition(
                    paths=[
                        BranchPath(
                            path_id='selected-path',
                            target_node_ids=['selected'],
                        ),
                        BranchPath(
                            path_id='inactive-path',
                            target_node_ids=['inactive'],
                        ),
                    ]
                ),
            ),
            Query(task_id='selected', query_str='selected', dependencies=['branch']),
            Query(task_id='inactive', query_str='inactive', dependencies=['branch']),
            Query(
                task_id='merger',
                query_str='merger',
                dependencies=['selected', 'inactive'],
            ),
        ]
    )

    await service._evaluate_and_mark_branch_paths('branch', work_item('branch'), plan)

    metadata_calls = {
        call.kwargs['job_id']: call.kwargs['metadata_updates']['branch_metadata']
        for call in service.repository.update_job_metadata.await_args_list
    }
    assert metadata_calls['branch']['selected_path_ids'] == ['selected-path']
    assert metadata_calls['selected']['selected_path_id'] == 'selected-path'
    assert metadata_calls['inactive']['skipped'] is True
    assert service.repository.mark_jobs_as_skipped.await_args.kwargs['job_ids'] == [
        'inactive'
    ]
    service.frontier.on_jobs_skipped.assert_awaited_once_with(['inactive'])


async def test_switch_marks_selected_case_and_skips_other_case() -> None:
    service = build_service()
    service._branch_evaluator = SimpleNamespace(
        evaluate_switch=AsyncMock(return_value=['invoice']),
        jsonpath_evaluator=Mock(evaluate=Mock(return_value='invoice')),
    )
    service.repository.mark_jobs_as_skipped.return_value = {'contract'}
    plan = QueryPlan(
        nodes=[
            Query(
                task_id='switch',
                query_str='switch',
                node_type=QueryType.SWITCH,
                definition=SwitchQueryDefinition(
                    switch_field='$.metadata.document_type',
                    cases={
                        'invoice': ['invoice'],
                        'contract': ['contract'],
                    },
                ),
            ),
            Query(task_id='invoice', query_str='invoice', dependencies=['switch']),
            Query(task_id='contract', query_str='contract', dependencies=['switch']),
        ]
    )

    await service._evaluate_and_mark_branch_paths('switch', work_item('switch'), plan)

    metadata_calls = {
        call.kwargs['job_id']: call.kwargs['metadata_updates']['branch_metadata']
        for call in service.repository.update_job_metadata.await_args_list
    }
    assert metadata_calls['switch']['switch_value'] == 'invoice'
    assert metadata_calls['invoice']['selected_case'] == 'invoice'
    assert metadata_calls['contract']['skipped'] is True
    assert service.repository.mark_jobs_as_skipped.await_args.kwargs['job_ids'] == [
        'contract'
    ]
    service.frontier.on_jobs_skipped.assert_awaited_once_with(['contract'])
