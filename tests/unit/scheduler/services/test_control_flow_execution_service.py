import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

import marie.scheduler.services.control_flow_execution_service as control_flow_module
from marie.logging_core.logger import MarieLogger
from marie.query_planner.base import Query, QueryPlan, QueryType
from marie.query_planner.branching import (
    BranchPath,
    BranchQueryDefinition,
    SwitchQueryDefinition,
)
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.services.control_flow_execution_service import (
    ControlFlowExecutionOutcome,
    ControlFlowExecutionService,
)
from marie.scheduler.services.scheduler_runtime import SchedulerRuntime
from marie.scheduler.state import WorkState


def build_service() -> ControlFlowExecutionService:
    frontier = AsyncMock()
    frontier.leased_until = {}
    return ControlFlowExecutionService(
        repository=AsyncMock(),
        frontier=frontier,
        dag_service=AsyncMock(),
        status_update_lock=AsyncJobLock(),
        topology_cache=DagTopologyCache(),
        job_cache={},
        lease_owner='scheduler-1',
        run_ttl_seconds=60,
        gateway_instance_id='gateway-1',
        notify_callback=AsyncMock(return_value=True),
        runtime=SchedulerRuntime(MarieLogger('control-flow-test')),
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


async def test_process_nodes_batches_simple_node_persistence() -> None:
    service = build_service()
    first = work_item('noop-1')
    second = work_item('merger-1')
    second.data = {'metadata': {'on': 'merger://default'}}
    for item in (first, second):
        item.state = WorkState.CREATED
        item.job_level = 1

    service.dag_service.active_dags = {'dag-1': object()}
    service._topology_cache = Mock(
        get_sorted_nodes_and_levels=Mock(
            return_value=(
                [],
                {'root': 2, 'noop-1': 1, 'merger-1': 1, 'child': 0},
            )
        )
    )
    service.repository.activate_from_lease.return_value = {
        'noop-1': 'attempt-1',
        'merger-1': 'attempt-2',
    }
    service.repository.complete_job_attempts.return_value = {
        'noop-1',
        'merger-1',
    }

    outcomes = await service.process_nodes([first, second])

    assert outcomes == [
        ControlFlowExecutionOutcome.COMPLETED,
        ControlFlowExecutionOutcome.COMPLETED,
    ]
    service.repository.activate_from_lease.assert_awaited_once_with(
        job_ids=['noop-1', 'merger-1'],
        owner='scheduler-1',
        run_ttl_seconds=60,
        gateway_instance_id='gateway-1',
    )
    service.repository.complete_job_attempts.assert_awaited_once_with(
        {
            'noop-1': ('extract', 'attempt-1'),
            'merger-1': ('extract', 'attempt-2'),
        },
        run_owner='scheduler-1',
        output_metadata={},
    )
    service.repository.complete_job.assert_not_awaited()
    service._notify_callback.assert_awaited_once_with()
    assert service.frontier.on_job_completed.await_count == 2


async def test_started_toast_does_not_block_node_completion(monkeypatch) -> None:
    service = build_service()
    item = work_item('noop')
    item.job_level = 2
    service.dag_service.active_dags = {'dag-1': object()}
    service._topology_cache = Mock(
        get_sorted_nodes_and_levels=Mock(
            return_value=([], {'noop': 2, 'downstream': 1})
        )
    )
    service._activate = AsyncMock(return_value=True)
    service._complete_attempt = AsyncMock(return_value=True)
    toast_started = asyncio.Event()
    release_toast = asyncio.Event()

    async def slow_toast(**_kwargs) -> bool:
        toast_started.set()
        await release_toast.wait()
        return True

    monkeypatch.setattr(control_flow_module, 'mark_as_started_toast', slow_toast)

    outcome = await asyncio.wait_for(service.process_node(item), timeout=0.1)

    assert outcome is ControlFlowExecutionOutcome.COMPLETED
    await asyncio.wait_for(toast_started.wait(), timeout=0.1)
    toast_tasks = [
        task
        for task in service._runtime.tasks()
        if task.get_name() == 'control-flow-toast-dag-1'
    ]
    assert len(toast_tasks) == 1
    assert not toast_tasks[0].done()
    release_toast.set()
    await asyncio.gather(*toast_tasks)


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


async def build_routing_service(
    node_type: str,
) -> tuple[ControlFlowExecutionService, WorkInfo, dict[str, WorkState]]:
    service = build_service()
    definition = (
        BranchQueryDefinition(
            paths=[
                BranchPath(path_id='selected-path', target_node_ids=['selected']),
                BranchPath(path_id='inactive-path', target_node_ids=['inactive']),
            ]
        )
        if node_type == 'branch'
        else SwitchQueryDefinition(
            switch_field='$.metadata.document_type',
            cases={'invoice': ['selected'], 'contract': ['inactive']},
        )
    )
    plan = QueryPlan(
        nodes=[
            Query(
                task_id='route',
                query_str='route',
                node_type=QueryType.BRANCH
                if node_type == 'branch'
                else QueryType.SWITCH,
                definition=definition,
            ),
            Query(task_id='selected', query_str='selected', dependencies=['route']),
            Query(task_id='inactive', query_str='inactive', dependencies=['route']),
            Query(
                task_id='inactive-child', query_str='child', dependencies=['inactive']
            ),
            Query(
                task_id='merger',
                query_str='merger',
                dependencies=['selected', 'inactive-child'],
            ),
        ]
    )
    jobs = []
    for node in plan.nodes:
        item = work_item(node.task_id)
        item.state = WorkState.ACTIVE if node.task_id == 'route' else WorkState.CREATED
        item.start_after = None
        item.dependencies = node.dependencies
        jobs.append(item)
    item = jobs[0]
    item.data['metadata'] = {'on': f'{node_type}://default', 'document_type': 'invoice'}
    item.run_owner = 'scheduler-1'
    item.run_attempt_id = 'attempt-1'
    service.frontier = MemoryFrontier()
    await service.frontier.add_dag(plan, jobs)
    service.dag_service.active_dags = {'dag-1': plan}
    service.dag_service.get_dag.return_value = plan
    states = {job.id: job.state for job in jobs}

    async def commit_branch_route(**kwargs: Any) -> tuple[bool, set[str]]:
        skipped = set(kwargs['skipped_job_ids'])
        for job_id in skipped:
            states[job_id] = WorkState.SKIPPED
        states[kwargs['job_id']] = WorkState.COMPLETED
        return True, skipped

    service.repository.commit_branch_route.side_effect = commit_branch_route
    return service, item, states


@pytest.mark.parametrize('node_type', ['branch', 'switch'])
@pytest.mark.parametrize('failure', ['database', 'closure'])
@pytest.mark.parametrize('batch', [False, True])
async def test_routing_failure_blocks_children_until_retry(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    failure: str,
    batch: bool,
) -> None:
    service, item, states = await build_routing_service(node_type)
    commit = service.repository.commit_branch_route.side_effect
    with monkeypatch.context() as patch:
        error = RuntimeError(f'{failure} failed')
        if failure == 'closure':
            patch.setattr(
                control_flow_module, 'exclusive_skip_closure', Mock(side_effect=error)
            )
        elif failure == 'database':
            service.repository.commit_branch_route.side_effect = error

        outcomes = (
            await service.process_nodes([item])
            if batch
            else [await service.process_node(item)]
        )

    assert outcomes == [ControlFlowExecutionOutcome.FAILED]
    service.repository.complete_job.assert_not_awaited()
    assert states['route'] is WorkState.ACTIVE
    assert await service.frontier.select_ready(10) == []
    rebuilt = MemoryFrontier()
    plan = service.dag_service.active_dags['dag-1']
    terminal = {
        WorkState.COMPLETED,
        WorkState.FAILED,
        WorkState.CANCELLED,
        WorkState.SKIPPED,
    }
    # Hydration loads only schedulable rows and excludes terminal dependencies.
    pending = [
        job.model_copy(
            update={
                'state': states[job.id],
                'dependencies': [
                    dep for dep in job.dependencies or [] if states[dep] not in terminal
                ],
            },
            deep=True,
        )
        for job in service.frontier.jobs_by_id.values()
        if states[job.id] in (WorkState.CREATED, WorkState.RETRY)
    ]
    await rebuilt.add_dag(plan, pending)
    assert await rebuilt.select_ready(10) == []

    service.repository.commit_branch_route.side_effect = commit
    # A recovered run attempt re-enters the frontier as a retry.
    retry = item.model_copy(
        update={
            'state': WorkState.RETRY,
            'run_owner': None,
            'run_attempt_id': None,
        },
        deep=True,
    )
    states[item.id] = WorkState.RETRY
    rebuilt = MemoryFrontier()
    await rebuilt.add_dag(plan, [retry, *pending])
    service.repository.activate_from_lease.return_value = {'route': 'attempt-2'}
    service.frontier = rebuilt
    assert await service.process_node(retry) is ControlFlowExecutionOutcome.COMPLETED
    assert states['inactive'] is WorkState.SKIPPED
    assert states['inactive-child'] is WorkState.SKIPPED
    assert [job.id for job in await rebuilt.peek_ready(10)] == ['selected']
    await rebuilt.on_job_completed('selected')
    assert [job.id for job in await rebuilt.peek_ready(10)] == ['merger']


@pytest.mark.parametrize('node_type', ['branch', 'switch'])
async def test_routing_frontier_failure_keeps_durable_skips(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
) -> None:
    service, item, states = await build_routing_service(node_type)
    monkeypatch.setattr(
        service.frontier,
        'on_jobs_skipped',
        AsyncMock(side_effect=RuntimeError('frontier failed')),
    )

    assert (
        await service.process_node(item)
        is ControlFlowExecutionOutcome.COMPLETED_WITH_ERROR
    )
    assert states['route'] is WorkState.COMPLETED
    assert states['inactive'] is WorkState.SKIPPED
    assert states['inactive-child'] is WorkState.SKIPPED


@pytest.mark.parametrize('node_type', ['branch', 'switch'])
async def test_stale_routing_attempt_does_not_update_frontier_or_metadata(
    node_type: str,
) -> None:
    service, item, states = await build_routing_service(node_type)
    service.repository.commit_branch_route.side_effect = None
    service.repository.commit_branch_route.return_value = False, set()

    assert (
        await service.process_node(item)
        is ControlFlowExecutionOutcome.COMPLETION_REJECTED
    )
    assert states['route'] is WorkState.ACTIVE
    assert states['inactive'] is WorkState.CREATED
    service.repository.update_job_metadata.assert_not_awaited()
    assert await service.frontier.select_ready(10) == []


@pytest.mark.parametrize('failure', ['database', 'closure'])
async def test_branch_callback_blocks_children_on_skip_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    service, item, _ = await build_routing_service('branch')
    error = RuntimeError('skip failed')
    if failure == 'database':
        service.repository.mark_jobs_as_skipped.side_effect = error
    else:
        monkeypatch.setattr(
            control_flow_module, 'exclusive_skip_closure', Mock(side_effect=error)
        )

    with pytest.raises(RuntimeError, match='skip failed'):
        await service.handle_successful_job_completion(item.id, item)

    assert await service.frontier.select_ready(10) == []
