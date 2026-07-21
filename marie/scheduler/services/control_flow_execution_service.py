from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from marie.logging_core.logger import MarieLogger
from marie.messaging import mark_as_started as mark_as_started_toast
from marie.query_planner.base import QueryPlan
from marie.query_planner.branching import (
    BranchQueryDefinition,
    SkipReason,
    SwitchQueryDefinition,
)
from marie.query_planner.guardrail import (
    GuardrailQueryDefinition,
    GuardrailRouteMetadata,
)
from marie.scheduler.branch_evaluator import BranchEvaluationContext, BranchEvaluator
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import WorkInfo
from marie.scheduler.planner_util import get_node_from_dag
from marie.scheduler.repository import JobRepository
from marie.scheduler.services.dag_management_service import DAGManagementService
from marie.scheduler.state import WorkState
from marie.utils.scheduler_trace import scheduler_trace
from marie.utils.utils import current_milli_time


def exclusive_skip_closure(
    dag_plan: QueryPlan, skipped_node_ids: list[str]
) -> list[str]:
    """Return skipped nodes whose every incoming path is also skipped."""
    skipped = set(skipped_node_ids)
    parents: dict[str, set[str]] = {}
    dependents: dict[str, list[str]] = defaultdict(list)

    for node in dag_plan.nodes:
        parents[node.task_id] = set(node.dependencies)
        for dependency_id in node.dependencies:
            dependents[dependency_id].append(node.task_id)

    pending = deque(skipped_node_ids)
    while pending:
        skipped_node_id = pending.popleft()
        for child_id in dependents.get(skipped_node_id, []):
            if child_id in skipped:
                continue
            if parents[child_id].issubset(skipped):
                skipped.add(child_id)
                pending.append(child_id)

    return [node.task_id for node in dag_plan.nodes if node.task_id in skipped]


class ControlFlowExecutionService:
    """Apply scheduler-owned control-flow transitions."""

    def __init__(
        self,
        repository: JobRepository,
        frontier: MemoryFrontier,
        dag_service: DAGManagementService,
        status_update_lock: AsyncJobLock,
        topology_cache: DagTopologyCache,
        job_cache: dict[str, WorkInfo],
        lease_owner: str,
        run_ttl_seconds: int,
        gateway_instance_id: str,
        notify_callback: Callable[[], Awaitable[bool]],
    ) -> None:
        self.logger = MarieLogger(ControlFlowExecutionService.__name__)
        self.repository = repository
        self.frontier = frontier
        self.dag_service = dag_service
        self._status_update_lock = status_update_lock
        self._topology_cache = topology_cache
        self._job_cache = job_cache
        self._lease_owner = lease_owner
        self._run_ttl_seconds = run_ttl_seconds
        self._gateway_instance_id = gateway_instance_id
        self._notify_callback = notify_callback
        self._branch_evaluator = BranchEvaluator()

    @staticmethod
    def _is_branch_node(node: Any) -> bool:
        return bool(
            node
            and hasattr(node, 'definition')
            and isinstance(
                node.definition, (BranchQueryDefinition, SwitchQueryDefinition)
            )
        )

    @staticmethod
    def _is_guardrail_node(node: Any) -> bool:
        return bool(
            node
            and hasattr(node, 'definition')
            and isinstance(node.definition, GuardrailQueryDefinition)
        )

    async def commit_guardrail_route_if_needed(
        self,
        job_id: str,
        work_item: WorkInfo,
        *,
        run_owner: str,
        run_attempt_id: str,
    ) -> tuple[bool, set[str], str | None] | None:
        """Commit an executable guardrail decision when the job is a guardrail."""
        dag_plan = await self.dag_service.get_dag(work_item.dag_id)
        if not dag_plan:
            return None

        node = get_node_from_dag(job_id, dag_plan)
        if not self._is_guardrail_node(node):
            return None

        report_decision = await self.repository.get_guardrail_report_decision(
            job_id=job_id,
            run_attempt_id=run_attempt_id,
        )
        if report_decision is None:
            raise ValueError(
                f"Guardrail attempt {run_attempt_id} produced no report asset"
            )

        selected_path_id = 'pass' if report_decision['outcome'] == 'VALID' else 'fail'
        route = GuardrailRouteMetadata(
            outcome=report_decision['outcome'],
            selected_path_ids=[selected_path_id],
            report_asset=report_decision['report_asset'],
            evaluated_at=report_decision['evaluated_at'],
        )
        route_metadata = route.model_dump(mode='json', by_alias=True)
        paths = {path.path_id: path for path in node.definition.paths}
        if selected_path_id not in paths:
            raise ValueError(
                f"Guardrail selected unknown path '{selected_path_id}' for job {job_id}"
            )

        selected_targets = set(paths[selected_path_id].target_node_ids)
        unselected_targets = {
            target_id
            for path_id, path in paths.items()
            if path_id != selected_path_id
            for target_id in path.target_node_ids
        }
        overlap = selected_targets & unselected_targets
        if overlap:
            raise ValueError(
                f"Guardrail paths overlap for job {job_id}: {sorted(overlap)}"
            )
        skipped_job_ids = exclusive_skip_closure(dag_plan, list(unselected_targets))

        async with self._status_update_lock[job_id]:
            committed, skipped_ids, reject_reason = (
                await self.repository.commit_guardrail_route(
                    job_id=job_id,
                    queue_name=work_item.name,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    branch_metadata=route_metadata,
                    skipped_job_ids=skipped_job_ids,
                )
            )

        if committed:
            work_item.branch_metadata = route_metadata
            work_item.state = WorkState.COMPLETED
            self._job_cache[job_id] = work_item
            await self.frontier.on_job_completed_with_skips(job_id, skipped_ids)
        return committed, skipped_ids, reject_reason

    async def handle_successful_job_completion(
        self, job_id: str, work_item: WorkInfo
    ) -> None:
        """Advance the frontier and apply a completed branch decision."""
        await self.frontier.on_job_completed(job_id)

        dag_plan = await self.dag_service.get_dag(work_item.dag_id)
        if not dag_plan:
            return

        node = get_node_from_dag(job_id, dag_plan)
        if node and self._is_branch_node(node):
            self.logger.info(
                f"Completed branch node detected: {job_id}. Evaluating paths..."
            )
            await self._evaluate_and_mark_branch_paths(job_id, work_item, dag_plan)

    async def process_node(self, work_item: WorkInfo) -> None:
        """Execute a scheduler-local control-flow node."""
        try:
            dag_id = work_item.dag_id
            entrypoint = work_item.data.get('metadata', {}).get('on', '')
            node_type = entrypoint.split('://', 1)[0].lower()
            scheduler_trace(
                'control_flow_started',
                job_id=work_item.id,
                dag_id=dag_id,
                node_type=node_type,
                job_name=work_item.name,
                job_level=work_item.job_level,
            )

            self.logger.debug(
                f"[CONTROL_FLOW] Processing {node_type} node: "
                f"{work_item.id} in DAG {dag_id}"
            )

            if dag_id not in self.dag_service.active_dags:
                dag = await self.dag_service.get_dag(dag_id)
                if not dag:
                    self.logger.error(
                        f"[CONTROL_FLOW] Missing DAG {dag_id} for "
                        f"{node_type} node {work_item.id}"
                    )
                    await self._release_lease(work_item.id)
                    return

                admitted = await self.dag_service.admit_dag(
                    dag_id, dag, source=f"control_flow:{node_type}"
                )
                if not admitted:
                    await self._release_lease(work_item.id)
                    return

            if not await self._activate(work_item):
                await self._release_lease(work_item.id)
                return

            _, job_levels = self._topology_cache.get_sorted_nodes_and_levels(
                self.dag_service.active_dags[dag_id], dag_id
            )

            if work_item.job_level == max(job_levels.values()):
                metadata = work_item.data.get('metadata', {})
                await mark_as_started_toast(
                    api_key=work_item.data.get('api_key'),
                    job_id=dag_id,
                    event_name=work_item.data.get('name', work_item.name),
                    job_tag=metadata.get('ref_type'),
                    status='OK',
                    timestamp=current_milli_time(),
                    payload=metadata,
                )

            if not await self._complete_attempt(work_item):
                return

            if node_type in ('branch', 'switch'):
                await self._evaluate_and_mark_branch_paths(
                    work_item.id,
                    work_item,
                    self.dag_service.active_dags[dag_id],
                )
            elif node_type not in ('noop', 'merger'):
                self.logger.warning(
                    f"[CONTROL_FLOW] Unknown control flow type: "
                    f"{node_type} for {work_item.id}"
                )

            self.frontier.leased_until.pop(work_item.id, None)
            await self.frontier.on_job_completed(work_item.id)
            await self._notify_callback()

            if job_levels.get(work_item.id, -1) == min(job_levels.values()):
                await self.dag_service.resolve_dag_status_with_retry(
                    work_item.id,
                    work_item,
                    source='control_flow',
                )

            scheduler_trace(
                'control_flow_completed',
                job_id=work_item.id,
                dag_id=dag_id,
                node_type=node_type,
                job_name=work_item.name,
                job_level=work_item.job_level,
            )
        except Exception as error:
            scheduler_trace(
                'control_flow_failed',
                job_id=work_item.id,
                dag_id=work_item.dag_id,
                job_name=work_item.name,
                error=repr(error),
            )
            self.logger.error(
                f"[CONTROL_FLOW] Error processing control flow node "
                f"{work_item.id}: {error}",
                exc_info=True,
            )
            try:
                await self._release_lease(work_item.id)
            except Exception as cleanup_error:
                self.logger.error(
                    f"[CONTROL_FLOW] Error during cleanup for "
                    f"{work_item.id}: {cleanup_error}"
                )

    async def _activate(self, work_item: WorkInfo) -> bool:
        if (
            work_item.state == WorkState.ACTIVE
            and work_item.run_owner == self._lease_owner
            and work_item.run_attempt_id
        ):
            self._job_cache[work_item.id] = work_item
            await self.frontier.update_job_state(work_item.id, WorkState.ACTIVE)
            return True

        activated = await self.repository.activate_from_lease(
            job_ids=[work_item.id],
            owner=self._lease_owner,
            run_ttl_seconds=self._run_ttl_seconds,
            gateway_instance_id=self._gateway_instance_id,
        )
        if work_item.id not in activated:
            self.logger.error(
                f"[CONTROL_FLOW] Failed to mark control flow node "
                f"{work_item.id} active"
            )
            return False

        work_item.run_owner = self._lease_owner
        work_item.run_attempt_id = activated[work_item.id]
        work_item.state = WorkState.ACTIVE
        self._job_cache[work_item.id] = work_item
        await self.frontier.update_job_state(work_item.id, WorkState.ACTIVE)
        return True

    async def _complete_attempt(self, work_item: WorkInfo) -> bool:
        if not work_item.run_owner or not work_item.run_attempt_id:
            self.logger.error(
                f"[CONTROL_FLOW] Missing run attempt for control flow node "
                f"{work_item.id}"
            )
            scheduler_trace(
                'control_flow_terminal_rejected',
                job_id=work_item.id,
                dag_id=work_item.dag_id,
                reason='missing_attempt',
            )
            return False

        async with self._status_update_lock[work_item.id]:
            completed = await self.repository.complete_job(
                job_id=work_item.id,
                queue_name=work_item.name,
                output_metadata={},
                run_owner=work_item.run_owner,
                run_attempt_id=work_item.run_attempt_id,
            )
        if completed:
            return True

        self.logger.warning(
            f"[CONTROL_FLOW] Terminal update rejected for control flow node "
            f"{work_item.id} (run_owner={work_item.run_owner}, "
            f"run_attempt_id={work_item.run_attempt_id})"
        )
        scheduler_trace(
            'control_flow_terminal_rejected',
            job_id=work_item.id,
            dag_id=work_item.dag_id,
            reason='attempt_mismatch',
            run_owner=work_item.run_owner,
            run_attempt_id=work_item.run_attempt_id,
        )
        return False

    async def _release_lease(self, job_id: str) -> None:
        await self.repository.release_lease(job_ids=[job_id])
        await self.frontier.release_lease_local(job_id)

    async def _evaluate_and_mark_branch_paths(
        self,
        branch_node_id: str,
        work_item: WorkInfo,
        dag_plan: QueryPlan,
    ) -> None:
        try:
            self.logger.info(f"Evaluating branch paths for node: {branch_node_id}")
            branch_node = get_node_from_dag(branch_node_id, dag_plan)
            if not branch_node or not self._is_branch_node(branch_node):
                self.logger.warning(
                    f"Node {branch_node_id} is not a branch node, "
                    'skipping evaluation'
                )
                return

            branch_definition = branch_node.definition
            context = BranchEvaluationContext(
                work_info=work_item,
                dag_plan=dag_plan,
                branch_node=branch_node,
                execution_results={},
            )

            active_path_ids: list[str] = []
            branch_metadata: dict[str, Any]
            if isinstance(branch_definition, BranchQueryDefinition):
                active_path_ids = await self._branch_evaluator.evaluate_branch(
                    branch_definition, context
                )
                evaluation_mode = branch_definition.evaluation_mode
                branch_metadata = {
                    'node_type': 'BRANCH',
                    'selected_path_ids': active_path_ids,
                    'evaluation_mode': (
                        evaluation_mode.value
                        if hasattr(evaluation_mode, 'value')
                        else evaluation_mode
                    ),
                    'default_path_id': branch_definition.default_path_id,
                    'all_paths': [path.path_id for path in branch_definition.paths],
                    'evaluated_at': datetime.now(timezone.utc).isoformat(),
                }
            elif isinstance(branch_definition, SwitchQueryDefinition):
                active_path_ids = (
                    await self._branch_evaluator.evaluate_switch(
                        branch_definition, context
                    )
                    or []
                )
                switch_value = self._branch_evaluator.jsonpath_evaluator.evaluate(
                    branch_definition.switch_field, context.context
                )
                branch_metadata = {
                    'node_type': 'SWITCH',
                    'switch_field': branch_definition.switch_field,
                    'switch_value': switch_value,
                    'selected_case': active_path_ids,
                    'all_cases': list(branch_definition.cases.keys()),
                    'evaluated_at': datetime.now(timezone.utc).isoformat(),
                }
            else:
                return

            await self._update_job_branch_metadata(
                job_id=branch_node_id,
                queue_name=work_item.name,
                branch_metadata=branch_metadata,
            )

            all_target_nodes: set[str] = set()
            active_target_nodes: set[str] = set()
            path_to_nodes: dict[str, list[str]] = {}

            if isinstance(branch_definition, BranchQueryDefinition):
                for path in branch_definition.paths:
                    path_to_nodes[path.path_id] = path.target_node_ids
                    all_target_nodes.update(path.target_node_ids)
                    if path.path_id in active_path_ids:
                        active_target_nodes.update(path.target_node_ids)
            else:
                for case_value, node_ids in branch_definition.cases.items():
                    path_to_nodes[str(case_value)] = node_ids
                    all_target_nodes.update(node_ids)
                active_target_nodes.update(active_path_ids)
                if branch_definition.default_case:
                    path_to_nodes['default'] = branch_definition.default_case
                    all_target_nodes.update(branch_definition.default_case)

            if active_target_nodes:
                await self._mark_selected_nodes(
                    branch_node_id,
                    work_item.name,
                    branch_definition,
                    active_path_ids,
                    active_target_nodes,
                    path_to_nodes,
                )

            skipped_target_nodes = all_target_nodes - active_target_nodes
            if skipped_target_nodes:
                skip_reason = SkipReason(
                    branch_node_id=branch_node_id,
                    reason=(
                        'Branch condition not met. ' f"Active paths: {active_path_ids}"
                    ),
                    evaluated_condition={'active_paths': active_path_ids},
                    selected_paths=active_path_ids,
                    timestamp=datetime.now(timezone.utc),
                )
                await self._mark_nodes_skipped(
                    list(skipped_target_nodes),
                    work_item.name,
                    skip_reason,
                    dag_plan,
                )
        except Exception as error:
            self.logger.error(
                f"Error evaluating branch paths for {branch_node_id}: {error}",
                exc_info=True,
            )

    async def _mark_selected_nodes(
        self,
        branch_node_id: str,
        queue_name: str,
        branch_definition: BranchQueryDefinition | SwitchQueryDefinition,
        active_path_ids: list[str],
        active_target_nodes: set[str],
        path_to_nodes: dict[str, list[str]],
    ) -> None:
        selected_at = datetime.now(timezone.utc).isoformat()
        if isinstance(branch_definition, BranchQueryDefinition):
            for path_id in active_path_ids:
                for node_id in path_to_nodes.get(path_id, []):
                    await self._update_job_branch_metadata(
                        job_id=node_id,
                        queue_name=queue_name,
                        branch_metadata={
                            'selected_by_branch': branch_node_id,
                            'selected_path_id': path_id,
                            'selected_at': selected_at,
                        },
                    )
            return

        for node_id in active_target_nodes:
            selected_case = next(
                (
                    str(case_value)
                    for case_value, node_ids in branch_definition.cases.items()
                    if node_id in node_ids
                ),
                None,
            )
            if (
                selected_case is None
                and branch_definition.default_case
                and node_id in branch_definition.default_case
            ):
                selected_case = 'default'
            await self._update_job_branch_metadata(
                job_id=node_id,
                queue_name=queue_name,
                branch_metadata={
                    'selected_by_switch': branch_node_id,
                    'selected_case': selected_case,
                    'selected_at': selected_at,
                },
            )

    async def _update_job_branch_metadata(
        self,
        job_id: str,
        queue_name: str,
        branch_metadata: dict[str, Any],
    ) -> None:
        try:
            await self.repository.update_job_metadata(
                job_id=job_id,
                queue_name=queue_name,
                metadata_updates={'branch_metadata': branch_metadata},
            )
        except Exception as error:
            self.logger.error(
                f"Error updating branch_metadata for job {job_id}: {error}",
                exc_info=True,
            )

    async def _mark_nodes_skipped(
        self,
        node_ids: list[str],
        queue_name: str,
        skip_reason: SkipReason,
        dag_plan: QueryPlan,
    ) -> None:
        if not node_ids:
            return

        try:
            skipped_node_ids = exclusive_skip_closure(dag_plan, node_ids)
            skipped_ids = await self.repository.mark_jobs_as_skipped(
                job_ids=skipped_node_ids,
                queue_name=queue_name,
                output_metadata={
                    'skip_reason': skip_reason.model_dump(mode='json'),
                    'skipped_at': skip_reason.timestamp.isoformat(),
                },
            )
            committed_node_ids = [
                node_id for node_id in skipped_node_ids if node_id in skipped_ids
            ]
            if not committed_node_ids:
                return

            for node_id in committed_node_ids:
                await self._update_job_branch_metadata(
                    job_id=node_id,
                    queue_name=queue_name,
                    branch_metadata={
                        'skip_reason': {
                            'branch_node_id': skip_reason.branch_node_id,
                            'reason': skip_reason.reason,
                            'selected_paths': skip_reason.selected_paths,
                            'evaluated_condition': skip_reason.evaluated_condition,
                            'timestamp': skip_reason.timestamp.isoformat(),
                        },
                        'skipped': True,
                    },
                )

            await self.frontier.on_jobs_skipped(committed_node_ids)
        except Exception as error:
            self.logger.error(f"Error marking nodes as skipped: {error}", exc_info=True)
