import asyncio
import socket
import time
import traceback
import uuid
import uuid as _uuid
from asyncio import Queue
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from math import inf
from typing import Any, Dict, List

import psycopg2

from marie.excepts import BadConfigSource, RuntimeFailToStart
from marie.helper import get_or_reuse_loop
from marie.job.common import JobStatus
from marie.job.job_manager import JobManager
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.messaging import mark_as_complete as mark_as_complete_toast
from marie.messaging import mark_as_started as mark_as_started_toast
from marie.query_planner.base import (
    QueryPlan,
    QueryType,
)
from marie.query_planner.branching import (
    BranchQueryDefinition,
    SkipReason,
    SwitchQueryDefinition,
)
from marie.query_planner.builtin import register_all_known_planners
from marie.query_planner.model import QueryPlannersConf
from marie.scheduler.branch_evaluator import BranchEvaluationContext, BranchEvaluator
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.fixtures import *
from marie.scheduler.global_execution_planner import GlobalPriorityExecutionPlanner
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.job_scheduler import JobScheduler, JobSubmissionRequest
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import ExistingWorkPolicy, HeartbeatConfig, WorkInfo
from marie.scheduler.planner_util import (
    _is_branch_query_definition,
    _is_noop_query_definition,
    debug_candidates_and_plan,
    get_node_from_dag,
    query_plan_work_items,
)
from marie.scheduler.repository import JobRepository
from marie.scheduler.scheduler_heartbeat import SchedulerHeartbeat
from marie.scheduler.scheduler_repository import SchedulerRepository
from marie.scheduler.services import (
    DAGManagementService,
    MaintenanceService,
    NotificationService,
)
from marie.scheduler.state import WorkState
from marie.scheduler.util import (
    adjust_backoff,
    available_slots_by_executor,
    convert_job_status_to_work_state,
)
from marie.serve.runtimes.servers.cluster_state import ClusterState
from marie.state.semaphore_store import SemaphoreStore
from marie.state.slot_capacity_manager import SlotCapacityManager
from marie.storage.database.postgres import PostgresqlMixin

INIT_POLL_PERIOD = 2.250  # 250ms
SHORT_POLL_INTERVAL = 1.0  # seconds, when slots exist but no work

MIN_POLL_PERIOD = 0.250
MAX_POLL_PERIOD = 8

MONITORING_POLL_PERIOD = 5.0  # 5s
SYNC_POLL_PERIOD = 5.0  # 5s

DEFAULT_SCHEMA = "marie_scheduler"
DEFAULT_JOB_TABLE = "job"


# FIXME : Today we are tracking at the executor level, however that might not be the best
# approach. We might want to track at the deployment level (endpoint level) instead.
# this will allow us to track the status of the deployment and not just the executor.


class PostgreSQLJobScheduler(PostgresqlMixin, JobScheduler):
    _mapper_warnings_shown = set()
    """A PostgreSQL-based job scheduler."""

    def __init__(
        self,
        config: Dict[str, Any],
        job_manager: JobManager,
        gateway_ready_event: asyncio.Event = None,
    ):
        super().__init__()
        self.logger = MarieLogger(PostgreSQLJobScheduler.__name__)
        if job_manager is None:
            raise BadConfigSource("job_manager argument is required for JobScheduler")

        self._gateway_ready_event = gateway_ready_event

        self.validate_config(config)
        self.config = config  # Store config for listener setup
        self._fetch_event = asyncio.Event()
        self._fetch_counter = 0
        self._debounced_notify = False

        self.known_queues = set(config.get("queue_names", []))
        self.running = False
        self._poll_task = None
        self._producer_task = None
        self._consumer_task = None
        self._heartbeat_task = None
        self.sync_task = None
        self.monitoring_task = None
        self._worker_tasks = None
        self._sync_dag_task = None
        self._cluster_state_monitor_task = None
        self._dag_state_listener_task = None
        self._listen_connection = None
        self._submission_count = 0
        self._pending_requests = {}  # Track pending requests by ID
        self._request_queue = Queue()  # Buffer up to 1000 requests

        self.scheduler_mode = config.get(
            "scheduler_mode", "parallel"
        )  # "serial" or "parallel"
        self.scheduler_mode = "serial"
        self.distributed_scheduler = config.get("distributed_scheduler", False)

        self._event_queue = Queue()
        self._status_update_lock = AsyncJobLock()

        self.max_workers = config.get("max_workers", 5)
        self._db_executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="db-executor"
        )
        self.logger.info(
            f"Using ThreadPoolExecutor for database operations with : {self.max_workers} workers."
        )
        if self.known_queues is None or len(self.known_queues) == 0:
            raise BadConfigSource("Queue names are required for JobScheduler")
        self.logger.info(f"Queue names to monitor: {self.known_queues}")

        self.job_manager = job_manager
        self._loop = get_or_reuse_loop()
        self._setup_event_subscriptions()
        self._setup_storage(config, connection_only=True)
        self._db = SchedulerRepository(config)

        self.repository = JobRepository(config, max_workers=self.max_workers)
        self.notification_service = NotificationService(config)

        # Initialize scheduler state (frontier and active_dags)
        self.frontier = MemoryFrontier()
        self.active_dags = {}

        # Initialize DAGManagementService for DAG lifecycle management
        # Service operates on scheduler's frontier and active_dags
        self.dag_service = DAGManagementService(
            repository=self.repository,
            frontier=self.frontier,
            active_dags=self.active_dags,
            loop=self._loop,
            executor=self._db_executor,
            notify_callback=self.notify_event,
        )

        # Register handler for DAG state changes (delegate to DAGManagementService)
        self.notification_service.register_handler(
            channel='dag_state_changed', handler=self.dag_service.handle_state_change
        )

        # Initialize MaintenanceService for periodic cleanup tasks
        maintenance_interval = config.get("maintenance_interval", 60)  # Default: 60s
        self.maintenance_service = MaintenanceService(
            repository=self.repository,
            loop=self._loop,
            executor=self._db_executor,
            notify_callback=self.notify_event,
            maintenance_interval=maintenance_interval,
        )

        self.execution_planner = GlobalPriorityExecutionPlanner()
        register_all_known_planners(
            QueryPlannersConf.from_dict(config.get("query_planners", {}))
        )

        # Initialize BranchEvaluator for conditional branching
        self.branch_evaluator = BranchEvaluator()

        dag_config = config.get("dag_manager", {})
        min_concurrent_dags = int(dag_config.get("min_concurrent_dags", 1))
        max_concurrent_dags = int(dag_config.get("max_concurrent_dags", 16))
        cache_ttl_seconds = int(dag_config.get("cache_ttl_seconds", 5))
        cache_ttl_seconds = int(dag_config.get("cache_ttl_seconds", 5))

        dag_cache_size = int(
            dag_config.get("dag_cache_size", 5000)
        )  # 5000 entries as this is what our fetch_next_job uses
        self._topology_cache = DagTopologyCache(maxsize=dag_cache_size)

        heartbeat_config_dict = config.get("heartbeat", {})
        self.heartbeat_config = HeartbeatConfig.from_dict(heartbeat_config_dict)
        self.logger.info(f"Heartbeat configuration: {self.heartbeat_config}")
        self.heartbeat = SchedulerHeartbeat(
            self, self.heartbeat_config, self._db, self.logger
        )

        self.max_concurrent_dags = max_concurrent_dags
        self._start_time = datetime.now(timezone.utc)

        self.frontier_batch_size = int(dag_config.get("frontier_batch_size", 1000))
        self.lease_ttl_seconds: int = int(config.get("lease_ttl_seconds", 5))
        self.run_ttl_seconds: int = int(config.get("run_ttl_seconds", 60))
        # unique, stable lease owner for this scheduler instance
        self.lease_owner: str = f"{socket.gethostname()}:{_uuid.uuid4()}"
        self.logger.info(
            f"Lease config: lease_ttl_seconds={self.lease_ttl_seconds}, "
            f"run_ttl_seconds={self.run_ttl_seconds}, owner='{self.lease_owner}'"
        )

        self._job_cache = {}
        self._job_cache_max_size = 5000

        # Semaphore-based capacity control, we hijaced the _etcd_client client here from job manager
        self._semaphore_store = SemaphoreStore(
            self.job_manager._etcd_client, default_lease_ttl=30
        )
        self._sem_default_ttl = 30
        self._sem_owner_prefix = f"{socket.gethostname()}"
        self._sem_owner_prefix = f""

        self.capacity_manager = SlotCapacityManager(
            semaphore_store=self._semaphore_store,
            logger=self.logger,
            # Optional mapping if slot types differ from executor names:
            # slot_type_resolver=lambda executor: {"extract_executor": "ocr.gpu"}.get(executor, executor),
        )
        self.cycle_log_every = 10

    def validate_config(self, config: Dict[str, Any]):
        # TODO :Implement full validation of required fields
        required_keys = ["queue_names"]
        for key in required_keys:
            if key not in config:
                raise BadConfigSource(f"Missing required config: {key}")

    async def handle_job_event(self, event_type: str, message: Any):
        """
        Handles a job event.

        :param event_type: The type of the event.
        :param message: The message associated with the event.
        """

        self.logger.debug(f"received message: {event_type} > {message}")

        if not isinstance(message, dict) or "job_id" not in message:
            self.logger.error(f"Invalid message format: {message}")
            return

        job_id = message.get("job_id")
        try:
            status = JobStatus(event_type)
            work_item: Optional[WorkInfo] = await self.get_job(job_id)

            if work_item is None:
                self.logger.error(f"WorkItem not found: {job_id}")
                raise ValueError(f"WorkItem not found: {job_id}")

            now = datetime.now(timezone.utc)
            work_state = convert_job_status_to_work_state(status)
            work_item.state = work_state

            # update storage cache and memory cache
            self._job_cache[job_id] = work_item
            await self.frontier.update_job_state(job_id, work_state)

            if status == JobStatus.PENDING:
                self.logger.debug(f"Job pending : {job_id}")
            elif status == JobStatus.SUCCEEDED:
                await self.complete(job_id, work_item)
                await self.frontier.on_job_completed(job_id)

                # Check if this is a branch node and evaluate paths if so
                dag_plan = await self.get_dag_by_id(work_item.dag_id)
                if dag_plan:
                    node = get_node_from_dag(job_id, dag_plan)
                    if node and self._is_branch_node(node):
                        self.logger.info(
                            f"Completed branch node detected: {job_id}. Evaluating paths..."
                        )
                        await self._evaluate_and_mark_branch_paths(
                            job_id, work_item, dag_plan
                        )
            elif status == JobStatus.FAILED:
                await self.fail(job_id, work_item)
                await self.frontier.on_job_failed(job_id)
            elif status == JobStatus.RUNNING:
                self.logger.debug(f"Job running : {job_id}")
                await self.put_status(job_id, work_state, now, None)
            else:
                self.logger.error(f"Unhandled job status: {status}. Marking as FAILED.")
                await self.fail(job_id, work_item)  # Fail-safe
                await self.frontier.on_job_failed(job_id)

            if status.is_terminal():
                self.logger.debug(
                    f"Job is in terminal state {status}, job_id: {job_id}"
                )

                self._status_update_lock.release(job_id)
                await self.resolve_dag_status(job_id, work_item, now, now)
                await self.notify_event()
        except Exception as e:
            self.logger.error(
                f"Error handling job event {event_type} for job {job_id}: {e}"
            )

    def _is_branch_node(self, node) -> bool:
        """Check if a node is a BRANCH or SWITCH node."""
        if not node or not hasattr(node, 'definition'):
            return False

        # Check if it's a BRANCH or SWITCH query type
        return isinstance(
            node.definition, (BranchQueryDefinition, SwitchQueryDefinition)
        )

    async def _process_control_flow_node(self, wi: WorkInfo) -> None:
        """
        Process a control flow node (NOOP, BRANCH, SWITCH, or MERGER).
        These nodes don't execute on executors - they're completed locally.

        :param wi: WorkInfo for the control flow node
        """
        try:
            dag_id = wi.dag_id
            ep = wi.data.get("metadata", {}).get("on", "")
            node_type = ep.split("://", 1)[0].lower()

            self.logger.info(
                f"[CONTROL_FLOW] Processing {node_type} node: {wi.id} in DAG {dag_id}"
            )

            # Ensure DAG is in active_dags
            if dag_id not in self.active_dags:
                dag = await self.get_dag_by_id(dag_id)
                if not dag:
                    self.logger.error(
                        f"[CONTROL_FLOW] Missing DAG {dag_id} for {node_type} node {wi.id}"
                    )
                    await self._release_lease_db([wi.id])
                    await self.frontier.release_lease_local(wi.id)
                    return

                await self.mark_as_active_dag(wi)
                self.active_dags[dag_id] = dag

            # Get the node from the DAG
            node = get_node_from_dag(wi.id, self.active_dags[dag_id])

            # Get job levels for root/leaf detection
            sorted_nodes, job_levels = self._topology_cache.get_sorted_nodes_and_levels(
                self.active_dags[dag_id], dag_id
            )

            # Check if this is a root node (emit DAG start event)
            is_root = wi.job_level == max(job_levels.values())
            if is_root:
                event_name = wi.data.get("name", wi.name)
                api_key = wi.data.get("api_key", None)
                metadata = wi.data.get("metadata", {})
                ref_type = metadata.get("ref_type")

                await mark_as_started_toast(
                    api_key=api_key,
                    job_id=wi.dag_id,
                    event_name=event_name,
                    job_tag=ref_type,
                    status="OK",
                    timestamp=int(time.time()),
                    payload=metadata,
                )

            # Handle based on node type
            if node_type in ("branch", "switch"):
                # BRANCH/SWITCH nodes need evaluation
                self.logger.info(
                    f"[CONTROL_FLOW] Evaluating {node_type} paths for {wi.id}"
                )

                # Complete the branch node first
                await self.complete(wi.id, wi, {}, force=True)

                # Evaluate and mark paths
                await self._evaluate_and_mark_branch_paths(
                    wi.id, wi, self.active_dags[dag_id]
                )

            elif node_type == "noop":
                # NOOP nodes just complete
                self.logger.info(f"[CONTROL_FLOW] Completing NOOP node {wi.id}")
                await self.complete(wi.id, wi, {}, force=True)

            elif node_type == "merger":
                # MERGER nodes wait for branches to complete via dependencies
                # The actual merge logic is handled by the dependency system
                # MERGER can complete immediately - dependencies prevent it from
                # running until all required branches are done
                self.logger.info(
                    f"[CONTROL_FLOW] Completing MERGER node {wi.id} "
                    "(merge logic handled by dependencies)"
                )
                await self.complete(wi.id, wi, {}, force=True)

            else:
                self.logger.warning(
                    f"[CONTROL_FLOW] Unknown control flow type: {node_type} for {wi.id}"
                )
                await self.complete(wi.id, wi, {}, force=True)

            # Clean up
            self.frontier.leased_until.pop(wi.id, None)
            await self.frontier.on_job_completed(wi.id)

            # Check if DAG is complete (leaf node check)
            if job_levels.get(wi.id, -1) == min(job_levels.values()):
                await self.resolve_dag_status(wi.id, wi)

            self.logger.info(
                f"[CONTROL_FLOW] Successfully processed {node_type} node {wi.id}"
            )

        except Exception as e:
            self.logger.error(
                f"[CONTROL_FLOW] Error processing control flow node {wi.id}: {e}",
                exc_info=True,
            )
            # Release leases on error
            try:
                await self._release_lease_db([wi.id])
                await self.frontier.release_lease_local(wi.id)
            except Exception as cleanup_error:
                self.logger.error(
                    f"[CONTROL_FLOW] Error during cleanup for {wi.id}: {cleanup_error}"
                )

    async def _evaluate_and_mark_branch_paths(
        self, branch_node_id: str, work_item: WorkInfo, dag_plan: QueryPlan
    ) -> None:
        """
        Evaluate a branch node and mark its child paths as READY or SKIPPED.
        Stores branch_metadata for tracking and debugging.

        :param branch_node_id: ID of the completed branch node
        :param work_item: WorkInfo of the branch node
        :param dag_plan: The DAG plan containing the branch
        """
        try:
            self.logger.info(f"Evaluating branch paths for node: {branch_node_id}")

            # Get the branch node from the DAG
            branch_node = get_node_from_dag(branch_node_id, dag_plan)
            if not branch_node or not self._is_branch_node(branch_node):
                self.logger.warning(
                    f"Node {branch_node_id} is not a branch node, skipping evaluation"
                )
                return

            branch_def = branch_node.definition

            # Build evaluation context
            # TODO: Gather execution results from previous nodes if needed
            execution_results = {}
            context = BranchEvaluationContext(
                work_info=work_item,
                dag_plan=dag_plan,
                branch_node=branch_node,
                execution_results=execution_results,
            )

            # Evaluate the branch to determine active paths
            active_path_ids = []
            branch_metadata = {}

            if isinstance(branch_def, BranchQueryDefinition):
                active_path_ids = await self.branch_evaluator.evaluate_branch(
                    branch_def, context
                )
                # Store BRANCH metadata for tracking
                branch_metadata = {
                    "node_type": "BRANCH",
                    "selected_path_ids": active_path_ids,
                    "evaluation_mode": (
                        branch_def.evaluation_mode.value
                        if hasattr(branch_def.evaluation_mode, 'value')
                        else branch_def.evaluation_mode
                    ),
                    "default_path_id": branch_def.default_path_id,
                    "all_paths": [p.path_id for p in branch_def.paths],
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                }
                self.logger.info(
                    f"BRANCH evaluation: selected_path_ids={active_path_ids}, "
                    f"evaluation_mode={branch_metadata['evaluation_mode']}"
                )

            elif isinstance(branch_def, SwitchQueryDefinition):
                active_path_ids = await self.branch_evaluator.evaluate_switch(
                    branch_def, context
                )
                # Get the evaluated switch value
                switch_value = self.branch_evaluator.jsonpath_evaluator.evaluate(
                    branch_def.switch_field, context.context
                )
                # Store SWITCH metadata for tracking
                branch_metadata = {
                    "node_type": "SWITCH",
                    "switch_field": branch_def.switch_field,
                    "switch_value": switch_value,
                    "selected_case": active_path_ids,
                    "all_cases": list(branch_def.cases.keys()),
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                }
                self.logger.info(
                    f"SWITCH evaluation: switch_value={switch_value}, "
                    f"selected_case={active_path_ids}"
                )
            else:
                self.logger.error(f"Unknown branch definition type: {type(branch_def)}")
                return

            # Store branch_metadata on the BRANCH/SWITCH node itself
            await self._update_job_branch_metadata(
                job_id=branch_node_id,
                queue_name=work_item.name,
                branch_metadata=branch_metadata,
            )

            self.logger.info(
                f"Branch evaluation complete. Active paths: {active_path_ids}"
            )

            # Mark active paths' target nodes as READY
            # Mark inactive paths' target nodes as SKIPPED
            all_target_nodes = set()
            active_target_nodes = set()
            path_to_nodes = {}  # Track which path leads to which nodes

            # Collect all target nodes based on branch type
            if isinstance(branch_def, BranchQueryDefinition):
                # BRANCH nodes have paths
                for path in branch_def.paths:
                    path_to_nodes[path.path_id] = path.target_node_ids
                    all_target_nodes.update(path.target_node_ids)
                    if path.path_id in active_path_ids:
                        active_target_nodes.update(path.target_node_ids)

            elif isinstance(branch_def, SwitchQueryDefinition):
                # SWITCH nodes have cases (Dict[value, List[node_ids]])
                for case_value, node_ids in branch_def.cases.items():
                    path_to_nodes[str(case_value)] = node_ids
                    all_target_nodes.update(node_ids)

                # Check if active_path_ids contains the selected nodes
                # For SWITCH, active_path_ids is a list of node IDs to activate
                if active_path_ids:
                    active_target_nodes.update(active_path_ids)

                # Add default case nodes to all targets
                if branch_def.default_case:
                    path_to_nodes['default'] = branch_def.default_case
                    all_target_nodes.update(branch_def.default_case)

            # Nodes to skip are all targets minus active targets
            skipped_target_nodes = all_target_nodes - active_target_nodes

            # Mark active nodes as READY and store branch_metadata
            if active_target_nodes:
                # Store metadata on active path nodes
                if isinstance(branch_def, BranchQueryDefinition):
                    # For BRANCH: active_path_ids are path IDs
                    for path_id in active_path_ids:
                        for node_id in path_to_nodes.get(path_id, []):
                            active_path_metadata = {
                                "selected_by_branch": branch_node_id,
                                "selected_path_id": path_id,
                                "selected_at": datetime.now(timezone.utc).isoformat(),
                            }
                            await self._update_job_branch_metadata(
                                job_id=node_id,
                                queue_name=work_item.name,
                                branch_metadata=active_path_metadata,
                            )
                elif isinstance(branch_def, SwitchQueryDefinition):
                    # For SWITCH: active_path_ids are the actual node IDs
                    for node_id in active_target_nodes:
                        # Find which case this node belongs to
                        selected_case = None
                        for case_value, node_ids in branch_def.cases.items():
                            if node_id in node_ids:
                                selected_case = str(case_value)
                                break
                        if (
                            not selected_case
                            and branch_def.default_case
                            and node_id in branch_def.default_case
                        ):
                            selected_case = "default"

                        active_path_metadata = {
                            "selected_by_switch": branch_node_id,
                            "selected_case": selected_case,
                            "selected_at": datetime.now(timezone.utc).isoformat(),
                        }
                        await self._update_job_branch_metadata(
                            job_id=node_id,
                            queue_name=work_item.name,
                            branch_metadata=active_path_metadata,
                        )

                await self._mark_nodes_ready(list(active_target_nodes), work_item.name)
                self.logger.info(
                    f"Marked {len(active_target_nodes)} nodes as READY with branch_metadata: {active_target_nodes}"
                )

            # Mark skipped nodes as SKIPPED and cascade to descendants
            if skipped_target_nodes:
                skip_reason = SkipReason(
                    branch_node_id=branch_node_id,
                    reason=f"Branch condition not met. Active paths: {active_path_ids}",
                    evaluated_condition={"active_paths": active_path_ids},
                    selected_paths=active_path_ids,
                    timestamp=datetime.now(timezone.utc),
                )
                await self._mark_nodes_skipped(
                    list(skipped_target_nodes),
                    work_item.name,
                    skip_reason,
                    dag_plan,
                )
                self.logger.info(
                    f"Marked {len(skipped_target_nodes)} nodes as SKIPPED with skip_reason: {skipped_target_nodes}"
                )

        except Exception as e:
            self.logger.error(
                f"Error evaluating branch paths for {branch_node_id}: {e}",
                exc_info=True,
            )

    async def _update_job_branch_metadata(
        self, job_id: str, queue_name: str, branch_metadata: Dict[str, Any]
    ) -> None:
        """
        Update job's branch_metadata field for tracking and debugging.

        :param job_id: Job ID to update
        :param queue_name: Queue name for the job
        :param branch_metadata: Metadata about branch evaluation/selection
        """
        try:
            # Update in repository (database is source of truth)
            await self.repository.update_job_metadata(
                job_id=job_id,
                queue_name=queue_name,
                metadata_updates={"branch_metadata": branch_metadata},
            )

            self.logger.debug(
                f"Updated branch_metadata for job {job_id}: {branch_metadata}"
            )
        except Exception as e:
            self.logger.error(
                f"Error updating branch_metadata for job {job_id}: {e}",
                exc_info=True,
            )

    async def _mark_nodes_ready(self, node_ids: list[str], queue_name: str) -> None:
        """Mark nodes as READY (keep them in CREATED state)."""
        # Nodes that should be executed remain in CREATED state
        # The scheduler will pick them up once their dependencies are met
        self.logger.debug(f"Nodes marked as ready: {node_ids}")
        # No database update needed - they're already in CREATED state

    async def _mark_nodes_skipped(
        self,
        node_ids: list[str],
        queue_name: str,
        skip_reason: SkipReason,
        dag_plan: QueryPlan,
    ) -> None:
        """
        Mark nodes as SKIPPED and cascade to descendants.
        Stores branch_metadata with skip_reason for tracking.

        :param node_ids: List of node IDs to mark as skipped
        :param queue_name: Queue name for the jobs
        :param skip_reason: Reason for skipping
        :param dag_plan: DAG plan to find descendants
        """
        if not node_ids:
            return

        try:
            # Mark nodes as SKIPPED in database
            skip_metadata = {
                "skip_reason": skip_reason.dict(),
                "skipped_at": skip_reason.timestamp.isoformat(),
            }

            await self.repository.mark_jobs_as_skipped(
                job_ids=node_ids,
                queue_name=queue_name,
                output_metadata=skip_metadata,
            )

            # Store branch_metadata with skip_reason for each skipped node
            for node_id in node_ids:
                # Store comprehensive skip information as branch_metadata
                skip_branch_metadata = {
                    "skip_reason": {
                        "branch_node_id": skip_reason.branch_node_id,
                        "reason": skip_reason.reason,
                        "selected_paths": skip_reason.selected_paths,
                        "evaluated_condition": skip_reason.evaluated_condition,
                        "timestamp": skip_reason.timestamp.isoformat(),
                    },
                    "skipped": True,
                }
                await self._update_job_branch_metadata(
                    job_id=node_id,
                    queue_name=queue_name,
                    branch_metadata=skip_branch_metadata,
                )

            # Update frontier to mark these as skipped
            for node_id in node_ids:
                await self.frontier.update_job_state(node_id, WorkState.SKIPPED)
                await self.frontier.on_job_completed(node_id)  # Remove from ready queue

            # Cascade skip to all descendants
            await self._cascade_skip_to_descendants(
                node_ids, queue_name, skip_reason, dag_plan
            )

        except Exception as e:
            self.logger.error(f"Error marking nodes as skipped: {e}", exc_info=True)

    async def _cascade_skip_to_descendants(
        self,
        skipped_node_ids: list[str],
        queue_name: str,
        skip_reason: SkipReason,
        dag_plan: QueryPlan,
    ) -> None:
        """
        Recursively mark all descendants of skipped nodes as SKIPPED.

        :param skipped_node_ids: List of skipped node IDs
        :param queue_name: Queue name
        :param skip_reason: Original skip reason
        :param dag_plan: DAG plan to traverse
        """
        if not skipped_node_ids:
            return

        descendants = set()

        # Find all descendants using the DAG structure
        for node_id in skipped_node_ids:
            node = get_node_from_dag(node_id, dag_plan)
            if not node:
                continue

            # Traverse the DAG to find all downstream nodes
            # This is a simplified traversal - in production, use topology cache
            for query in dag_plan.queries:
                if node_id in query.depends_on:
                    descendants.add(query.query)

        if descendants:
            # Create cascaded skip reason
            cascaded_reason = SkipReason(
                branch_node_id=skip_reason.branch_node_id,
                reason=f"Ancestor node(s) skipped: {skipped_node_ids}",
                evaluated_condition=skip_reason.evaluated_condition,
                selected_paths=skip_reason.selected_paths,
                timestamp=datetime.now(timezone.utc),
            )

            # Mark descendants as skipped
            await self._mark_nodes_skipped(
                list(descendants), queue_name, cascaded_reason, dag_plan
            )

    async def _handle_dag_state_notification(self, payload: dict):
        """
        Handle a DAG state change notification from PostgreSQL.

        Optimized payload structure:
        - UPDATE: {'dag_id': '<id>', 'state': '<new_state>', 'op': 'UPDATE'}
        - DELETE: {'dag_id': '<id>', 'op': 'DELETE'}

        :param payload: The notification payload with minimal fields (dag_id, state, op)
        """
        try:
            op = payload.get("op")
            dag_id: str = payload.get("dag_id", "00000000-0000-0000-0000-000000000000")

            if not dag_id or dag_id == "00000000-0000-0000-0000-000000000000":
                self.logger.warning(f"Received notification without dag_id: {payload}")
                return

            self.logger.info(
                f"Received DAG state notification: op={op}, dag_id={dag_id}"
            )

            if op == "DELETE":
                self.logger.info(
                    f"DAG {dag_id} was deleted, removing from memory frontier"
                )
                stats = await self.frontier.finalize_dag(dag_id)
                self.logger.info(f"Finalized DAG {dag_id} from memory: {stats}")

                if dag_id in self.active_dags:
                    del self.active_dags[dag_id]
                    self.logger.info(f"Removed DAG {dag_id} from active_dags")

            elif op == "UPDATE":
                new_state = payload.get("state")
                self.logger.info(f"DAG {dag_id} state changed to: {new_state}")

                if new_state == "created":
                    # DAG was reset (via reset_all or similar)
                    # Remove from memory and re-hydrate from DB
                    self.logger.warning(
                        f"DAG {dag_id} reset to 'created' - removing from memory and re-hydrating from DB"
                    )
                    stats = await self.frontier.finalize_dag(dag_id)
                    self.logger.info(
                        f"Removed DAG {dag_id} from memory frontier: {stats}"
                    )

                    if dag_id in self.active_dags:
                        del self.active_dags[dag_id]
                        self.logger.info(f"Removed DAG {dag_id} from active_dags")

                    hydrated = await self.hydrate_single_dag_from_db(dag_id)
                    if hydrated:
                        self.logger.info(
                            f"Successfully re-hydrated DAG {dag_id} from database"
                        )
                    else:
                        self.logger.warning(
                            f"Could not re-hydrate DAG {dag_id} - may not have eligible jobs"
                        )

                elif new_state == "cancelled":
                    self.logger.info(
                        f"DAG {dag_id} cancelled - removing from memory and active processing"
                    )
                    stats = await self.frontier.finalize_dag(dag_id)
                    self.logger.info(
                        f"Removed cancelled DAG {dag_id} from memory: {stats}"
                    )

                    if dag_id in self.active_dags:
                        del self.active_dags[dag_id]
                        self.logger.info(
                            f"Removed cancelled DAG {dag_id} from active_dags"
                        )

                elif new_state == "suspended":
                    self.logger.info(
                        f"DAG {dag_id} suspended - removing from active execution"
                    )
                    stats = await self.frontier.finalize_dag(dag_id)
                    self.logger.info(
                        f"Removed suspended DAG {dag_id} from memory: {stats}"
                    )
                elif new_state in ["completed", "failed"]:
                    self.logger.info(
                        f"DAG {dag_id} finished with state '{new_state}' - cleaning up memory"
                    )
                    stats = await self.frontier.finalize_dag(dag_id)
                    self.logger.info(
                        f"Cleaned up finished DAG {dag_id} from memory: {stats}"
                    )

                    if dag_id in self.active_dags:
                        del self.active_dags[dag_id]
                        self.logger.info(
                            f"Removed finished DAG {dag_id} from active_dags"
                        )

                elif new_state in ["running", "pending"]:
                    # DAG should be active - if not in memory, it might need hydration
                    if dag_id not in self.active_dags:
                        self.logger.warning(
                            f"DAG {dag_id} is in '{new_state}' state but not in active_dags. "
                            "It will be hydrated on next scheduler cycle."
                        )

                else:
                    self.logger.warning(
                        f"DAG {dag_id} changed to unknown state '{new_state}' - no action taken"
                    )

            await self.notify_event()

        except Exception as e:
            self.logger.error(f"Error handling DAG state notification: {e}")
            traceback.print_exc()

    # ==================== Schema Management (Delegated to Repository) ====================

    def create_tables(self, schema: str):
        """
        Create all database tables, functions, and triggers.
        Delegates to JobRepository.

        :param schema: The name of the schema where the tables will be created
        :return: None
        """
        self.repository.create_tables(schema)

    async def wipe(self) -> None:
        """
        Clear all data from job and archive tables.
        Delegates to JobRepository.

        :return: None
        """
        await self.repository.wipe(DEFAULT_SCHEMA)

    async def is_installed(self) -> bool:
        """
        Check if the scheduler tables are installed.
        Delegates to JobRepository.

        :return: True if tables are installed, False otherwise
        """
        return await self.repository.is_installed(DEFAULT_SCHEMA)

    async def create_queue(self, queue_name: str) -> None:
        """
        Create a new queue.
        Delegates to JobRepository.

        :param queue_name: Name of the queue to create
        :return: None
        """
        await self.repository.create_queue(queue_name)

    async def _get_defined_queues(self) -> set[str]:
        """
        Get all defined queues from the database.
        Delegates to JobRepository.

        :return: Set of queue names
        """
        return await self.repository.get_defined_queues(DEFAULT_SCHEMA)

    async def start(self) -> None:
        """
        Starts the job scheduling agent.

        :return: None
        """
        logger.info("Starting job scheduling agent")
        # Check if tables are installed and create if needed (delegate to repository)
        installed = await self.repository.is_installed(DEFAULT_SCHEMA)
        logger.info(f"Tables installed: {installed}")
        if not installed:
            self.repository.create_tables(DEFAULT_SCHEMA)

        # Get defined queues from repository
        defined_queues = await self.repository.get_defined_queues(DEFAULT_SCHEMA)
        for work_queue in self.known_queues.difference(defined_queues):
            self.logger.info(f"Create queue: {work_queue}")
            await self.repository.create_queue(work_queue)
            await self.repository.create_queue(f"${work_queue}_dlq")

        # We need to display the status
        await self.hydrate_from_db()

        self.running = True
        # self.sync_task = asyncio.create_task(self._sync())
        # self.monitoring_task = asyncio.create_task(self._monitor())
        self.monitoring_task = None

        # self._heartbeat_task = asyncio.create_task(
        #     self._heartbeat_loop(self.heartbeat_config)
        # )

        # TODO : Heartbeat currently disabled
        self.logger.warning("Heartbeat is currently disabled")
        # await self.heartbeat.start()

        self._poll_task = asyncio.create_task(self._poll())
        self._cluster_state_monitor_task = asyncio.create_task(
            self.__monitor_deployment_updates()
        )

        self._worker_tasks = [
            asyncio.create_task(self._process_submission_queue(worker_id))
            for worker_id in range(self.max_workers)
        ]

        # Start the NotificationService for DAG state change notifications
        # The service handles PostgreSQL LISTEN/NOTIFY in a non-blocking way
        try:
            await self.notification_service.start()
            self.logger.info(
                "Started NotificationService for DAG state change notifications"
            )
        except RuntimeFailToStart as e:
            self.logger.error(f"Critical: NotificationService failed to start: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error starting NotificationService: {e}")
            raise RuntimeFailToStart(f"NotificationService failed to start: {e}") from e

        # Start the MaintenanceService for periodic cleanup tasks
        try:
            await self.maintenance_service.start()
            self.logger.info(
                f"Started MaintenanceService (interval: {self.maintenance_service.maintenance_interval}s)"
            )
        except Exception as e:
            self.logger.error(f"Error starting MaintenanceService: {e}")
            # Non-critical - continue without maintenance service

        # self._sync_dag_task = asyncio.create_task(self._sync_dag())
        await self.notify_event()

    async def _poll(self):
        """
        Handles the polling, scheduling, and execution of jobs in an asynchronous job scheduler
        until the scheduler is stopped. Coordinates job management by interacting with a frontier
        (queue system), execution planner, and database for job leasing and dispatch.

        The method:
        * Periodically polls for ready work from the frontier.
        * Filters and plans the dispatching of work based on available slots
          for executors, active DAGs, and job dependencies.
        * Manages job soft-leases (local and database-level) to ensure claim
          consistency.
        * Executes or schedules ready jobs, including handling NOOP jobs that
          require local processing.

        The method dynamically adjusts its sleep intervals in case of failed
        or delayed operations, to achieve a backoff mechanism during low activity.

        Planner-first loop:
          1) wait (debounced) for wake/event
          2) read cluster slots
          3) peek ready candidates from frontier (executor-agnostic)
          4) let planner choose a plan
          5) take the chosen ids from frontier, soft-lease, DB-lease
          6) NOOPs -> complete locally; normal jobs -> dispatch and activate_from_lease

        :raises asyncio.TimeoutError: When the operation times out.
        :raises Exception: If any unexpected error occurs during job scheduling or execution processing.
        """

        self.logger.info("Starting job scheduler")
        wait_time = INIT_POLL_PERIOD
        batch_size = self.frontier_batch_size
        max_concurrent_dags = self.max_concurrent_dags
        lease_ttl = self.lease_ttl_seconds

        failures = 0
        idle_streak = 0
        _cycle_idx = 0

        cycle_log_every = self.cycle_log_every
        cycle_stats = {
            "count": 0,
            "sum_total": 0.0,
            "sum_active": 0.0,
            "min_total": inf,
            "max_total": 0.0,
            "min_active": inf,
            "max_active": 0.0,
        }

        while self.running:
            scheduled_any = False
            t_cycle_start = time.perf_counter()
            t_active_start = None

            try:
                self.logger.debug(
                    f"Polling : {wait_time:.2f}s — Queue size: {self._event_queue.qsize()} — Idle streak: {idle_streak}"
                )
                try:
                    await asyncio.wait_for(self._event_queue.get(), timeout=wait_time)
                    self._debounced_notify = False
                    wait_time = MIN_POLL_PERIOD
                except asyncio.TimeoutError:
                    pass

                # Check if gateway is ready before attempting to dispatch work
                if (
                    self._gateway_ready_event is not None
                    and not self._gateway_ready_event.is_set()
                ):
                    if _cycle_idx % 10 == 0:  # Log every 10 cycles to avoid spam
                        self.logger.warning(
                            f"[WORK_DIST] Gateway not ready yet. Scheduler will wait. "
                            f"Queue size: {self._event_queue.qsize()}"
                        )
                    idle_streak += 1
                    wait_time = adjust_backoff(
                        wait_time,
                        idle_streak,
                        scheduled=False,
                        min_poll_period=MIN_POLL_PERIOD,
                    )
                    continue

                t_active_start = time.perf_counter()
                slots_by_executor = available_slots_by_executor(
                    self._semaphore_store
                ).copy()

                if not any(slots_by_executor.values()):
                    self.logger.debug(
                        f"[WORK_DIST] No available executor slots. Backing off. "
                        f"Slots by executor: {slots_by_executor} | "
                        f"Idle streak: {idle_streak} | "
                        f"Wait time: {wait_time:.2f}s"
                    )
                    idle_streak += 1
                    wait_time = adjust_backoff(
                        wait_time,
                        idle_streak,
                        scheduled=False,
                        min_poll_period=MIN_POLL_PERIOD,
                    )
                    continue

                self.logger.debug(f"[WORK_DIST] Available slots: {slots_by_executor}")

                # FETCH READY CANDIDATES (executor-agnostic)
                # frontier should not filter by executors; let planner decide
                candidates_wi: list[WorkInfo] = await self.frontier.peek_ready(
                    batch_size,  # filter_fn=slot_filter
                )

                if not candidates_wi or len(candidates_wi) == 0:
                    frontier_summary = self.frontier.summary(detail=False)
                    self.logger.debug(
                        f"[WORK_DIST] No ready work in frontier. Short sleep. "
                        f"Batch size: {batch_size} | "
                        f"Frontier summary: {frontier_summary} | "
                        f"Idle streak: {idle_streak} | "
                        f"Wait time: {wait_time:.2f}s"
                    )
                    await asyncio.sleep(SHORT_POLL_INTERVAL)
                    idle_streak += 1
                    wait_time = adjust_backoff(
                        wait_time,
                        idle_streak,
                        scheduled=False,
                        min_poll_period=MIN_POLL_PERIOD,
                    )
                    continue

                self.logger.debug(
                    f"[WORK_DIST] Fetched {len(candidates_wi)} candidates from frontier. "
                )

                # Separate control flow nodes (NOOP/BRANCH/SWITCH) from regular jobs
                # Control flow nodes don't need executor slots and should be processed immediately
                control_flow_jobs: list[WorkInfo] = []
                regular_candidates: list[WorkInfo] = []

                for wi in candidates_wi:
                    ep = wi.data.get("metadata", {}).get("on", "")
                    if not ep:
                        self.logger.error(
                            f"[WORK_DIST] Job without entrypoint 'on': {wi.id}"
                        )
                        continue

                    # Check if this is a control flow node (noop, branch, switch, merger)
                    exe = ep.split("://", 1)[0].lower()
                    if exe in ("noop", "branch", "switch", "merger"):
                        control_flow_jobs.append(wi)
                    else:
                        regular_candidates.append(wi)

                # Process control flow nodes immediately (they don't need slots)
                if control_flow_jobs:
                    self.logger.info(
                        f"[WORK_DIST] Processing {len(control_flow_jobs)} control flow nodes immediately"
                    )
                    for wi in control_flow_jobs:
                        # Take from frontier and lease in DB
                        taken_wis = await self.frontier.take([wi.id])
                        if not taken_wis:
                            self.logger.warning(
                                f"[WORK_DIST] Failed to take control flow node {wi.id} from frontier"
                            )
                            continue

                        # Try to lease in DB
                        try:
                            leased_ids = await self._lease_jobs_db(wi.name, [wi.id])
                            if not leased_ids:
                                self.logger.warning(
                                    f"[WORK_DIST] Failed to lease control flow node {wi.id} in DB"
                                )
                                await self.frontier.release_lease_local(wi.id)
                                continue
                        except Exception as e:
                            self.logger.error(
                                f"[WORK_DIST] Error leasing control flow node {wi.id}: {e}"
                            )
                            await self.frontier.release_lease_local(wi.id)
                            continue

                        # Process the control flow node
                        asyncio.create_task(self._process_control_flow_node(wi))
                        scheduled_any = True

                # Build (entrypoint, wi) tuples for planner input (only regular jobs)
                planner_candidates: list[tuple[str, WorkInfo]] = []
                for wi in regular_candidates:
                    ep = wi.data.get("metadata", {}).get("on", "")
                    planner_candidates.append((ep, wi))

                self.logger.info(
                    f"[WORK_DIST] Built {len(planner_candidates)} planner candidates from {len(regular_candidates)} regular jobs "
                    f"(+{len(control_flow_jobs)} control flow nodes processed). "
                    f"Executors needed: {set(ep for ep, _ in planner_candidates)}"
                )

                # If all candidates were control flow nodes, skip planner and continue
                if not planner_candidates:
                    if scheduled_any:
                        # We processed control flow nodes, reset idle streak
                        idle_streak = 0
                        wait_time = MIN_POLL_PERIOD
                    self.logger.debug(
                        f"[WORK_DIST] No regular jobs to plan (processed {len(control_flow_jobs)} control flow nodes)"
                    )
                    continue

                # Give the planner: candidates + a COPY of slots + active_dags
                pick_slots = slots_by_executor.copy()
                planned: list[tuple[str, WorkInfo]] = self.execution_planner.plan(
                    planner_candidates,
                    pick_slots,
                    self.active_dags,
                    exclude_blocked=True,
                )

                await debug_candidates_and_plan(
                    candidates_wi, planned, pick_slots, self.active_dags, self.frontier
                )
                if not planned:
                    # Group candidates by executor for detailed analysis
                    candidates_by_executor = defaultdict(list)
                    for ep, wi in planner_candidates:
                        exe = ep.split("://", 1)[0]
                        candidates_by_executor[exe].append(wi.id)

                    active_dag_count = len(self.active_dags)
                    self.logger.debug(
                        f"[WORK_DIST] Planner returned NO picks. Short sleep. "
                        f"Candidates count: {len(planner_candidates)} | "
                        f"Candidates by executor: {dict(candidates_by_executor)} | "
                        f"Available slots: {pick_slots} | "
                        f"Active DAGs: {active_dag_count}/{max_concurrent_dags} | "
                        f"Idle streak: {idle_streak}"
                    )
                    await asyncio.sleep(SHORT_POLL_INTERVAL)
                    idle_streak += 1
                    wait_time = adjust_backoff(
                        wait_time,
                        idle_streak,
                        scheduled=False,
                        min_poll_period=MIN_POLL_PERIOD,
                    )
                    continue

                self.logger.debug(
                    f"[WORK_DIST] Planner selected {len(planned)} jobs to schedule. "
                    f"Job IDs: {[wi.id for _, wi in planned[:10]]}"
                )

                # TAKE + SOFT-LEASE
                selected_ids = [wi.id for _, wi in planned]
                selected_wis: List[WorkInfo] = await self.frontier.take(selected_ids)

                taken = len(selected_wis)
                requested = len(selected_ids)
                if taken != requested:
                    taken_ids = {wi.id for wi in selected_wis}
                    missing = list(set(selected_ids) - taken_ids)
                    self.logger.warning(
                        f"[WORK_DIST] Not all jobs taken from frontier: taken={taken}/{requested}. "
                        f"Missing IDs: {missing[:10]}{'...' if len(missing) > 10 else ''}"
                    )
                else:
                    self.logger.info(
                        f"[WORK_DIST] Successfully took {taken} jobs from frontier for soft-lease"
                    )

                planned_by_id = {wi.id: (ep, wi) for ep, wi in planned}
                ids_by_job_name: dict[str, list[str]] = defaultdict(list)

                for wi in selected_wis:
                    ids_by_job_name[wi.name].append(wi.id)

                leased_ids: set[str] = set()
                for job_name, ids in ids_by_job_name.items():
                    try:
                        self.logger.info(
                            f'[WORK_DIST] Attempting DB lease for job={job_name}, count={len(ids)}'
                        )
                        got = await self._lease_jobs_db(job_name, ids)
                        leased_ids.update(got)
                        self.logger.info(
                            f'[WORK_DIST] DB lease result for job={job_name}: leased {len(got)}/{len(ids)}'
                        )
                        if len(got) < len(ids):
                            missing_ids = set(ids) - set(got)
                            self.logger.warning(
                                f"[WORK_DIST] DB lease shortfall for '{job_name}': got {len(got)}/{len(ids)}. "
                                f"Missing IDs: {list(missing_ids)[:5]}"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"[WORK_DIST] DB lease FAILED for '{job_name}' ({len(ids)} ids): {e}",
                            exc_info=True,
                        )
                # put *everything* back
                if not leased_ids:
                    for wi in selected_wis:
                        await self.frontier.release_lease_local(wi.id)
                    self.logger.warning(
                        f"[WORK_DIST] NO candidates could be leased in DB; backing off. "
                        f"Attempted {len(selected_wis)} jobs across {len(ids_by_job_name)} job names. "
                        f"Job names: {list(ids_by_job_name.keys())}"
                    )
                    await asyncio.sleep(SHORT_POLL_INTERVAL)
                    idle_streak += 1
                    wait_time = adjust_backoff(
                        wait_time,
                        idle_streak,
                        scheduled=False,
                        min_poll_period=MIN_POLL_PERIOD,
                    )
                    continue

                self.logger.info(
                    f"[WORK_DIST] Successfully leased {len(leased_ids)} jobs in DB. "
                    f"Processing leased jobs now..."
                )

                # return non-leased back to frontier immediately
                for wi in selected_wis:
                    if wi.id not in leased_ids:
                        await self.frontier.release_lease_local(wi.id)

                # only process those that we leased in DB
                leased_jobs: list[tuple[str, WorkInfo]] = [
                    planned_by_id[jid] for jid in leased_ids if jid in planned_by_id
                ]

                #  PROCESS LEASED JOBS
                scheduled_any = False
                jobs_scheduled_this_cycle = defaultdict(int)
                enqueue_tasks = []

                self.logger.info(
                    f"[WORK_DIST] Processing {len(leased_jobs)} leased jobs..."
                )

                for entrypoint, wi in leased_jobs:
                    dag_id = wi.dag_id
                    if (
                        dag_id not in self.active_dags
                        and len(self.active_dags) >= max_concurrent_dags
                    ):
                        self.logger.debug(
                            f"[WORK_DIST] Max DAG limit reached ({len(self.active_dags)}/{max_concurrent_dags}). "
                            f"Skipping job {wi.id} (DAG: {dag_id}). "
                            # f"Active DAGs: {list(self.active_dags.keys())}"
                        )
                        await self._release_lease_db([wi.id])
                        await self.frontier.release_lease_local(wi.id)
                        continue

                    # Ensure DAG cached/active
                    if dag_id not in self.active_dags:
                        dag = await self.get_dag_by_id(dag_id)
                        if not dag:
                            self.logger.warning(
                                f"Missing DAG {dag_id} for job {wi.id}; releasing lease."
                            )
                            await self._release_lease_db([wi.id])
                            await self.frontier.release_lease_local(wi.id)
                            continue

                        self.logger.debug(
                            f"Marking active dag : {len(self.active_dags)}"
                        )
                        await self.mark_as_active_dag(wi)
                        self.active_dags[dag_id] = dag

                    # NOTE: NOOP/BRANCH/SWITCH nodes are handled earlier in the pipeline
                    # (before planner) and should never reach this point.
                    # If they do, it's a bug - but we'll handle gracefully

                    # Normal job: check slots then dispatch
                    exe = entrypoint.split("://", 1)[0]
                    if slots_by_executor.get(exe, 0) <= 0:
                        self.logger.warning(
                            f"[WORK_DIST] No slots available for executor={exe}, delaying job {wi.id}. "
                            f"Current slots_by_executor: {slots_by_executor}"
                        )
                        await self._release_lease_db([wi.id])
                        await self.frontier.release_lease_local(wi.id)
                        continue

                    # Reserve capacity via semaphore before dispatch to avoid async races
                    slot_type = exe
                    # owner = f"{self._sem_owner_prefix}:{wi.id}"
                    owner = f"{wi.id}"
                    reserved = False
                    try:
                        self.logger.debug(
                            f"[WORK_DIST] Attempting semaphore reservation for job={wi.id}, executor={slot_type}"
                        )
                        reserved = await asyncio.to_thread(
                            self._semaphore_store.reserve,
                            slot_type,
                            wi.id,  # ticket_id
                            node='',  # at this time we don't know the placement where the job wil be executed yet
                            ttl=self._sem_default_ttl,
                            owner=wi.id,
                        )
                        if reserved:
                            self.logger.info(
                                f"[WORK_DIST] Semaphore reserved successfully for job={wi.id}, executor={slot_type}"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"[WORK_DIST] Semaphore reserve ERROR for job={wi.id}, executor={slot_type}: {e}",
                            exc_info=True,
                        )
                        reserved = False

                    if not reserved:
                        self.logger.warning(
                            f"[WORK_DIST] NO semaphore capacity for executor={slot_type}; releasing lease for job={wi.id}. "
                            f"slots_by_executor={slots_by_executor}"
                        )
                        await self._release_lease_db([wi.id])
                        await self.frontier.release_lease_local(wi.id)
                        continue

                    slots_by_executor[exe] = max(0, slots_by_executor.get(exe, 0) - 1)
                    enqueue_tasks.append(
                        {
                            "task": asyncio.create_task(
                                self._activate_and_enqueue_job(wi)
                            ),
                            "wi": wi,
                            "exe": exe,
                            "owner": owner,
                        }
                    )

                if enqueue_tasks:
                    self.logger.info(
                        f"[WORK_DIST] Dispatching {len(enqueue_tasks)} jobs via _activate_and_enqueue_job..."
                    )
                    results = await asyncio.gather(
                        *[t["task"] for t in enqueue_tasks], return_exceptions=True
                    )
                    self.logger.info(
                        f"[WORK_DIST] Dispatch completed. Processing {len(results)} results..."
                    )
                    for i, result in enumerate(results):
                        wi = enqueue_tasks[i]["wi"]
                        exe = enqueue_tasks[i]["exe"]
                        owner = enqueue_tasks[i]["owner"]

                        if isinstance(result, Exception) or not result:
                            # dispatch failed → release lease & requeue
                            self.logger.error(
                                f"[WORK_DIST] Dispatch FAILED for job={wi.id}, executor={exe}: {result}",
                                exc_info=(
                                    True if isinstance(result, Exception) else False
                                ),
                            )

                            raise Exception(f"Dispatch failed for job {wi.id}")
                            await self._release_lease_db([wi.id])
                            await self.frontier.release_lease_local(wi.id)

                            try:
                                released = await asyncio.to_thread(
                                    self._semaphore_store.release_owned,
                                    exe,
                                    wi.id,
                                    owner=owner,
                                )
                                self.logger.debug(
                                    f"[sem] release on dispatch-fail {wi.id}@{slot_type} -> {released}"
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"[sem] release error after dispatch-fail {wi.id}@{slot_type}: {e}"
                                )
                            continue

                        # runner accepted → activate from lease
                        activated = await self._activate_from_lease_db([wi.id])
                        if wi.id in activated:
                            jobs_scheduled_this_cycle[exe] += 1
                            scheduled_any = True
                        else:
                            self.logger.error(
                                f"Failed to activate job {wi.id}; releasing lease."
                            )
                            await self._release_lease_db([wi.id])
                            await self.frontier.release_lease_local(wi.id)

                if jobs_scheduled_this_cycle:
                    self.logger.info("Scheduling summary for this cycle:")
                    for exe, cnt in sorted(jobs_scheduled_this_cycle.items()):
                        self.logger.info(f"  - {exe}: {cnt} scheduled")

                if scheduled_any:
                    await self.notify_event()

                # maintain frontier heap
                if (_cycle_idx % 20) == 0:
                    removed = await self.frontier.compact_ready_heap(max_scan=10000)
                    if removed:
                        self.logger.debug(f"Frontier heap compacted: removed={removed}")

                idle_streak = 0 if scheduled_any else idle_streak + 1
                wait_time = adjust_backoff(
                    wait_time,
                    idle_streak,
                    scheduled_any,
                    min_poll_period=MIN_POLL_PERIOD,
                )
                failures = 0

            except Exception as e:
                self.logger.error("Poll loop exception", exc_info=True)
                failures += 1
                if failures >= 5:
                    self.logger.warning("Too many failures — entering cooldown")
                    await asyncio.sleep(60)
                    failures = 0
            finally:
                # ---- timing ----
                t_end = time.perf_counter()
                dt_total = t_end - t_cycle_start
                dt_active = (t_end - t_active_start) if t_active_start else 0.0

                cycle_stats["count"] += 1
                cycle_stats["sum_total"] += dt_total
                cycle_stats["sum_active"] += dt_active
                cycle_stats["min_total"] = min(cycle_stats["min_total"], dt_total)
                cycle_stats["max_total"] = max(cycle_stats["max_total"], dt_total)
                cycle_stats["min_active"] = min(cycle_stats["min_active"], dt_active)
                cycle_stats["max_active"] = max(cycle_stats["max_active"], dt_active)

                _cycle_idx += 1
                if (_cycle_idx % cycle_log_every) == 0:
                    avg_total = cycle_stats["sum_total"] / cycle_stats["count"]
                    avg_active = cycle_stats["sum_active"] / cycle_stats["count"]

                    self.logger.info(
                        "[poll] Cycle stats (last %d): total=%.1f ms (avg %.1f–%.1f) | "
                        "active=%.1f ms (avg %.1f–%.1f) | wait=%.1fs | idle_streak=%d",
                        cycle_stats["count"],
                        avg_total * 1000,
                        cycle_stats["min_total"] * 1000,
                        cycle_stats["max_total"] * 1000,
                        avg_active * 1000,
                        cycle_stats["min_active"] * 1000,
                        cycle_stats["max_active"] * 1000,
                        wait_time,
                        idle_streak,
                    )

                    # reset rolling window
                    cycle_stats = {
                        "count": 0,
                        "sum_total": 0.0,
                        "sum_active": 0.0,
                        "min_total": inf,
                        "max_total": 0.0,
                        "min_active": inf,
                        "max_active": 0.0,
                    }

    async def _activate_and_enqueue_job(self, wi: WorkInfo) -> bool:
        """Marks a job as active in the database and then enqueues it to a worker."""
        await self.mark_as_active(wi)
        return await self.enqueue(wi)

    async def stop(self, timeout: float = 2.0) -> None:
        self.logger.info("Stopping job scheduling agent")
        self.running = False

        tasks = [self.monitoring_task]
        tasks = tasks + [self._producer_task]

        if self._heartbeat_task:
            tasks.append(self._heartbeat_task)

        if self._cluster_state_monitor_task:
            tasks.append(self._cluster_state_monitor_task)

        if self._worker_tasks:
            for task in self._worker_tasks:
                tasks.append(task)

        if self.sync_task:
            tasks.append(self.sync_task)

        if self._sync_dag_task:
            tasks.append(self._sync_dag_task)

        # Stop NotificationService
        try:
            await self.notification_service.stop()
            self.logger.info("Stopped NotificationService")
        except Exception as e:
            self.logger.error(f"Error stopping NotificationService: {e}")

        # Stop MaintenanceService
        try:
            await self.maintenance_service.stop()
            self.logger.info("Stopped MaintenanceService")
        except Exception as e:
            self.logger.error(f"Error stopping MaintenanceService: {e}")

        for task in tasks:
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, timeout)
                except asyncio.TimeoutError:
                    task_name = getattr(task, '_name', task.__class__.__name__)
                    self.logger.warning(
                        f"Task did not complete in time, cancelling it : {task_name}"
                    )
                    task.cancel()
                    try:
                        await task  # Wait for cancellation
                    except asyncio.CancelledError:
                        self.logger.debug("Task cancelled successfully")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(f"Unexpected error during task shutdown: {e}")

    def debug_info(self):
        """
        Return comprehensive debugging information about the scheduler's current state.

        Returns:
            dict: Dictionary containing various debugging information including:
                - Scheduler status and configuration
                - Task states and counters
                - Queue information
                - Database connection status
                - Active DAGs and jobs summary
        """

        current_time = datetime.now(timezone.utc)

        debug_data = {
            "scheduler_info": {
                "running": self.running,
                "scheduler_mode": self.scheduler_mode,
                "max_concurrent_dags": self.max_concurrent_dags,
                "known_queues": list(self.known_queues) if self.known_queues else [],
                "active_dags_count": len(self.active_dags) if self.active_dags else 0,
            },
            "timing_info": {
                "current_time": current_time.isoformat(),
                "start_time": self._start_time.isoformat(),
                "uptime_seconds": (current_time - self._start_time).total_seconds(),
                "uptime_human": str(current_time - self._start_time),
            },
            "counters": {
                "fetch_counter": self._fetch_counter,
                "submission_count": self._submission_count,
                "pending_requests": (
                    len(self._pending_requests) if self._pending_requests else 0
                ),
            },
            "queues": {
                "request_queue_size": (
                    self._request_queue.qsize() if self._request_queue else 0
                ),
                "event_queue_size": (
                    self._event_queue.qsize() if self._event_queue else 0
                ),
            },
            "execution_planning": {
                "execution_planner_available": self.execution_planner is not None,
            },
        }

        # Add active DAGs information if available
        if self.active_dags:
            active = {}
            for dag_id, dag_info in self.active_dags.items():
                status_val = "unknown"
                try:
                    status_val = dag_info.status
                except Exception:
                    pass
                active[dag_id] = {"dag_id": dag_id, "status": status_val}
            debug_data["active_dags"] = active

        # Add queue status information
        try:
            debug_data["queue_status"] = self.get_queue_status()
        except Exception as e:
            debug_data["queue_status_error"] = str(e)

        try:
            debug_data["job_state_counts"] = self._db.count_job_states()
        except Exception as e:
            debug_data["job_state_counts_error"] = str(e)

        try:
            debug_data["dag_state_counts"] = self._db.count_dag_states()
        except Exception as e:
            debug_data["dag_state_counts_error"] = str(e)

        # Include detailed frontier summary without using getattr
        frontier_info = {"available": self.frontier is not None}
        if self.frontier:
            try:
                # Prefer detailed view with top-N stalest items
                frontier_info["summary"] = self.frontier.summary(detail=True, top_n=10)
            except TypeError:
                # Fallback to default signature
                try:
                    frontier_info["summary"] = self.frontier.summary(detail=True)
                except Exception as e:
                    frontier_info["summary_error"] = str(e)
            except Exception as e:
                frontier_info["summary_error"] = str(e)

            # Known scheduler-level frontier settings
            try:
                frontier_info["batch_size"] = self.frontier_batch_size
            except Exception:
                pass
            try:
                frontier_info["lease_ttl_seconds"] = self.lease_ttl_seconds
            except Exception:
                pass

        debug_data["frontier"] = frontier_info

        return debug_data

    async def enqueue(self, work_info: WorkInfo) -> bool:
        """
        Tries to dispatch a work item to an executor and waits for confirmation.
        This method does NOT change the job state in the database.

        :param work_info: The information about the work item to be processed.
        :return: True if successfully dispatched and confirmed, False otherwise.
        """
        self.logger.info(f"Attempting to dispatch work item: {work_info.id}")
        confirmation_event = asyncio.Event()
        submission_id = work_info.id
        entrypoint = work_info.data.get("metadata", {}).get("on")

        if not entrypoint:
            self.logger.error(
                f"The entrypoint 'on' is not defined in metadata for job {submission_id}"
            )
            return False

        try:
            await self.job_manager.submit_job(
                entrypoint=entrypoint,
                submission_id=submission_id,
                metadata=work_info.data,
                confirmation_event=confirmation_event,
            )

            # Wait for the supervisor to confirm it has received the job and is running.
            # A short timeout is critical, it should be less than the lease TTL.
            await asyncio.wait_for(
                confirmation_event.wait(), timeout=self.lease_ttl_seconds - 1
            )
            self.logger.debug(f"Dispatch confirmed for job: {submission_id}")
            return True

        except asyncio.TimeoutError:
            self.logger.error(
                f"Timeout waiting for dispatch confirmation for job {submission_id}"
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Failed to dispatch job {submission_id}: {e}", exc_info=True
            )
            return False

    async def get_job(self, job_id: str) -> Optional[WorkInfo]:
        """
        Get a job by its ID from cache or database.
        :param job_id: The ID of the job to retrieve.
        """
        # Fast path - check cache first
        if job_id in self._job_cache:
            # Move to end to signify it's recently used (LRU)
            self._job_cache[job_id] = self._job_cache.pop(job_id)
            return self._job_cache[job_id]

        # Cache miss - fetch from repository
        work_item = await self.repository.get_job_by_id(job_id)

        # Update cache if found
        if work_item:
            self._job_cache[job_id] = work_item
            # Evict oldest if cache is over size
            if len(self._job_cache) > self._job_cache_max_size:
                self._job_cache.pop(next(iter(self._job_cache)))

        return work_item

    async def get_job_for_policy(self, work_info: WorkInfo) -> Optional[WorkInfo]:
        """
        Find a job by its name and data (used for policy checks).
        :param work_info: WorkInfo containing metadata with ref_type and ref_id
        :return: WorkInfo if found, None otherwise
        """
        ref_type = work_info.data.get("metadata", {}).get("ref_type", "")
        ref_id = work_info.data.get("metadata", {}).get("ref_id", "")

        return await self.repository.get_job_by_policy(ref_type, ref_id)

    async def list_jobs(
        self, state: Optional[str | list[str]] = None, batch_size: int = 0
    ) -> Dict[str, WorkInfo]:
        work_items = {}
        schema = DEFAULT_SCHEMA
        table = DEFAULT_JOB_TABLE
        cursor = None
        conn = None

        if state is not None:
            if isinstance(state, str):
                state = [state]
            invalid_states = [
                s for s in state if s.upper() not in WorkState.__members__
            ]
            if invalid_states:
                raise ValueError(f"Invalid state(s): {', '.join(invalid_states)}")
            states = "','".join(s.lower() for s in state)
        else:
            states = "','".join(s.lower() for s in WorkState.__members__.keys())

        try:
            conn = self._get_connection()
            cursor = conn.cursor("doc_iterator")
            cursor.itersize = 10000
            cursor.execute(
                f"""
                SELECT id,name, priority,state,retry_limit,start_after,expire_in,data,retry_delay,retry_backoff,keep_until,dag_id,job_level
                FROM {schema}.{table} 
                WHERE state IN ('{states}')
                {f"LIMIT {batch_size}" if batch_size > 0 else ""}
                """
            )
            for record in cursor:
                work_items[record[0]] = self.record_to_work_info(record)
            conn.commit()
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error listing jobs: {error}")
            conn.rollback()
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)
        return work_items

    async def submit_job(self, work_info: WorkInfo, overwrite: bool = True) -> str:
        """
        Inserts a new work item into the scheduler.
        :param work_info: The work item to insert.
        :param overwrite: Whether to overwrite the work item if it already exists.
        :return: The ID of the inserted work item.
        :raises ValueError: If the job submission fails or if the job already exists.
        """
        self.logger.info(f"Submitting job : {work_info.id}")

        work_queue = work_info.name
        if work_info.name not in self.known_queues:
            self.logger.info(f"Checking for queue: {work_queue}")
            await self.create_queue(work_queue)
            await self.create_queue(f"${work_queue}_dlq")
            self.known_queues.add(work_queue)

        result_future = asyncio.Future()
        request_id = str(uuid.uuid4())

        submission_request = JobSubmissionRequest(
            work_info=work_info,
            overwrite=overwrite,
            request_id=request_id,
            result_future=result_future,
        )

        self._pending_requests[request_id] = submission_request

        try:
            self._request_queue.put_nowait(submission_request)
            self.logger.debug(
                f"Job {work_info.id} queued successfully (request: {request_id})"
            )
            # sync_mode = work_info.data.get("metadata", {}).get("sync_mode", False)
            sync_mode = False
            if sync_mode:
                # Wait for the result
                result = await result_future
                return result

            return work_info.id
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            result_future.set_exception(e)
            raise

    async def _handle_priority_refresh(self):
        """Handle priority refresh"""
        refresh_interval = getattr(self, 'priority_refresh_interval', 10)

        if self._submission_count % refresh_interval == 0:
            await self._refresh_job_priorities()
            self.logger.info(
                f"Refreshed job priorities after {self._submission_count} submissions "
                f"(interval: {refresh_interval})"
            )

    async def _process_submission_queue(self, worker_id: int) -> None:
        """Background worker that processes queued job submissions"""
        self.logger.info(f"Background job submission worker started # {worker_id}")

        while self.running:
            request = None
            try:
                request = await self._request_queue.get()
                try:
                    result = await self.__submit_job(
                        request.work_info, request.overwrite
                    )
                    self._submission_count += 1
                    await self._handle_priority_refresh()

                    if not request.result_future.done():
                        request.result_future.set_result(result)

                    queue_size = self._request_queue.qsize()
                    self.logger.debug(
                        f"Successfully processed job: {request.work_info.id} (queue size: {queue_size})"
                    )

                except ValueError as e:
                    self.logger.warning(
                        f"Job submission issue for {request.work_info.id}: {e}"
                    )
                    if not request.result_future.done():
                        request.result_future.set_result(request.work_info.id)
                except Exception as e:
                    if not request.result_future.done():
                        request.result_future.set_exception(e)
                    self.logger.error(
                        f"Failed to process job {request.work_info.id}: {e}"
                    )
                finally:
                    self._pending_requests.pop(request.request_id, None)

            except asyncio.CancelledError:
                self.logger.info("Background job submission worker cancelled")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in submission worker: {e}")
                await asyncio.sleep(1)
            finally:
                if request:
                    self._request_queue.task_done()

    def get_queue_status(self) -> dict:
        """Get current status of the submission queue"""
        active_workers = 0
        total_workers = len(self._worker_tasks) if self._worker_tasks else 0

        if self._worker_tasks:
            active_workers = sum(
                1 for task in self._worker_tasks if task and not task.done()
            )

        return {
            "queue_size": self._request_queue.qsize(),
            "pending_requests": len(self._pending_requests),
            "total_submissions": self._submission_count,
            "workers": {
                "total": total_workers,
                "active": active_workers,
                "utilization": (
                    f"{(active_workers / total_workers) * 100:.1f}%"
                    if total_workers > 0
                    else "0%"
                ),
            },
        }

    async def __submit_job(self, work_info: WorkInfo, overwrite: bool = True) -> str:
        """
        :param work_info: WorkInfo object containing job details
        :param overwrite:
        :return:
        """
        submission_id = work_info.id
        submission_policy = ExistingWorkPolicy.create(
            work_info.policy, default_policy=ExistingWorkPolicy.REJECT_DUPLICATE
        )

        is_valid = await self.is_valid_submission(work_info, submission_policy)
        if not is_valid:
            raise ValueError(
                f"Job with submission_id {submission_id} already exists."
                f"For work item : {work_info}."
            )

        # Build plan & nodes once (used by DB and memory)
        plan, dag_nodes = query_plan_work_items(work_info)

        # Set dag_id on all nodes
        for dag_work_info in dag_nodes:
            dag_work_info.dag_id = submission_id

        # Delegate DAG and job creation to repository
        new_key_added, new_dag_key = await self.repository.create_dag_with_jobs(
            dag_id=submission_id,
            plan=plan,
            dag_nodes=dag_nodes,
            work_info=work_info,
        )

        if not new_key_added:
            raise ValueError(
                f"Job with submission_id {submission_id} already exists. "
                "Please use a different submission_id."
            )

        await self.frontier.add_dag(plan, dag_nodes)
        await self.notify_event()
        return submission_id

    async def _refresh_job_priorities(self):
        """Execute the job priority refresh SQL function"""
        try:
            self.logger.info(
                f'Refreshing job priorities...  : {self._submission_count}'
            )
            #   SELECT marie_scheduler.refresh_job_priority();
        except Exception as e:
            self.logger.error(f"Failed to refresh job priorities: {e}")

    async def mark_as_active(self, work_info: WorkInfo) -> bool:
        """
        Mark a job as active.
        Delegates to JobRepository.

        :param work_info: WorkInfo containing job ID and name
        :return: True if successful, False otherwise
        """
        self.logger.debug(f"Marking as active : {work_info.id}")
        count = await self.repository.mark_jobs_as_active(
            job_ids=[work_info.id], job_name=work_info.name
        )
        return count > 0

    async def mark_as_active_dag(self, work_info: WorkInfo) -> bool:
        """
        Mark a DAG as active.
        Delegates to JobRepository.

        :param work_info: WorkInfo containing DAG ID
        :return: True if successful, False otherwise
        """
        return await self.repository.mark_dag_as_active(work_info.dag_id)

    async def is_valid_submission(
        self, work_info: WorkInfo, policy: ExistingWorkPolicy
    ) -> bool:
        """
        Validates a work submission based on the specified policy.

        :param work_info: Information about the work to be checked for validity
        :param policy: Policy that dictates the rules for the work submission
        :return: True if the submission is valid according to the policy, False otherwise
        :raises ValueError: If an unsupported policy is provided
        """
        try:
            if policy in (
                ExistingWorkPolicy.ALLOW_ALL,
                ExistingWorkPolicy.ALLOW_DUPLICATE,
            ):
                return True

            if policy == ExistingWorkPolicy.REJECT_ALL:
                return False

            existing_job = await self._loop.run_in_executor(
                self._db_executor, self.get_job_for_policy, work_info
            )

            if policy == ExistingWorkPolicy.REJECT_DUPLICATE:
                return existing_job is None

            if policy == ExistingWorkPolicy.REPLACE:
                return not existing_job or (
                    existing_job.state is not None and existing_job.state.is_terminal()
                )

            raise ValueError(f"Unsupported policy: {policy}")

        except Exception as e:
            logger.error(
                f"Error validating submission for work '{work_info.name}' "
                f"with policy '{policy}': {str(e)}"
            )
            return False

    def stop_job(self, job_id: str) -> bool:
        """Request a job to exit, fire and forget.
        Returns whether or not the job was running.
        """
        raise NotImplementedError

    async def delete_job(self, job_id: str):
        """Deletes the job with the given job_id."""
        ...

        raise NotImplementedError

    async def cancel_job(self, job_id: str, work_item: WorkInfo) -> None:
        """
        Cancel a job by its ID.
        Delegates to JobRepository.

        :param job_id: The ID of the job.
        :param work_item: The work item to cancel.
        """
        async with self._status_update_lock[job_id]:
            await self.repository.cancel_job(
                job_id=job_id,
                queue_name=work_item.name,
                schema=DEFAULT_SCHEMA,
            )

    async def resume_job(self, job_id: str) -> None:
        """
        Resume a job by its ID.
        Delegates to JobRepository.

        :param job_id: The ID of the job to resume
        """
        # TODO: This queue name is a placeholder - should be determined from job metadata
        queue_name = "extract"
        await self.repository.resume_job(
            job_id=job_id,
            queue_name=queue_name,
            schema=DEFAULT_SCHEMA,
        )

    async def put_status(
        self,
        job_id: str,
        status: WorkState,
        started_on: Optional[datetime] = None,
        completed_on: Optional[datetime] = None,
    ):
        """
        Update the status of a job.
        Delegates to JobRepository.

        :param job_id: The ID of the job.
        :param status: The new status of the job.
        :param started_on: Optional start time of the job.
        :param completed_on: Optional completion time of the job.
        """
        async with self._status_update_lock[job_id]:
            await self.repository.update_job_state(
                job_id=job_id,
                state=status,
                started_on=started_on,
                completed_on=completed_on,
            )
            # self._job_cache.pop(job_id, None)

    async def maintenance(self):
        """
        Performs the maintenance process, including expiring, archiving, and purging.
        Delegates to MaintenanceService.

        :return: None
        """
        await self.maintenance_service.maintenance()

    async def expire(self):
        """
        Expire jobs with expired leases.
        Delegates to MaintenanceService.
        """
        await self.maintenance_service.expire()

    async def archive(self):
        """
        Archive completed jobs.
        Delegates to MaintenanceService.
        """
        await self.maintenance_service.archive()

    async def purge(self):
        """
        Purge old archived jobs.
        Delegates to MaintenanceService.
        """
        await self.maintenance_service.purge()

    def _setup_event_subscriptions(self):
        self.job_manager.event_publisher.subscribe(
            [
                JobStatus.RUNNING,
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.PENDING,
                JobStatus.STOPPED,
            ],
            self.handle_job_event,
        )

    def record_to_work_info(self, record: Any) -> WorkInfo:
        """
        Convert a record to a WorkInfo object.
        :param record:
        :return:
        """
        return WorkInfo(
            id=record[0],
            name=record[1],
            priority=record[2],
            state=WorkState(record[3]) if record[3] else None,
            retry_limit=record[4],
            start_after=record[5],
            expire_in_seconds=0,  # record[6], # FIXME this is wrong type
            data=record[7],
            retry_delay=record[8],
            retry_backoff=record[9],
            keep_until=record[10],
            dag_id=record[11],
            job_level=record[12],
        )

    async def _monitor(self):
        """
        Background monitoring loop that updates the monitor timestamp.
        Delegates to JobRepository.
        """
        wait_time = MONITORING_POLL_PERIOD
        while self.running:
            self.logger.debug(f"Polling jobs status : {wait_time}")
            await asyncio.sleep(wait_time)

            try:
                # Delegate to repository to update monitor time
                monitored_on = await self.repository.update_monitor_time(
                    monitor_state_interval_seconds=int(MONITORING_POLL_PERIOD)
                )

                if monitored_on is None:
                    self.logger.error("Error setting monitor time")
                    continue
            except Exception as e:
                logger.error(f"Error monitoring jobs: {e}")
                traceback.print_exc()
                # TODO: emit error event

    async def complete(
        self,
        job_id: str,
        work_item: WorkInfo,
        output_metadata: dict = None,
        force=False,
    ):
        """
        Mark a job as completed.
        Delegates to JobRepository.

        :param job_id: The ID of the job to complete
        :param work_item: The work item containing queue name
        :param output_metadata: Optional metadata to store with completion
        :param force: If True, complete job regardless of current state
        """
        async with self._status_update_lock[job_id]:
            await self.repository.complete_job(
                job_id=job_id,
                queue_name=work_item.name,
                output_metadata=output_metadata,
                force=force,
                schema=DEFAULT_SCHEMA,
            )
            # self._job_cache.pop(job_id, None) # invalidate cache

    async def fail(
        self, job_id: str, work_item: WorkInfo, output_metadata: dict = None
    ):
        """
        Mark a job as failed.
        Delegates to JobRepository.

        :param job_id: The ID of the job to mark as failed
        :param work_item: The work item containing queue name
        :param output_metadata: Optional metadata to store with failure
        """
        async with self._status_update_lock[job_id]:
            await self.repository.fail_job(
                job_id=job_id,
                queue_name=work_item.name,
                output_metadata=output_metadata,
                schema=DEFAULT_SCHEMA,
            )
            # self._job_cache.pop(job_id, None) # invalidate cache

    async def _sync(self):
        """
        Synchronizes job status between the local job tracking system and db.
        This function runs in a loop to periodically check the status of active jobs and update their state
        locally based on the external source.
        """
        wait_time = SYNC_POLL_PERIOD
        job_info_client = self.job_manager.job_info_client()
        min_sync_interval_seconds = 300  # 5 minutes in seconds

        while self.running:
            self.logger.info(f"Syncing job status every {wait_time} seconds")
            await asyncio.sleep(wait_time)
            try:
                active_jobs = await self.list_jobs(
                    state=[WorkState.ACTIVE.value, WorkState.CREATED.value]
                )
                if not active_jobs:
                    continue

                for job_id, work_item in active_jobs.items():
                    self.logger.info(f"Syncing job: {job_id}")
                    job_info = await job_info_client.get_info(job_id)
                    if job_info is None:
                        self.logger.error(f"Job to synchronize not found: {job_id}")
                        continue

                    if not job_info.status:
                        self.logger.warning(
                            f"Missing status for job: {job_id}, skipping."
                        )
                        continue

                    job_info_state = convert_job_status_to_work_state(job_info.status)
                    if (
                        job_info.status.is_terminal()
                        and work_item.state != job_info_state
                    ):
                        self.logger.info(
                            f"State mismatch for job {job_id}: "
                            f"WorkState={work_item.state}, JobInfoState={job_info_state}. Updating."
                        )

                        synchronize = False
                        remaining_time = None
                        now = datetime.now(tz=timezone.utc)

                        if job_info.end_time is not None:
                            timestamp_ms = job_info.end_time  # Unix timestamp in ms
                            end_time = datetime.fromtimestamp(
                                timestamp_ms / 1000, tz=timezone.utc
                            )
                            remaining_time = end_time - now

                            if end_time < now - timedelta(
                                seconds=min_sync_interval_seconds
                            ):
                                synchronize = True

                        if not synchronize:
                            seconds = (
                                remaining_time.total_seconds()
                                if remaining_time
                                else "unknown"
                            )
                            self.logger.info(
                                f"Job has not ended more than {min_sync_interval_seconds} seconds ago, skipping sync. "
                                f"{job_id}: {seconds} seconds since end."
                            )
                            continue

                        meta = {"synced": True}
                        if job_info.status == JobStatus.SUCCEEDED:
                            await self.complete(job_id, work_item, meta, force=True)
                        elif job_info.status == JobStatus.FAILED:
                            await self.fail(job_id, work_item, meta)
                        elif job_info.status == JobStatus.STOPPED:
                            await self.cancel_job(job_id, work_item)
                        else:
                            self.logger.error(
                                f"Unhandled job status: {job_info.status}. Marking as FAILED."
                            )
                            await self.fail(job_id, work_item)

                        self.logger.info(
                            f"Synchronized job {job_id} is in terminal state {job_info.status}"
                        )
                        await self.resolve_dag_status(job_id, work_item, now, now)

            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error syncing jobs: {error}")
                self.logger.error(traceback.format_exc())

    async def _sync_dag(self):
        self.logger.info("Starting DAG synchronization")
        # https://github.com/marieai/marie-ai/issues/134
        await self._loop.run_in_executor(self._db_executor, self._blocking_sync_dag)

    def _blocking_sync_dag(self, interval: int = 30) -> None:
        """
        Validate that DAGs in memory still exist and are active in database.
        This runs in a background thread and delegates DB access to repository.
        """
        self.logger.info(f"Starting DAG sync polling (interval: {interval}s)")

        while self.running:
            try:
                if not self.active_dags:
                    self.logger.debug("No active DAGs in memory to validate")
                    time.sleep(interval)
                    continue

                memory_dag_ids = list(self.active_dags.keys())
                self.logger.debug(f"Validating {len(memory_dag_ids)} DAGs in memory")

                # Delegate to repository to get active DAG IDs
                # We need to run this async method from sync context
                future = asyncio.run_coroutine_threadsafe(
                    self.repository.get_active_dag_ids(memory_dag_ids), self._loop
                )
                valid_db_dags = future.result(timeout=30)

                invalid_dags = set(memory_dag_ids) - valid_db_dags

                if invalid_dags:
                    self.logger.info(
                        f"Found {len(invalid_dags)} invalid DAGs in memory"
                    )
                    for dag_id in invalid_dags:
                        self._remove_dag_from_memory(
                            dag_id, "no longer active or deleted in database"
                        )
                else:
                    self.logger.debug("All DAGs in memory are still valid")

            except Exception as error:
                self.logger.error(f"Error validating DAGs: {error}")

            time.sleep(interval)
        self.logger.debug(f"DAG sync polling stopped")

    def _remove_dag_from_memory(self, dag_id: str, reason: str):
        """
        Centralized method to remove DAG from memory with logging.
        Delegates to DAGManagementService.
        """
        self.dag_service.remove_dag(dag_id, reason)

    async def notify_event(self) -> bool:
        if self._debounced_notify:
            return False
        self._debounced_notify = True
        try:
            self._event_queue.put_nowait("wake")
        except asyncio.QueueFull:
            pass
        return True

    async def resolve_dag_status(
        self,
        job_id: str,
        work_info: WorkInfo,
        started_on: Optional[datetime] = None,
        completed_on: Optional[datetime] = None,
    ) -> bool:
        """
        Resolves the status of a directed acyclic graph (DAG). This method checks
        if the DAG has completed execution by querying its current state and handles
        the corresponding logic for the DAG lifecycle, including sending notification
        about the completion or failure of the DAG.
        """
        self.logger.info(f"Resolving DAG status: {work_info.dag_id}")

        try:
            # Delegate to repository to resolve DAG state
            dag_state = await self.repository.resolve_dag_state(work_info.dag_id)

            self.logger.info(f"Resolved DAG state: {dag_state}")
            if dag_state not in ("completed", "failed"):
                self.logger.debug(f"DAG is still in progress: {work_info.dag_id}")
                return False

            if work_info.dag_id in self.active_dags:
                del self.active_dags[work_info.dag_id]
                self.logger.debug(
                    f"Removed DAG from cache: {work_info.dag_id}, size = {len(self.active_dags)}"
                )

            self.logger.info(
                f"Resolved DAG status: {work_info.dag_id}, status={dag_state}, active_dag = {len(self.active_dags)}"
            )
            # notification
            event_name = work_info.data.get("name", work_info.name)
            api_key = work_info.data.get("api_key", None)
            metadata = work_info.data.get("metadata", {})
            ref_type = metadata.get("ref_type")

            if not api_key or not event_name:
                self.logger.warning(
                    f"Missing API key or event name: api_key={api_key}, event_name={event_name}"
                )
                raise ValueError(
                    f"Missing API key or event name: api_key={api_key}, event_name={event_name}"
                )

            status = "OK" if dag_state == "completed" else "FAILED"

            if status == "OK":
                fs = await self.frontier.finalize_dag(work_info.dag_id)

            await mark_as_complete_toast(
                api_key=api_key,
                job_id=work_info.dag_id,
                event_name=event_name,
                job_tag=ref_type,
                status=status,
                timestamp=int(time.time()),
                payload=metadata,
            )

            self.logger.debug(
                f"DAG notification sent: {work_info.dag_id}, status={status}"
            )
            return True
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error resolving DAG status: {error}")
            raise error

    async def get_dag_by_id(self, dag_id: str) -> QueryPlan | None:
        """
        Retrieves a DAG by its ID, using in-memory cache if available.
        Falls back to loading from db if missing.
        Delegates to DAGManagementService.
        """
        return await self.dag_service.get_dag(dag_id)

    def get_available_slots(self) -> dict[str, int]:
        return available_slots_by_executor(ClusterState.deployments)

    async def reset_active_dags(self):
        """
        Reset the active DAGs dictionary, clearing all currently tracked DAGs.
        This can be useful for debugging or when you need to force a fresh state.
        Delegates to DAGManagementService.

        Returns:
            dict: Information about the reset operation including count of cleared DAGs
        """
        return await self.dag_service.reset_all_dags()

    async def _lease_jobs_db(self, job_name: str, ids: list[str]) -> set[str]:
        """
        Try to lease the given job ids for this scheduler instance in the DB.
        Returns the subset of ids that were successfully leased.
        """
        if not ids:
            return set()

        # Skip leasing if not in distributed mode
        if not self.distributed_scheduler:
            return set(ids)

        return await self.repository.lease_jobs(
            job_ids=ids,
            owner=self.lease_owner,
            ttl_seconds=self.lease_ttl_seconds,
            job_name=job_name,
        )

    async def _activate_from_lease_db(self, ids: list[str]) -> set[str]:
        """
        Promote leased jobs to active in DB once dispatch is acknowledged.
        """
        if not ids:
            return set()

        # Skip activation if not in distributed mode
        if not self.distributed_scheduler:
            return set(ids)

        return await self.repository.activate_from_lease(
            job_ids=ids, owner=self.lease_owner, run_ttl_seconds=self.run_ttl_seconds
        )

    async def _release_lease_db(self, ids: list[str]) -> set[str]:
        """
        Release DB leases for the given job ids if dispatch fails or needs retry.
        """
        if not ids:
            return set()

        # Skip release if not in distributed mode
        if not self.distributed_scheduler:
            return set(ids)

        return await self.repository.release_lease(job_ids=ids)

    async def hydrate_single_dag_from_db(self, dag_id: str) -> bool:
        """
        Hydrate a specific DAG from the database into the MemoryFrontier.
        Delegates to DAGManagementService.

        :param dag_id: The ID of the DAG to hydrate
        :return: True if DAG was hydrated, False if not found or failed
        """
        return await self.dag_service.hydrate_single_dag(dag_id)

    async def hydrate_from_db(
        self,
        dag_batch_size: int = 1000,
        itersize: int = 5000,
        log_every_seconds: float = 2.0,
    ) -> None:
        """
        Rebuild MemoryFrontier from DB in two phases with progress & timing logs.
        Delegates to DAGManagementService.

        :param dag_batch_size: Number of DAGs to process in each batch
        :param itersize: Cursor iteration size for streaming
        :param log_every_seconds: How often to log progress
        """
        await self.dag_service.hydrate_bulk(
            dag_batch_size=dag_batch_size,
            itersize=itersize,
            log_every_seconds=log_every_seconds,
        )

    async def __monitor_deployment_updates(self):
        """
        Reactively monitors the ClusterState update event and wakes up the
        _poll loop whenever a deployment's state changes.
        """
        self.logger.info("Starting deployment update monitor.")
        while self.running:
            try:
                await ClusterState.deployment_update_event.wait()
                self.logger.debug(
                    "Deployment update event received, notifying scheduler."
                )
                await self.notify_event()
            except asyncio.CancelledError:
                self.logger.warning("Deployment update monitor task cancelled.")
                break
            except Exception as e:
                self.logger.error(
                    f"Error in deployment update monitor: {e}", exc_info=True
                )
                await asyncio.sleep(5)
