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
from typing import Any, Dict, List, Optional

import psycopg

from marie.excepts import BadConfigSource, RuntimeFailToStart
from marie.helper import get_or_reuse_loop
from marie.job.common import JobInfo, JobStatus
from marie.job.job_manager import JobManager
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.messaging import mark_as_complete as mark_as_complete_toast
from marie.messaging import mark_as_failed as mark_as_failed_toast
from marie.messaging import mark_as_started as mark_as_started_toast
from marie.query_planner.base import (
    QueryPlan,
)
from marie.query_planner.branching import (
    BranchQueryDefinition,
    SkipReason,
    SwitchQueryDefinition,
)
from marie.query_planner.builtin import register_all_known_planners
from marie.query_planner.guardrail import GuardrailQueryDefinition
from marie.query_planner.model import QueryPlannersConf
from marie.scheduler.branch_evaluator import BranchEvaluationContext, BranchEvaluator
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.fixtures import *
from marie.scheduler.global_execution_planner import GlobalPriorityExecutionPlanner
from marie.scheduler.guardrail_evaluator import (
    GuardrailEvaluationContext,
    GuardrailEvaluator,
)
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.job_scheduler import JobScheduler, JobSubmissionRequest
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import (
    ExistingWorkPolicy,
    HeartbeatConfig,
    RecoveredRunLease,
    WorkInfo,
)
from marie.scheduler.planner_util import (
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
    frontier_candidate_window,
    frontier_slot_filter,
    is_control_flow_entrypoint,
    ordered_leased_jobs,
)
from marie.serve.discovery.registry import _is_known_connection_error
from marie.serve.runtimes.servers.cluster_state import ClusterState
from marie.state.semaphore_store import SemaphoreStore
from marie.state.slot_capacity_manager import SlotCapacityManager
from marie.storage.database.postgres import PostgresqlMixin
from marie.utils.scheduler_trace import scheduler_trace
from marie.utils.utils import current_milli_time

INIT_POLL_PERIOD = 0.5  # initial idle wait before the first scheduler wake
SHORT_POLL_INTERVAL = 0.250  # fallback wait when a wake is missed or no work is visible
SLOT_POLL_INTERVAL = 0.100  # busy wait while executor work is blocked only by slots

MIN_POLL_PERIOD = 0.250
MAX_POLL_PERIOD = 8
CONTROL_FLOW_DRAIN_MAX_PASSES = 8

MONITORING_POLL_PERIOD = 5.0  # 5s
SYNC_POLL_PERIOD = 60.0  # 60s — safety net, not primary dispatch path

DEFAULT_SCHEMA = "marie_scheduler"
DEFAULT_JOB_TABLE = "job"
RUN_LEASE_EXTEND_STALE_ATTEMPT_TOTAL = "run_lease_extend_stale_attempt_total"
TERMINAL_EVENT_STALE_ATTEMPT_TOTAL = "terminal_event_stale_attempt_total"
RUN_LEASE_RECOVERED_RETRY_TOTAL = "run_lease_recovered_retry_total"
RUN_LEASE_RECOVERED_FAILED_TOTAL = "run_lease_recovered_failed_total"


def limit_planned_jobs_to_available_slots(
    planned: List[tuple[str, WorkInfo]],
    slots_by_executor: Dict[str, int],
) -> List[tuple[str, WorkInfo]]:
    """
    Keep planner order, but never select more regular jobs than the current
    executor-slot snapshot can actually run.

    This avoids taking and DB-leasing a large tail of jobs that will be
    immediately released once we discover there is no remaining capacity for
    their executor.
    """
    if not planned:
        return []

    remaining = {
        executor: max(0, int(count)) for executor, count in slots_by_executor.items()
    }
    selected: List[tuple[str, WorkInfo]] = []

    for entrypoint, wi in planned:
        executor = entrypoint.split("://", 1)[0]
        if executor == "noop":
            selected.append((entrypoint, wi))
            continue

        if remaining.get(executor, 0) <= 0:
            continue

        remaining[executor] -= 1
        selected.append((entrypoint, wi))

    return selected


def regular_candidates_cover_available_slots(
    candidates: List[WorkInfo], slots_by_executor: Dict[str, int]
) -> bool:
    remaining = {
        executor: max(0, int(count))
        for executor, count in slots_by_executor.items()
        if int(count) > 0
    }
    if not remaining:
        return bool(candidates)

    for wi in candidates:
        metadata = wi.data.get("metadata", {}) if isinstance(wi.data, dict) else {}
        entrypoint = metadata.get("on", "") if isinstance(metadata, dict) else ""
        executor = entrypoint.split("://", 1)[0]
        if remaining.get(executor, 0) > 0:
            remaining[executor] -= 1

    return all(count <= 0 for count in remaining.values())


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
        self._paused = False
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
        self._scheduler_counters = defaultdict(int)

        self.scheduler_mode = config.get(
            "scheduler_mode", "parallel"
        )  # "serial" or "parallel"
        self.distributed_scheduler = config.get("distributed_scheduler", True)
        if self.distributed_scheduler is False:
            raise BadConfigSource(
                "distributed_scheduler=false is no longer supported; "
                "durable DB leasing is required."
            )

        self._event_queue = Queue()
        self._status_update_lock = AsyncJobLock()
        self._dag_resolution_lock = AsyncJobLock()
        self._terminal_dag_states: dict[str, str] = {}

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

        self.sla_priority_interval_seconds = max(
            1, int(config.get("sla_priority_interval_seconds", 15 * 60))
        )

        # Initialize scheduler state (frontier and active_dags)
        self.frontier = MemoryFrontier(
            sla_priority_interval_seconds=self.sla_priority_interval_seconds
        )
        self.active_dags = {}
        self._dag_admission_lock = asyncio.Lock()

        dag_config = config.get("dag_manager", {})
        self.max_concurrent_dags = int(dag_config.get("max_concurrent_dags", 16))
        if self.max_concurrent_dags <= 0:
            raise BadConfigSource(
                "dag_manager.max_concurrent_dags must be greater than zero"
            )
        self._dag_resolution_retry_limit = int(
            dag_config.get("dag_resolution_retry_limit", 3)
        )
        self._dag_resolution_retry_delay = float(
            dag_config.get("dag_resolution_retry_delay", 1.0)
        )
        self._dag_resolution_retry_backoff = bool(
            dag_config.get("dag_resolution_retry_backoff", True)
        )
        self._dag_resolution_retry_max_delay = float(
            dag_config.get("dag_resolution_retry_max_delay", 30.0)
        )

        # Initialize DAGManagementService for DAG lifecycle management
        # Service operates on scheduler's frontier and active_dags
        self.dag_service = DAGManagementService(
            repository=self.repository,
            frontier=self.frontier,
            active_dags=self.active_dags,
            loop=self._loop,
            executor=self._db_executor,
            notify_callback=self.notify_event,
            max_active_dags=self.max_concurrent_dags,
            admission_lock=self._dag_admission_lock,
            slot_snapshot_provider=self.get_available_slots,
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
            recovery_callback=self._reconcile_recovered_run_leases,
            maintenance_interval=maintenance_interval,
        )

        self.execution_planner = GlobalPriorityExecutionPlanner(
            sla_priority_interval_seconds=self.sla_priority_interval_seconds
        )
        self.logger.info(
            "SLA priority interval configured to %ss",
            self.sla_priority_interval_seconds,
        )
        register_all_known_planners(
            QueryPlannersConf.from_dict(config.get("query_planners", {}))
        )

        self.branch_evaluator = BranchEvaluator()
        self.guardrail_evaluator = GuardrailEvaluator()

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

        self._start_time = datetime.now(timezone.utc)
        self.hard_sla_policy = str(config.get("hard_sla_policy", "track_only")).lower()
        if self.hard_sla_policy not in {
            "track_only",
            "escalate_only",
            "expire_unfinished",
        }:
            self.logger.warning(
                f"Unknown hard_sla_policy='{self.hard_sla_policy}', falling back to 'track_only'"
            )
            self.hard_sla_policy = "track_only"
        self.sla_warning_top_n = int(config.get("sla_warning_top_n", 5))
        self.priority_refresh_interval = int(
            config.get("priority_refresh_interval", 10)
        )
        self.priority_refresh_interval_seconds = float(
            config.get("priority_refresh_interval_seconds", 5.0)
        )
        self.priority_refresh_hydrate_limit = int(
            config.get("priority_refresh_hydrate_limit", 100)
        )
        self._priority_refresh_seq = 0
        self._next_priority_refresh_at = (
            time.monotonic() + self.priority_refresh_interval_seconds
        )

        self.frontier_batch_size = int(dag_config.get("frontier_batch_size", 1000))
        self.lease_ttl_seconds: int = int(config.get("lease_ttl_seconds", 5))
        self.run_ttl_seconds: int = int(config.get("run_ttl_seconds", 60))
        # unique, stable lease owner for this scheduler instance
        self.lease_owner: str = f"{socket.gethostname()}:{_uuid.uuid4()}"
        self.gateway_instance_id: str = str(
            config.get("gateway_instance_id") or self.lease_owner
        )
        self.logger.info(
            f"Lease config: lease_ttl_seconds={self.lease_ttl_seconds}, "
            f"run_ttl_seconds={self.run_ttl_seconds}, owner='{self.lease_owner}', "
            f"gateway_instance_id='{self.gateway_instance_id}'"
        )

        self._job_cache = {}
        self._job_cache_max_size = 5000

        # Semaphore-based capacity control, we hijaced the _etcd_client client here from job manager
        self._semaphore_store = SemaphoreStore(
            self.job_manager._etcd_client, default_lease_ttl=30
        )
        self._sem_default_ttl = 30
        self._sem_owner_prefix = f"{socket.gethostname()}"
        self._sem_owner_prefix = ""

        self.capacity_manager = SlotCapacityManager(
            semaphore_store=self._semaphore_store,
            logger=self.logger,
            # Optional mapping if slot types differ from executor names:
            # slot_type_resolver=lambda executor: {"extract_executor": "ocr.gpu"}.get(executor, executor),
        )
        self.cycle_log_every = 10

    def _ha_trace_fields(self) -> dict[str, str]:
        return {
            "gateway_instance_id": self.gateway_instance_id,
            "scheduler_lease_owner": self.lease_owner,
        }

    def validate_config(self, config: Dict[str, Any]):
        # TODO :Implement full validation of required fields
        required_keys = ["queue_names"]
        for key in required_keys:
            if key not in config:
                raise BadConfigSource(f"Missing required config: {key}")

    def _scheduler_counter(self, name: str, **fields: Any) -> None:
        counters = getattr(self, "_scheduler_counters", None)
        if counters is None:
            counters = defaultdict(int)
            self._scheduler_counters = counters

        counters[name] += 1
        scheduler_trace(name, count=counters[name], **fields)

    async def _record_terminal_attempt_audit(
        self,
        *,
        job_id: str,
        work_item: WorkInfo,
        status: JobStatus,
        run_owner: str,
        run_attempt_id: str,
        terminal_work_state: str | None,
        source: str,
        accepted: bool,
        reject_reason: str | None = None,
    ) -> None:
        try:
            await self.repository.record_job_attempt_terminal(
                job_id=job_id,
                job_name=work_item.name,
                dag_id=str(work_item.dag_id),
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                scheduler_lease_owner=self.lease_owner,
                gateway_instance_id=self.gateway_instance_id,
                terminal_status=status.value,
                terminal_work_state=terminal_work_state,
                source=source,
                accepted=accepted,
                reject_reason=reject_reason,
            )
        except Exception as audit_error:
            scheduler_trace(
                "job_attempt_audit_failed",
                job_id=job_id,
                dag_id=work_item.dag_id,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                source=source,
                accepted=accepted,
                error=repr(audit_error),
                **self._ha_trace_fields(),
            )
            self.logger.warning(
                f"Failed to record terminal audit for attempt {run_attempt_id}: {audit_error}"
            )

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
            run_owner = message.get("run_owner")
            run_attempt_id = message.get("run_attempt_id")

            # Track actual work state for failures (may be 'retry' or 'failed')
            actual_work_state: Optional[str] = None

            if status == JobStatus.PENDING:
                self.logger.debug(f"Job pending : {job_id}")
            elif status == JobStatus.SUCCEEDED:
                if not run_owner or not run_attempt_id:
                    scheduler_trace(
                        "job_terminal_attempt_rejected",
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        status=status.value,
                        reason="missing_attempt",
                        **self._ha_trace_fields(),
                    )
                    self.logger.warning(
                        f"Ignoring terminal job event without run attempt: job_id={job_id}, status={status}"
                    )
                    return

                completed = await self.complete(
                    job_id,
                    work_item,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                )
                if completed <= 0:
                    scheduler_trace(
                        "job_terminal_attempt_rejected",
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        status=status.value,
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        reason="db_update_zero_rows",
                        **self._ha_trace_fields(),
                    )
                    await self._record_terminal_attempt_audit(
                        job_id=job_id,
                        work_item=work_item,
                        status=status,
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        terminal_work_state=None,
                        source="job_event",
                        accepted=False,
                        reject_reason="db_update_zero_rows",
                    )
                    self._scheduler_counter(
                        TERMINAL_EVENT_STALE_ATTEMPT_TOTAL,
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        status=status.value,
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        source="job_event",
                    )
                    return
                work_item.state = WorkState.COMPLETED
                self._job_cache[job_id] = work_item
                await self._record_terminal_attempt_audit(
                    job_id=job_id,
                    work_item=work_item,
                    status=status,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    terminal_work_state=WorkState.COMPLETED.value,
                    source="job_event",
                    accepted=True,
                )
                scheduler_trace(
                    "job_terminal_attempt_accepted",
                    job_id=job_id,
                    dag_id=work_item.dag_id,
                    status=status.value,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    **self._ha_trace_fields(),
                )
                await self._handle_successful_job_completion(job_id, work_item)
            elif status == JobStatus.FAILED:
                if not run_owner or not run_attempt_id:
                    scheduler_trace(
                        "job_terminal_attempt_rejected",
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        status=status.value,
                        reason="missing_attempt",
                        **self._ha_trace_fields(),
                    )
                    self.logger.warning(
                        f"Ignoring failed job event without run attempt: job_id={job_id}"
                    )
                    return

                actual_work_state = await self.fail(
                    job_id,
                    work_item,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                )
                if actual_work_state is None:
                    scheduler_trace(
                        "job_terminal_attempt_rejected",
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        status=status.value,
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        reason="db_update_zero_rows",
                        **self._ha_trace_fields(),
                    )
                    await self._record_terminal_attempt_audit(
                        job_id=job_id,
                        work_item=work_item,
                        status=status,
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        terminal_work_state=None,
                        source="job_event",
                        accepted=False,
                        reject_reason="db_update_zero_rows",
                    )
                    self._scheduler_counter(
                        TERMINAL_EVENT_STALE_ATTEMPT_TOTAL,
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        status=status.value,
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        source="job_event",
                    )
                    return
                work_item.state = WorkState(actual_work_state)
                self._job_cache[job_id] = work_item
                await self._record_terminal_attempt_audit(
                    job_id=job_id,
                    work_item=work_item,
                    status=status,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    terminal_work_state=actual_work_state,
                    source="job_event",
                    accepted=True,
                )
                scheduler_trace(
                    "job_terminal_attempt_accepted",
                    job_id=job_id,
                    dag_id=work_item.dag_id,
                    status=status.value,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    final_state=actual_work_state,
                    **self._ha_trace_fields(),
                )
                if actual_work_state == WorkState.RETRY.value:
                    await self.frontier.on_job_retry(job_id, work_item)
                else:
                    await self.frontier.on_job_failed(job_id)
            elif status == JobStatus.STOPPED:
                await self.cancel_job(job_id, work_item)
                work_item.state = work_state
                self._job_cache[job_id] = work_item
                await self.frontier.on_job_cancelled(job_id)
            elif status == JobStatus.RUNNING:
                if not run_owner or not run_attempt_id:
                    scheduler_trace(
                        "run_lease_extend_rejected",
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        status=status.value,
                        reason="missing_attempt",
                        **self._ha_trace_fields(),
                    )
                    self.logger.warning(
                        f"Ignoring running job event without run attempt: job_id={job_id}"
                    )
                    return

                extended = await self._extend_run_lease_db(
                    [job_id], run_owner=run_owner, run_attempt_id=run_attempt_id
                )
                if job_id not in extended:
                    scheduler_trace(
                        "run_lease_extend_rejected",
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        status=status.value,
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        reason="db_update_zero_rows",
                        **self._ha_trace_fields(),
                    )
                    self._scheduler_counter(
                        RUN_LEASE_EXTEND_STALE_ATTEMPT_TOTAL,
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        status=status.value,
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        source="job_event",
                    )
                    return

                work_item.state = work_state
                work_item.run_owner = run_owner
                work_item.run_attempt_id = run_attempt_id
                self._job_cache[job_id] = work_item
                await self.frontier.update_job_state(job_id, work_state)
                self.logger.debug(f"Job running : {job_id}")
            else:
                self.logger.error(f"Unhandled job status: {status}. Marking as FAILED.")
                actual_work_state = await self.fail(
                    job_id,
                    work_item,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                )  # Fail-safe
                if actual_work_state is None:
                    return
                work_item.state = WorkState(actual_work_state)
                self._job_cache[job_id] = work_item
                if actual_work_state == WorkState.RETRY.value:
                    await self.frontier.on_job_retry(job_id, work_item)
                else:
                    await self.frontier.on_job_failed(job_id)

            # Only resolve DAG status for truly terminal states
            # For FAILED jobs, check if they went to 'failed' (not 'retry')
            is_truly_terminal = (
                status == JobStatus.SUCCEEDED
                or status == JobStatus.STOPPED
                or (
                    status == JobStatus.FAILED
                    and actual_work_state == WorkState.FAILED.value
                )
            )

            if is_truly_terminal:
                self.logger.debug(
                    f"Job is in terminal state {status} (work_state={actual_work_state}), job_id: {job_id}"
                )

                self._status_update_lock.release(job_id)
                await self._resolve_dag_status_with_retry(
                    job_id,
                    work_item,
                    now,
                    now,
                    source="job_event",
                )
                await self.notify_event()
            elif (
                status == JobStatus.FAILED
                and actual_work_state == WorkState.RETRY.value
            ):
                self.logger.info(f"Job {job_id} will be retried, DAG remains active")
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

    def _is_guardrail_node(self, node) -> bool:
        """Check if a node is a GUARDRAIL node."""
        if not node or not hasattr(node, 'definition'):
            return False

        return isinstance(node.definition, GuardrailQueryDefinition)

    async def _handle_successful_job_completion(
        self, job_id: str, work_item: WorkInfo
    ) -> None:
        await self.frontier.on_job_completed(job_id)

        dag_plan = await self.get_dag_by_id(work_item.dag_id)
        if not dag_plan:
            return

        node = get_node_from_dag(job_id, dag_plan)
        if node and self._is_branch_node(node):
            self.logger.info(
                f"Completed branch node detected: {job_id}. Evaluating paths..."
            )
            await self._evaluate_and_mark_branch_paths(job_id, work_item, dag_plan)
        elif node and self._is_guardrail_node(node):
            self.logger.info(
                f"Completed guardrail node detected: {job_id}. Evaluating metrics..."
            )
            await self._evaluate_and_mark_guardrail_paths(job_id, work_item, dag_plan)

    async def _activate_control_flow_job(self, wi: WorkInfo) -> bool:
        if (
            wi.state == WorkState.ACTIVE
            and wi.run_owner == self.lease_owner
            and wi.run_attempt_id
        ):
            self._job_cache[wi.id] = wi
            await self.frontier.update_job_state(wi.id, WorkState.ACTIVE)
            return True

        activated = await self._activate_from_lease_db([wi.id])
        marked_active = wi.id in activated
        if marked_active:
            wi.run_owner = self.lease_owner
            wi.run_attempt_id = activated[wi.id]
            wi.state = WorkState.ACTIVE
            self._job_cache[wi.id] = wi

        if not marked_active:
            self.logger.error(
                f"[CONTROL_FLOW] Failed to mark control flow node {wi.id} active"
            )
            return False

        await self.frontier.update_job_state(wi.id, WorkState.ACTIVE)
        return True

    async def _complete_control_flow_attempt(self, wi: WorkInfo) -> bool:
        if not wi.run_owner or not wi.run_attempt_id:
            self.logger.error(
                f"[CONTROL_FLOW] Missing run attempt for control flow node {wi.id}"
            )
            scheduler_trace(
                "control_flow_terminal_rejected",
                job_id=wi.id,
                dag_id=wi.dag_id,
                reason="missing_attempt",
            )
            return False

        completed = await self.complete(
            wi.id,
            wi,
            {},
            run_owner=wi.run_owner,
            run_attempt_id=wi.run_attempt_id,
        )
        if completed:
            return True

        self.logger.warning(
            f"[CONTROL_FLOW] Terminal update rejected for control flow node {wi.id} "
            f"(run_owner={wi.run_owner}, run_attempt_id={wi.run_attempt_id})"
        )
        scheduler_trace(
            "control_flow_terminal_rejected",
            job_id=wi.id,
            dag_id=wi.dag_id,
            reason="attempt_mismatch",
            run_owner=wi.run_owner,
            run_attempt_id=wi.run_attempt_id,
        )
        return False

    async def _reconcile_control_flow_lease_miss(
        self, wi: WorkInfo, db_wi: WorkInfo | None
    ) -> bool:
        return await self._reconcile_db_lease_miss(wi, db_wi, context="control_flow")

    async def _reconcile_db_lease_miss(
        self, wi: WorkInfo, db_wi: WorkInfo | None, *, context: str
    ) -> bool:
        if db_wi is None:
            await self._evict_missing_db_work_item(wi, context=context)
            return True

        self._job_cache[wi.id] = db_wi
        state = db_wi.state
        trace_name = (
            "control_flow_db_lease_miss_reconciled"
            if context == "control_flow"
            else "db_lease_miss_reconciled"
        )
        scheduler_trace(
            trace_name,
            job_id=wi.id,
            dag_id=wi.dag_id,
            db_state=state.value if isinstance(state, WorkState) else str(state),
            context=context,
        )

        if state == WorkState.COMPLETED:
            await self.frontier.on_job_completed(wi.id)
            return True
        if state == WorkState.SKIPPED:
            await self.frontier.on_job_skipped(wi.id)
            return True
        if state in (WorkState.FAILED, WorkState.EXPIRED):
            await self.frontier.on_job_failed(wi.id)
            return True
        if state == WorkState.CANCELLED:
            await self.frontier.on_job_cancelled(wi.id)
            return True
        if state == WorkState.ACTIVE:
            await self.frontier.update_job_state(wi.id, WorkState.ACTIVE)
            await self.frontier.release_lease_local(wi.id)
            return True

        await self.frontier.release_lease_local(wi.id)
        return False

    async def _evict_missing_db_work_item(self, wi: WorkInfo, *, context: str) -> bool:
        self._job_cache.pop(wi.id, None)
        reason = f"{context}: work item {wi.id} missing from database"
        self.logger.warning(
            f"[WORK_DIST] Evicting stale in-memory work item {wi.id} "
            f"(dag_id={wi.dag_id}, context={context}); database row is missing"
        )
        scheduler_trace(
            "db_missing_work_item_evicted",
            job_id=wi.id,
            dag_id=wi.dag_id,
            context=context,
        )

        if wi.dag_id:
            removed = await self._evict_dag_from_memory(wi.dag_id, reason)
        else:
            await self.frontier.on_job_cancelled(wi.id)
            removed = True

        await self.notify_event()
        return removed

    async def _reconcile_db_lease_shortfall(
        self, selected_wis: list[WorkInfo], leased_ids: set[str]
    ) -> int:
        reconciled = 0
        evicted_dag_ids: set[str] = set()
        for wi in selected_wis:
            if wi.id in leased_ids:
                continue

            try:
                db_wi = await self.repository.get_job_by_id(wi.id)
            except Exception as exc:
                self.logger.error(
                    f"[WORK_DIST] Failed to inspect DB lease shortfall for {wi.id}: {exc}",
                    exc_info=True,
                )
                await self.frontier.release_lease_local(wi.id)
                continue

            reconciled += int(
                await self._reconcile_db_lease_miss(
                    wi, db_wi, context="dispatch_shortfall"
                )
            )
            if db_wi is None and wi.dag_id:
                evicted_dag_ids.add(wi.dag_id)

        if evicted_dag_ids and leased_ids:
            stale_leased_ids = [
                wi.id
                for wi in selected_wis
                if wi.id in leased_ids and wi.dag_id in evicted_dag_ids
            ]
            if stale_leased_ids:
                try:
                    await self._release_lease_db(stale_leased_ids)
                except Exception as exc:
                    self.logger.error(
                        f"[WORK_DIST] Failed to release leased jobs from evicted DAGs "
                        f"{sorted(evicted_dag_ids)}: {exc}",
                        exc_info=True,
                    )
                leased_ids.difference_update(stale_leased_ids)

        return reconciled

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
            scheduler_trace(
                "control_flow_started",
                job_id=wi.id,
                dag_id=dag_id,
                node_type=node_type,
                job_name=wi.name,
                job_level=wi.job_level,
            )

            self.logger.debug(
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

                admitted = await self._admit_dag(
                    wi, dag, source=f"control_flow:{node_type}"
                )
                if not admitted:
                    await self._release_lease_db([wi.id])
                    await self.frontier.release_lease_local(wi.id)
                    return

            if not await self._activate_control_flow_job(wi):
                await self._release_lease_db([wi.id])
                await self.frontier.release_lease_local(wi.id)
                return

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
                    timestamp=current_milli_time(),
                    payload=metadata,
                )

            # Handle based on node type
            if node_type in ("branch", "switch"):
                # BRANCH/SWITCH nodes need evaluation
                self.logger.info(
                    f"[CONTROL_FLOW] Evaluating {node_type} paths for {wi.id}"
                )

                # Complete the branch node first
                if not await self._complete_control_flow_attempt(wi):
                    return

                # Evaluate and mark paths
                await self._evaluate_and_mark_branch_paths(
                    wi.id, wi, self.active_dags[dag_id]
                )

            elif node_type == "guardrail":
                # GUARDRAIL nodes need quality validation evaluation
                self.logger.info(
                    f"[CONTROL_FLOW] Evaluating guardrail metrics for {wi.id}"
                )

                # Complete the guardrail node first
                if not await self._complete_control_flow_attempt(wi):
                    return

                # Evaluate and mark paths based on pass/fail
                await self._evaluate_and_mark_guardrail_paths(
                    wi.id, wi, self.active_dags[dag_id]
                )

            elif node_type == "noop":
                # NOOP nodes just complete
                self.logger.debug(f"[CONTROL_FLOW] Completing NOOP node {wi.id}")
                if not await self._complete_control_flow_attempt(wi):
                    return

            elif node_type == "merger":
                # MERGER nodes wait for branches to complete via dependencies
                # The actual merge logic is handled by the dependency system
                # MERGER can complete immediately - dependencies prevent it from
                # running until all required branches are done
                self.logger.debug(
                    f"[CONTROL_FLOW] Completing MERGER node {wi.id} "
                    "(merge logic handled by dependencies)"
                )
                if not await self._complete_control_flow_attempt(wi):
                    return

            else:
                self.logger.warning(
                    f"[CONTROL_FLOW] Unknown control flow type: {node_type} for {wi.id}"
                )
                if not await self._complete_control_flow_attempt(wi):
                    return

            # Clean up
            self.frontier.leased_until.pop(wi.id, None)
            await self.frontier.on_job_completed(wi.id)
            await self.notify_event()

            # Check if DAG is complete (leaf node check)
            if job_levels.get(wi.id, -1) == min(job_levels.values()):
                await self._resolve_dag_status_with_retry(
                    wi.id,
                    wi,
                    source="control_flow",
                )

            self.logger.debug(
                f"[CONTROL_FLOW] Successfully processed {node_type} node {wi.id}"
            )
            scheduler_trace(
                "control_flow_completed",
                job_id=wi.id,
                dag_id=dag_id,
                node_type=node_type,
                job_name=wi.name,
                job_level=wi.job_level,
            )

        except Exception as e:
            scheduler_trace(
                "control_flow_failed",
                job_id=wi.id,
                dag_id=wi.dag_id,
                job_name=wi.name,
                error=repr(e),
            )
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

    async def _process_control_flow_candidates(
        self, control_flow_jobs: list[WorkInfo], lease_ttl: float
    ) -> int:
        if not control_flow_jobs:
            return 0

        tasks: list[asyncio.Task[None]] = []
        reconciled_count = 0

        for wi in control_flow_jobs:
            taken_wis = await self.frontier.take([wi.id], lease_ttl=lease_ttl)
            if not taken_wis:
                self.logger.warning(
                    f"[WORK_DIST] Failed to take control flow node {wi.id} from frontier"
                )
                continue

            try:
                leased_ids = await self._lease_jobs_db(wi.name, [wi.id])
            except Exception as e:
                self.logger.error(
                    f"[WORK_DIST] Error leasing control flow node {wi.id}: {e}"
                )
                await self.frontier.release_lease_local(wi.id)
                continue

            if not leased_ids:
                db_wi = await self.repository.get_job_by_id(wi.id)
                db_state = db_wi.state if db_wi else None
                self.logger.warning(
                    f"[WORK_DIST] Failed to lease control flow node {wi.id} in DB "
                    f"(frontier_state={wi.state}, db_state={db_state})"
                )
                if (
                    db_wi
                    and db_wi.state == WorkState.ACTIVE
                    and db_wi.run_owner == self.lease_owner
                    and db_wi.run_attempt_id
                ):
                    await self.frontier.update_job_state(wi.id, WorkState.ACTIVE)
                    tasks.append(
                        asyncio.create_task(self._process_control_flow_node(db_wi))
                    )
                    continue

                reconciled = await self._reconcile_control_flow_lease_miss(wi, db_wi)
                reconciled_count += int(reconciled)
                continue

            tasks.append(asyncio.create_task(self._process_control_flow_node(wi)))

        if not tasks:
            return reconciled_count

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(
                    f"[WORK_DIST] Control flow drain task failed: {result}",
                    exc_info=(type(result), result, result.__traceback__),
                )

        return len(tasks) + reconciled_count

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
                "skip_reason": skip_reason.model_dump(),
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

            # Update frontier to mark these as skipped (without unblocking children)
            for node_id in node_ids:
                await self.frontier.on_job_skipped(node_id)

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

    async def _evaluate_and_mark_guardrail_paths(
        self, guardrail_node_id: str, work_item: WorkInfo, dag_plan: QueryPlan
    ) -> None:
        """
        Evaluate a guardrail node and mark its paths as READY or SKIPPED.
        Stores guardrail_metadata for tracking and debugging.

        :param guardrail_node_id: ID of the completed guardrail node
        :param work_item: WorkInfo of the guardrail node
        :param dag_plan: The DAG plan containing the guardrail
        """
        try:
            self.logger.info(
                f"Evaluating guardrail metrics for node: {guardrail_node_id}"
            )

            # Get the guardrail node from the DAG
            guardrail_node = get_node_from_dag(guardrail_node_id, dag_plan)
            if not guardrail_node or not self._is_guardrail_node(guardrail_node):
                self.logger.warning(
                    f"Node {guardrail_node_id} is not a guardrail node, skipping evaluation"
                )
                return

            guardrail_def = guardrail_node.definition

            # Build evaluation context
            # TODO: Gather execution results from previous nodes if needed
            execution_results = {}
            context = GuardrailEvaluationContext(
                work_info=work_item,
                dag_plan=dag_plan,
                guardrail_node=guardrail_node,
                execution_results=execution_results,
            )

            # Evaluate guardrail metrics with timeout protection
            try:
                result = await asyncio.wait_for(
                    self.guardrail_evaluator.evaluate(guardrail_def, context),
                    timeout=guardrail_def.evaluation_timeout,
                )
            except asyncio.TimeoutError:
                self.logger.error(
                    f"Guardrail evaluation timed out for {guardrail_node_id} "
                    f"after {guardrail_def.evaluation_timeout}s"
                )
                # Default to fail path on timeout
                pass_path = guardrail_def.get_pass_path()
                fail_path = guardrail_def.get_fail_path()

                from marie.query_planner.guardrail import GuardrailEvaluationResult

                result = GuardrailEvaluationResult(
                    overall_passed=False,
                    overall_score=0.0,
                    individual_results=[],
                    selected_path_id="fail",
                    active_target_nodes=fail_path.target_node_ids if fail_path else [],
                    skipped_target_nodes=pass_path.target_node_ids if pass_path else [],
                    total_execution_time_ms=guardrail_def.evaluation_timeout * 1000,
                    error=f"Evaluation timed out after {guardrail_def.evaluation_timeout}s",
                )

            # Store guardrail metadata for audit
            guardrail_metadata = {
                "node_type": "GUARDRAIL",
                "overall_passed": result.overall_passed,
                "overall_score": result.overall_score,
                "selected_path_id": result.selected_path_id,
                "individual_results": [
                    r.model_dump() for r in result.individual_results
                ],
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "error": result.error,
            }

            await self._update_job_branch_metadata(
                job_id=guardrail_node_id,
                queue_name=work_item.name,
                branch_metadata=guardrail_metadata,
            )

            self.logger.info(
                f"Guardrail evaluation complete: passed={result.overall_passed}, "
                f"score={result.overall_score:.2f}, path={result.selected_path_id}"
            )

            # Mark active paths' target nodes as READY
            if result.active_target_nodes:
                active_path_metadata = {
                    "selected_by_guardrail": guardrail_node_id,
                    "selected_path_id": result.selected_path_id,
                    "guardrail_score": result.overall_score,
                    "selected_at": datetime.now(timezone.utc).isoformat(),
                }
                for node_id in result.active_target_nodes:
                    await self._update_job_branch_metadata(
                        job_id=node_id,
                        queue_name=work_item.name,
                        branch_metadata=active_path_metadata,
                    )

                await self._mark_nodes_ready(result.active_target_nodes, work_item.name)
                self.logger.info(
                    f"Marked {len(result.active_target_nodes)} nodes as READY: "
                    f"{result.active_target_nodes}"
                )

            # Mark skipped nodes as SKIPPED and cascade to descendants
            if result.skipped_target_nodes:
                skip_reason = SkipReason(
                    branch_node_id=guardrail_node_id,
                    reason=f"Guardrail {'passed' if result.overall_passed else 'failed'} "
                    f"(score: {result.overall_score:.2f})",
                    evaluated_condition={
                        "metrics": [r.metric_name for r in result.individual_results]
                    },
                    selected_paths=[result.selected_path_id],
                    timestamp=datetime.now(timezone.utc),
                )
                await self._mark_nodes_skipped(
                    result.skipped_target_nodes,
                    work_item.name,
                    skip_reason,
                    dag_plan,
                )
                self.logger.info(
                    f"Marked {len(result.skipped_target_nodes)} nodes as SKIPPED: "
                    f"{result.skipped_target_nodes}"
                )

        except Exception as e:
            self.logger.error(
                f"Error evaluating guardrail metrics for {guardrail_node_id}: {e}",
                exc_info=True,
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
                self._terminal_dag_states.pop(dag_id, None)
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
                    self._terminal_dag_states.pop(dag_id, None)
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
                    self._terminal_dag_states.setdefault(dag_id, new_state)
                    stats = await self.frontier.finalize_dag(dag_id)
                    self.logger.info(
                        f"Cleaned up finished DAG {dag_id} from memory: {stats}"
                    )

                    if dag_id in self.active_dags:
                        del self.active_dags[dag_id]
                        self.logger.info(
                            f"Removed finished DAG {dag_id} from active_dags"
                        )

                elif new_state in ["active", "running", "pending"]:
                    if dag_id not in self.active_dags:
                        self.logger.debug(
                            f"DAG {dag_id} is '{new_state}' in DB but is not local "
                            "to this scheduler yet; it may be admitted by the current "
                            "cycle, owned by another scheduler, or hydrated later."
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

        await self.repository.validate_durable_scheduler_schema(DEFAULT_SCHEMA)

        # Get defined queues from repository
        defined_queues = await self.repository.get_defined_queues(DEFAULT_SCHEMA)
        for work_queue in self.known_queues.difference(defined_queues):
            self.logger.info(f"Create queue: {work_queue}")
            await self.repository.create_queue(work_queue)
            await self.repository.create_queue(f"${work_queue}_dlq")

        # Start the NotificationService before hydrating or polling so DAG
        # state transitions cannot race ahead of the LISTEN connection.
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

        # We need to display the status
        await self.hydrate_from_db()

        self.running = True
        self.sync_task = asyncio.create_task(self._sync())
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

        # Start the MaintenanceService for periodic cleanup tasks
        try:
            await self.maintenance_service.start()
            self.logger.info(
                f"Started MaintenanceService (interval: {self.maintenance_service.maintenance_interval}s)"
            )
        except Exception as e:
            self.logger.error(f"Error starting MaintenanceService: {e}")
            # Non-critical - continue without maintenance service

        self._sync_dag_task = asyncio.create_task(self._sync_dag())
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
                priority_due_in = max(
                    0.0, self._next_priority_refresh_at - time.monotonic()
                )
                effective_wait_time = min(wait_time, priority_due_in)
                self.logger.debug(
                    f"Polling : {effective_wait_time:.2f}s — Queue size: {self._event_queue.qsize()} — Idle streak: {idle_streak}"
                )
                woke = await self._wait_for_dispatch_wake(effective_wait_time)
                if woke:
                    wait_time = MIN_POLL_PERIOD

                if time.monotonic() >= self._next_priority_refresh_at:
                    refresh_started = time.perf_counter()
                    scheduler_trace(
                        "scheduler_priority_refresh_due",
                        source="scheduler_loop",
                        submission_count=self._submission_count,
                        refresh_interval_seconds=self.priority_refresh_interval_seconds,
                        request_queue_size=self._request_queue.qsize(),
                        pending_requests=len(self._pending_requests),
                    )
                    refresh_id = await self._refresh_job_priorities(
                        source="scheduler_loop"
                    )
                    scheduler_trace(
                        "scheduler_priority_refresh_returned",
                        source="scheduler_loop",
                        refresh_id=refresh_id,
                        submission_count=self._submission_count,
                        refresh_interval_seconds=self.priority_refresh_interval_seconds,
                        request_queue_size=self._request_queue.qsize(),
                        pending_requests=len(self._pending_requests),
                        elapsed_ms=(time.perf_counter() - refresh_started) * 1000.0,
                    )

                reaped_soft_leases = await self.frontier.reap_expired_soft_leases()
                if reaped_soft_leases:
                    self.logger.warning(
                        f"[WORK_DIST] Reaped {reaped_soft_leases} expired local frontier lease(s)"
                    )

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

                # Check if scheduler is paused before dispatching work
                if self._paused:
                    if _cycle_idx % 10 == 0:
                        self.logger.info(
                            f"[WORK_DIST] Scheduler is paused. Skipping dispatch. "
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

                no_executor_slots = not any(slots_by_executor.values())
                self.logger.debug(f"[WORK_DIST] Available slots: {slots_by_executor}")

                # Fetch candidates from frontier.  Even when no executor slots
                # are available we must still peek so that control flow nodes
                # (NOOP/BRANCH/SWITCH) — which do NOT consume slots can be
                # dispatched.  Skipping the peek previously caused a deadlock
                # where noops starved while all executor slots were occupied.
                candidate_window = frontier_candidate_window(
                    batch_size, slots_by_executor
                )
                slot_filter = frontier_slot_filter(slots_by_executor)

                candidates_wi: list[WorkInfo] = []
                regular_candidates: list[WorkInfo] = []
                control_flow_seen_total = 0
                control_flow_processed_total = 0
                control_flow_drain_passes = 0
                no_ready_candidates = False

                def dag_admission_filter(wi: WorkInfo) -> bool:
                    return (
                        wi.dag_id in self.active_dags
                        or len(self.active_dags) < max_concurrent_dags
                    )

                def schedulable_slot_filter(wi: WorkInfo) -> bool:
                    return dag_admission_filter(wi) and slot_filter(wi)

                def regular_slot_filter(wi: WorkInfo) -> bool:
                    ep = wi.data.get("metadata", {}).get("on", "")
                    return (
                        bool(ep)
                        and not is_control_flow_entrypoint(ep)
                        and schedulable_slot_filter(wi)
                    )

                if not no_executor_slots:
                    regular_candidates = await self.frontier.peek_ready(
                        candidate_window,
                        filter_fn=regular_slot_filter,
                    )
                    candidates_wi = regular_candidates

                for drain_pass in range(CONTROL_FLOW_DRAIN_MAX_PASSES):
                    if regular_candidates_cover_available_slots(
                        regular_candidates, slots_by_executor
                    ):
                        break

                    control_flow_drain_passes = drain_pass + 1
                    candidates_wi = await self.frontier.peek_ready(
                        candidate_window,
                        filter_fn=schedulable_slot_filter,
                    )

                    if not candidates_wi:
                        no_ready_candidates = True
                        break

                    control_flow_jobs: list[WorkInfo] = []
                    current_regular_candidates: list[WorkInfo] = []

                    for wi in candidates_wi:
                        ep = wi.data.get("metadata", {}).get("on", "")
                        if not ep:
                            self.logger.error(
                                f"[WORK_DIST] Job without entrypoint 'on': {wi.id}"
                            )
                            continue

                        if is_control_flow_entrypoint(ep):
                            control_flow_jobs.append(wi)
                        else:
                            current_regular_candidates.append(wi)

                    regular_candidates = current_regular_candidates
                    control_flow_seen_total += len(control_flow_jobs)

                    if control_flow_jobs:
                        admission_slots = max(
                            0, max_concurrent_dags - len(self.active_dags)
                        )
                        selected_control_flow: list[WorkInfo] = []
                        selected_new_dags: set[str] = set()

                        for wi in control_flow_jobs:
                            if wi.dag_id in self.active_dags:
                                selected_control_flow.append(wi)
                                continue

                            if (
                                admission_slots > 0
                                and wi.dag_id not in selected_new_dags
                            ):
                                selected_control_flow.append(wi)
                                selected_new_dags.add(wi.dag_id)
                                admission_slots -= 1

                        control_flow_jobs = selected_control_flow

                    if control_flow_jobs:
                        self.logger.debug(
                            f"[WORK_DIST] Draining {len(control_flow_jobs)} control flow nodes "
                            f"(pass {control_flow_drain_passes}/{CONTROL_FLOW_DRAIN_MAX_PASSES})"
                        )
                        processed = await self._process_control_flow_candidates(
                            control_flow_jobs, lease_ttl
                        )
                        control_flow_processed_total += processed
                        scheduled_any = scheduled_any or processed > 0

                    if not no_executor_slots and not regular_candidates:
                        regular_candidates = await self.frontier.peek_ready(
                            candidate_window,
                            filter_fn=regular_slot_filter,
                        )
                        if regular_candidates:
                            candidates_wi = regular_candidates
                            if regular_candidates_cover_available_slots(
                                regular_candidates, slots_by_executor
                            ):
                                break

                    if not control_flow_jobs:
                        break

                if (
                    no_ready_candidates
                    and not regular_candidates
                    and control_flow_processed_total == 0
                ):
                    if no_executor_slots:
                        self.logger.debug(
                            f"[WORK_DIST] No available executor slots and no control flow nodes. Backing off. "
                            f"Slots by executor: {slots_by_executor} | "
                            f"Idle streak: {idle_streak} | "
                            f"Wait time: {wait_time:.2f}s"
                        )
                    else:
                        frontier_summary = self.frontier.summary(detail=False)
                        self.logger.debug(
                            f"[WORK_DIST] No ready work in frontier. Short sleep. "
                            f"Batch size: {batch_size} | "
                            f"Candidate window: {candidate_window} | "
                            f"Frontier summary: {frontier_summary} | "
                            f"Idle streak: {idle_streak} | "
                            f"Wait time: {wait_time:.2f}s"
                        )
                    poll_interval = (
                        SLOT_POLL_INTERVAL if no_executor_slots else SHORT_POLL_INTERVAL
                    )
                    woke = await self._wait_for_dispatch_wake(poll_interval)
                    idle_streak = 0 if woke else idle_streak + 1
                    wait_time = 0.0
                    continue

                self.logger.debug(
                    f"[WORK_DIST] Fetched {len(candidates_wi)} candidates from frontier. "
                )

                # Build (entrypoint, wi) tuples for planner input (only regular jobs)
                planner_candidates: list[tuple[str, WorkInfo]] = []
                for wi in regular_candidates:
                    ep = wi.data.get("metadata", {}).get("on", "")
                    planner_candidates.append((ep, wi))

                self.logger.info(
                    f"[WORK_DIST] Built {len(planner_candidates)} planner candidates from {len(regular_candidates)} regular jobs "
                    f"(+{control_flow_processed_total}/{control_flow_seen_total} control flow nodes processed "
                    f"over {control_flow_drain_passes} drain pass(es)). "
                    f"Executors needed: {set(ep for ep, _ in planner_candidates)}"
                )
                scheduler_trace(
                    "candidate_built",
                    candidates=len(planner_candidates),
                    regular_jobs=len(regular_candidates),
                    control_flow_jobs=control_flow_processed_total,
                    control_flow_seen=control_flow_seen_total,
                    control_flow_drain_passes=control_flow_drain_passes,
                    executors=sorted(
                        {ep.split("://", 1)[0] for ep, _ in planner_candidates}
                    ),
                    slots_by_executor=dict(slots_by_executor),
                    active_dags=len(self.active_dags),
                    max_concurrent_dags=max_concurrent_dags,
                    job_ids=[wi.id for _, wi in planner_candidates],
                )

                # If no regular jobs to plan (either all were control flow or
                # no executor slots were available), skip the planner.
                if not planner_candidates:
                    if scheduled_any:
                        idle_streak = 0
                        wait_time = 0.0
                    elif no_executor_slots:
                        idle_streak += 1
                        wait_time = adjust_backoff(
                            wait_time,
                            idle_streak,
                            scheduled=False,
                            min_poll_period=MIN_POLL_PERIOD,
                        )
                    self.logger.debug(
                        f"[WORK_DIST] No regular jobs to plan (processed {control_flow_processed_total}/"
                        f"{control_flow_seen_total} control flow nodes over {control_flow_drain_passes} drain pass(es))"
                    )
                    continue

                # Give the planner: candidates + a COPY of slots + active_dags
                pick_slots = slots_by_executor.copy()
                dag_remaining = self.frontier.dag_remaining_counts()
                planned: list[tuple[str, WorkInfo]] = self.execution_planner.plan(
                    planner_candidates,
                    pick_slots,
                    self.active_dags,
                    exclude_blocked=True,
                    dag_remaining=dag_remaining,
                )

                await debug_candidates_and_plan(
                    candidates_wi, planned, pick_slots, self.active_dags, self.frontier
                )
                limited_planned = limit_planned_jobs_to_available_slots(
                    planned, pick_slots
                )
                scheduler_trace(
                    "planner_selected",
                    planned=len(planned),
                    limited=len(limited_planned),
                    slots=dict(pick_slots),
                    job_ids=[wi.id for _, wi in limited_planned],
                )
                if len(limited_planned) < len(planned):
                    self.logger.info(
                        f"[WORK_DIST] Trimmed planner selection from {len(planned)} "
                        f"to {len(limited_planned)} based on live slot capacity. "
                        f"Slots: {pick_slots}"
                    )
                planned = limited_planned
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
                    woke = await self._wait_for_dispatch_wake(SHORT_POLL_INTERVAL)
                    idle_streak = 0 if woke else idle_streak + 1
                    wait_time = 0.0
                    continue

                self.logger.debug(
                    f"[WORK_DIST] Planner selected {len(planned)} jobs to schedule. "
                    f"Job IDs: {[wi.id for _, wi in planned[:10]]}"
                )

                # TAKE + SOFT-LEASE
                selected_ids = [wi.id for _, wi in planned]
                selected_wis: List[WorkInfo] = await self.frontier.take(
                    selected_ids, lease_ttl=lease_ttl
                )

                taken = len(selected_wis)
                requested = len(selected_ids)
                scheduler_trace(
                    "frontier_taken",
                    requested=requested,
                    taken=taken,
                    job_ids=[wi.id for wi in selected_wis],
                )
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
                        scheduler_trace(
                            "db_leased",
                            job_name=job_name,
                            requested=len(ids),
                            leased=len(got),
                            job_ids=list(got),
                        )
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
                if len(leased_ids) < len(selected_wis):
                    reconciled_shortfall = await self._reconcile_db_lease_shortfall(
                        selected_wis, leased_ids
                    )
                    if reconciled_shortfall:
                        self.logger.info(
                            f"[WORK_DIST] Reconciled {reconciled_shortfall} DB lease shortfall(s)"
                        )

                if not leased_ids:
                    self.logger.warning(
                        f"[WORK_DIST] NO candidates could be leased in DB; backing off. "
                        f"Attempted {len(selected_wis)} jobs across {len(ids_by_job_name)} job names. "
                        f"Job names: {list(ids_by_job_name.keys())}"
                    )
                    woke = await self._wait_for_dispatch_wake(SHORT_POLL_INTERVAL)
                    idle_streak = 0 if woke else idle_streak + 1
                    wait_time = 0.0
                    continue

                self.logger.info(
                    f"[WORK_DIST] Successfully leased {len(leased_ids)} jobs in DB. "
                    f"Processing leased jobs now..."
                )

                # only process those that we leased in DB
                leased_jobs: list[tuple[str, WorkInfo]] = ordered_leased_jobs(
                    planned, leased_ids
                )

                #  PROCESS LEASED JOBS
                scheduled_any = False
                jobs_scheduled_this_cycle = defaultdict(int)
                enqueue_tasks = []
                reservable_jobs_by_executor: dict[str, list[WorkInfo]] = defaultdict(
                    list
                )
                slots_before_by_job: dict[str, int] = {}

                self.logger.debug(
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

                        admitted = await self._admit_dag(wi, dag, source="dispatch")
                        if not admitted:
                            await self._release_lease_db([wi.id])
                            await self.frontier.release_lease_local(wi.id)
                            continue

                    # NOTE: NOOP/BRANCH/SWITCH nodes are handled earlier in the pipeline
                    # (before planner) and should never reach this point.
                    # If they do, it's a bug - but we'll handle gracefully

                    # Normal job: check slots then dispatch
                    exe = entrypoint.split("://", 1)[0]
                    slots_before = slots_by_executor.get(exe, 0)
                    if slots_before <= 0:
                        self.logger.debug(
                            f"[WORK_DIST] No slots available for executor={exe}, delaying job {wi.id}. "
                            f"Current slots_by_executor: {slots_by_executor}"
                        )
                        await self._release_lease_db([wi.id])
                        await self.frontier.release_lease_local(wi.id)
                        continue

                    slots_before_by_job[wi.id] = slots_before
                    slots_by_executor[exe] = max(0, slots_by_executor.get(exe, 0) - 1)
                    reservable_jobs_by_executor[exe].append(wi)

                for slot_type, jobs in reservable_jobs_by_executor.items():
                    reserved_ids = await self._reserve_semaphore_slots(slot_type, jobs)
                    reserved_jobs: list[WorkInfo] = []
                    for wi in jobs:
                        owner = wi.id
                        if wi.id not in reserved_ids:
                            scheduler_trace(
                                "slot_unavailable",
                                job_id=wi.id,
                                dag_id=wi.dag_id,
                                executor=slot_type,
                                slots_by_executor=dict(slots_by_executor),
                            )
                            self.logger.warning(
                                f"[WORK_DIST] NO semaphore capacity for executor={slot_type}; releasing lease for job={wi.id}. "
                                f"slots_by_executor={slots_by_executor}"
                            )
                            await self._release_lease_db([wi.id])
                            await self.frontier.release_lease_local(wi.id)
                            continue

                        scheduler_trace(
                            "slot_reserved",
                            job_id=wi.id,
                            dag_id=wi.dag_id,
                            executor=slot_type,
                            owner=owner,
                            slots_before=slots_before_by_job.get(wi.id),
                            ttl_seconds=self._sem_default_ttl,
                        )
                        reserved_jobs.append(wi)

                    if not reserved_jobs:
                        continue

                    try:
                        attempts = await self._activate_from_lease_db(
                            [wi.id for wi in reserved_jobs]
                        )
                    except Exception as error:
                        attempts = {}
                        self.logger.error(
                            f"[WORK_DIST] DB activation failed for executor={slot_type}: {error}",
                            exc_info=True,
                        )

                    for wi in reserved_jobs:
                        owner = wi.id
                        run_attempt_id = attempts.get(wi.id)
                        if not run_attempt_id:
                            scheduler_trace(
                                "job_db_activate_failed",
                                job_id=wi.id,
                                dag_id=wi.dag_id,
                                executor=slot_type,
                                run_owner=self.lease_owner,
                            )
                            await self._release_lease_db([wi.id])
                            await self.frontier.release_lease_local(wi.id)
                            try:
                                await asyncio.to_thread(
                                    self._semaphore_store.release_owned,
                                    slot_type,
                                    wi.id,
                                    owner=owner,
                                )
                            except Exception as release_error:
                                self.logger.warning(
                                    f"[sem] release error after activation-fail {wi.id}@{slot_type}: {release_error}"
                                )
                            continue

                        scheduler_trace(
                            "job_run_attempt_started",
                            job_id=wi.id,
                            dag_id=wi.dag_id,
                            executor=slot_type,
                            run_owner=self.lease_owner,
                            run_attempt_id=run_attempt_id,
                            **self._ha_trace_fields(),
                        )
                        enqueue_tasks.append(
                            {
                                "task": asyncio.create_task(
                                    self._activate_and_enqueue_job(
                                        wi,
                                        run_owner=self.lease_owner,
                                        run_attempt_id=run_attempt_id,
                                    )
                                ),
                                "wi": wi,
                                "exe": slot_type,
                                "owner": owner,
                                "run_owner": self.lease_owner,
                                "run_attempt_id": run_attempt_id,
                            }
                        )

                if enqueue_tasks:
                    scheduler_trace(
                        "dispatch_batch_start",
                        count=len(enqueue_tasks),
                        job_ids=[item["wi"].id for item in enqueue_tasks],
                    )
                    self.logger.info(
                        f"[WORK_DIST] Dispatching {len(enqueue_tasks)} jobs via _activate_and_enqueue_job..."
                    )
                    results = await asyncio.gather(
                        *[t["task"] for t in enqueue_tasks], return_exceptions=True
                    )
                    scheduler_trace(
                        "dispatch_batch_complete",
                        count=len(results),
                        failures=sum(
                            1
                            for result in results
                            if isinstance(result, Exception) or not result
                        ),
                    )
                    self.logger.info(
                        f"[WORK_DIST] Dispatch completed. Processing {len(results)} results..."
                    )
                    for i, result in enumerate(results):
                        wi = enqueue_tasks[i]["wi"]
                        exe = enqueue_tasks[i]["exe"]
                        owner = enqueue_tasks[i]["owner"]
                        run_owner = enqueue_tasks[i]["run_owner"]
                        run_attempt_id = enqueue_tasks[i]["run_attempt_id"]

                        if isinstance(result, Exception) or not result:
                            # dispatch failed → release lease & requeue
                            self.logger.error(
                                f"[WORK_DIST] Dispatch FAILED for job={wi.id}, executor={exe}: {result}",
                                exc_info=(
                                    True if isinstance(result, Exception) else False
                                ),
                            )
                            await self._handle_dispatch_failure(
                                wi,
                                exe,
                                owner,
                                result,
                                run_owner=run_owner,
                                run_attempt_id=run_attempt_id,
                            )
                            continue

                        jobs_scheduled_this_cycle[exe] += 1
                        scheduled_any = True

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
                if _is_known_connection_error(e):
                    self.logger.warning(
                        "Poll loop: ETCD connection unavailable, waiting for reconnect"
                    )
                    await asyncio.sleep(3)
                    continue

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

    async def _activate_and_enqueue_job(
        self, wi: WorkInfo, *, run_owner: str, run_attempt_id: str
    ) -> bool:
        is_retry = wi.state == WorkState.RETRY or wi.state == WorkState.RETRY.value
        scheduler_trace(
            "job_enqueue_start",
            job_id=wi.id,
            dag_id=wi.dag_id,
            job_name=wi.name,
            is_retry=is_retry,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
            **self._ha_trace_fields(),
        )
        wi.run_owner = run_owner
        wi.run_attempt_id = run_attempt_id
        wi.state = WorkState.ACTIVE
        self._job_cache[wi.id] = wi
        await self.frontier.update_job_state(wi.id, WorkState.ACTIVE)
        scheduler_trace(
            "job_active_marked",
            job_id=wi.id,
            dag_id=wi.dag_id,
            job_name=wi.name,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
            **self._ha_trace_fields(),
        )
        enqueued = await self.enqueue(
            wi,
            is_retry=is_retry,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
        )
        scheduler_trace(
            "job_enqueue_result",
            job_id=wi.id,
            dag_id=wi.dag_id,
            job_name=wi.name,
            success=enqueued,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
            **self._ha_trace_fields(),
        )
        return enqueued

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
                "paused": self._paused,
                "scheduler_mode": self.scheduler_mode,
                "gateway_instance_id": self.gateway_instance_id,
                "scheduler_lease_owner": self.lease_owner,
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
            "sla_policy": {
                "hard_sla_policy": self.hard_sla_policy,
                "warning_top_n": self.sla_warning_top_n,
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
            debug_data["frontier_summary"] = self.frontier.summary(detail=True)
        except Exception as e:
            debug_data["frontier_summary_error"] = str(e)

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

    async def enqueue(
        self,
        work_info: WorkInfo,
        *,
        is_retry: Optional[bool] = None,
        run_owner: str | None = None,
        run_attempt_id: str | None = None,
    ) -> bool:
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
            scheduler_trace(
                "gateway_dispatch_rejected",
                job_id=submission_id,
                dag_id=work_info.dag_id,
                reason="missing_entrypoint",
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                **self._ha_trace_fields(),
            )
            self.logger.error(
                f"The entrypoint 'on' is not defined in metadata for job {submission_id}"
            )
            return False

        try:
            dispatch_started = time.perf_counter()
            executor = entrypoint.split("://", 1)[0]
            if run_owner and run_attempt_id:
                try:
                    await self.repository.record_job_attempt_dispatch_started(
                        job_id=submission_id,
                        job_name=work_info.name,
                        dag_id=str(work_info.dag_id),
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        scheduler_lease_owner=self.lease_owner,
                        gateway_instance_id=self.gateway_instance_id,
                        executor=executor,
                    )
                except Exception as audit_error:
                    self.logger.warning(
                        f"Failed to record dispatch start for attempt {run_attempt_id}: {audit_error}"
                    )
            scheduler_trace(
                "gateway_dispatch_start",
                job_id=submission_id,
                dag_id=work_info.dag_id,
                entrypoint=entrypoint,
                is_retry=is_retry,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                **self._ha_trace_fields(),
            )
            # Inject DAG tracking parameters into metadata for asset tracking
            # These are needed by executors to record asset materializations
            job_metadata = work_info.data.copy()
            if work_info.dag_id:
                job_metadata['dag_id'] = work_info.dag_id
                job_metadata['node_task_id'] = (
                    work_info.id
                )  # job ID serves as node task ID
            if run_owner:
                job_metadata["run_owner"] = run_owner
            if run_attempt_id:
                job_metadata["run_attempt_id"] = run_attempt_id

            # Detect if this is a retry (job was previously run and failed)
            if is_retry is None:
                is_retry = (
                    work_info.state == WorkState.RETRY
                    or work_info.state == WorkState.RETRY.value
                )

            async def _submit_and_confirm() -> None:
                await self.job_manager.submit_job(
                    entrypoint=entrypoint,
                    submission_id=submission_id,
                    metadata=job_metadata,
                    confirmation_event=confirmation_event,
                    is_retry=is_retry,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                )
                scheduler_trace(
                    "gateway_dispatch_submitted",
                    job_id=submission_id,
                    dag_id=work_info.dag_id,
                    entrypoint=entrypoint,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    **self._ha_trace_fields(),
                )

                # Wait for the supervisor to confirm it has received the job and is running.
                # The timeout covers submit + confirmation and must stay inside the lease TTL.
                await confirmation_event.wait()
                scheduler_trace(
                    "gateway_dispatch_confirmed",
                    job_id=submission_id,
                    dag_id=work_info.dag_id,
                    entrypoint=entrypoint,
                    elapsed_ms=(time.perf_counter() - dispatch_started) * 1000.0,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    **self._ha_trace_fields(),
                )
                if run_attempt_id:
                    try:
                        await self.repository.record_job_attempt_dispatch_result(
                            run_attempt_id=run_attempt_id,
                            confirmed=True,
                        )
                    except Exception as audit_error:
                        self.logger.warning(
                            f"Failed to record dispatch confirmation for attempt {run_attempt_id}: {audit_error}"
                        )

            dispatch_timeout = max(0.1, float(self.lease_ttl_seconds) - 1.0)
            await asyncio.wait_for(_submit_and_confirm(), timeout=dispatch_timeout)
            self.logger.debug(f"Dispatch confirmed for job: {submission_id}")
            return True

        except asyncio.TimeoutError:
            scheduler_trace(
                "gateway_dispatch_timeout",
                job_id=submission_id,
                dag_id=work_info.dag_id,
                entrypoint=entrypoint,
                timeout_seconds=max(0.1, float(self.lease_ttl_seconds) - 1.0),
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                **self._ha_trace_fields(),
            )
            if run_attempt_id:
                try:
                    await self.repository.record_job_attempt_dispatch_result(
                        run_attempt_id=run_attempt_id,
                        confirmed=False,
                        error="dispatch_timeout",
                    )
                except Exception as audit_error:
                    self.logger.warning(
                        f"Failed to record dispatch timeout for attempt {run_attempt_id}: {audit_error}"
                    )
            self.logger.error(
                f"Timeout waiting for dispatch confirmation for job {submission_id}"
            )
            return False
        except Exception as e:
            scheduler_trace(
                "gateway_dispatch_failed",
                job_id=submission_id,
                dag_id=work_info.dag_id,
                entrypoint=entrypoint,
                error=repr(e),
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                **self._ha_trace_fields(),
            )
            if run_attempt_id:
                try:
                    await self.repository.record_job_attempt_dispatch_result(
                        run_attempt_id=run_attempt_id,
                        confirmed=False,
                        error=repr(e),
                    )
                except Exception as audit_error:
                    self.logger.warning(
                        f"Failed to record dispatch failure for attempt {run_attempt_id}: {audit_error}"
                    )
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
        except (Exception, psycopg.Error) as error:
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
        # sync_mode = work_info.data.get("metadata", {}).get("sync_mode", False)
        sync_mode = False

        submission_request = JobSubmissionRequest(
            work_info=work_info,
            overwrite=overwrite,
            request_id=request_id,
            result_future=result_future,
            wait_for_result=sync_mode,
        )

        self._pending_requests[request_id] = submission_request

        try:
            queue_size_before = self._request_queue.qsize()
            self._request_queue.put_nowait(submission_request)
            scheduler_trace(
                "scheduler_submission_enqueued",
                job_id=work_info.id,
                dag_id=work_info.id,
                job_name=work_info.name,
                request_id=request_id,
                queue_size_before=queue_size_before,
                queue_size=self._request_queue.qsize(),
            )
            self.logger.debug(
                f"Job {work_info.id} queued successfully (request: {request_id})"
            )
            if sync_mode:
                # Wait for the result
                result = await result_future
                return result

            return work_info.id
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            if sync_mode and not result_future.done():
                result_future.set_exception(e)
            raise

    async def _handle_priority_refresh(self):
        """Handle priority refresh"""
        refresh_interval = self.priority_refresh_interval

        if self._submission_count % refresh_interval == 0:
            now = time.monotonic()
            due_in = self._next_priority_refresh_at - now
            should_wake = due_in <= 0
            scheduler_trace(
                "scheduler_priority_refresh_requested",
                source="submission_worker",
                submission_count=self._submission_count,
                refresh_interval=refresh_interval,
                request_queue_size=self._request_queue.qsize(),
                pending_requests=len(self._pending_requests),
                due_in_ms=max(0.0, due_in * 1000.0),
                wake_scheduler=should_wake,
            )
            if should_wake:
                await self.notify_event()
            self.logger.info(
                f"Requested job priority refresh after {self._submission_count} submissions "
                f"(interval: {refresh_interval}, due_in={max(0.0, due_in):.3f}s)"
            )

    async def _send_submission_failure_toast(self, work_info, error: Exception) -> None:
        """Send a failed toast event when job submission fails."""
        try:
            event_name = work_info.data.get("name", work_info.name)
            api_key = work_info.data.get("api_key", None)
            metadata = work_info.data.get("metadata", {})
            ref_type = metadata.get("ref_type")

            if not api_key or not event_name:
                self.logger.warning(
                    f"Cannot send failure toast for {work_info.id}: "
                    f"missing api_key={api_key} or event_name={event_name}"
                )
                return

            await mark_as_failed_toast(
                api_key=api_key,
                job_id=work_info.id,
                event_name=event_name,
                job_tag=ref_type,
                status="FAILED",
                timestamp=current_milli_time(),
                payload={**metadata, "error": str(error)},
            )
        except Exception as toast_err:
            self.logger.error(
                f"Failed to send failure toast for {work_info.id}: {toast_err}"
            )

    async def _process_submission_queue(self, worker_id: int) -> None:
        """Background worker that processes queued job submissions"""
        self.logger.info(f"Background job submission worker started # {worker_id}")

        while self.running:
            request = None
            try:
                request = await self._request_queue.get()
                scheduler_trace(
                    "scheduler_submission_dequeued",
                    job_id=request.work_info.id,
                    dag_id=request.work_info.id,
                    job_name=request.work_info.name,
                    request_id=request.request_id,
                    worker_id=worker_id,
                    queue_size=self._request_queue.qsize(),
                )
                try:
                    result = await self.__submit_job(
                        request.work_info, request.overwrite
                    )
                    self._submission_count += 1
                    await self._handle_priority_refresh()

                    if request.wait_for_result and not request.result_future.done():
                        request.result_future.set_result(result)

                    queue_size = self._request_queue.qsize()
                    self.logger.debug(
                        f"Successfully processed job: {request.work_info.id} (queue size: {queue_size})"
                    )

                except ValueError as e:
                    self.logger.error(
                        f"Job submission failed for {request.work_info.id}: {e}"
                    )
                    if request.wait_for_result and not request.result_future.done():
                        request.result_future.set_exception(e)
                except Exception as e:
                    if request.wait_for_result and not request.result_future.done():
                        request.result_future.set_exception(e)
                    self.logger.error(
                        f"Failed to process job {request.work_info.id}: {e}"
                    )
                    await self._send_submission_failure_toast(request.work_info, e)
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
        scheduler_trace(
            "dag_plan_built",
            job_id=submission_id,
            dag_id=submission_id,
            job_name=work_info.name,
            job_count=len(dag_nodes),
        )

        # Set dag_id on all nodes
        for dag_work_info in dag_nodes:
            dag_work_info.dag_id = submission_id

        # Delegate DAG and job creation to repository
        scheduler_trace(
            "dag_persist_start",
            job_id=submission_id,
            dag_id=submission_id,
            job_name=work_info.name,
            job_count=len(dag_nodes),
        )
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

        scheduler_trace(
            "dag_persisted",
            job_id=submission_id,
            dag_id=submission_id,
            job_name=work_info.name,
            job_count=len(dag_nodes),
            new_dag_key=new_dag_key,
        )
        scheduler_trace(
            "dag_frontier_add_start",
            job_id=submission_id,
            dag_id=submission_id,
            job_name=work_info.name,
            job_count=len(dag_nodes),
        )
        await self.frontier.add_dag(plan, dag_nodes)
        scheduler_trace(
            "dag_frontier_added",
            job_id=submission_id,
            dag_id=submission_id,
            job_name=work_info.name,
            job_count=len(dag_nodes),
        )
        await self.notify_event()
        return submission_id

    async def _refresh_job_priorities(self, source: str = "unknown") -> int:
        """Sync DB priority edits into memory and log current SLA pressure."""
        self._priority_refresh_seq += 1
        refresh_id = self._priority_refresh_seq
        refresh_started = time.perf_counter()
        scheduler_trace(
            "scheduler_priority_refresh_start",
            source=source,
            refresh_id=refresh_id,
            submission_count=self._submission_count,
            hydrate_missing_limit=self.priority_refresh_hydrate_limit,
            request_queue_size=self._request_queue.qsize(),
            pending_requests=len(self._pending_requests),
        )
        try:
            phase_started = time.perf_counter()
            scheduler_trace(
                "scheduler_priority_refresh_frontier_start",
                source=source,
                refresh_id=refresh_id,
                submission_count=self._submission_count,
                hydrate_missing_limit=self.priority_refresh_hydrate_limit,
            )
            refresh_stats = await self.dag_service.refresh_frontier_priorities(
                hydrate_missing_limit=self.priority_refresh_hydrate_limit,
                refresh_id=refresh_id,
                source=source,
            )
            scheduler_trace(
                "scheduler_priority_refresh_frontier_done",
                source=source,
                refresh_id=refresh_id,
                submission_count=self._submission_count,
                elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
                tracked=refresh_stats.get("tracked", 0),
                fetched=refresh_stats.get("fetched", 0),
                changed=refresh_stats.get("changed", 0),
                hydrated_missing=refresh_stats.get("hydrated_missing", 0),
            )

            phase_started = time.perf_counter()
            scheduler_trace(
                "scheduler_priority_refresh_ready_ordering_start",
                source=source,
                refresh_id=refresh_id,
                submission_count=self._submission_count,
            )
            await self.frontier.refresh_ready_ordering()
            scheduler_trace(
                "scheduler_priority_refresh_ready_ordering_done",
                source=source,
                refresh_id=refresh_id,
                submission_count=self._submission_count,
                elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
            )
            self._next_priority_refresh_at = (
                time.monotonic() + self.priority_refresh_interval_seconds
            )

            phase_started = time.perf_counter()
            scheduler_trace(
                "scheduler_priority_refresh_summary_start",
                source=source,
                refresh_id=refresh_id,
                submission_count=self._submission_count,
                top_n=self.sla_warning_top_n,
            )
            frontier_summary = self.frontier.summary(
                detail=True, top_n=self.sla_warning_top_n
            )
            sla_summary = frontier_summary.get("sla", {})
            scheduler_trace(
                "scheduler_priority_refresh_summary_done",
                source=source,
                refresh_id=refresh_id,
                submission_count=self._submission_count,
                elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
                frontier_jobs=frontier_summary.get("totals", {}).get("jobs", 0),
                frontier_dags=frontier_summary.get("totals", {}).get("dags", 0),
                ready=frontier_summary.get("totals", {}).get("ready", 0),
                blocked=frontier_summary.get("totals", {}).get("blocked", 0),
                soft_missed=sla_summary.get("soft_missed", 0),
                hard_missed=sla_summary.get("hard_missed", 0),
                highest_bucket=sla_summary.get("highest_bucket", 0),
            )
            if not sla_summary:
                self.logger.info(
                    "[SLA] Refresh checkpoint: "
                    f"db_tracked={refresh_stats.get('tracked', 0)}, "
                    f"db_fetched={refresh_stats.get('fetched', 0)}, "
                    f"changed={refresh_stats.get('changed', 0)}, "
                    f"hydrated_missing={refresh_stats.get('hydrated_missing', 0)}, "
                    "no frontier-tracked jobs"
                )
                scheduler_trace(
                    "scheduler_priority_refresh_done",
                    source=source,
                    refresh_id=refresh_id,
                    submission_count=self._submission_count,
                    elapsed_ms=(time.perf_counter() - refresh_started) * 1000.0,
                    tracked=refresh_stats.get("tracked", 0),
                    fetched=refresh_stats.get("fetched", 0),
                    changed=refresh_stats.get("changed", 0),
                    hydrated_missing=refresh_stats.get("hydrated_missing", 0),
                    has_sla_summary=False,
                )
                return refresh_id

            self.logger.info(
                "[SLA] Refresh checkpoint: "
                f"db_tracked={refresh_stats.get('tracked', 0)}, "
                f"db_fetched={refresh_stats.get('fetched', 0)}, "
                f"changed={refresh_stats.get('changed', 0)}, "
                f"hydrated_missing={refresh_stats.get('hydrated_missing', 0)}, "
                f"frontier_tracked={sla_summary.get('tracked', 0)}, "
                f"approaching={sla_summary.get('approaching_soft', 0)}, "
                f"soft_missed={sla_summary.get('soft_missed', 0)}, "
                f"hard_missed={sla_summary.get('hard_missed', 0)}, "
                f"highest_bucket={sla_summary.get('highest_bucket', 0)}"
            )
            top_urgent = sla_summary.get("top_urgent", [])
            if top_urgent:
                self.logger.warning(f"[SLA] Top urgent jobs: {top_urgent}")

            phase_started = time.perf_counter()
            scheduler_trace(
                "scheduler_priority_refresh_hard_sla_policy_start",
                source=source,
                refresh_id=refresh_id,
                submission_count=self._submission_count,
                hard_missed=sla_summary.get("hard_missed", 0),
                policy=self.hard_sla_policy,
            )
            await self._handle_hard_sla_policy(sla_summary)
            scheduler_trace(
                "scheduler_priority_refresh_hard_sla_policy_done",
                source=source,
                refresh_id=refresh_id,
                submission_count=self._submission_count,
                elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
                hard_missed=sla_summary.get("hard_missed", 0),
                policy=self.hard_sla_policy,
            )
            scheduler_trace(
                "scheduler_priority_refresh_done",
                source=source,
                refresh_id=refresh_id,
                submission_count=self._submission_count,
                elapsed_ms=(time.perf_counter() - refresh_started) * 1000.0,
                tracked=refresh_stats.get("tracked", 0),
                fetched=refresh_stats.get("fetched", 0),
                changed=refresh_stats.get("changed", 0),
                hydrated_missing=refresh_stats.get("hydrated_missing", 0),
                has_sla_summary=True,
                soft_missed=sla_summary.get("soft_missed", 0),
                hard_missed=sla_summary.get("hard_missed", 0),
                highest_bucket=sla_summary.get("highest_bucket", 0),
            )
        except Exception as e:
            scheduler_trace(
                "scheduler_priority_refresh_failed",
                source=source,
                refresh_id=refresh_id,
                submission_count=self._submission_count,
                elapsed_ms=(time.perf_counter() - refresh_started) * 1000.0,
                error=repr(e),
            )
            self.logger.error(f"Failed to refresh job priorities: {e}")
        return refresh_id

    async def _handle_hard_sla_policy(self, sla_summary: Dict[str, Any]) -> None:
        """Current hard-SLA behavior hook for the in-memory scheduler."""
        hard_missed = int(sla_summary.get("hard_missed", 0))
        if hard_missed <= 0:
            return

        if self.hard_sla_policy == "track_only":
            self.logger.warning(
                f"[SLA] {hard_missed} jobs have missed hard SLA; policy=track_only"
            )
            return

        if self.hard_sla_policy == "escalate_only":
            self.logger.warning(
                f"[SLA] {hard_missed} jobs have missed hard SLA; policy=escalate_only "
                "and planner ranking will continue to prefer them"
            )
            return

        self.logger.error(
            f"[SLA] {hard_missed} jobs have missed hard SLA; policy=expire_unfinished "
            "is configured but not yet implemented in the in-memory scheduler"
        )

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

    async def _admit_dag(
        self, work_info: WorkInfo, dag: QueryPlan, *, source: str
    ) -> bool:
        dag_id = work_info.dag_id

        async with self._dag_admission_lock:
            if dag_id in self.active_dags:
                return True

            if len(self.active_dags) >= self.max_concurrent_dags:
                self.logger.debug(
                    f"[DAG_ADMISSION] Skipping DAG {dag_id} from {source}; "
                    f"active_dags={len(self.active_dags)}/{self.max_concurrent_dags}"
                )
                return False

            marked_active = await self.mark_as_active_dag(work_info)
            if not marked_active:
                self.logger.warning(
                    f"[DAG_ADMISSION] Failed to mark DAG {dag_id} as active in DB "
                    f"from {source}; leaving it out of active_dags"
                )
                return False
            self.active_dags[dag_id] = dag
            return True

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

            existing_job = await self.get_job_for_policy(work_info)

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

    async def _reconcile_recovered_run_leases(
        self, recovered: list[RecoveredRunLease]
    ) -> None:
        for recovery in recovered:
            work_item = await self.repository.get_job_by_id(recovery.id)
            if work_item is None:
                self.logger.warning(
                    f"Recovered run lease for missing job: {recovery.id}"
                )
                continue

            self._job_cache[recovery.id] = work_item
            scheduler_trace(
                "run_lease_recovered",
                job_id=recovery.id,
                dag_id=recovery.dag_id,
                recovered_state=recovery.recovered_state,
                previous_run_owner=recovery.previous_run_owner,
                previous_run_attempt_id=recovery.previous_run_attempt_id,
                reason_code=recovery.reason_code,
                **self._ha_trace_fields(),
            )

            if recovery.recovered_state == "retry":
                self._scheduler_counter(
                    RUN_LEASE_RECOVERED_RETRY_TOTAL,
                    job_id=recovery.id,
                    dag_id=recovery.dag_id,
                    previous_run_owner=recovery.previous_run_owner,
                    previous_run_attempt_id=recovery.previous_run_attempt_id,
                    reason_code=recovery.reason_code,
                )
                await self.frontier.on_job_retry(
                    recovery.id, work_item, start_after=recovery.start_after
                )
                continue

            self._scheduler_counter(
                RUN_LEASE_RECOVERED_FAILED_TOTAL,
                job_id=recovery.id,
                dag_id=recovery.dag_id,
                previous_run_owner=recovery.previous_run_owner,
                previous_run_attempt_id=recovery.previous_run_attempt_id,
                reason_code=recovery.reason_code,
            )
            await self.frontier.on_job_failed(recovery.id)
            await self._resolve_dag_status_with_retry(
                recovery.id,
                work_item,
                source="run_lease_recovery",
            )

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
            id=str(record[0]),
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
            dag_id=str(record[11]) if record[11] is not None else None,
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
        run_owner: str | None = None,
        run_attempt_id: str | None = None,
    ) -> int:
        """
        Mark a job as completed.
        Delegates to JobRepository.

        :param job_id: The ID of the job to complete
        :param work_item: The work item containing queue name
        :param output_metadata: Optional metadata to store with completion
        :param force: If True, complete job regardless of current state
        """
        async with self._status_update_lock[job_id]:
            count = await self.repository.complete_job(
                job_id=job_id,
                queue_name=work_item.name,
                output_metadata=output_metadata,
                force=force,
                schema=DEFAULT_SCHEMA,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
            )
            # self._job_cache.pop(job_id, None) # invalidate cache
            return count

    async def fail(
        self,
        job_id: str,
        work_item: WorkInfo,
        output_metadata: dict = None,
        run_owner: str | None = None,
        run_attempt_id: str | None = None,
    ) -> Optional[str]:
        """
        Mark a job as failed or for retry.
        Delegates to JobRepository.

        :param job_id: The ID of the job to mark as failed
        :param work_item: The work item containing queue name
        :param output_metadata: Optional metadata to store with failure
        :return: The actual final state ('retry' or 'failed'), or None on error
        """
        async with self._status_update_lock[job_id]:
            count, final_state = await self.repository.fail_job(
                job_id=job_id,
                queue_name=work_item.name,
                output_metadata=output_metadata,
                schema=DEFAULT_SCHEMA,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
            )
            # self._job_cache.pop(job_id, None) # invalidate cache
            return final_state

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
                active_jobs = await self.list_jobs(state=[WorkState.ACTIVE.value])
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

                    if job_info.status == JobStatus.RUNNING:
                        run_owner = job_info.run_owner
                        run_attempt_id = job_info.run_attempt_id
                        if not run_owner or not run_attempt_id:
                            scheduler_trace(
                                "run_lease_extend_rejected",
                                job_id=job_id,
                                dag_id=work_item.dag_id,
                                status=job_info.status.value,
                                reason="sync_missing_attempt",
                            )
                            continue

                        extended = await self._extend_run_lease_db(
                            [job_id],
                            run_owner=run_owner,
                            run_attempt_id=run_attempt_id,
                        )
                        if job_id not in extended:
                            scheduler_trace(
                                "run_lease_extend_rejected",
                                job_id=job_id,
                                dag_id=work_item.dag_id,
                                status=job_info.status.value,
                                run_owner=run_owner,
                                run_attempt_id=run_attempt_id,
                                reason="sync_db_update_zero_rows",
                            )
                            self._scheduler_counter(
                                RUN_LEASE_EXTEND_STALE_ATTEMPT_TOTAL,
                                job_id=job_id,
                                dag_id=work_item.dag_id,
                                status=job_info.status.value,
                                run_owner=run_owner,
                                run_attempt_id=run_attempt_id,
                                source="storage_sync",
                            )
                        continue

                    await self._sync_terminal_job_state(
                        job_id,
                        work_item,
                        job_info,
                        min_sync_interval_seconds=min_sync_interval_seconds,
                    )

            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error syncing jobs: {error}")
                self.logger.error(traceback.format_exc())

    async def _sync_terminal_job_state(
        self,
        job_id: str,
        work_item: WorkInfo,
        job_info: JobInfo,
        *,
        min_sync_interval_seconds: int,
    ) -> bool:
        job_info_state = convert_job_status_to_work_state(job_info.status)
        if not job_info.status.is_terminal() or work_item.state == job_info_state:
            return False

        self.logger.info(
            f"State mismatch for job {job_id}: "
            f"WorkState={work_item.state}, JobInfoState={job_info_state}. Updating."
        )

        synchronize = False
        remaining_time = None
        now = datetime.now(tz=timezone.utc)

        if job_info.end_time is not None:
            timestamp_ms = job_info.end_time
            end_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            remaining_time = end_time - now

            if end_time < now - timedelta(seconds=min_sync_interval_seconds):
                synchronize = True

        if not synchronize:
            seconds = remaining_time.total_seconds() if remaining_time else "unknown"
            self.logger.info(
                f"Job has not ended more than {min_sync_interval_seconds} seconds ago, skipping sync. "
                f"{job_id}: {seconds} seconds since end."
            )
            return False

        meta = {"synced": True}
        actual_work_state: Optional[str] = None
        run_owner = job_info.run_owner
        run_attempt_id = job_info.run_attempt_id

        if job_info.status == JobStatus.SUCCEEDED:
            if not run_owner or not run_attempt_id:
                scheduler_trace(
                    "job_terminal_attempt_rejected",
                    job_id=job_id,
                    dag_id=work_item.dag_id,
                    status=job_info.status.value,
                    reason="sync_missing_attempt",
                    **self._ha_trace_fields(),
                )
                return False
            completed = await self.complete(
                job_id,
                work_item,
                meta,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
            )
            if completed <= 0:
                scheduler_trace(
                    "job_terminal_attempt_rejected",
                    job_id=job_id,
                    dag_id=work_item.dag_id,
                    status=job_info.status.value,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    reason="sync_db_update_zero_rows",
                    **self._ha_trace_fields(),
                )
                await self._record_terminal_attempt_audit(
                    job_id=job_id,
                    work_item=work_item,
                    status=job_info.status,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    terminal_work_state=None,
                    source="storage_sync",
                    accepted=False,
                    reject_reason="sync_db_update_zero_rows",
                )
                self._scheduler_counter(
                    TERMINAL_EVENT_STALE_ATTEMPT_TOTAL,
                    job_id=job_id,
                    dag_id=work_item.dag_id,
                    status=job_info.status.value,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    source="storage_sync",
                )
                return False
            await self._record_terminal_attempt_audit(
                job_id=job_id,
                work_item=work_item,
                status=job_info.status,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                terminal_work_state=WorkState.COMPLETED.value,
                source="storage_sync",
                accepted=True,
            )
            scheduler_trace(
                "job_terminal_attempt_accepted",
                job_id=job_id,
                dag_id=work_item.dag_id,
                status=job_info.status.value,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                source="storage_sync",
                **self._ha_trace_fields(),
            )
            await self._handle_successful_job_completion(job_id, work_item)
        elif job_info.status == JobStatus.FAILED:
            if not run_owner or not run_attempt_id:
                scheduler_trace(
                    "job_terminal_attempt_rejected",
                    job_id=job_id,
                    dag_id=work_item.dag_id,
                    status=job_info.status.value,
                    reason="sync_missing_attempt",
                    **self._ha_trace_fields(),
                )
                return False
            actual_work_state = await self.fail(
                job_id,
                work_item,
                meta,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
            )
            if actual_work_state is None:
                scheduler_trace(
                    "job_terminal_attempt_rejected",
                    job_id=job_id,
                    dag_id=work_item.dag_id,
                    status=job_info.status.value,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    reason="sync_db_update_zero_rows",
                    **self._ha_trace_fields(),
                )
                await self._record_terminal_attempt_audit(
                    job_id=job_id,
                    work_item=work_item,
                    status=job_info.status,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    terminal_work_state=None,
                    source="storage_sync",
                    accepted=False,
                    reject_reason="sync_db_update_zero_rows",
                )
                self._scheduler_counter(
                    TERMINAL_EVENT_STALE_ATTEMPT_TOTAL,
                    job_id=job_id,
                    dag_id=work_item.dag_id,
                    status=job_info.status.value,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    source="storage_sync",
                )
                return False
            await self._record_terminal_attempt_audit(
                job_id=job_id,
                work_item=work_item,
                status=job_info.status,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                terminal_work_state=actual_work_state,
                source="storage_sync",
                accepted=True,
            )
            scheduler_trace(
                "job_terminal_attempt_accepted",
                job_id=job_id,
                dag_id=work_item.dag_id,
                status=job_info.status.value,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                final_state=actual_work_state,
                source="storage_sync",
                **self._ha_trace_fields(),
            )
            if actual_work_state == WorkState.RETRY.value:
                await self.frontier.on_job_retry(job_id, work_item)
            else:
                await self.frontier.on_job_failed(job_id)
        elif job_info.status == JobStatus.STOPPED:
            await self.cancel_job(job_id, work_item)
            await self.frontier.on_job_cancelled(job_id)
        else:
            self.logger.error(
                f"Unhandled job status: {job_info.status}. Marking as FAILED."
            )
            actual_work_state = await self.fail(job_id, work_item)
            if actual_work_state == WorkState.RETRY.value:
                await self.frontier.on_job_retry(job_id, work_item)
            else:
                await self.frontier.on_job_failed(job_id)

        is_truly_terminal = (
            job_info.status == JobStatus.SUCCEEDED
            or job_info.status == JobStatus.STOPPED
            or (
                job_info.status == JobStatus.FAILED
                and actual_work_state == WorkState.FAILED.value
            )
        )

        if is_truly_terminal:
            self.logger.info(
                f"Synchronized job {job_id} is in terminal state {job_info.status}"
            )
            self._status_update_lock.release(job_id)
            await self._resolve_dag_status_with_retry(
                job_id,
                work_item,
                now,
                now,
                source="storage_sync",
            )
            await self.notify_event()
        elif (
            job_info.status == JobStatus.FAILED
            and actual_work_state == WorkState.RETRY.value
        ):
            self.logger.info(
                f"Synchronized job {job_id} marked for retry, DAG remains active"
            )
            await self.notify_event()

        return True

    async def _sync_dag(self):
        self.logger.info("Starting DAG synchronization")
        # https://github.com/marieai/marie-ai/issues/134
        await self._sync_dag_loop()

    async def _sync_dag_loop(self, interval: int = 30) -> None:
        """
        Validate in-memory DAGs without monopolizing the scheduler DB executor.
        """
        self.logger.info(f"Starting DAG sync polling (interval: {interval}s)")
        scheduler_trace("scheduler_dag_sync_loop_start", interval=interval)

        while self.running:
            try:
                await self._sync_dag_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                scheduler_trace("scheduler_dag_sync_cycle_failed", error=repr(error))
                self.logger.error(f"Error validating DAGs: {error}")

            await asyncio.sleep(interval)

        scheduler_trace("scheduler_dag_sync_loop_stopped")
        self.logger.debug("DAG sync polling stopped")

    async def _sync_dag_once(self) -> None:
        if not self.active_dags:
            scheduler_trace("scheduler_dag_sync_cycle_skipped", reason="no_active_dags")
            self.logger.debug("No active DAGs in memory to validate")
            return

        memory_dag_ids = list(self.active_dags.keys())
        scheduler_trace(
            "scheduler_dag_sync_cycle_start",
            active_dags=len(memory_dag_ids),
        )
        self.logger.debug(f"Validating {len(memory_dag_ids)} DAGs in memory")

        resolved_terminal_dags: set[str] = set()
        for dag_id in memory_dag_ids:
            try:
                dag_state = await self.repository.resolve_dag_state(dag_id)
                if dag_state in ("completed", "failed"):
                    resolved_terminal_dags.add(dag_id)
            except Exception as resolve_error:
                self.logger.warning(
                    f"[DAG_SYNC] Failed to resolve DAG state for {dag_id}: "
                    f"{resolve_error}"
                )

        valid_db_dags = await self.repository.get_active_dag_ids(memory_dag_ids)
        invalid_dags = (set(memory_dag_ids) - valid_db_dags).union(
            resolved_terminal_dags
        )

        if invalid_dags:
            self.logger.info(f"Found {len(invalid_dags)} invalid DAGs in memory")
            for dag_id in invalid_dags:
                await self._evict_dag_from_memory(
                    dag_id, "no longer active or deleted in database"
                )
            await self.notify_event()
        else:
            self.logger.debug("All DAGs in memory are still valid")

        scheduler_trace(
            "scheduler_dag_sync_cycle_done",
            active_dags=len(memory_dag_ids),
            valid_dags=len(valid_db_dags),
            terminal_dags=len(resolved_terminal_dags),
            invalid_dags=len(invalid_dags),
        )

    async def _evict_dag_from_memory(self, dag_id: str, reason: str) -> bool:
        self._terminal_dag_states.pop(dag_id, None)
        dag_jobs = await self.frontier.get_jobs_by_dag_id(dag_id)
        stats = await self.frontier.finalize_dag(dag_id)
        for dag_job in dag_jobs:
            self._job_cache.pop(dag_job.id, None)
        removed = self.dag_service.remove_dag(dag_id, reason)
        self.logger.info(
            f"[DAG_SYNC] Evicted DAG {dag_id} from memory ({reason}), "
            f"removed={removed}, finalize_stats={stats}"
        )
        return removed

    async def notify_event(self) -> bool:
        if self._debounced_notify:
            return False
        self._debounced_notify = True
        try:
            self._event_queue.put_nowait("wake")
        except asyncio.QueueFull:
            pass
        return True

    async def _wait_for_dispatch_wake(self, timeout: float) -> bool:
        if timeout <= 0:
            try:
                self._event_queue.get_nowait()
                self._debounced_notify = False
                return True
            except asyncio.QueueEmpty:
                return False

        try:
            await asyncio.wait_for(self._event_queue.get(), timeout=timeout)
            self._debounced_notify = False
            return True
        except asyncio.TimeoutError:
            return False

    async def _emit_dag_terminal_event(
        self, dag_state: str, work_info: WorkInfo
    ) -> None:
        event_name = work_info.data.get("name", work_info.name)
        api_key = work_info.data.get("api_key")
        metadata = work_info.data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        ref_type = metadata.get("ref_type")

        if not api_key or not event_name:
            self.logger.warning(
                f"Skipping DAG terminal event for {work_info.dag_id}: "
                f"missing api_key={api_key} or event_name={event_name}"
            )
            return

        status = "OK" if dag_state == "completed" else "FAILED"
        notifier = (
            mark_as_complete_toast if dag_state == "completed" else mark_as_failed_toast
        )

        try:
            await notifier(
                api_key=api_key,
                job_id=work_info.dag_id,
                event_name=event_name,
                job_tag=ref_type,
                status=status,
                timestamp=current_milli_time(),
                payload=metadata,
            )
            self.logger.debug(
                f"DAG notification sent: {work_info.dag_id}, status={status}"
            )
        except Exception as toast_error:
            self.logger.error(
                f"Failed to send DAG terminal event for {work_info.dag_id}: {toast_error}"
            )

    async def _handle_dispatch_failure(
        self,
        wi: WorkInfo,
        executor: str,
        owner: str,
        error: Any,
        *,
        run_owner: str | None = None,
        run_attempt_id: str | None = None,
    ) -> None:
        error_message = str(error) if error is not None else "dispatch failed"

        actual_work_state = await self.fail(
            wi.id,
            wi,
            {
                "dispatch_failed": True,
                "dispatch_error": error_message,
                "failure_stage": "enqueue",
            },
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
        )
        if actual_work_state is not None and run_owner and run_attempt_id:
            await self._record_terminal_attempt_audit(
                job_id=wi.id,
                work_item=wi,
                status=JobStatus.FAILED,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                terminal_work_state=actual_work_state,
                source="dispatch_failure",
                accepted=True,
            )
        elif run_owner and run_attempt_id:
            await self._record_terminal_attempt_audit(
                job_id=wi.id,
                work_item=wi,
                status=JobStatus.FAILED,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                terminal_work_state=None,
                source="dispatch_failure",
                accepted=False,
                reject_reason="db_update_zero_rows",
            )

        if actual_work_state == WorkState.RETRY.value:
            await self.frontier.on_job_retry(wi.id, wi)
        elif actual_work_state == WorkState.FAILED.value:
            await self.frontier.on_job_failed(wi.id)
            await self._resolve_dag_status_with_retry(
                wi.id,
                wi,
                source="dispatch_failure",
            )
        else:
            self.logger.error(
                f"Dispatch failure cleanup could not transition job {wi.id}; "
                f"actual_work_state={actual_work_state}"
            )
            await self.frontier.release_lease_local(wi.id)

        try:
            released = await asyncio.to_thread(
                self._semaphore_store.release_owned,
                executor,
                wi.id,
                owner=owner,
            )
            self.logger.debug(
                f"[sem] release on dispatch-fail {wi.id}@{executor} -> {released}"
            )
        except Exception as release_error:
            self.logger.warning(
                f"[sem] release error after dispatch-fail {wi.id}@{executor}: "
                f"{release_error}"
            )

        await self.notify_event()

    def _get_dag_resolution_retry_delay(self, retry_number: int) -> float:
        delay = max(0.0, self._dag_resolution_retry_delay)
        if not self._dag_resolution_retry_backoff or retry_number <= 1:
            return delay

        return min(
            self._dag_resolution_retry_max_delay,
            delay * (2 ** (retry_number - 1)),
        )

    async def _resolve_dag_status_with_retry(
        self,
        job_id: str,
        work_info: WorkInfo,
        started_on: Optional[datetime] = None,
        completed_on: Optional[datetime] = None,
        *,
        source: str,
    ) -> bool:
        """
        Retry DAG terminal resolution a small number of times for transient
        database or cleanup errors without introducing a separate queue.
        """
        retry_number = 0
        while True:
            try:
                return await self.resolve_dag_status(
                    job_id, work_info, started_on, completed_on
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                retry_number += 1
                if retry_number > self._dag_resolution_retry_limit:
                    self.logger.error(
                        f"[DAG_RESOLVE] Exhausted {self._dag_resolution_retry_limit} "
                        f"retries for dag={work_info.dag_id}, job={job_id}, "
                        f"source={source}: {error}"
                    )
                    return False

                delay = self._get_dag_resolution_retry_delay(retry_number)
                self.logger.warning(
                    f"[DAG_RESOLVE] Retry {retry_number}/"
                    f"{self._dag_resolution_retry_limit} for dag={work_info.dag_id}, "
                    f"job={job_id}, source={source} after error: {error}. "
                    f"Waiting {delay:.2f}s"
                )
                if delay > 0:
                    await asyncio.sleep(delay)

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
        dag_id = work_info.dag_id
        self.logger.info(f"Resolving DAG status: {dag_id}")

        if not dag_id:
            self.logger.warning(
                f"Skipping DAG status resolution for job without dag_id: {job_id}"
            )
            return False

        dag_lock = self._dag_resolution_lock[dag_id]
        try:
            async with dag_lock:
                claimed_terminal_state = False
                try:
                    dag_state = await self.repository.resolve_dag_state(dag_id)

                    self.logger.info(f"Resolved DAG state: {dag_state}")
                    if dag_state not in ("completed", "failed"):
                        self.logger.debug(f"DAG is still in progress: {dag_id}")
                        return False

                    previous_state = self._terminal_dag_states.get(dag_id)
                    if previous_state is not None:
                        self.logger.debug(
                            f"DAG {dag_id} already handled as terminal state "
                            f"'{previous_state}', skipping duplicate resolution"
                        )
                        return False

                    self._terminal_dag_states[dag_id] = dag_state
                    claimed_terminal_state = True

                    if dag_id in self.active_dags:
                        del self.active_dags[dag_id]
                        self.logger.debug(
                            f"Removed DAG from cache: {dag_id}, size = {len(self.active_dags)}"
                        )

                    if dag_state == "failed":
                        cancelled = await self.repository.cancel_pending_jobs_for_dag(
                            dag_id=dag_id,
                            output_metadata={
                                "on_complete": "failed",
                                "cancel_reason": "dag_failed",
                                "terminal_dag_state": dag_state,
                                "resolved_by_job_id": job_id,
                            },
                        )
                        self.logger.info(
                            f"Cancelled {cancelled} pending jobs for failed DAG {dag_id}"
                        )

                    dag_jobs = await self.frontier.get_jobs_by_dag_id(dag_id)
                    stats = await self.frontier.finalize_dag(dag_id)
                    for dag_job in dag_jobs:
                        self._job_cache.pop(dag_job.id, None)

                    self.logger.info(
                        f"Resolved DAG status: {dag_id}, status={dag_state}, "
                        f"active_dag = {len(self.active_dags)}, finalize_stats={stats}"
                    )

                    await self._emit_dag_terminal_event(dag_state, work_info)
                    return True
                except (Exception, psycopg.Error):
                    if claimed_terminal_state:
                        self._terminal_dag_states.pop(dag_id, None)
                    raise
        except (Exception, psycopg.Error) as error:
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
        return available_slots_by_executor(self._semaphore_store)

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

        return await self.repository.lease_jobs(
            job_ids=ids,
            owner=self.lease_owner,
            ttl_seconds=self.lease_ttl_seconds,
            job_name=job_name,
        )

    async def _activate_from_lease_db(self, ids: list[str]) -> dict[str, str]:
        """
        Promote leased jobs to active in DB once dispatch is acknowledged.
        """
        if not ids:
            return {}

        return await self.repository.activate_from_lease(
            job_ids=ids,
            owner=self.lease_owner,
            run_ttl_seconds=self.run_ttl_seconds,
            gateway_instance_id=self.gateway_instance_id,
        )

    async def _extend_run_lease_db(
        self, ids: list[str], *, run_owner: str, run_attempt_id: str
    ) -> set[str]:
        """
        Extend active run leases for the current durable attempt.
        """
        if not ids:
            return set()

        return await self.repository.extend_run_lease(
            job_ids=ids,
            owner=run_owner,
            run_attempt_id=run_attempt_id,
            extend_seconds=self.run_ttl_seconds,
        )

    async def _release_lease_db(self, ids: list[str]) -> set[str]:
        """
        Release DB leases for the given job ids if dispatch fails or needs retry.
        """
        if not ids:
            return set()

        return await self.repository.release_lease(job_ids=ids)

    async def _reserve_semaphore_slots_serial(
        self, executor: str, jobs: list[WorkInfo]
    ) -> set[str]:
        reserved: set[str] = set()
        for wi in jobs:
            try:
                ok = await asyncio.to_thread(
                    self._semaphore_store.reserve,
                    executor,
                    wi.id,
                    node='',
                    ttl=self._sem_default_ttl,
                    owner=wi.id,
                )
            except Exception as error:
                scheduler_trace(
                    "slot_reserve_failed",
                    job_id=wi.id,
                    dag_id=wi.dag_id,
                    executor=executor,
                    owner=wi.id,
                    error=repr(error),
                )
                self.logger.error(
                    f"[WORK_DIST] Semaphore reserve ERROR for job={wi.id}, executor={executor}: {error}",
                    exc_info=True,
                )
                continue
            if ok:
                reserved.add(wi.id)
        return reserved

    async def _reserve_semaphore_slots(
        self, executor: str, jobs: list[WorkInfo]
    ) -> set[str]:
        if not jobs:
            return set()

        job_ids = [wi.id for wi in jobs]
        started = time.perf_counter()
        fallback_used = False
        error: str | None = None
        scheduler_trace(
            "semaphore_reserve_batch_start",
            executor=executor,
            requested=len(job_ids),
            job_ids=job_ids,
        )
        try:
            reserved = await asyncio.to_thread(
                self._semaphore_store.reserve_many,
                executor,
                job_ids,
                node='',
                ttl=self._sem_default_ttl,
                owner_by_ticket={wi.id: wi.id for wi in jobs},
            )
        except Exception as exc:
            fallback_used = True
            error = repr(exc)
            self.logger.error(
                f"[WORK_DIST] Batch semaphore reserve failed for executor={executor}; falling back to serial reserve: {exc}",
                exc_info=True,
            )
            reserved = await self._reserve_semaphore_slots_serial(executor, jobs)

        scheduler_trace(
            "semaphore_reserve_batch_done",
            executor=executor,
            requested=len(job_ids),
            reserved=len(reserved),
            fallback_used=fallback_used,
            error=error,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            job_ids=list(reserved),
        )
        return set(reserved)

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
