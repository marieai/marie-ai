import asyncio
import random
import socket
import time
import traceback
import uuid as _uuid
from asyncio import Queue
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import ceil
from typing import Any, Dict, List, Optional

import psycopg

from marie.constants import JOB_STATUS_NOTIFICATION_CHANNEL
from marie.excepts import BadConfigSource, RuntimeFailToStart
from marie.job.common import JobInfo, JobStatus
from marie.job.job_manager import JobManager
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.messaging import mark_as_complete as mark_as_complete_toast
from marie.messaging import mark_as_failed as mark_as_failed_toast
from marie.query_planner.base import (
    QueryPlan,
)
from marie.query_planner.builtin import register_all_known_planners
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.in_memory_scheduling_engine import InMemorySchedulingEngine
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.job_scheduler import JobScheduler
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import (
    ExistingWorkPolicy,
    RecoveredRunLease,
    WorkInfo,
)
from marie.scheduler.planner_util import (
    debug_candidates_and_plan,
)
from marie.scheduler.postgres_scheduler_config import PostgreSQLSchedulerConfig
from marie.scheduler.repository import JobRepository
from marie.scheduler.services import (
    TERMINAL_EVENT_STALE_ATTEMPT_TOTAL,
    AttemptLifecycleService,
    ControlFlowExecutionOutcome,
    ControlFlowExecutionService,
    DAGManagementService,
    DagSubmissionService,
    MaintenanceService,
    NotificationService,
    SchedulerDiagnostics,
    SchedulerRuntime,
)
from marie.scheduler.state import WorkState
from marie.scheduler.util import (
    adjust_backoff,
    available_slots_by_executor,
    convert_job_status_to_work_state,
    executor_name,
    frontier_candidate_window,
    is_control_flow_entrypoint,
    ordered_leased_jobs,
)
from marie.serve.discovery.registry import _is_known_connection_error
from marie.serve.runtimes.servers.cluster_state import ClusterState
from marie.state.semaphore_store import SemaphoreStore
from marie.state.slot_capacity_manager import SlotCapacityManager
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool
from marie.utils.scheduler_trace import scheduler_trace, scheduler_trace_enabled
from marie.utils.utils import current_milli_time

INIT_POLL_PERIOD = 0.5  # initial idle wait before the first scheduler wake
SHORT_POLL_INTERVAL = 0.250  # fallback wait when a wake is missed or no work is visible
SLOT_POLL_INTERVAL = 0.100  # busy wait while executor work is blocked only by slots

MIN_POLL_PERIOD = 0.250
MAX_POLL_PERIOD = 8

SYNC_POLL_PERIOD = 60.0  # 60s — safety net, not primary dispatch path
EVENT_LOOP_LAG_INTERVAL_SECONDS = 0.1
EVENT_LOOP_TASK_SAMPLE_INTERVAL = 10
EVENT_LOOP_TASK_GROUP_LIMIT = 10

_DYNAMIC_TASK_NAME_PREFIXES = (
    'finalize:',
    'job:',
    'scheduler-dispatch-',
    'supervisor:',
)

DEFAULT_SCHEMA = "marie_scheduler"
DEFAULT_JOB_TABLE = "job"
RUN_LEASE_EXTEND_STALE_ATTEMPT_TOTAL = "run_lease_extend_stale_attempt_total"
RUN_LEASE_RECOVERED_RETRY_TOTAL = "run_lease_recovered_retry_total"
RUN_LEASE_RECOVERED_FAILED_TOTAL = "run_lease_recovered_failed_total"


def _task_group_name(task: asyncio.Task[Any]) -> str:
    name = task.get_name()
    for prefix in _DYNAMIC_TASK_NAME_PREFIXES:
        if name.startswith(prefix):
            return f'{prefix}*'
    if name.startswith('Task-') and name[5:].isdigit():
        coroutine = task.get_coro()
        return str(getattr(coroutine, '__qualname__', type(coroutine).__qualname__))
    return name


def _bounded_task_name_counts(
    tasks: set[asyncio.Task[Any]],
    limit: int = EVENT_LOOP_TASK_GROUP_LIMIT,
) -> tuple[dict[str, int], int]:
    counts = Counter(_task_group_name(task) for task in tasks)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    selected = dict(ranked[:limit])
    other = sum(count for _, count in ranked[limit:])
    return selected, other


@dataclass(frozen=True, slots=True)
class DispatchCycleResult:
    """Outcome used by the poll loop to choose its next wait."""

    scheduled: bool
    wait_interval: float | None = None


@dataclass(frozen=True, slots=True)
class _PendingDispatch:
    work_info: WorkInfo
    executor: str
    semaphore_owner: str
    run_owner: str
    run_attempt_id: str


class SemaphoreReservationStatus(str, Enum):
    RESERVED = "reserved"
    CAPACITY_FULL = "capacity_full"
    TICKET_EXISTS = "ticket_exists"
    CONTENTION = "contention"
    STORE_ERROR = "store_error"


@dataclass(frozen=True, slots=True)
class ControlFlowBatchResult:
    outcomes: tuple[ControlFlowExecutionOutcome, ...] = ()
    reconciled: int = 0

    @property
    def completed(self) -> int:
        return sum(outcome.made_progress for outcome in self.outcomes)

    @property
    def made_progress(self) -> bool:
        return self.completed > 0 or self.reconciled > 0


@dataclass(frozen=True, slots=True)
class PriorityRefreshResult:
    refresh_id: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


# FIXME : Today we are tracking at the executor level, however that might not be the best
# approach. We might want to track at the deployment level (endpoint level) instead.
# this will allow us to track the status of the deployment and not just the executor.


class PostgreSQLJobScheduler(JobScheduler):
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
        scheduler_config = self.validate_config(config)
        self.scheduler_config = scheduler_config
        self.config = config  # Store config for listener setup
        self._fetch_counter = 0
        self._debounced_notify = False

        self.known_queues = set(scheduler_config.queue_names)
        self.running = False
        self._paused = False
        self._lifecycle_lock = asyncio.Lock()
        self._event_subscriptions_active = False
        self.runtime = SchedulerRuntime(self.logger)
        self._scheduler_counters = defaultdict(int)

        self._event_queue = Queue()
        self._status_update_lock = AsyncJobLock()

        self.logger.info(f"Queue names to monitor: {self.known_queues}")

        self.job_manager = job_manager
        self.dispatch_confirmation_max_in_flight = (
            scheduler_config.dispatch_confirmation_max_in_flight
        )
        self._pending_dispatches: dict[str, asyncio.Task[None]] = {}
        self._db_pool = AsyncPostgresConnectionPool()
        self.repository = JobRepository(
            config,
            pool=self._db_pool,
        )
        self.notification_service = NotificationService(config)

        self.sla_priority_interval_seconds = (
            scheduler_config.sla_priority_interval_seconds
        )

        # The scheduler reads active_dags for planning; DAGManagementService mutates it.
        self.frontier = MemoryFrontier(
            sla_priority_interval_seconds=self.sla_priority_interval_seconds
        )
        self.active_dags = {}
        self._job_cache: dict[str, WorkInfo] = {}
        self._dag_admission_lock = asyncio.Lock()

        self.max_concurrent_dags = scheduler_config.max_concurrent_dags
        self._dag_resolution_retry_limit = scheduler_config.dag_resolution_retry_limit
        self._dag_resolution_retry_delay = scheduler_config.dag_resolution_retry_delay
        self._dag_resolution_retry_backoff = (
            scheduler_config.dag_resolution_retry_backoff
        )
        self._dag_resolution_retry_max_delay = (
            scheduler_config.dag_resolution_retry_max_delay
        )

        self.dag_service = DAGManagementService(
            repository=self.repository,
            frontier=self.frontier,
            active_dags=self.active_dags,
            notify_callback=self.notify_event,
            max_active_dags=self.max_concurrent_dags,
            admission_lock=self._dag_admission_lock,
            slot_snapshot_provider=self.get_available_slots,
            job_cache=self._job_cache,
            terminal_event_callback=self._emit_dag_terminal_event,
            resolution_retry_limit=self._dag_resolution_retry_limit,
            resolution_retry_delay=self._dag_resolution_retry_delay,
            resolution_retry_backoff=self._dag_resolution_retry_backoff,
            resolution_retry_max_delay=self._dag_resolution_retry_max_delay,
            admission_batch_size=scheduler_config.priority_refresh_hydrate_limit,
            sla_priority_interval_seconds=self.sla_priority_interval_seconds,
        )

        # Register handler for DAG state changes (delegate to DAGManagementService)
        self.notification_service.register_handler(
            channel='dag_state_changed', handler=self.dag_service.handle_state_change
        )
        self.notification_service.register_handler(
            channel=JOB_STATUS_NOTIFICATION_CHANNEL,
            handler=self.job_manager.handle_job_status_notification,
        )

        # Initialize MaintenanceService for periodic cleanup tasks
        self._maintenance_interval = scheduler_config.maintenance_interval
        self.maintenance_service = MaintenanceService(
            repository=self.repository,
            notify_callback=self.notify_event,
            recovery_callback=self._reconcile_recovered_run_leases,
            maintenance_interval=self._maintenance_interval,
        )

        self.scheduling_engine = InMemorySchedulingEngine(
            self.frontier,
            sla_priority_interval_seconds=self.sla_priority_interval_seconds,
        )
        self.logger.info(
            "SLA priority interval configured to %ss",
            self.sla_priority_interval_seconds,
        )
        register_all_known_planners(scheduler_config.query_planners)

        self._topology_cache = DagTopologyCache(maxsize=scheduler_config.dag_cache_size)

        self._resources_closed = False

        self._start_time = datetime.now(timezone.utc)
        self.sla_warning_top_n = scheduler_config.sla_warning_top_n
        self.priority_refresh_enabled = scheduler_config.priority_refresh_enabled
        self.priority_refresh_interval = scheduler_config.priority_refresh_interval
        self.priority_refresh_interval_seconds = (
            scheduler_config.priority_refresh_interval_seconds
        )
        self.priority_refresh_timeout_seconds = (
            scheduler_config.priority_refresh_timeout_seconds
        )
        self.priority_refresh_hydrate_limit = (
            scheduler_config.priority_refresh_hydrate_limit
        )
        self._priority_refresh_seq = 0
        self._priority_refresh_event = asyncio.Event()
        self._priority_refresh_source = "startup"
        self._priority_refresh_running = False
        self._next_priority_refresh_at = (
            time.monotonic() + self.priority_refresh_interval_seconds
            if self.priority_refresh_enabled
            else float('inf')
        )

        self.frontier_batch_size = scheduler_config.frontier_batch_size
        self.lease_ttl_seconds = scheduler_config.lease_ttl_seconds
        self.run_ttl_seconds = scheduler_config.run_ttl_seconds
        self.run_lease_renewal_interval_seconds = (
            scheduler_config.run_lease_renewal_interval_seconds
        )
        # unique, stable lease owner for this scheduler instance
        self.lease_owner: str = f"{socket.gethostname()}:{_uuid.uuid4()}"
        self.gateway_instance_id = (
            scheduler_config.gateway_instance_id or self.lease_owner
        )
        self.control_flow_service = self._build_control_flow_service()
        self.attempt_lifecycle_service = self._build_attempt_lifecycle_service()
        self.logger.info(
            f"Lease config: lease_ttl_seconds={self.lease_ttl_seconds}, "
            f"run_ttl_seconds={self.run_ttl_seconds}, "
            "run_lease_renewal_interval_seconds="
            f"{self.run_lease_renewal_interval_seconds}, "
            f"owner='{self.lease_owner}', "
            f"gateway_instance_id='{self.gateway_instance_id}'"
        )

        # Semaphore-based capacity control, we hijaced the _etcd_client client here from job manager
        self._semaphore_store = SemaphoreStore(
            self.job_manager._etcd_client, default_lease_ttl=30
        )
        self._sem_default_ttl = 30
        self._ticket_collision_counts: dict[str, int] = {}

        self.capacity_manager = SlotCapacityManager(
            semaphore_store=self._semaphore_store,
            logger=self.logger,
            # Optional mapping if slot types differ from executor names:
            # slot_type_resolver=lambda executor: {"extract_executor": "ocr.gpu"}.get(executor, executor),
        )
        self.cycle_log_interval_seconds = 10.0
        self.submission_service = self._build_submission_service()
        self.diagnostics = self._build_diagnostics()

    def _build_submission_service(
        self,
        *,
        initial_submission_count: int = 0,
    ) -> DagSubmissionService:
        return DagSubmissionService(
            repository=self.repository,
            dag_admission_callback=self.dag_service.request_admission,
            known_queues=self.known_queues,
            notify_callback=self.notify_event,
            is_running=lambda: self.running,
            submission_processed_callback=(
                self._handle_priority_refresh if self.priority_refresh_enabled else None
            ),
            logger=self.logger,
            initial_submission_count=initial_submission_count,
        )

    def _build_diagnostics(self) -> SchedulerDiagnostics:
        return SchedulerDiagnostics(
            repository=self.repository,
            frontier=self.frontier,
            submission_service=self.submission_service,
            event_queue=self._event_queue,
            active_dags=self.active_dags,
            known_queues=self.known_queues,
            scheduling_engine=self.scheduling_engine,
            gateway_instance_id=self.gateway_instance_id,
            lease_owner=self.lease_owner,
            max_concurrent_dags=self.max_concurrent_dags,
            start_time=self._start_time,
            sla_warning_top_n=self.sla_warning_top_n,
            frontier_batch_size=self.frontier_batch_size,
            lease_ttl_seconds=self.lease_ttl_seconds,
        )

    def _build_control_flow_service(self) -> ControlFlowExecutionService:
        return ControlFlowExecutionService(
            repository=self.repository,
            frontier=self.frontier,
            dag_service=self.dag_service,
            status_update_lock=self._status_update_lock,
            topology_cache=self._topology_cache,
            job_cache=self._job_cache,
            lease_owner=self.lease_owner,
            run_ttl_seconds=self.run_ttl_seconds,
            gateway_instance_id=self.gateway_instance_id,
            notify_callback=self.notify_event,
            runtime=self.runtime,
        )

    def _build_attempt_lifecycle_service(self) -> AttemptLifecycleService:
        return AttemptLifecycleService(
            repository=self.repository,
            frontier=self.frontier,
            dag_service=self.dag_service,
            control_flow_service=self.control_flow_service,
            status_update_lock=self._status_update_lock,
            job_cache=self._job_cache,
            scheduler_lease_owner=self.lease_owner,
            gateway_instance_id=self.gateway_instance_id,
            notify_callback=self.notify_event,
            counter_callback=self._scheduler_counter,
        )

    def _ha_trace_fields(self) -> dict[str, str]:
        return {
            "gateway_instance_id": self.gateway_instance_id,
            "scheduler_lease_owner": self.lease_owner,
        }

    def validate_config(
        self,
        config: Dict[str, Any],
    ) -> PostgreSQLSchedulerConfig:
        return PostgreSQLSchedulerConfig.from_dict(config)

    def _scheduler_counter(self, name: str, **fields: Any) -> None:
        self._scheduler_counters[name] += 1
        scheduler_trace(name, count=self._scheduler_counters[name], **fields)

    async def handle_job_event(self, event_type: str, message: Any) -> None:
        """Apply a job event delivered by JobManager's keyed publisher."""
        self.logger.debug(f"received message: {event_type} > {message}")

        if not isinstance(message, dict) or "job_id" not in message:
            self.logger.error(f"Invalid message format: {message}")
            return

        job_id = message.get("job_id")
        status = JobStatus(event_type)
        scheduler_trace(
            "scheduler_job_event_received",
            job_id=job_id,
            status=status.value,
        )
        work_item: Optional[WorkInfo] = await self.get_job(job_id)

        if work_item is None:
            raise ValueError(f"WorkItem not found: {job_id}")

        run_owner = message.get("run_owner")
        run_attempt_id = message.get("run_attempt_id")

        if status == JobStatus.PENDING:
            self.logger.debug(f"Job pending : {job_id}")
        elif status.is_terminal():
            replace_kwargs = message.get("jobinfo_replace_kwargs")
            runtime_env = (
                replace_kwargs.get("runtime_env")
                if isinstance(replace_kwargs, dict)
                else None
            )
            await self.attempt_lifecycle_service.transition_terminal(
                job_id,
                work_item,
                status,
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                source="job_event",
                message=message.get("message"),
                runtime_env=runtime_env,
            )
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

            work_state = convert_job_status_to_work_state(status)
            work_item.state = work_state
            work_item.run_owner = run_owner
            work_item.run_attempt_id = run_attempt_id
            self._job_cache[job_id] = work_item
            await self.frontier.update_job_state(job_id, work_state)
            self.logger.debug(f"Job running : {job_id}")
        else:
            self.logger.error(f"Unhandled job status: {status}")

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
            removed = await self.dag_service.evict_dag(wi.dag_id, reason)
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

    async def _process_control_flow_candidates(
        self, control_flow_jobs: list[WorkInfo], lease_ttl: float
    ) -> ControlFlowBatchResult:
        if not control_flow_jobs:
            return ControlFlowBatchResult()

        processable_jobs: list[WorkInfo] = []
        reconciled_count = 0

        requested_ids = [wi.id for wi in control_flow_jobs]
        taken_wis = await self.frontier.take(requested_ids, lease_ttl=lease_ttl)
        taken_ids = {wi.id for wi in taken_wis}
        missing_ids = [job_id for job_id in requested_ids if job_id not in taken_ids]
        if missing_ids:
            self.logger.warning(
                "[WORK_DIST] Failed to take %d/%d control flow nodes from frontier: %s",
                len(missing_ids),
                len(requested_ids),
                missing_ids[:10],
            )

        jobs_by_name: dict[str, list[WorkInfo]] = defaultdict(list)
        for wi in taken_wis:
            jobs_by_name[wi.name].append(wi)

        for job_name, jobs in jobs_by_name.items():
            job_ids = [wi.id for wi in jobs]
            try:
                leased_ids = await self._lease_jobs_db(job_name, job_ids)
            except Exception as e:
                self.logger.error(
                    f"[WORK_DIST] Error leasing {len(job_ids)} control flow nodes "
                    f"for job name {job_name}: {e}"
                )
                for wi in jobs:
                    await self.frontier.release_lease_local(wi.id)
                continue

            for wi in jobs:
                if wi.id in leased_ids:
                    processable_jobs.append(wi)
                    continue

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
                    processable_jobs.append(db_wi)
                    continue

                reconciled = await self._reconcile_control_flow_lease_miss(wi, db_wi)
                reconciled_count += int(reconciled)

        if not processable_jobs:
            return ControlFlowBatchResult(reconciled=reconciled_count)

        outcomes = await self.control_flow_service.process_nodes(processable_jobs)

        return ControlFlowBatchResult(
            outcomes=tuple(outcomes),
            reconciled=reconciled_count,
        )

    # ==================== Schema Management (Delegated to Repository) ====================

    async def create_tables(self, schema: str) -> None:
        """
        Create all database tables, functions, and triggers.
        Delegates to JobRepository.

        :param schema: The name of the schema where the tables will be created
        :return: None
        """
        await self.repository.create_tables(schema)

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
        """Start the scheduler unless it is already running."""
        async with self._lifecycle_lock:
            if self.running:
                self.logger.warning("Job scheduler is already running")
                return
            if self._resources_closed:
                await self._reopen_runtime_resources()
            try:
                await self._start_locked()
                self._setup_event_subscriptions()
            except BaseException:
                try:
                    await self._stop_locked(timeout=2.0)
                except Exception as rollback_error:
                    self.logger.error(
                        f"Scheduler startup rollback failed: {rollback_error}",
                        exc_info=True,
                    )
                raise

    async def _start_locked(self) -> None:
        """
        Starts the job scheduling agent.

        :return: None
        """
        logger.info("Starting job scheduling agent")
        await self.repository.initialize()
        # Check if tables are installed and create if needed (delegate to repository)
        installed = await self.repository.is_installed(DEFAULT_SCHEMA)
        logger.info(f"Tables installed: {installed}")
        if not installed:
            await self.repository.create_tables(DEFAULT_SCHEMA)

        await self.repository.validate_durable_scheduler_schema(DEFAULT_SCHEMA)

        # Get defined queues from repository
        defined_queues = await self.repository.get_defined_queues(DEFAULT_SCHEMA)
        for work_queue in self.known_queues.difference(defined_queues):
            self.logger.info(f"Create queue: {work_queue}")
            await self.repository.create_queue(work_queue)
            await self.repository.create_queue(f"${work_queue}_dlq")

        # Start the NotificationService before admission or polling so DAG
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

        reconcile_summary = await asyncio.to_thread(
            self._semaphore_store.reconcile_all,
            delete_orphan_holders=True,
            fix_counters=True,
        )
        self.logger.info(f"[sem] startup reconciliation: {reconcile_summary}")

        await self.dag_service.start_admission()

        await self._renew_active_run_leases()
        await self.maintenance_service.start()
        self.logger.info(
            f"Started MaintenanceService (interval: {self.maintenance_service.maintenance_interval}s)"
        )
        await self.dag_service.start_sync()

        if self.priority_refresh_enabled:
            self._priority_refresh_event.clear()
        self.running = True
        if self.priority_refresh_enabled:
            self.runtime.create_task(
                self._priority_refresh_loop(),
                name="scheduler-priority-refresh",
            )
        self.runtime.create_task(self._sync(), name="scheduler-sync")
        self.runtime.create_task(
            self._renew_run_leases(), name="scheduler-run-lease-renewal"
        )

        self.runtime.create_task(self._poll(), name="scheduler-poll")
        self.runtime.create_task(
            self.__monitor_deployment_updates(),
            name="scheduler-deployment-monitor",
        )
        if scheduler_trace_enabled():
            self.runtime.create_task(
                self._event_loop_lag_watchdog(),
                name="gateway-event-loop-lag",
            )

        await self.notify_event()

    async def _event_loop_lag_watchdog(
        self,
        interval_seconds: float = EVENT_LOOP_LAG_INTERVAL_SECONDS,
    ) -> None:
        loop = asyncio.get_running_loop()
        sample_count = 0
        while self.running:
            expected_at = loop.time() + interval_seconds
            await asyncio.sleep(interval_seconds)
            observed_at = loop.time()
            sample_count += 1
            trace_fields: dict[str, Any] = {
                "lag_ms": max(0.0, observed_at - expected_at) * 1000.0,
                "interval_ms": interval_seconds * 1000.0,
            }
            if sample_count % EVENT_LOOP_TASK_SAMPLE_INTERVAL == 0:
                tasks = asyncio.all_tasks(loop)
                task_names, other = _bounded_task_name_counts(tasks)
                trace_fields.update(
                    task_count=len(tasks),
                    task_names=task_names,
                    task_names_other=other,
                )
            scheduler_trace("gateway_event_loop_lag", **trace_fields)

    async def run_dispatch_cycle(self, cycle_index: int) -> DispatchCycleResult:
        """Run one candidate, lease, activation, and dispatch attempt."""
        cycle_phase_started = time.perf_counter()
        batch_size = self.frontier_batch_size
        max_concurrent_dags = self.max_concurrent_dags
        lease_ttl = self.lease_ttl_seconds
        scheduled_any = False
        compact_ready_heap = cycle_index % 20 == 0

        if self.running:
            try:
                if (
                    self.priority_refresh_enabled
                    and time.monotonic() >= self._next_priority_refresh_at
                ):
                    scheduler_trace(
                        "scheduler_priority_refresh_due",
                        source="scheduler_loop",
                        submission_count=self.submission_service.submission_count,
                        refresh_interval_seconds=self.priority_refresh_interval_seconds,
                    )
                    self._request_priority_refresh(source="scheduler_loop")

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
                    if cycle_index % 10 == 0:
                        self.logger.warning(
                            f"[WORK_DIST] Gateway not ready yet. Scheduler will wait. "
                            f"Queue size: {self._event_queue.qsize()}"
                        )
                    return DispatchCycleResult(scheduled=False)

                # Check if scheduler is paused before dispatching work
                if self._paused:
                    if cycle_index % 10 == 0:
                        self.logger.info(
                            f"[WORK_DIST] Scheduler is paused. Skipping dispatch. "
                            f"Queue size: {self._event_queue.qsize()}"
                        )
                    return DispatchCycleResult(scheduled=False)

                capacity_started = time.perf_counter()
                slots_by_executor = available_slots_by_executor(
                    self._semaphore_store
                ).copy()
                dispatch_capacity = max(
                    0,
                    self.dispatch_confirmation_max_in_flight
                    - len(self._pending_dispatches),
                )
                if dispatch_capacity == 0:
                    scheduler_trace(
                        "dispatch_confirmation_backpressure",
                        pending=len(self._pending_dispatches),
                        limit=self.dispatch_confirmation_max_in_flight,
                    )
                    slots_by_executor = {executor: 0 for executor in slots_by_executor}

                no_executor_slots = not any(slots_by_executor.values())
                capacity_captured_at = time.perf_counter()
                scheduler_trace(
                    "scheduler_dispatch_capacity_snapshot",
                    cycle_index=cycle_index,
                    elapsed_ms=(capacity_captured_at - capacity_started) * 1000.0,
                    cycle_elapsed_ms=(capacity_captured_at - cycle_phase_started)
                    * 1000.0,
                    slots_by_executor=dict(slots_by_executor),
                    available_slots=sum(slots_by_executor.values()),
                    dispatch_capacity=dispatch_capacity,
                    pending_dispatches=len(self._pending_dispatches),
                    active_dags=len(self.active_dags),
                )
                self.logger.debug(f"[WORK_DIST] Available slots: {slots_by_executor}")

                # Control-flow work does not consume executor slots. Process one
                # ready wave before regular selection; another ready wave causes
                # the poll loop to run an immediate drain cycle.
                candidate_window = frontier_candidate_window(
                    batch_size, slots_by_executor
                )

                control_flow_outcomes: Counter[ControlFlowExecutionOutcome] = Counter()
                control_flow_reconciled_total = 0

                def dag_admission_filter(wi: WorkInfo) -> bool:
                    return (
                        wi.dag_id in self.active_dags
                        or len(self.active_dags) < max_concurrent_dags
                    )

                def control_flow_filter(wi: WorkInfo) -> bool:
                    metadata = (
                        wi.data.get("metadata", {}) if isinstance(wi.data, dict) else {}
                    )
                    entrypoint = (
                        metadata.get("on", "") if isinstance(metadata, dict) else ""
                    )
                    return dag_admission_filter(wi) and is_control_flow_entrypoint(
                        entrypoint
                    )

                peek_started = time.perf_counter()
                visible_control_flow = await self.frontier.peek_ready(
                    candidate_window,
                    filter_fn=control_flow_filter,
                )
                control_flow_jobs = [
                    wi
                    for wi in visible_control_flow
                    if is_control_flow_entrypoint(
                        wi.data.get("metadata", {}).get("on", "")
                    )
                ]
                scheduler_trace(
                    "scheduler_control_flow_peek_completed",
                    elapsed_ms=(time.perf_counter() - peek_started) * 1000.0,
                    candidate_window=candidate_window,
                    visible_jobs=len(control_flow_jobs),
                    no_executor_slots=no_executor_slots,
                )

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

                        if admission_slots > 0 and wi.dag_id not in selected_new_dags:
                            selected_control_flow.append(wi)
                            selected_new_dags.add(wi.dag_id)
                            admission_slots -= 1

                    control_flow_jobs = selected_control_flow

                if control_flow_jobs:
                    self.logger.debug(
                        f"[WORK_DIST] Processing {len(control_flow_jobs)} ready control flow nodes"
                    )
                    control_flow_started = time.perf_counter()
                    batch_result = await self._process_control_flow_candidates(
                        control_flow_jobs, lease_ttl
                    )
                    control_flow_outcomes.update(batch_result.outcomes)
                    control_flow_reconciled_total += batch_result.reconciled
                    scheduled_any = scheduled_any or batch_result.made_progress
                    scheduler_trace(
                        "scheduler_control_flow_batch_completed",
                        jobs=len(control_flow_jobs),
                        completed=batch_result.completed,
                        reconciled=batch_result.reconciled,
                        made_progress=batch_result.made_progress,
                        outcomes={
                            outcome.value: count
                            for outcome, count in Counter(batch_result.outcomes).items()
                        },
                        elapsed_ms=(time.perf_counter() - control_flow_started)
                        * 1000.0,
                    )

                control_flow_completed_total = sum(
                    count
                    for outcome, count in control_flow_outcomes.items()
                    if outcome.made_progress
                )
                control_flow_progress_total = (
                    control_flow_completed_total + control_flow_reconciled_total
                )
                control_flow_outcome_counts = {
                    outcome.value: count
                    for outcome, count in sorted(
                        control_flow_outcomes.items(), key=lambda item: item[0].value
                    )
                }

                selection_started = time.perf_counter()
                selection = await self.scheduling_engine.select_ready(
                    slots_by_executor=slots_by_executor.copy(),
                    batch_size=batch_size,
                    dispatch_capacity=dispatch_capacity,
                    lease_ttl=lease_ttl,
                    resident_dag_ids=set(self.active_dags),
                    max_resident_dags=max_concurrent_dags,
                )
                candidates_captured_at = time.perf_counter()
                candidate_window = selection.candidate_window
                regular_candidates = list(selection.candidates)
                scheduler_trace(
                    "scheduler_dispatch_candidate_capture_completed",
                    cycle_index=cycle_index,
                    elapsed_ms=(candidates_captured_at - selection_started) * 1000.0,
                    capacity_to_capture_ms=(
                        candidates_captured_at - capacity_captured_at
                    )
                    * 1000.0,
                    cycle_elapsed_ms=(candidates_captured_at - cycle_phase_started)
                    * 1000.0,
                    candidate_window=candidate_window,
                    candidates=len(regular_candidates),
                    control_flow_seen=len(control_flow_jobs),
                    control_flow_batch=bool(control_flow_jobs),
                    no_executor_slots=no_executor_slots,
                )

                if not regular_candidates and control_flow_progress_total == 0:
                    if no_executor_slots:
                        self.logger.debug(
                            f"[WORK_DIST] No available executor slots and no control flow nodes. Backing off. "
                            f"Slots by executor: {slots_by_executor}"
                        )
                    else:
                        frontier_summary = self.frontier.summary(detail=False)
                        self.logger.debug(
                            f"[WORK_DIST] No ready work in frontier. Short sleep. "
                            f"Batch size: {batch_size} | "
                            f"Candidate window: {candidate_window} | "
                            f"Frontier summary: {frontier_summary}"
                        )
                    poll_interval = (
                        SLOT_POLL_INTERVAL if no_executor_slots else SHORT_POLL_INTERVAL
                    )
                    return DispatchCycleResult(
                        scheduled=False,
                        wait_interval=poll_interval,
                    )

                self.logger.debug(
                    f"[WORK_DIST] Fetched {len(regular_candidates)} candidates from frontier. "
                )

                self.logger.debug(
                    f"[WORK_DIST] Built {len(regular_candidates)} planner candidates "
                    f"(+{control_flow_completed_total} completed, {control_flow_reconciled_total} reconciled "
                    f"of {len(control_flow_jobs)} control flow nodes; outcomes={control_flow_outcome_counts}). "
                    f"Executors needed: {set(ep for ep, _ in selection.ranked)}"
                )
                scheduler_trace(
                    "candidate_built",
                    candidates=len(regular_candidates),
                    regular_jobs=len(regular_candidates),
                    control_flow_jobs=control_flow_completed_total,
                    control_flow_completed=control_flow_completed_total,
                    control_flow_reconciled=control_flow_reconciled_total,
                    control_flow_outcomes=control_flow_outcome_counts,
                    control_flow_seen=len(control_flow_jobs),
                    control_flow_batch=bool(control_flow_jobs),
                    executors=sorted(
                        {ep.split("://", 1)[0] for ep, _ in selection.ranked}
                    ),
                    slots_by_executor=dict(selection.slots_by_executor),
                    active_dags=len(self.active_dags),
                    max_concurrent_dags=max_concurrent_dags,
                    job_ids=list(selection.candidate_ids),
                    capture_limit=candidate_window,
                    capture_eligible_by_executor=dict(selection.eligible_by_executor),
                    capture_selected_by_executor=dict(selection.captured_by_executor),
                    capture_eligible_by_dag=dict(selection.eligible_by_dag),
                    capture_selected_by_dag=dict(selection.captured_by_dag),
                )

                if not regular_candidates:
                    self.logger.debug(
                        f"[WORK_DIST] No regular jobs to plan ({control_flow_completed_total} completed, "
                        f"{control_flow_reconciled_total} reconciled of {len(control_flow_jobs)} "
                        f"control flow nodes; outcomes={control_flow_outcome_counts})"
                    )
                    return DispatchCycleResult(
                        scheduled=scheduled_any,
                        wait_interval=0.0 if scheduled_any else None,
                    )

                pick_slots = dict(selection.slots_by_executor)
                await debug_candidates_and_plan(
                    regular_candidates,
                    list(selection.ranked),
                    pick_slots,
                    self.active_dags,
                    self.frontier,
                )
                scheduler_trace(
                    "planner_selected",
                    planned=len(selection.ranked),
                    limited=len(selection.requested),
                    slots=dict(pick_slots),
                    dispatch_capacity=dispatch_capacity,
                    pending_dispatches=len(self._pending_dispatches),
                    dispatch_confirmation_limit=(
                        self.dispatch_confirmation_max_in_flight
                    ),
                    job_ids=list(selection.requested_ids),
                )
                if len(selection.requested) < len(selection.ranked):
                    self.logger.debug(
                        f"[WORK_DIST] Trimmed planner selection from {len(selection.ranked)} "
                        f"to {len(selection.requested)} based on live slot and "
                        f"dispatch-confirmation capacity. "
                        f"Slots: {pick_slots}"
                    )
                planned = list(selection.requested)
                if not planned:
                    candidates_by_executor = defaultdict(list)
                    for ep, wi in selection.ranked:
                        exe = ep.split("://", 1)[0]
                        candidates_by_executor[exe].append(wi.id)

                    active_dag_count = len(self.active_dags)
                    self.logger.debug(
                        f"[WORK_DIST] Planner returned NO picks. Short sleep. "
                        f"Candidates count: {len(regular_candidates)} | "
                        f"Candidates by executor: {dict(candidates_by_executor)} | "
                        f"Available slots: {pick_slots} | "
                        f"Active DAGs: {active_dag_count}/{max_concurrent_dags}"
                    )
                    return DispatchCycleResult(
                        scheduled=scheduled_any,
                        wait_interval=(0.0 if scheduled_any else SHORT_POLL_INTERVAL),
                    )

                self.logger.debug(
                    f"[WORK_DIST] Planner selected {len(planned)} jobs to schedule. "
                    f"Job IDs: {[wi.id for _, wi in planned[:10]]}"
                )

                selected_wis = list(selection.selected)
                taken = len(selected_wis)
                requested = len(selection.requested)
                scheduler_trace(
                    "frontier_taken",
                    requested=requested,
                    taken=taken,
                    job_ids=[wi.id for wi in selected_wis],
                )
                if taken != requested:
                    selected_ids = {wi.id for wi in selected_wis}
                    missing = [
                        job_id
                        for job_id in selection.requested_ids
                        if job_id not in selected_ids
                    ]
                    self.logger.warning(
                        f"[WORK_DIST] Not all jobs taken from frontier: taken={taken}/{requested}. "
                        f"Missing IDs: {missing[:10]}{'...' if len(missing) > 10 else ''}"
                    )

                ids_by_job_name: dict[str, list[str]] = defaultdict(list)
                for wi in selected_wis:
                    ids_by_job_name[wi.name].append(wi.id)

                leased_ids: set[str] = set()
                for job_name, ids in ids_by_job_name.items():
                    try:
                        self.logger.debug(
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
                    return DispatchCycleResult(
                        scheduled=scheduled_any,
                        wait_interval=(0.0 if scheduled_any else SHORT_POLL_INTERVAL),
                    )

                leased_jobs: list[tuple[str, WorkInfo]] = ordered_leased_jobs(
                    planned, leased_ids
                )
                jobs_scheduled_this_cycle = defaultdict(int)
                pending_dispatches: list[_PendingDispatch] = []
                reservable_jobs_by_executor: dict[str, list[WorkInfo]] = defaultdict(
                    list
                )
                slots_before_by_job: dict[str, int] = {}

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

                        admitted = await self.dag_service.admit_dag(
                            dag_id, dag, source="dispatch"
                        )
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
                    run_attempt_ids = {wi.id: str(_uuid.uuid4()) for wi in jobs}
                    reservation_results = await self._reserve_semaphore_slots(
                        slot_type, jobs, run_attempt_ids
                    )
                    reserved_jobs: list[WorkInfo] = []
                    for wi in jobs:
                        owner = wi.id
                        reservation_status = reservation_results.get(
                            wi.id, SemaphoreReservationStatus.CONTENTION
                        )
                        if (
                            reservation_status
                            != SemaphoreReservationStatus.TICKET_EXISTS
                        ):
                            self._ticket_collision_counts.pop(wi.id, None)
                        if reservation_status != SemaphoreReservationStatus.RESERVED:
                            scheduler_trace(
                                "slot_unavailable",
                                job_id=wi.id,
                                dag_id=wi.dag_id,
                                executor=slot_type,
                                reason=reservation_status.value,
                                slots_by_executor=dict(slots_by_executor),
                            )
                            self.logger.warning(
                                f"[WORK_DIST] Semaphore reservation rejected for "
                                f"executor={slot_type}, job={wi.id}, "
                                f"reason={reservation_status.value}; "
                                f"slots_by_executor={slots_by_executor}"
                            )
                            if (
                                reservation_status
                                == SemaphoreReservationStatus.TICKET_EXISTS
                            ):
                                collision_count = (
                                    self._ticket_collision_counts.get(wi.id, 0) + 1
                                )
                                self._ticket_collision_counts[wi.id] = collision_count
                                base_delay = min(
                                    30.0, 2.0 ** min(collision_count - 1, 5)
                                )
                                delay_seconds = min(
                                    30.0, base_delay * random.uniform(0.8, 1.2)
                                )

                                deferred = await self.repository.defer_leased_job(
                                    job_id=wi.id,
                                    owner=self.lease_owner,
                                    delay_seconds=delay_seconds,
                                )
                                if deferred:
                                    wi.start_after = datetime.now(
                                        timezone.utc
                                    ) + timedelta(seconds=delay_seconds)
                                    scheduler_trace(
                                        "slot_ticket_conflict_deferred",
                                        job_id=wi.id,
                                        dag_id=wi.dag_id,
                                        executor=slot_type,
                                        delay_seconds=delay_seconds,
                                        collision_count=collision_count,
                                    )
                                else:
                                    await self._release_lease_db([wi.id])
                            else:
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
                        reserved_attempt_ids = {
                            wi.id: run_attempt_ids[wi.id] for wi in reserved_jobs
                        }
                        attempts = await self._activate_from_lease_db(
                            [wi.id for wi in reserved_jobs],
                            reserved_attempt_ids,
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
                                    run_attempt_id=run_attempt_ids[wi.id],
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
                        pending_dispatches.append(
                            _PendingDispatch(
                                work_info=wi,
                                executor=slot_type,
                                semaphore_owner=owner,
                                run_owner=self.lease_owner,
                                run_attempt_id=run_attempt_id,
                            )
                        )

                if pending_dispatches:
                    scheduler_trace(
                        "dispatch_batch_start",
                        count=len(pending_dispatches),
                        job_ids=[item.work_info.id for item in pending_dispatches],
                    )
                    self.logger.debug(
                        f"[WORK_DIST] Launching {len(pending_dispatches)} dispatch confirmations"
                    )
                    for pending in pending_dispatches:
                        self._start_pending_dispatch(pending)
                        jobs_scheduled_this_cycle[pending.executor] += 1
                    scheduled_any = True
                    scheduler_trace(
                        "dispatch_batch_launched",
                        count=len(pending_dispatches),
                        pending=len(self._pending_dispatches),
                        limit=self.dispatch_confirmation_max_in_flight,
                    )

                if jobs_scheduled_this_cycle:
                    self.logger.debug("Scheduling summary for this cycle:")
                    for exe, cnt in sorted(jobs_scheduled_this_cycle.items()):
                        self.logger.debug(f"  - {exe}: {cnt} scheduled")

                return DispatchCycleResult(scheduled=scheduled_any)
            finally:
                if compact_ready_heap:
                    removed = await self.frontier.compact_ready_heap(max_scan=10000)
                    if removed:
                        self.logger.debug(f"Frontier heap compacted: removed={removed}")

        return DispatchCycleResult(scheduled=False)

    def _start_pending_dispatch(self, pending: _PendingDispatch) -> None:
        if len(self._pending_dispatches) >= self.dispatch_confirmation_max_in_flight:
            raise RuntimeError("Dispatch confirmation capacity exhausted")

        task = asyncio.create_task(
            self._settle_pending_dispatch(pending, time.perf_counter()),
            name=f"scheduler-dispatch-{pending.run_attempt_id}",
        )
        self._pending_dispatches[pending.run_attempt_id] = task
        self.runtime.track_event_task(task)
        task.add_done_callback(
            lambda done, attempt_id=pending.run_attempt_id: (
                self._discard_pending_dispatch(attempt_id, done)
            )
        )

    def _discard_pending_dispatch(
        self, run_attempt_id: str, task: asyncio.Task[None]
    ) -> None:
        if self._pending_dispatches.get(run_attempt_id) is task:
            self._pending_dispatches.pop(run_attempt_id, None)
        self.runtime.discard_event_task(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.logger.error(
                "Dispatch settlement task failed for run_attempt_id=%s: %s",
                run_attempt_id,
                error,
            )

    async def _settle_pending_dispatch(
        self, pending: _PendingDispatch, launched_at: float
    ) -> None:
        wi = pending.work_info
        try:
            result = await self._activate_and_enqueue_job(
                wi,
                run_owner=pending.run_owner,
                run_attempt_id=pending.run_attempt_id,
            )
        except asyncio.CancelledError:
            scheduler_trace(
                "dispatch_confirmation_settled",
                job_id=wi.id,
                dag_id=wi.dag_id,
                executor=pending.executor,
                run_attempt_id=pending.run_attempt_id,
                outcome="cancelled",
                elapsed_ms=(time.perf_counter() - launched_at) * 1000.0,
            )
            raise
        except Exception as error:
            result = error

        if isinstance(result, Exception) or not result:
            self.logger.error(
                f"[WORK_DIST] Dispatch FAILED for job={wi.id}, "
                f"executor={pending.executor}: {result}",
                exc_info=isinstance(result, Exception),
            )
            await self._handle_dispatch_failure(
                wi,
                pending.executor,
                pending.semaphore_owner,
                result,
                run_owner=pending.run_owner,
                run_attempt_id=pending.run_attempt_id,
            )
            outcome = "failed"
        else:
            outcome = "confirmed"

        scheduler_trace(
            "dispatch_confirmation_settled",
            job_id=wi.id,
            dag_id=wi.dag_id,
            executor=pending.executor,
            run_attempt_id=pending.run_attempt_id,
            outcome=outcome,
            elapsed_ms=(time.perf_counter() - launched_at) * 1000.0,
        )
        if self.running:
            await self.notify_event()

    async def _poll(self) -> None:
        """Wait for scheduler wakes and run dispatch cycles until stopped."""
        self.logger.info("Starting job scheduler")
        wait_time = INIT_POLL_PERIOD
        failures = 0
        idle_streak = 0
        cycle_index = 0
        drain_ready = False
        cycle_stats_started_at = time.perf_counter()
        cycle_total_samples: list[float] = []
        cycle_active_samples: list[float] = []
        trace_dispatch_cycles = scheduler_trace_enabled()

        while self.running:
            cycle_started_at = time.perf_counter()
            active_started_at: float | None = None
            wake_completed_at: float | None = None
            cycle_trigger = "drain"
            try:
                if not drain_ready:
                    effective_wait_time = wait_time
                    if self.priority_refresh_enabled:
                        priority_due_in = max(
                            0.0, self._next_priority_refresh_at - time.monotonic()
                        )
                        effective_wait_time = min(wait_time, priority_due_in)
                    self.logger.debug(
                        f"Polling : {effective_wait_time:.2f}s — "
                        f"Queue size: {self._event_queue.qsize()} — "
                        f"Idle streak: {idle_streak}"
                    )
                    wait_started_at = (
                        time.perf_counter() if trace_dispatch_cycles else None
                    )
                    woke = await self._wait_for_dispatch_wake(effective_wait_time)
                    cycle_trigger = "wake" if woke else "timeout"
                    if trace_dispatch_cycles:
                        wake_completed_at = time.perf_counter()
                        scheduler_trace(
                            "scheduler_dispatch_wait_completed",
                            cycle_index=cycle_index,
                            outcome=cycle_trigger,
                            elapsed_ms=(wake_completed_at - wait_started_at) * 1000.0,
                            requested_wait_ms=effective_wait_time * 1000.0,
                            queue_size=self._event_queue.qsize(),
                        )
                    if woke:
                        idle_streak = 0
                        wait_time = MIN_POLL_PERIOD

                active_started_at = time.perf_counter()
                if trace_dispatch_cycles:
                    trace_fields: dict[str, Any] = {
                        "cycle_index": cycle_index,
                        "trigger": cycle_trigger,
                        "queue_size": self._event_queue.qsize(),
                    }
                    if wake_completed_at is not None:
                        trace_fields["wait_to_cycle_ms"] = (
                            active_started_at - wake_completed_at
                        ) * 1000.0
                    scheduler_trace(
                        "scheduler_dispatch_cycle_started",
                        **trace_fields,
                    )
                self._fetch_counter += 1
                result = await self.run_dispatch_cycle(cycle_index)
                drain_ready = result.scheduled
                idle_streak = 0 if result.scheduled else idle_streak + 1
                if result.wait_interval is not None:
                    wait_time = result.wait_interval
                else:
                    wait_time = adjust_backoff(
                        wait_time,
                        idle_streak,
                        result.scheduled,
                        min_poll_period=MIN_POLL_PERIOD,
                        max_poll_period=MAX_POLL_PERIOD,
                    )
                failures = 0
            except Exception as error:
                drain_ready = False
                if _is_known_connection_error(error):
                    self.logger.warning(
                        "Poll loop: ETCD connection unavailable, waiting for reconnect"
                    )
                    wait_time = 3.0
                else:
                    self.logger.error("Poll loop exception", exc_info=True)
                    failures += 1
                    if failures >= 5:
                        self.logger.warning("Too many failures — entering cooldown")
                        wait_time = 60.0
                        failures = 0
            finally:
                cycle_ended_at = time.perf_counter()
                total_seconds = cycle_ended_at - cycle_started_at
                active_seconds = (
                    cycle_ended_at - active_started_at if active_started_at else 0.0
                )
                cycle_total_samples.append(total_seconds)
                cycle_active_samples.append(active_seconds)

                cycle_index += 1
                stats_window_seconds = cycle_ended_at - cycle_stats_started_at
                if stats_window_seconds >= self.cycle_log_interval_seconds:
                    p95_index = ceil(len(cycle_total_samples) * 0.95) - 1
                    sorted_total = sorted(cycle_total_samples)
                    sorted_active = sorted(cycle_active_samples)
                    self.logger.info(
                        "[poll] Cycle stats (%.1fs, %d cycles): "
                        "total_ms(avg/p95/max)=%.1f/%.1f/%.1f | "
                        "active_ms(avg/p95/max)=%.1f/%.1f/%.1f | "
                        "wait=%.1fs | idle_streak=%d",
                        stats_window_seconds,
                        len(cycle_total_samples),
                        sum(cycle_total_samples) / len(cycle_total_samples) * 1000,
                        sorted_total[p95_index] * 1000,
                        sorted_total[-1] * 1000,
                        sum(cycle_active_samples) / len(cycle_active_samples) * 1000,
                        sorted_active[p95_index] * 1000,
                        sorted_active[-1] * 1000,
                        wait_time,
                        idle_streak,
                    )
                    cycle_stats_started_at = cycle_ended_at
                    cycle_total_samples.clear()
                    cycle_active_samples.clear()

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
        """Stop scheduler-owned work and close its database resources."""
        async with self._lifecycle_lock:
            if not self.running and self._resources_closed:
                return
            await self._stop_locked(timeout)

    async def _stop_locked(self, timeout: float) -> None:
        self.logger.info("Stopping job scheduling agent")
        self.running = False
        await self._drain_pending_dispatches(timeout)
        await self.job_manager.event_publisher.join()
        self._remove_event_subscriptions()
        await self.runtime.stop(
            {
                'scheduler-notification-stop': self.notification_service.stop(),
                'scheduler-maintenance-stop': self.maintenance_service.stop(),
                'scheduler-dag-sync-stop': self.dag_service.stop_sync(),
                'scheduler-dag-admission-stop': self.dag_service.stop_admission(),
            },
            timeout=timeout,
        )
        await self._close_runtime_resources()

    async def _drain_pending_dispatches(self, timeout: float) -> None:
        tasks = [task for task in self._pending_dispatches.values() if not task.done()]
        if not tasks:
            return

        scheduler_trace(
            "dispatch_confirmation_shutdown_drain_start",
            pending=len(tasks),
            timeout_seconds=max(0.0, timeout),
        )
        _, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout))
        scheduler_trace(
            "dispatch_confirmation_shutdown_drain_done",
            completed=len(tasks) - len(pending),
            pending=len(pending),
        )
        if pending:
            self.logger.warning(
                "Scheduler shutdown left %s dispatch confirmation(s) unresolved; "
                "durable attempt recovery will reconcile them",
                len(pending),
            )

    async def _close_runtime_resources(self) -> None:
        if self._resources_closed:
            return

        try:
            await self.repository.close()
        except Exception as error:
            self.logger.error(f"Error closing job repository: {error}")

        try:
            await self._db_pool.close()
        except Exception as error:
            self.logger.error(f"Error closing scheduler PostgreSQL pool: {error}")

        self._resources_closed = True

    async def _reopen_runtime_resources(self) -> None:
        submission_count = self.submission_service.submission_count
        self._db_pool = AsyncPostgresConnectionPool()
        self.repository = JobRepository(
            self.config,
            pool=self._db_pool,
        )
        await self.repository.initialize()
        self.notification_service = NotificationService(self.config)
        self.dag_service = DAGManagementService(
            repository=self.repository,
            frontier=self.frontier,
            active_dags=self.active_dags,
            notify_callback=self.notify_event,
            max_active_dags=self.max_concurrent_dags,
            admission_lock=self._dag_admission_lock,
            slot_snapshot_provider=self.get_available_slots,
            job_cache=self._job_cache,
            terminal_event_callback=self._emit_dag_terminal_event,
            resolution_retry_limit=self._dag_resolution_retry_limit,
            resolution_retry_delay=self._dag_resolution_retry_delay,
            resolution_retry_backoff=self._dag_resolution_retry_backoff,
            resolution_retry_max_delay=self._dag_resolution_retry_max_delay,
            admission_batch_size=self.priority_refresh_hydrate_limit,
            sla_priority_interval_seconds=self.sla_priority_interval_seconds,
        )
        self.notification_service.register_handler(
            channel='dag_state_changed', handler=self.dag_service.handle_state_change
        )
        self.notification_service.register_handler(
            channel=JOB_STATUS_NOTIFICATION_CHANNEL,
            handler=self.job_manager.handle_job_status_notification,
        )
        self.maintenance_service = MaintenanceService(
            repository=self.repository,
            notify_callback=self.notify_event,
            recovery_callback=self._reconcile_recovered_run_leases,
            maintenance_interval=self._maintenance_interval,
        )
        self.control_flow_service = self._build_control_flow_service()
        self.attempt_lifecycle_service = self._build_attempt_lifecycle_service()
        self.submission_service = self._build_submission_service(
            initial_submission_count=submission_count
        )
        self.diagnostics = self._build_diagnostics()
        self._resources_closed = False

    async def debug_info(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the scheduler runtime."""
        snapshot = await self.diagnostics.snapshot(
            running=self.running,
            paused=self._paused,
            fetch_counter=self._fetch_counter,
        )
        pending = len(self._pending_dispatches)
        limit = self.dispatch_confirmation_max_in_flight
        snapshot['dispatch'] = {
            'pending_confirmations': pending,
            'confirmation_limit': limit,
            'available_confirmations': max(0, limit - pending),
            'utilization_pct': pending / limit * 100.0 if limit > 0 else None,
            'counters': dict(sorted(self._scheduler_counters.items())),
        }
        return snapshot

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
        self.logger.debug(f"Attempting to dispatch work item: {work_info.id}")
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

                # Wait for the supervisor's pre-send admission signal. Executor receipt and
                # worker acknowledgement are traced independently by the supervisor/runtime.
                # The timeout covers submit + admission and must stay inside the run lease TTL.
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

            dispatch_timeout = max(0.1, float(self.run_ttl_seconds) - 1.0)
            await asyncio.wait_for(_submit_and_confirm(), timeout=dispatch_timeout)
            self.logger.debug(f"Dispatch confirmed for job: {submission_id}")
            return True

        except asyncio.TimeoutError:
            scheduler_trace(
                "gateway_dispatch_timeout",
                job_id=submission_id,
                dag_id=work_info.dag_id,
                entrypoint=entrypoint,
                timeout_seconds=max(0.1, float(self.run_ttl_seconds) - 1.0),
                run_owner=run_owner,
                run_attempt_id=run_attempt_id,
                **self._ha_trace_fields(),
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
            self.logger.error(
                f"Failed to dispatch job {submission_id}: {e}", exc_info=True
            )
            return False

    async def get_job(self, job_id: str) -> Optional[WorkInfo]:
        """Get a job from the active projection or PostgreSQL."""
        cached = self._job_cache.get(job_id)
        if cached is not None:
            return cached

        work_item = await self.repository.get_job_by_id(job_id)
        if work_item:
            self._job_cache[job_id] = work_item
        return work_item

    async def list_jobs(
        self, state: Optional[str | list[str]] = None, batch_size: int = 0
    ) -> Dict[str, WorkInfo]:
        if state is not None:
            if isinstance(state, str):
                state = [state]
            invalid_states = [
                s for s in state if s.upper() not in WorkState.__members__
            ]
            if invalid_states:
                raise ValueError(f"Invalid state(s): {', '.join(invalid_states)}")
            states = [s.lower() for s in state]
        else:
            states = None

        work_items = await self.repository.list_jobs(state=states, limit=batch_size)
        return {work_item.id: work_item for work_item in work_items}

    async def submit_job(self, work_info: WorkInfo, overwrite: bool = True) -> str:
        """Persist a DAG before acknowledging its submission."""
        return await self.submission_service.submit(work_info, overwrite)

    async def _handle_priority_refresh(self, submission_count: int) -> None:
        if not self.priority_refresh_enabled:
            return
        refresh_interval = self.priority_refresh_interval
        if submission_count % refresh_interval == 0:
            now = time.monotonic()
            due_in = self._next_priority_refresh_at - now
            if due_in <= 0:
                self._request_priority_refresh(source='submission')
            self.logger.info(
                f'Requested job priority refresh after {submission_count} submissions '
                f'(interval: {refresh_interval}, due_in={max(0.0, due_in):.3f}s)'
            )

    def _request_priority_refresh(self, source: str) -> None:
        """Queue at most one additional priority refresh."""
        if not self.priority_refresh_enabled:
            return
        already_pending = self._priority_refresh_event.is_set()
        self._priority_refresh_source = source
        self._priority_refresh_event.set()
        self._next_priority_refresh_at = (
            time.monotonic() + self.priority_refresh_interval_seconds
        )
        scheduler_trace(
            "scheduler_priority_refresh_requested",
            source=source,
            coalesced=already_pending,
            refresh_running=self._priority_refresh_running,
            submission_count=self.submission_service.submission_count,
        )

    async def _priority_refresh_loop(self) -> None:
        """Run bounded priority refreshes outside the dispatch loop."""
        while self.running:
            try:
                await self._priority_refresh_event.wait()
                self._priority_refresh_event.clear()
                source = self._priority_refresh_source
                refresh_started = time.perf_counter()
                self._priority_refresh_running = True
                try:
                    result = await asyncio.wait_for(
                        self._refresh_job_priorities(source=source),
                        timeout=self.priority_refresh_timeout_seconds,
                    )
                    if result.succeeded:
                        scheduler_trace(
                            "scheduler_priority_refresh_completed",
                            source=source,
                            refresh_id=result.refresh_id,
                            submission_count=self.submission_service.submission_count,
                            refresh_interval_seconds=self.priority_refresh_interval_seconds,
                            elapsed_ms=(time.perf_counter() - refresh_started) * 1000.0,
                        )
                    else:
                        scheduler_trace(
                            "scheduler_priority_refresh_failed",
                            source=source,
                            refresh_id=result.refresh_id,
                            submission_count=self.submission_service.submission_count,
                            elapsed_ms=(time.perf_counter() - refresh_started) * 1000.0,
                            error=result.error,
                        )
                        self.logger.error(
                            f"Failed to refresh job priorities: {result.error}"
                        )
                except asyncio.TimeoutError:
                    scheduler_trace(
                        "scheduler_priority_refresh_failed",
                        source=source,
                        submission_count=self.submission_service.submission_count,
                        elapsed_ms=(time.perf_counter() - refresh_started) * 1000.0,
                        error="timeout",
                        timeout_seconds=self.priority_refresh_timeout_seconds,
                    )
                    self.logger.warning(
                        "Priority refresh timed out after %.1fs",
                        self.priority_refresh_timeout_seconds,
                    )
                finally:
                    self._priority_refresh_running = False
            except asyncio.CancelledError:
                break

    async def _refresh_job_priorities(
        self, source: str = "unknown"
    ) -> PriorityRefreshResult:
        """Sync DB priority edits into memory and log current SLA pressure."""
        self._priority_refresh_seq += 1
        refresh_id = self._priority_refresh_seq
        refresh_started = time.perf_counter()
        scheduler_trace(
            "scheduler_priority_refresh_start",
            source=source,
            refresh_id=refresh_id,
            submission_count=self.submission_service.submission_count,
            hydrate_missing_limit=self.priority_refresh_hydrate_limit,
        )
        try:
            phase_started = time.perf_counter()
            scheduler_trace(
                "scheduler_priority_refresh_frontier_start",
                source=source,
                refresh_id=refresh_id,
                submission_count=self.submission_service.submission_count,
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
                submission_count=self.submission_service.submission_count,
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
                submission_count=self.submission_service.submission_count,
            )
            await self.frontier.refresh_ready_ordering()
            scheduler_trace(
                "scheduler_priority_refresh_ready_ordering_done",
                source=source,
                refresh_id=refresh_id,
                submission_count=self.submission_service.submission_count,
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
                submission_count=self.submission_service.submission_count,
                top_n=self.sla_warning_top_n,
            )
            frontier_summary = await self.frontier.priority_refresh_summary(
                top_n=self.sla_warning_top_n
            )
            sla_summary = frontier_summary.get("sla", {})
            scheduler_trace(
                "scheduler_priority_refresh_summary_done",
                source=source,
                refresh_id=refresh_id,
                submission_count=self.submission_service.submission_count,
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
                return PriorityRefreshResult(refresh_id=refresh_id)

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
            hard_missed = int(sla_summary.get("hard_missed", 0))
            if hard_missed > 0:
                self.logger.warning(
                    f"[SLA] {hard_missed} jobs have missed hard SLA; "
                    "planner ranking continues to prefer them"
                )
            return PriorityRefreshResult(refresh_id=refresh_id)
        except Exception as e:
            return PriorityRefreshResult(
                refresh_id=refresh_id,
                error=str(e),
            )

    async def mark_as_active(self, work_info: WorkInfo) -> bool:
        self.logger.debug(f"Marking as active : {work_info.id}")
        count = await self.repository.mark_jobs_as_active(
            job_ids=[work_info.id], job_name=work_info.name
        )
        return count > 0

    async def is_valid_submission(
        self, work_info: WorkInfo, policy: ExistingWorkPolicy
    ) -> bool:
        """Validate a submission against the configured existing-work policy."""
        return await self.submission_service.is_valid_submission(work_info, policy)

    async def cancel_job(self, job_id: str, work_item: WorkInfo) -> int:
        """
        Cancel a job by its ID.
        Delegates to JobRepository.

        :param job_id: The ID of the job.
        :param work_item: The work item to cancel.
        """
        async with self._status_update_lock[job_id]:
            return await self.repository.cancel_job(
                job_id=job_id,
                queue_name=work_item.name,
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

    async def _reconcile_recovered_run_leases(
        self, recovered: list[RecoveredRunLease]
    ) -> None:
        admission_required = False
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
                if recovery.id not in self.frontier.jobs_by_id:
                    dag_id = recovery.dag_id or work_item.dag_id
                    if dag_id:
                        await self.hydrate_single_dag_from_db(dag_id)
                if recovery.id not in self.frontier.jobs_by_id:
                    admission_required = True
                    continue
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
            await self.dag_service.resolve_dag_status_with_retry(
                recovery.id,
                work_item,
                source="run_lease_recovery",
            )

        if admission_required:
            await self.dag_service.request_admission("run_lease_recovery")

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
        if self._event_subscriptions_active:
            return
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
        self._event_subscriptions_active = True

    def _remove_event_subscriptions(self) -> None:
        if not self._event_subscriptions_active:
            return
        for status in (
            JobStatus.RUNNING,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.PENDING,
            JobStatus.STOPPED,
        ):
            self.job_manager.event_publisher.unsubscribe(status, self.handle_job_event)
        self._event_subscriptions_active = False

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
            return final_state

    async def _sync(self):
        """
        Synchronizes job status between the local job tracking system and db.
        This function runs in a loop to periodically check the status of active jobs and update their state
        locally based on the external source.
        """
        wait_time = SYNC_POLL_PERIOD
        job_info_client = self.job_manager.job_info_client()
        min_sync_interval_seconds = 300

        while self.running:
            self.logger.info(f"Syncing job status every {wait_time} seconds")
            await asyncio.sleep(wait_time)
            try:
                active_jobs = await self.list_jobs(state=[WorkState.ACTIVE.value])
                if not active_jobs:
                    continue

                sync_missing = 0
                sync_without_status = 0
                sync_running = 0
                sync_terminal = 0
                for job_id, work_item in active_jobs.items():
                    job_info = await job_info_client.get_info(job_id)
                    if job_info is None:
                        sync_missing += 1
                        continue

                    if not job_info.status:
                        sync_without_status += 1
                        continue

                    if job_info.status == JobStatus.RUNNING:
                        sync_running += 1
                        continue

                    sync_terminal += 1
                    await self._sync_terminal_job_state(
                        job_id,
                        work_item,
                        job_info,
                        min_sync_interval_seconds=min_sync_interval_seconds,
                    )

                self.logger.info(
                    "Synchronized active jobs: total=%d running=%d terminal=%d "
                    "missing=%d without_status=%d",
                    len(active_jobs),
                    sync_running,
                    sync_terminal,
                    sync_missing,
                    sync_without_status,
                )

            except (Exception, psycopg.Error) as error:
                self.logger.error(f"Error syncing jobs: {error}")
                self.logger.error(traceback.format_exc())

    async def _renew_run_leases(self) -> None:
        while self.running:
            started = time.monotonic()
            try:
                await self._renew_active_run_leases()
            except Exception as error:
                self.logger.error(
                    f"Error renewing active run leases: {error}", exc_info=True
                )
            elapsed = time.monotonic() - started
            await asyncio.sleep(
                max(0.0, self.run_lease_renewal_interval_seconds - elapsed)
            )

    async def _renew_active_run_leases(self) -> None:
        started = time.perf_counter()
        active_jobs = await self.list_jobs(state=[WorkState.ACTIVE.value])
        if not active_jobs:
            return

        job_info_client = self.job_manager.job_info_client()
        concurrency = asyncio.Semaphore(16)

        async def renew(job_id: str, work_item: WorkInfo) -> str:
            async with concurrency:
                run_owner = work_item.run_owner
                run_attempt_id = work_item.run_attempt_id
                if not run_owner or not run_attempt_id:
                    return "missing_identity"
                if run_owner != self.lease_owner:
                    scheduler_trace(
                        "run_lease_extend_rejected",
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        reason="foreign_owner",
                        **self._ha_trace_fields(),
                    )
                    return "foreign_owner"

                job_info = await job_info_client.get_info(job_id)
                if job_info is None:
                    return "missing_job_info"
                if job_info.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                    return "not_running"
                if (
                    job_info.run_owner != run_owner
                    or job_info.run_attempt_id != run_attempt_id
                ):
                    scheduler_trace(
                        "run_lease_extend_rejected",
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        status=job_info.status.value,
                        run_owner=job_info.run_owner,
                        run_attempt_id=job_info.run_attempt_id,
                        reason="renewal_attempt_mismatch",
                        **self._ha_trace_fields(),
                    )
                    return "attempt_mismatch"

                if job_info.status == JobStatus.PENDING:
                    metadata = (
                        work_item.data.get("metadata", {})
                        if isinstance(work_item.data, dict)
                        else {}
                    )
                    entrypoint = (
                        metadata.get("on", "") if isinstance(metadata, dict) else ""
                    )
                    if not entrypoint:
                        return "missing_entrypoint"
                    semaphore_renewed = await asyncio.to_thread(
                        self._semaphore_store.renew,
                        executor_name(entrypoint),
                        job_id,
                        owner=job_id,
                        run_attempt_id=run_attempt_id,
                    )
                    if not semaphore_renewed:
                        scheduler_trace(
                            "scheduler_semaphore_renew_rejected",
                            job_id=job_id,
                            dag_id=work_item.dag_id,
                            status=job_info.status.value,
                            run_owner=run_owner,
                            run_attempt_id=run_attempt_id,
                            reason="pending_ticket_missing_or_stale",
                            **self._ha_trace_fields(),
                        )
                        return "semaphore_rejected"

                extended = await self._extend_run_lease_db(
                    [job_id],
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                )
                if job_id in extended:
                    scheduler_trace(
                        "run_lease_extended",
                        job_id=job_id,
                        dag_id=work_item.dag_id,
                        run_owner=run_owner,
                        run_attempt_id=run_attempt_id,
                        source="renewal_loop",
                        **self._ha_trace_fields(),
                    )
                    return "extended"

                self._scheduler_counter(
                    RUN_LEASE_EXTEND_STALE_ATTEMPT_TOTAL,
                    job_id=job_id,
                    dag_id=work_item.dag_id,
                    status=job_info.status.value,
                    run_owner=run_owner,
                    run_attempt_id=run_attempt_id,
                    source="renewal_loop",
                )
                return "rejected"

        results = await asyncio.gather(
            *(renew(job_id, work_item) for job_id, work_item in active_jobs.items()),
            return_exceptions=True,
        )
        outcomes: Counter[str] = Counter()
        for result in results:
            if isinstance(result, BaseException):
                outcomes["error"] += 1
                self.logger.error(
                    f"Run lease renewal failed for one active job: {result}"
                )
            else:
                outcomes[result] += 1
        scheduler_trace(
            "run_lease_renewal_pass",
            active=len(active_jobs),
            attempted=(
                outcomes["extended"]
                + outcomes["rejected"]
                + outcomes["attempt_mismatch"]
            ),
            extended=outcomes["extended"],
            rejected=outcomes["rejected"] + outcomes["attempt_mismatch"],
            missing_identity=outcomes["missing_identity"],
            missing_job_info=outcomes["missing_job_info"],
            not_running=outcomes["not_running"],
            missing_entrypoint=outcomes["missing_entrypoint"],
            semaphore_rejected=outcomes["semaphore_rejected"],
            foreign_owner=outcomes["foreign_owner"],
            errors=outcomes["error"],
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            **self._ha_trace_fields(),
        )

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

        now = datetime.now(tz=timezone.utc)
        terminal_age_seconds = None
        if job_info.end_time is not None:
            timestamp_ms = job_info.end_time
            end_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            terminal_age_seconds = (now - end_time).total_seconds()

        current_work_item = await self.repository.get_job_by_id(job_id)
        if current_work_item is None or current_work_item.state != WorkState.ACTIVE:
            return False

        if (
            terminal_age_seconds is None
            or terminal_age_seconds < min_sync_interval_seconds
        ):
            age = (
                f"{terminal_age_seconds:.3f}"
                if terminal_age_seconds is not None
                else "unknown"
            )
            self.logger.info(
                f"Terminal state mismatch detected for job {job_id}: "
                f"WorkState={current_work_item.state}, "
                f"JobInfoState={job_info_state}, "
                f"terminal_age_seconds={age}, "
                f"repair_grace_seconds={min_sync_interval_seconds}; "
                "deferring repair."
            )
            return False

        self.logger.info(
            f"Repairing terminal state mismatch for job {job_id}: "
            f"WorkState={current_work_item.state}, JobInfoState={job_info_state}, "
            f"terminal_age_seconds={terminal_age_seconds:.3f}, "
            f"repair_grace_seconds={min_sync_interval_seconds}."
        )

        return await self.attempt_lifecycle_service.transition_terminal(
            job_id,
            current_work_item,
            job_info.status,
            run_owner=job_info.run_owner,
            run_attempt_id=job_info.run_attempt_id,
            source="storage_sync",
            output_metadata={"synced": True},
            message=job_info.message,
            runtime_env=job_info.runtime_env,
        )

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

        transitioned = await self.attempt_lifecycle_service.transition_terminal(
            wi.id,
            wi,
            JobStatus.FAILED,
            run_owner=run_owner,
            run_attempt_id=run_attempt_id,
            source="dispatch_failure",
            output_metadata={
                "dispatch_failed": True,
                "dispatch_error": error_message,
                "failure_stage": "enqueue",
            },
            message=error_message,
        )
        if not transitioned:
            self.logger.error(
                f"Dispatch failure cleanup could not transition job {wi.id}; "
                f"run_attempt_id={run_attempt_id}"
            )
            await self.frontier.release_lease_local(wi.id)

        try:
            released = await asyncio.to_thread(
                self._semaphore_store.release_owned,
                executor,
                wi.id,
                owner=owner,
                run_attempt_id=run_attempt_id,
            )
            self.logger.debug(
                f"[sem] release on dispatch-fail {wi.id}@{executor} -> {released}"
            )
        except Exception as release_error:
            self.logger.warning(
                f"[sem] release error after dispatch-fail {wi.id}@{executor}: "
                f"{release_error}"
            )

        if not transitioned:
            await self.notify_event()

    async def get_dag_by_id(self, dag_id: str) -> QueryPlan | None:
        return await self.dag_service.get_dag(dag_id)

    def get_available_slots(self) -> dict[str, int]:
        return available_slots_by_executor(self._semaphore_store)

    async def reset_active_dags(self):
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

    async def _activate_from_lease_db(
        self, ids: list[str], run_attempt_ids: dict[str, str]
    ) -> dict[str, str]:
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
            run_attempt_ids=run_attempt_ids,
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
        self,
        executor: str,
        jobs: list[WorkInfo],
        run_attempt_ids: dict[str, str],
    ) -> dict[str, SemaphoreReservationStatus]:
        results: dict[str, SemaphoreReservationStatus] = {}
        for wi in jobs:
            try:
                ok = await asyncio.to_thread(
                    self._semaphore_store.reserve,
                    executor,
                    wi.id,
                    node='',
                    ttl=self._sem_default_ttl,
                    owner=wi.id,
                    run_attempt_id=run_attempt_ids[wi.id],
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
                results[wi.id] = SemaphoreReservationStatus.STORE_ERROR
                continue
            if ok:
                results[wi.id] = SemaphoreReservationStatus.RESERVED
            else:
                results[wi.id] = await self._classify_reservation_miss(executor, wi.id)
        return results

    async def _classify_reservation_miss(
        self, executor: str, job_id: str
    ) -> SemaphoreReservationStatus:
        try:
            holder = await asyncio.to_thread(
                self._semaphore_store.get_holder, executor, job_id
            )
            if holder is not None:
                return SemaphoreReservationStatus.TICKET_EXISTS
            available = await asyncio.to_thread(
                self._semaphore_store.available_slot_count, executor
            )
        except Exception:
            return SemaphoreReservationStatus.STORE_ERROR
        if available <= 0:
            return SemaphoreReservationStatus.CAPACITY_FULL
        return SemaphoreReservationStatus.CONTENTION

    async def _reserve_semaphore_slots(
        self,
        executor: str,
        jobs: list[WorkInfo],
        run_attempt_ids: dict[str, str],
    ) -> dict[str, SemaphoreReservationStatus]:
        if not jobs:
            return {}

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
                run_attempt_id_by_ticket=run_attempt_ids,
            )
        except Exception as exc:
            fallback_used = True
            error = repr(exc)
            self.logger.error(
                f"[WORK_DIST] Batch semaphore reserve failed for executor={executor}; falling back to serial reserve: {exc}",
                exc_info=True,
            )
            results = await self._reserve_semaphore_slots_serial(
                executor, jobs, run_attempt_ids
            )
        else:
            results = {
                job_id: SemaphoreReservationStatus.RESERVED for job_id in reserved
            }
            for wi in jobs:
                if wi.id not in results:
                    results[wi.id] = await self._classify_reservation_miss(
                        executor, wi.id
                    )

        scheduler_trace(
            "semaphore_reserve_batch_done",
            executor=executor,
            requested=len(job_ids),
            reserved=sum(
                status == SemaphoreReservationStatus.RESERVED
                for status in results.values()
            ),
            fallback_used=fallback_used,
            error=error,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            outcomes={job_id: status.value for job_id, status in results.items()},
        )
        return results

    async def hydrate_single_dag_from_db(self, dag_id: str) -> bool:
        """
        Hydrate a specific DAG from the database into the MemoryFrontier.
        Delegates to DAGManagementService.

        :param dag_id: The ID of the DAG to hydrate
        :return: True if DAG was hydrated, False if not found or failed
        """
        return await self.dag_service.hydrate_single_dag(dag_id)

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
                await self.dag_service.request_admission("deployment_update")
                await self.notify_event()
            except asyncio.CancelledError:
                self.logger.debug("Deployment update monitor task cancelled.")
                break
            except Exception as e:
                self.logger.error(
                    f"Error in deployment update monitor: {e}", exc_info=True
                )
                await asyncio.sleep(5)

    def stop_job(self, job_id: str) -> bool:
        raise NotImplementedError

    async def delete_job(self, job_id: str) -> None:
        raise NotImplementedError
