import asyncio
import contextlib
import copy
import itertools
import random
import time
import traceback
import uuid
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, NamedTuple, Tuple

import psycopg2
from rich.console import Console
from rich.table import Table

from marie.constants import (
    __default_extract_dir__,
    __default_psql_dir__,
    __default_schema_dir__,
)
from marie.excepts import BadConfigSource, RuntimeFailToStart
from marie.helper import get_or_reuse_loop
from marie.job.common import JobStatus
from marie.job.job_manager import JobManager
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.messaging import mark_as_complete
from marie.query_planner.base import (
    NoopQueryDefinition,
    Query,
    QueryDefinition,
    QueryPlan,
)
from marie.query_planner.builtin import register_all_known_planners
from marie.query_planner.mapper import JobMetadata, has_mapper_config
from marie.query_planner.model import QueryPlannersConf
from marie.query_planner.planner import (
    PlannerInfo,
    compute_job_levels,
    plan_to_yaml,
    query_planner,
    topological_sort,
)
from marie.scheduler.fixtures import *
from marie.scheduler.global_execution_planner import GlobalPriorityExecutionPlanner
from marie.scheduler.job_scheduler import JobScheduler
from marie.scheduler.models import ExistingWorkPolicy, WorkInfo
from marie.scheduler.plans import (
    cancel_jobs,
    complete_jobs,
    complete_jobs_by_id,
    count_states,
    create_queue,
    fail_jobs_by_id,
    fetch_next_job,
    insert_dag,
    insert_job,
    insert_version,
    load_dag,
    mark_as_active_dags,
    mark_as_active_jobs,
    resume_jobs,
    to_timestamp_with_tz,
    try_set_monitor_time,
    version_table_exists,
)
from marie.scheduler.state import WorkState
from marie.scheduler.util import available_slots_by_executor
from marie.serve.runtimes.servers.cluster_state import ClusterState
from marie.storage.database.postgres import PostgresqlMixin

INIT_POLL_PERIOD = 2.250  # 250ms

SHORT_POLL_INTERVAL = 1.0  # seconds, when slots exist but no work

MIN_POLL_PERIOD = 0.5
MAX_POLL_PERIOD = 16

MONITORING_POLL_PERIOD = 5.0  # 5s
SYNC_POLL_PERIOD = 5.0  # 5s

DEFAULT_SCHEMA = "marie_scheduler"
DEFAULT_JOB_TABLE = "job"
COMPLETION_JOB_PREFIX = f"__state__{WorkState.COMPLETED.value}__"


def convert_job_status_to_work_state(job_status: JobStatus) -> WorkState:
    """
    Convert a JobStatus to a WorkState.
    :param job_status:
    :return:
    """
    if job_status == JobStatus.PENDING:
        return WorkState.CREATED
    elif job_status == JobStatus.RUNNING:
        return WorkState.ACTIVE
    elif job_status == JobStatus.SUCCEEDED:
        return WorkState.COMPLETED
    elif job_status == JobStatus.FAILED:
        return WorkState.FAILED
    elif job_status == JobStatus.STOPPED:
        return WorkState.CANCELLED
    else:
        raise ValueError(f"Unknown JobStatus: {job_status}")


def adjust_backoff(wait_time: float, idle_streak: int, scheduled: bool) -> float:
    if scheduled:
        return max(wait_time * 0.5, MIN_POLL_PERIOD)
    jitter = random.uniform(0.9, 1.1)
    return min(wait_time * (1.5 + 0.1 * idle_streak), MAX_POLL_PERIOD) * jitter


def print_state_summary(job_states_data: Dict[str, Any]):
    try:
        console = Console()
        table = Table(
            title="📊 Consolidated Job States for All Queues",
            border_style="green",
            title_style="bold white on blue",
            header_style="bold yellow",
            show_lines=True,  # Adds separating lines between rows for better readability
        )

        table.add_column("Queue", justify="left", style="cyan", no_wrap=False, width=16)
        metrics = [
            "created",
            "retry",
            "active",
            "completed",
            "expired",
            "cancelled",
            "failed",
            "all",
        ]
        for metric in metrics:
            table.add_column(
                metric.capitalize(),
                justify="center",
                style="magenta",
                no_wrap=False,
                width=16,
            )

        if job_states_data.get("queues"):
            for queue_name, queue_data in job_states_data["queues"].items():
                row_values = [queue_name.capitalize()]
                for metric in metrics:
                    row_values.append(str(queue_data.get(metric, 0)))
                table.add_row(*row_values)

            summary_values = {metric: 0 for metric in metrics}
            for queue_data in job_states_data["queues"].values():
                for metric, value in queue_data.items():
                    if metric in summary_values:
                        summary_values[metric] += value

            table.add_row(
                "Summary",
                *[str(summary_values[metric]) for metric in metrics],
                style="bold green",
            )
        else:
            table.add_row("No Data", *["0" for _ in metrics], style="bold red")

        console.print(table)
    except Exception as e:
        logger.error(f"Error printing state summary: {e}")
        logger.error(traceback.format_exc())


def print_slots_table(slots: dict[str, int]) -> None:
    console = Console()
    table = Table(
        title="⚙️  Available Slots",
        border_style="green",
        title_style="bold white on blue",
        header_style="bold yellow",
        show_lines=True,
    )
    table.add_column("Slot Type", justify="left", style="cyan", no_wrap=False)
    table.add_column("Count", justify="center", style="magenta", no_wrap=False)

    slots_s = dict(sorted(slots.items()))
    for slot_type, count in slots_s.items():
        table.add_row(slot_type, str(count))

    console.print(table)


# FIXME : Today we are tracking at the executor level, however that might not be the best
# approach. We might want to track at the deployment level (endpoint level) instead.
# this will allow us to track the status of the deployment and not just the executor.


class JobSubmissionRequest(NamedTuple):
    work_info: WorkInfo
    overwrite: bool
    request_id: str
    result_future: asyncio.Future


class PostgreSQLJobScheduler(PostgresqlMixin, JobScheduler):
    """A PostgreSQL-based job scheduler."""

    def __init__(self, config: Dict[str, Any], job_manager: JobManager):
        super().__init__()
        self.logger = MarieLogger(PostgreSQLJobScheduler.__name__)
        if job_manager is None:
            raise BadConfigSource("job_manager argument is required for JobScheduler")

        self._fetch_event = asyncio.Event()
        self._fetch_counter = 0
        self._debounced_notify = False

        self.known_queues = set(config.get("queue_names", []))
        self.running = False
        self.task = None
        self._producer_task = None
        self._consumer_task = None
        self._heartbeat_task = None
        self.sync_task = None
        self.monitoring_task = None
        self._worker_task = None
        self._sync_dag_task = None
        self._submission_count = 0
        self._pending_requests = {}  # Track pending requests by ID
        self._request_queue = Queue()  # Buffer up to 1000 requests

        self.scheduler_mode = config.get(
            "scheduler_mode", "parallel"
        )  # "serial" or "parallel"
        self.scheduler_mode = "serial"
        self._event_queue = Queue()

        lock_free = True
        self._lock = (
            contextlib.AsyncExitStack() if lock_free else asyncio.Lock()
        )  # Lock to prevent concurrent access to the database

        max_workers = config.get("max_workers", 1)
        self._db_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="sync-db-executor"
        )

        if self.known_queues is None or len(self.known_queues) == 0:
            raise BadConfigSource("Queue names are required for JobScheduler")
        self.logger.info(f"Queue names to monitor: {self.known_queues}")

        self.active_dags = {}
        self.job_manager = job_manager
        self._loop = get_or_reuse_loop()
        self._setup_event_subscriptions()
        self._setup_storage(config, connection_only=True)

        # self.execution_planner = SJFSExecutionPlanner()
        self.execution_planner = GlobalPriorityExecutionPlanner()
        register_all_known_planners(
            QueryPlannersConf.from_dict(config.get("query_planners", {}))
        )

        dag_config = config.get("dag_manager", {})
        dag_strategy = dag_config.get("strategy", 'fixed')  # fixed or dynamic
        min_concurrent_dags = int(dag_config.get("min_concurrent_dags", 1))
        max_concurrent_dags = int(dag_config.get("max_concurrent_dags", 16))
        cache_ttl_seconds = int(dag_config.get("cache_ttl_seconds", 5))

        # TODO : Implement DagConcurrencyManager properly
        self.max_concurrent_dags = max_concurrent_dags
        # self.dag_concurrency_manager = DagConcurrencyManager(
        #     self,
        #     strategy=dag_strategy,
        #     min_concurrent_dags=min_concurrent_dags,
        #     max_concurrent_dags=max_concurrent_dags,
        #     cache_ttl_seconds=cache_ttl_seconds,
        # )
        #
        # count = self.dag_concurrency_manager.calculate_max_concurrent_dags()
        # print(f'dag_config : ', dag_config)
        # print(f'count  = {count}')
        # print(self.dag_concurrency_manager.get_configuration_summary())
        self._start_time = datetime.now(timezone.utc)

    async def handle_job_event(self, event_type: str, message: Any):
        """
        Handles a job event.

        :param event_type: The type of the event.
        :param message: The message associated with the event.
        """

        self.logger.info(f"received message: {event_type} > {message}")
        if not isinstance(message, dict) or "job_id" not in message:
            self.logger.error(f"Invalid message format: {message}")
            return

        async with self._lock:
            try:
                job_id = message.get("job_id")
                status = JobStatus(event_type)
                work_item: WorkInfo = await self.get_job(job_id)

                if work_item is None:
                    self.logger.error(f"WorkItem not found: {job_id}")
                    return
                work_state = convert_job_status_to_work_state(status)
                now = datetime.now()

                if status == JobStatus.PENDING:
                    self.logger.debug(f"Job pending : {job_id}")
                elif status == JobStatus.SUCCEEDED:
                    await self.complete(job_id, work_item)
                elif status == JobStatus.FAILED:
                    await self.fail(job_id, work_item)
                elif status == JobStatus.RUNNING:
                    self.logger.debug(f"Job running : {job_id}")
                    await self.put_status(job_id, work_state, now, None)
                else:
                    self.logger.error(
                        f"Unhandled job status: {status}. Marking as FAILED."
                    )
                    await self.fail(job_id, work_item)  # Fail-safe

                if status.is_terminal():
                    self.logger.debug(
                        f"Job is in terminal state {status}, job_id: {job_id}"
                    )
                    await self.resolve_dag_status(job_id, work_item, now, now)
                    await self.notify_event()
            except Exception as e:
                self.logger.error(
                    f"Error handling job event {event_type} for job {job_id}: {e}"
                )

    def create_tables(self, schema: str):
        """
        :param schema: The name of the schema where the tables will be created.
        :return: None
        """
        version = 1
        commands = [
            create_schema(schema),
            create_version_table(schema),
            create_table_queue(schema),
            create_job_state_enum(schema),
            create_job_table(schema),
            create_primary_key_job(schema),
            create_job_history_table(schema),
            create_job_update_trigger_function(schema),
            create_job_update_trigger(schema),
            clone_job_table_for_archive(schema),
            create_schedule_table(schema),
            create_subscription_table(schema),
            add_archived_on_to_archive(schema),
            add_archived_on_index_to_archive(schema),
            add_id_index_to_archive(schema),
            # create_index_singleton_on(schema),
            # create_index_singleton_key_on(schema),
            create_index_job_name(schema),
            create_index_job_fetch(schema),
            create_queue_function(schema),
            delete_queue_function(schema),
            insert_version(schema, version),
            create_exponential_backoff_function(schema),
            create_dag_table(schema),
            create_dag_table_history(schema),
            create_dag_history_trigger_function(schema),
            # create_dag_resolve_state_function(schema),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "job_dependencies.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "resolve_dag_state.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "fetch_next_job.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "refresh_job_priority.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "delete_dag_and_jobs.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "delete_failed_dags_and_jobs.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "delete_orphaned_jobs.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(
                    __default_schema_dir__, "jobs_with_unmet_dependencies.sql"
                ),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "notify_dag_state_change.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "purge_non_started_work.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "ready_jobs_view.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "refresh_dag_durations.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "refresh_job_durations.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "reset_active_dags_and_jobs.sql"),
            ),
            create_sql_from_file(
                schema, os.path.join(__default_schema_dir__, "reset_all.sql")
            ),
            create_sql_from_file(
                schema,
                os.path.join(
                    __default_schema_dir__, "reset_completed_dags_and_jobs.sql"
                ),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "reset_failed_dags_and_jobs.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "suspend_non_started_work.sql"),
            ),
            create_sql_from_file(
                schema,
                os.path.join(__default_schema_dir__, "unsuspend_work.sql"),
            ),
            create_sql_from_file(
                schema, os.path.join(__default_psql_dir__, "cron_job_init.sql")
            ),
        ]

        query = ";\n".join(commands)

        locked_query = f"""
           BEGIN;
           SET LOCAL statement_timeout = '30s';
           SELECT pg_try_advisory_lock(1);
           {query};
           SELECT pg_advisory_unlock(1);
           COMMIT;
           """

        with self:
            try:
                self._execute_sql_gracefully(locked_query)
            except (Exception, psycopg2.Error) as error:
                if isinstance(error, psycopg2.errors.DuplicateTable):
                    self.logger.warning("Tables already exist, skipping creation.")
                else:
                    self.logger.error(f"Error creating tables: {error}")
                    raise RuntimeFailToStart(
                        f"Failed to create tables in schema '{schema}': {error}"
                    )

    async def wipe(self) -> None:
        """Clears the schedule storage."""
        schema = DEFAULT_SCHEMA
        query = f"""
           TRUNCATE {schema}.job, {schema}.archive
           """
        with self:
            self._execute_sql_gracefully(query)

    async def is_installed(self) -> bool:
        """check if the tables are installed"""
        schema = DEFAULT_SCHEMA
        with self:
            cursor = None
            try:
                cursor = self._execute_sql_gracefully(
                    version_table_exists(schema), return_cursor=True
                )
                if cursor and cursor.rowcount > 0:
                    result = cursor.fetchone()
                    if result and result[0] is not None:
                        return True
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error checking tables: {error}")
                raise RuntimeFailToStart(
                    f"Unable to check installation in schema '{schema}': {error}"
                )
            finally:
                self._close_cursor(cursor)

    async def create_queue(self, queue_name: str) -> None:
        """Setup the queue for the scheduler."""
        with self:
            self._execute_sql_gracefully(create_queue(DEFAULT_SCHEMA, queue_name, {}))

    async def start(self) -> None:
        """
        Starts the job scheduling agent.

        :return: None
        """
        logger.info("Starting job scheduling agent")
        installed = await self.is_installed()
        logger.info(f"Tables installed: {installed}")
        if not installed:
            self.create_tables(DEFAULT_SCHEMA)

            for work_queue in self.known_queues:
                self.logger.info(f"Create queue: {work_queue}")
                await self.create_queue(work_queue)
                await self.create_queue(f"${work_queue}_dlq")

        self.running = True
        # self.sync_task = asyncio.create_task(self._sync())
        # self.monitoring_task = asyncio.create_task(self._monitor())

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop()
        )  # THIS SHOULD BE IN OWN EVENT LOOP

        if self.scheduler_mode == "parallelXXX":
            self._producer_task = asyncio.create_task(self._producer_loop())
            self._consumer_task = asyncio.create_task(self._consumer_loop())
            pass
        else:
            self.task = asyncio.create_task(self._poll())

        self._worker_task = asyncio.create_task(self._process_submission_queue())
        # self._sync_dag_task = asyncio.create_task(self._sync_dag())

        await self.notify_event()

    async def _heartbeat_loop(self, interval: float = 5.0):
        """Periodic heartbeat logger showing scheduler state."""
        while self.running:
            try:
                queue_size = self._event_queue.qsize()
                slot_info = available_slots_by_executor(ClusterState.deployments)
                active_dags = list(self.active_dags.keys())

                self.logger.info("🔄  Scheduler Heartbeat")
                self.logger.info(f"  🧭  Mode              : {self.scheduler_mode}")
                self.logger.info(f"  📦  Queue Size        : {queue_size}")
                self.logger.info(f"  ⚙️   Available Slots ")
                print_slots_table(slot_info)
                self.logger.info(f"  🧠  Active DAGs        : {len(active_dags)}")

                if active_dags:
                    shown = ', '.join(active_dags[:5])
                    suffix = '...' if len(active_dags) > 5 else ''
                    self.logger.debug(f"     DAG IDs          : [{shown}{suffix}]")

                states = self.count_states()
                # print_state_summary(states)
                # self.diagnose_pool()

                # print_dag_concurrency_status_compact(self.dag_concurrency_manager.get_configuration_summary())

                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"❌ Heartbeat loop error: {e}")
                await asyncio.sleep(5)

    async def _producer_loop(self):
        raise NotImplementedError("Producer loop is not implemented yet.")

    async def _consumer_loop(self):
        raise NotImplementedError("Consumer loop is not implemented yet.")

    async def _poll(self):
        self.logger.info("Starting database job scheduler (serial mode)")
        wait_time = INIT_POLL_PERIOD
        batch_size = 25000
        failures = 0
        idle_streak = 0

        last_deployments_timestamp = ClusterState.deployments_last_updated
        slots_by_executor = available_slots_by_executor(ClusterState.deployments).copy()
        recently_activated_dags = set()
        ACTIVATION_TIMEOUT = 60  # Seconds to consider DAG recently activated

        def cleanup_recently_activated_dags():
            """Removes DAGs that were activated too long ago from the set"""
            current_time = time.time()
            for dag_id, ts in list(recently_activated_dags):
                if current_time - ts > ACTIVATION_TIMEOUT:
                    recently_activated_dags.remove((dag_id, ts))

        max_concurrent_dags = self.max_concurrent_dags
        while self.running:
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

                # FIXME : this is a hack
                # refresh slots
                if ClusterState.deployments_last_updated != last_deployments_timestamp:
                    last_deployments_timestamp = ClusterState.deployments_last_updated
                    slots_by_executor = available_slots_by_executor(
                        ClusterState.deployments
                    ).copy()
                    self.logger.debug(
                        "Detected deployment update — refreshing slot state"
                    )

                # back off if no slots
                self.logger.debug(f"Available slots: {slots_by_executor}")
                if not any(slots_by_executor.values()):
                    self.logger.debug("No available executor slots. Backing off.")
                    idle_streak += 1
                    wait_time = adjust_backoff(wait_time, idle_streak, scheduled=False)
                    continue

                # get jobs from the database that are ready to be scheduled, aka have their dependencies met
                records_by_queue = await self.get_work_items_by_queue(limit=batch_size)
                flat_jobs: list[Tuple[str, WorkInfo]] = []

                for rec in itertools.chain.from_iterable(records_by_queue.values()):
                    wi = self.record_to_work_info(rec)
                    ep = wi.data.get("metadata", {}).get("on", "")
                    if "://" not in ep:
                        self.logger.warning(
                            f"Skipping job with invalid entrypoint: {wi.id}"
                        )
                        continue
                    flat_jobs.append((ep, wi))

                cleanup_recently_activated_dags()
                recently_activated_dag_ids = set(
                    dag_id for dag_id, _ in recently_activated_dags
                )
                flat_jobs = self.execution_planner.plan(
                    flat_jobs,
                    slots_by_executor,
                    self.active_dags,
                    recently_activated_dag_ids,
                )
                scheduled_any = False

                if not flat_jobs:
                    self.logger.debug(
                        "Slots available but no ready jobs; sleeping briefly before next poll."
                    )
                    await asyncio.sleep(SHORT_POLL_INTERVAL)
                    continue

                await self.debug_work_plans(flat_jobs, records_by_queue)

                for entrypoint, work_info in flat_jobs:
                    executor = entrypoint.split("://")[0]
                    dag_id = work_info.dag_id

                    if (
                        dag_id not in self.active_dags
                        and len(self.active_dags) >= max_concurrent_dags
                    ):
                        self.logger.debug(
                            f"Max DAG limit reached "
                            f"({len(self.active_dags)}/{max_concurrent_dags}). "
                            f"Skipping DAG {dag_id}"
                        )
                        continue

                    dag = self.get_dag_by_id(dag_id)

                    if dag is None:
                        raise ValueError(f"DAG not found: {dag_id}")

                    if dag_id not in self.active_dags:
                        await self.mark_as_active_dag(work_info)
                        self.active_dags[dag_id] = dag
                        recently_activated_dags.add((dag_id, time.time()))
                        self.logger.debug(f"DAG activated: {dag_id}")

                    node = self.get_node_from_dag(work_info.id, dag)

                    if self._is_noop_query_definition(node):
                        now = datetime.now()

                        await self.put_status(
                            work_info.id, WorkState.COMPLETED, now, now
                        )
                        await self.complete(work_info.id, work_info, {}, force=True)
                        await self.resolve_dag_status(work_info.id, work_info)
                        await self.notify_event()

                        continue

                    if slots_by_executor.get(executor, 0) <= 0:
                        self.logger.debug(
                            f"No slots for {executor}, delaying job {work_info.id}"
                        )
                        continue

                    await self.mark_as_active(work_info)
                    enqueue_id = await self.enqueue(work_info)

                    if enqueue_id:
                        slots_by_executor[executor] -= 1
                        self.logger.info(f"Job scheduled: {enqueue_id} on {executor}")
                        scheduled_any = True
                    else:
                        self.logger.error(f"Failed to enqueue job: {work_info.id}")

                # acknowledgments, we are waiting for the jobs to be scheduled on the executor and get response via ETCD
                if scheduled_any:
                    try:
                        await asyncio.wait_for(
                            ClusterState.scheduled_event.wait(), timeout=1
                        )  # THIS SHOULD BE SAME AS ETCD lease time
                    except asyncio.TimeoutError:
                        self.logger.warning("Timeout waiting for schedule confirmation")
                    finally:
                        ClusterState.scheduled_event.clear()
                        await self.notify_event()

                idle_streak = 0 if scheduled_any else idle_streak + 1
                wait_time = adjust_backoff(wait_time, idle_streak, scheduled_any)
                failures = 0
            except Exception as e:
                self.logger.error("Poll loop exception", exc_info=True)
                failures += 1
                if failures >= 5:
                    self.logger.warning("Too many failures — entering cooldown")
                    await asyncio.sleep(60)
                    failures = 0

    async def debug_work_plans(self, flat_jobs, records_by_queue):
        # Debug: Write the flat jobs plan to a file
        from datetime import datetime

        os.makedirs("/tmp/marie/plans", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file_path = f"/tmp/marie/plans/flat_jobs_plan_debug_{timestamp}.txt"
        try:
            with open(debug_file_path, 'w') as debug_file:
                for queue_name, records in records_by_queue.items():
                    debug_file.write(f"Queue: {queue_name}\n")
                    for record in records:
                        debug_file.write(f"  Records:  {record},\n")
                debug_file.write("\n\n")
                debug_file.write("Jobs Plan:\n")
                for entrypoint, work_info in flat_jobs:
                    debug_file.write(
                        f"Entrypoint: {entrypoint}, "
                        f"Work ID: {work_info.id}, "
                        f"Priority: {work_info.priority}, "
                        f"Job Level: {work_info.job_level}, "
                        f"DAG ID: {work_info.dag_id}\n"
                    )
            self.logger.debug(
                f"Flat jobs plan written to {debug_file_path} for analysis."
            )
        except Exception as debug_error:
            self.logger.error(
                f"Failed to write flat jobs plan to debug file: {debug_error}"
            )

    async def stop(self, timeout: float = 2.0) -> None:
        self.logger.info("Stopping job scheduling agent")
        self.running = False

        tasks = [self.monitoring_task]
        if self.scheduler_mode == "parallel":
            tasks = tasks + [self._producer_task, self._consumer_task]
        else:
            tasks = tasks + [self.task]

        if self._heartbeat_task:
            tasks.append(self._heartbeat_task)

        if self._worker_task:
            tasks.append(self._worker_task)

        if self.sync_task:
            tasks.append(self.sync_task)

        if self._sync_dag_task:
            tasks.append(self._sync_dag_task)

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
            debug_data["active_dags"] = {
                dag_id: {
                    "dag_id": dag_id,
                    "status": (
                        getattr(dag_info, 'status', 'unknown')
                        if hasattr(dag_info, 'status')
                        else 'unknown'
                    ),
                }
                for dag_id, dag_info in self.active_dags.items()
            }

        # Add queue status information
        try:
            queue_status = self.get_queue_status()
            debug_data["queue_status"] = queue_status
        except Exception as e:
            debug_data["queue_status_error"] = str(e)

        # Add job state counts if available
        try:
            state_counts = self.count_states()
            debug_data["job_state_counts"] = state_counts
        except Exception as e:
            debug_data["job_state_counts_error"] = str(e)

        return debug_data

    async def enqueue(self, work_info: WorkInfo) -> str | None:
        """
        Enqueues a work item for processing on the next available executor.

        :param work_info: The information about the work item to be processed.
        :return: The ID of the work item if successfully enqueued, None otherwise.
        """
        self.logger.debug(f"Enqueuing work item : {work_info.id}")
        submission_id = work_info.id
        entrypoint = work_info.data.get("metadata", {}).get("on")
        if not entrypoint:
            raise ValueError("The entrypoint 'on' is not defined in metadata")

        # FIXME : This is a hack to allow the job to be re-submitted after a failure
        await self.job_manager.job_info_client().delete_info(submission_id)
        try:
            returned_id = await self.job_manager.submit_job(
                entrypoint=entrypoint,
                submission_id=submission_id,
                metadata=work_info.data,
            )
        except ValueError as e:
            self.logger.error(f"Error submitting job: {e}")
            return None
        return returned_id

    async def get_work_items_by_queue(
        self,
        limit: int = 1,
        stop_event: asyncio.Event = None,
    ) -> dict[str, list[Any]]:
        """Get the Jobs from the PSQL database.

        :param limit: the maximal number records to get
        :param stop_event: an event to signal when to stop iterating over the records
        :return:
        """

        records_by_queue = {}
        # FIXME : Change how we check for known queues
        if not self.known_queues:
            self.logger.info("No known queues, skipping fetching jobs.")
            return records_by_queue

        async with self._lock:
            with self:
                try:
                    fetch_query_def = fetch_next_job(DEFAULT_SCHEMA)
                    for queue in self.known_queues:
                        query = fetch_query_def(
                            name=queue,
                            batch_size=limit,
                            include_metadata=False,
                            priority=True,
                            mark_as_active=False,
                        )
                        # we can't use named cursors as it will throw an error
                        cursor = self.connection.cursor()
                        cursor.itersize = limit
                        cursor.execute(f"{query}")
                        records = [record for record in cursor]
                        if records:
                            records_by_queue[queue] = records
                            self.logger.debug(
                                f"Fetched jobs from queue: {queue} > {len(records)}"
                            )
                        else:
                            self.logger.debug(f"No jobs found in queue: {queue}")

                    return records_by_queue
                except (Exception, psycopg2.Error) as error:
                    self.logger.error(f"Error fetching next job: {error}")
                    self.connection.rollback()
                finally:
                    self.connection.commit()

    async def get_job(self, job_id: str) -> Optional[WorkInfo]:
        """
        Get a job by its ID.
        :param job_id:
        """
        schema = DEFAULT_SCHEMA
        table = DEFAULT_JOB_TABLE

        with self:
            try:
                cursor = self.connection.cursor()
                cursor.execute(
                    f"""
                    SELECT
                          id,
                          name,
                          priority,
                          state,
                          retry_limit,
                          start_after,
                          expire_in,
                          data,
                          retry_delay,
                          retry_backoff,
                          keep_until,
                          dag_id,
                          job_level
                    FROM {schema}.{table}
                    WHERE id = '{job_id}'
                    """
                )
                record = cursor.fetchone()
                if record:
                    return self.record_to_work_info(record)
                return None
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error importing snapshot: {error}")
                self.connection.rollback()
            finally:
                self.connection.commit()

    async def get_job_for_policy(self, work_info: WorkInfo) -> Optional[WorkInfo]:
        """
        Find a job by its name and data.
        :param work_info:
        """
        schema = DEFAULT_SCHEMA
        table = DEFAULT_JOB_TABLE
        ref_type = work_info.data.get("metadata", {}).get("ref_type", "")
        ref_id = work_info.data.get("metadata", {}).get("ref_id", "")

        with self:
            try:
                cursor = self.connection.cursor()
                cursor.execute(
                    f"""
                        SELECT
                        id,
                        name,
                        priority,
                        state,
                        retry_limit,
                        start_after,
                        expire_in,
                        data,
                        retry_delay,
                        retry_backoff,
                        keep_until,
                        dag_id,
                        job_level
                    FROM {schema}.{table}
                    WHERE data->'metadata'->>'ref_type' = '{ref_type}'
                    AND data->'metadata'->>'ref_id' = '{ref_id}'
                    """
                )
                record = cursor.fetchone()
                if record:
                    return self.record_to_work_info(record)
                return None
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error getting job: {error}")
                self.connection.rollback()
            finally:
                self.connection.commit()

    async def list_jobs(
        self, state: Optional[str | list[str]] = None, batch_size: int = 0
    ) -> Dict[str, WorkInfo]:
        work_items = {}
        schema = DEFAULT_SCHEMA
        table = DEFAULT_JOB_TABLE

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
            states = "','".join(WorkState.__members__.keys()).lower()

        with self:
            try:
                cursor = self.connection.cursor("doc_iterator")
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
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error listing jobs: {error}")
                self.connection.rollback()
            finally:
                self.connection.commit()
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

    async def _process_submission_queue(self):
        """Background worker that processes queued job submissions"""
        self.logger.info("Background job submission worker started")

        while True:
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
                    self.logger.debug(
                        f"Successfully processed job: {request.work_info.id}"
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
        return {
            "queue_size": self._request_queue.qsize(),
            "pending_requests": len(self._pending_requests),
            "total_submissions": self._submission_count,
            "worker_running": self._worker_task and not self._worker_task.done(),
        }

    def _create_dag_and_jobs_sync(
        self, work_info: WorkInfo, submission_id: str
    ) -> Tuple[bool, str]:
        """Synchronous method for blocking database operations
        It is important to run this in a thread to prevent blocking the main event loop.

        :param work_info: WorkInfo object containing job details
        :param submission_id: Unique identifier for the job submission
        """
        # TODO: generate a unique dag_id if not provided
        dag_id = submission_id  # generate_job_id()
        dag_name = f"{dag_id}_dag"
        work_info.dag_id = dag_id
        query_plan_dag, topological_sorted_nodes = query_plan_work_items(work_info)
        connection = None

        try:
            # important that we use new connection for each job submission
            try:
                # insert the DAG, we will serialize the DAG to JSON and Pickle
                # this will allow us to re-create the DAG from the database without having to re-plan the DAG
                json_serialized_dag = query_plan_dag.model_dump()
                connection = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    insert_dag(DEFAULT_SCHEMA, dag_id, dag_name, json_serialized_dag),
                    connection=connection,
                    return_cursor=True,
                )
                # we will use the cursor to get the new dag id but it should be the same as dag_id
                new_dag_key = (
                    result[0] if cursor and (result := cursor.fetchone()) else None
                )
                self.logger.info(f"DAG inserted with ID: {new_dag_key}")
                self._close_cursor(cursor)

                for i, dag_work_info in enumerate(topological_sorted_nodes):
                    cursor = self._execute_sql_gracefully(
                        insert_job(
                            DEFAULT_SCHEMA,
                            dag_work_info,
                        ),
                        connection=connection,
                        return_cursor=True,
                    )
                    new_dag_key = cursor is not None and cursor.rowcount > 0
                    if i == 0:
                        new_key_added = new_dag_key

                    self._close_cursor(cursor)

                return new_key_added, new_dag_key

            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error creating job: {error}")
                raise ValueError(
                    f"Job creation for submission_id {submission_id} failed. "
                    f"Please check the logs for more information. {error}"
                )
        finally:
            self._close_connection(connection)

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

        # until we upgrade to pycog3 and convert to async we have to run this in a thread to prevent blocking of main loop
        new_key_added, new_dag_key = await self._loop.run_in_executor(
            self._db_executor, self._create_dag_and_jobs_sync, work_info, submission_id
        )

        if not new_key_added:
            raise ValueError(
                f"Job with submission_id {submission_id} already exists. "
                "Please use a different submission_id."
            )

        await self.notify_event()
        return submission_id

    async def _refresh_job_priorities(self):
        """Execute the job priority refresh SQL function"""
        try:
            print(f'Refreshing job priorities...  : {self._submission_count}')
            #   SELECT marie_scheduler.refresh_job_priority();
        except Exception as e:
            self.logger.error(f"Failed to refresh job priorities: {e}")

    async def mark_as_active(self, work_info: WorkInfo) -> bool:
        return await self._mark_active(
            "job", mark_as_active_jobs(DEFAULT_SCHEMA, work_info.name, [work_info.id])
        )

    async def mark_as_active_dag(self, work_info: WorkInfo) -> bool:
        return await self._mark_active(
            "dag", mark_as_active_dags(DEFAULT_SCHEMA, [work_info.dag_id])
        )

    async def _mark_active(self, label: str, query: str) -> bool:
        self.logger.debug(f"Marking {label} as active")
        with self:
            cursor = None
            try:
                cursor = self._execute_sql_gracefully(query, return_cursor=True)
                return result[0] if cursor and (result := cursor.fetchone()) else False
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error updating {label}: {error}")
                return False
            finally:
                self._close_cursor(cursor)

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
        :param job_id: The ID of the job.
        :param work_item: The work item to cancel.
        """
        with self:
            try:
                self.logger.info(f"Cancelling job: {job_id}")
                self._execute_sql_gracefully(
                    cancel_jobs(DEFAULT_SCHEMA, work_item.name, [job_id])
                )
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error handling job event: {error}")

    async def resume_job(self, job_id: str) -> None:
        """
        Resume a job by its ID.
        :param job_id:
        """
        name = "extract"  # TODO this is a placeholder
        with self:
            try:
                self.logger.info(f"Resuming job: {job_id}")
                self._execute_sql_gracefully(
                    resume_jobs(DEFAULT_SCHEMA, name, [job_id])
                )
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error handling job event: {error}")

    async def put_status(
        self,
        job_id: str,
        status: WorkState,
        started_on: Optional[datetime] = None,
        completed_on: Optional[datetime] = None,
    ):
        """
        Update the status of a job.
        :param job_id: The ID of the job.
        :param status: The new status of the job.
        :param started_on: Optional start time of the job.
        :param completed_on: Optional completion time of the job.
        """
        schema = DEFAULT_SCHEMA
        table = "job"

        update_fields = [f"state = '{status.value}'"]
        if started_on:
            update_fields.append(
                f"started_on = '{to_timestamp_with_tz(started_on)}'::timestamp with time zone "
            )
        if completed_on:
            update_fields.append(
                f"completed_on = '{to_timestamp_with_tz(completed_on)}'::timestamp with time zone "
            )

        update_query = f"""
        UPDATE {schema}.{table}
        SET {', '.join(update_fields)}
        WHERE id = '{job_id}'
        """

        with self:
            try:
                self._execute_sql_gracefully(update_query)
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error handling job event: {error}")

    async def maintenance(self):
        """
        Performs the maintenance process, including expiring, archiving, and purging.

        :return: None
        """
        try:
            with self:
                await self.expire()
                await self.archive()
                await self.purge()
        except Exception as e:
            self.logger.error(f"Error in maintenance: {e}")

    async def expire(self):
        print("Expiring jobs")

    async def archive(self):
        print("Archiving jobs")

    async def purge(self):
        print("Purging jobs")

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

    def count_states(self) -> Dict[str, Dict[str, int]]:
        """
        Fetch and count job states from the database.

        :return: A dictionary with queue names as keys and state counts as values.
        """
        state_count_default = {key.lower(): 0 for key in WorkState.__members__.keys()}
        counts = []
        cursor = None

        with self:
            try:
                cursor = self._execute_sql_gracefully(
                    count_states(DEFAULT_SCHEMA), return_cursor=True
                )
                counts = cursor.fetchall()
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error handling job event: {error}")
            finally:
                self._close_cursor(cursor)

        states = {"queues": {}}
        for item in counts:
            name, state, size = item
            if name:
                if name not in states["queues"]:
                    states["queues"][name] = state_count_default.copy()
            queue = states["queues"].get(name, states)
            queue[state or "all"] = int(size)

        return states

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
            state=record[3],
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
        wait_time = MONITORING_POLL_PERIOD
        while self.running:
            self.logger.debug(f"Polling jobs status : {wait_time}")
            await asyncio.sleep(wait_time)

            try:
                monitored_on = None
                cursor = None

                try:
                    cursor = self._execute_sql_gracefully(
                        try_set_monitor_time(
                            DEFAULT_SCHEMA,
                            monitor_state_interval_seconds=int(MONITORING_POLL_PERIOD),
                        ),
                        return_cursor=True,
                    )
                    monitored_on = cursor.fetchone()
                except (Exception, psycopg2.Error) as error:
                    self.logger.error(f"Error handling job event: {error}")
                finally:
                    self._close_cursor(cursor)

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
        # self.logger.info(f"Job completed : {job_id}, {work_item}")
        self.logger.info(f"Job completed : {job_id}")
        with self:

            def complete_jobs_wrapper(
                schema: str, name: str, ids: list, output: dict, _force: bool
            ):
                if _force:
                    return complete_jobs_by_id(schema, name, ids, output)
                else:
                    return complete_jobs(schema, name, ids, output)

            cursor = None

            try:
                cursor = self._execute_sql_gracefully(
                    complete_jobs_wrapper(
                        DEFAULT_SCHEMA,
                        work_item.name,
                        [job_id],
                        {"on_complete": "done", **(output_metadata or {})},
                        force,
                    ),
                    return_cursor=True,
                )
                counts = cursor.fetchone()[0]
                if counts > 0:
                    self.logger.debug(f"Completed job: {job_id} : {counts}")
                else:
                    self.logger.error(f"Error completing job: {job_id}")
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error completing job: {error}")
            finally:
                self._close_cursor(cursor)

    async def fail(
        self, job_id: str, work_item: WorkInfo, output_metadata: dict = None
    ):
        self.logger.info(f"Job failed : {job_id}")
        with self:
            try:
                cursor = self._execute_sql_gracefully(
                    fail_jobs_by_id(
                        DEFAULT_SCHEMA,
                        work_item.name,
                        [job_id],
                        {"on_complete": "failed", **(output_metadata or {})},
                    )
                )
                counts = cursor.fetchone()[0]
                if counts > 0:
                    self.logger.info(f"Completed failed job: {job_id}")
                else:
                    self.logger.error(f"Error completing failed job: {job_id}")
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error completing failed job: {error}")

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

                        if job_info.end_time is not None:
                            timestamp_ms = job_info.end_time  # Unix timestamp in ms
                            end_time = datetime.fromtimestamp(
                                timestamp_ms / 1000, tz=timezone.utc
                            )
                            now = datetime.now(tz=timezone.utc)
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
        await self._loop.run_in_executor(None, self._blocking_sync_dag)

    def _blocking_sync_dag(self, interval: int = 30) -> None:
        """
        Validate that DAGs in memory still exist and are active in database
        """
        self.logger.info(f"Starting DAG sync polling (interval: {interval}s)")

        while self.running:
            cursor = None

            try:
                if not self.active_dags:
                    self.logger.debug("No active DAGs in memory to validate")
                    time.sleep(interval)
                    continue

                memory_dag_ids = list(self.active_dags.keys())
                self.logger.debug(f"Validating {len(memory_dag_ids)} DAGs in memory")

                placeholders = ','.join(['%s'] * len(memory_dag_ids))
                query = f"""
                    SELECT id FROM marie_scheduler.dag 
                    WHERE id IN ({placeholders}) AND state = 'active'
                """

                cursor = self._execute_sql_gracefully(
                    query, memory_dag_ids, return_cursor=True
                )
                if not cursor:
                    self.logger.warning("No result from DAG validation query")
                    time.sleep(interval)
                    continue

                valid_dag_records = cursor.fetchall()
                valid_db_dags = {record[0] for record in valid_dag_records}
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
            finally:
                self._close_cursor(cursor)
            time.sleep(interval)
        self.logger.debug(f"DAG sync polling stopped")

    def _remove_dag_from_memory(self, dag_id: str, reason: str):
        """Centralized method to remove DAG from memory with logging"""
        if dag_id in self.active_dags:
            del self.active_dags[dag_id]
            self.logger.warning(f"Removed DAG {dag_id} from active_dags ({reason})")
        else:
            self.logger.debug(f"DAG {dag_id} not in active_dags ({reason})")

    def _blocking_sync_dag_events(self):
        self.logger.info("Starting BLOCKING-DAG synchronization")
        import json
        import select

        conn = self._get_connection()
        try:
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            cur.execute("LISTEN dag_state_changed;")
            self.logger.info("Listening for DAG changes...")

            while self.running:
                self.logger.info(
                    "Waiting for notifications on 'dag_state_changed' channel..."
                )
                if select.select([conn], [], [], 5) == ([], [], []):
                    continue  # Timeout, no notifications
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    self.logger.info(
                        f"Received NOTIFY: {notify.channel} - {notify.payload}"
                    )

                    try:
                        payload_data = json.loads(notify.payload)
                        operation = payload_data.get('operation')
                        dag_id = payload_data.get('dag_id')

                        if operation == 'DELETE':
                            # Handle DAG deletion
                            if dag_id in self.active_dags:
                                del self.active_dags[dag_id]
                                self.logger.warning(
                                    f"Removed deleted DAG {dag_id} from active_dags"
                                )
                        elif operation == 'UPDATE':
                            if True:
                                continue  # Skip updates for now

                            old_state = payload_data.get('old_state')
                            new_state = payload_data.get('new_state')
                            self.logger.info(
                                f"DAG {dag_id} state changed: {old_state} -> {new_state}"
                            )
                            # Remove from active_dags if no longer active
                            if new_state != 'active' and dag_id in self.active_dags:
                                del self.active_dags[dag_id]
                                self.logger.warning(
                                    f"Removed DAG {dag_id} from active_dags (state: {new_state})"
                                )

                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing notification payload: {e}")
                    except Exception as e:
                        self.logger.error(f"Error handling notification: {e}")

        except Exception as e:
            self.logger.error(f"Error in listener: {e}")
        finally:
            self._close_connection(conn)

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
        self.logger.debug(f"🔍 Resolving DAG status: {work_info.dag_id}")

        with self:
            cursor = None

            try:
                resolve_query = f"SELECT {DEFAULT_SCHEMA}.resolve_dag_state('{work_info.dag_id}'::uuid);"
                cursor = self._execute_sql_gracefully(resolve_query, return_cursor=True)
                dag_state = (
                    result[0] if cursor and (result := cursor.fetchone()) else None
                )
                self.logger.debug(f"Resolved DAG state: {dag_state}")

                if dag_state not in ("completed", "failed"):
                    self.logger.debug(f"DAG is still in progress: {work_info.dag_id}")
                    return False

                if work_info.dag_id in self.active_dags:
                    del self.active_dags[work_info.dag_id]
                    self.logger.debug(f"Removed DAG from cache: {work_info.dag_id}")

                self.logger.info(
                    f"Resolved DAG status: {work_info.dag_id}, status={dag_state}"
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
                    return False

                status = "OK" if dag_state == "completed" else "FAILED"

                await mark_as_complete(
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
            finally:
                self._close_cursor(cursor)

    def get_dag_by_id(self, dag_id: str) -> QueryPlan | None:
        """
        Retrieves a DAG by its ID, using in-memory cache if available.
        Falls back to loading from db if missing.
        """
        # Return from cache if present
        if dag_id in self.active_dags:
            return self.active_dags[dag_id]

        try:
            cursor = None
            with self:
                cursor = self._execute_sql_gracefully(
                    load_dag(DEFAULT_SCHEMA, dag_id), return_cursor=True
                )
                result = cursor.fetchone()
                if result and result[0]:
                    dag_definition = result[0]
                    dag = QueryPlan(**dag_definition)
                    self.logger.debug(f"📥 Loaded DAG from DB: {dag_id}")
                    return dag
                else:
                    self.logger.warning(f"DAG not found in DB: {dag_id}")
                    return None
        except Exception as e:
            self.logger.error(f"❌ Error loading DAG {dag_id}: {e}")
            return None
        finally:
            self._close_cursor(cursor)

    def _is_noop_query_definition(self, node: QueryDefinition) -> bool:
        """
        Checks if the given node is a NoopQueryDefinition node.
        NoopQueryDefinition is a special type of node that does not perform any operation and are used for
        aggregation or as placeholders in the query plan.
        """
        try:
            return isinstance(node.definition, NoopQueryDefinition)
        except ImportError:
            # If import fails, try to check by class name
            return node.__class__.__name__ == "NoopQueryDefinition" or (
                hasattr(node, 'definition')
                and node.query_definition.__class__.__name__ == "NoopQueryDefinition"
            )

    def get_node_from_dag(self, work_id: str, dag: QueryPlan) -> Query:
        """
        Retrieves a node from the DAG by its ID.

        Args:
            work_id: The ID of the node to retrieve
            dag: The DAG to search in

        Returns:
            The node if found, None otherwise
        """
        for node in dag.nodes:
            if node.task_id == work_id:
                return node
        raise ValueError(f"Node with ID {work_id} not found in DAG")

    def get_available_slots(self) -> dict[str, int]:
        return available_slots_by_executor(ClusterState.deployments)

    def reset_active_dags(self):
        """
        Reset the active DAGs dictionary, clearing all currently tracked DAGs.
        This can be useful for debugging or when you need to force a fresh state.

        Returns:
            dict: Information about the reset operation including count of cleared DAGs
        """
        try:
            cleared_count = len(self.active_dags) if self.active_dags else 0
            cleared_dags = list(self.active_dags.keys()) if self.active_dags else []

            # Reset the active DAGs
            self.active_dags = {}

            self.logger.info(f"Reset active DAGs: cleared {cleared_count} DAGs")
            if cleared_dags:
                self.logger.debug(f"Cleared DAGs: {cleared_dags}")

            return {
                "success": True,
                "cleared_count": cleared_count,
                "cleared_dags": cleared_dags,
                "message": f"Successfully reset active DAGs, cleared {cleared_count} DAGs",
            }
        except Exception as e:
            error_msg = f"Failed to reset active DAGs: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "cleared_count": 0,
                "cleared_dags": [],
            }


def query_plan_work_items(work_info: WorkInfo) -> tuple[QueryPlan, list[WorkInfo]]:
    # from metadata or fallback to name They can be this same
    query_planner_name = work_info.data.get("metadata", {}).get(
        "planner", work_info.name
    )

    planner_info = PlannerInfo(name=query_planner_name, base_id=work_info.id)
    plan = query_planner(planner_info)
    # pprint(plan.model_dump())
    # visualize_query_plan_graph(plan)
    yml_str = plan_to_yaml(plan)

    sorted_nodes = topological_sort(plan)
    job_levels = compute_job_levels(sorted_nodes, plan)
    # print("Topologically sorted nodes:", sorted_nodes)
    # print_sorted_nodes(sorted_nodes, plan)

    dag_nodes = []
    node_dict = {node.task_id: node for node in plan.nodes}

    for i, task_id in enumerate(sorted_nodes):
        node = node_dict.get(task_id)
        wi = copy.deepcopy(work_info)
        wi.id = node.task_id
        wi.job_level = job_levels[task_id]

        if has_mapper_config(__default_extract_dir__, query_planner_name):
            meta = JobMetadata.from_task(node, query_planner_name)
            meta_dict = meta.model_dump()  # need plain dict
            metadata = meta_dict["metadata"]
            wi.data['metadata'].update(metadata)
        else:
            logger.warning(
                f"No mapper configuration found for {query_planner_name}, "
                "using default metadata."
            )

        if i == 0:
            # this should already been handled by the query planner
            if node.dependencies:
                raise ValueError(
                    f"Root node has dependencies: {node.dependencies}, expected none"
                )
            work_info.dependencies = []  # Root node has no dependencies
            wi.dependencies = []
        else:
            wi.dependencies = node.dependencies
        dag_nodes.append(wi)

    return plan, dag_nodes


# https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/mit6_042js15_session17.pdf
