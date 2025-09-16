import asyncio
import contextlib
import copy
import itertools
import json
import random
import time
import traceback
import uuid
from asyncio import Queue
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, NamedTuple, Tuple, Union

import psycopg2

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
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.fixtures import *
from marie.scheduler.global_execution_planner import GlobalPriorityExecutionPlanner
from marie.scheduler.job_scheduler import JobScheduler
from marie.scheduler.models import ExistingWorkPolicy, HeartbeatConfig, WorkInfo
from marie.scheduler.plans import (
    cancel_jobs,
    complete_jobs,
    complete_jobs_by_id,
    count_dag_states,
    count_job_states,
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
from marie.scheduler.printers import (
    print_dag_state_summary,
    print_job_state_summary,
    print_slots_table,
)
from marie.scheduler.state import WorkState
from marie.scheduler.util import available_slots_by_executor
from marie.serve.runtimes.servers.cluster_state import ClusterState
from marie.storage.database.postgres import PostgresqlMixin

INIT_POLL_PERIOD = 2.250  # 250ms
SHORT_POLL_INTERVAL = 1.0  # seconds, when slots exist but no work

MIN_POLL_PERIOD = 0.250
MAX_POLL_PERIOD = 8

MONITORING_POLL_PERIOD = 5.0  # 5s
SYNC_POLL_PERIOD = 5.0  # 5s

DEFAULT_SCHEMA = "marie_scheduler"
DEFAULT_JOB_TABLE = "job"
COMPLETION_JOB_PREFIX = f"__state__{WorkState.COMPLETED.value}__"

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


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
    """
    Adjusts the backoff time for a polling mechanism based on the provided wait time,
    the number of consecutive idle streaks, and whether the task is scheduled. The
    resulting wait time ensures it stays within predefined minimum and maximum periods.
    In cases where the task is scheduled, the wait time is reduced. For non-scheduled
    tasks, it considers random jitter and adjusts based on idle streaks.
    """
    if scheduled:
        return max(wait_time * 0.5, MIN_POLL_PERIOD)
    jitter = random.uniform(0.9, 1.1)
    return min(wait_time * (1.5 + 0.1 * idle_streak), MAX_POLL_PERIOD) * jitter


# FIXME : Today we are tracking at the executor level, however that might not be the best
# approach. We might want to track at the deployment level (endpoint level) instead.
# this will allow us to track the status of the deployment and not just the executor.


class JobSubmissionRequest(NamedTuple):
    work_info: WorkInfo
    overwrite: bool
    request_id: str
    result_future: asyncio.Future


class PostgreSQLJobScheduler(PostgresqlMixin, JobScheduler):
    _mapper_warnings_shown = set()

    """A PostgreSQL-based job scheduler."""

    def __init__(self, config: Dict[str, Any], job_manager: JobManager):
        super().__init__()
        self.logger = MarieLogger(PostgreSQLJobScheduler.__name__)
        if job_manager is None:
            raise BadConfigSource("job_manager argument is required for JobScheduler")

        self.validate_config(config)
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

        self.active_dags = {}
        self.job_manager = job_manager
        self._loop = get_or_reuse_loop()
        self._setup_event_subscriptions()
        self._setup_storage(config, connection_only=True)

        self.execution_planner = GlobalPriorityExecutionPlanner()
        register_all_known_planners(
            QueryPlannersConf.from_dict(config.get("query_planners", {}))
        )

        dag_config = config.get("dag_manager", {})
        dag_strategy = dag_config.get("strategy", 'fixed')  # fixed or dynamic
        min_concurrent_dags = int(dag_config.get("min_concurrent_dags", 1))
        max_concurrent_dags = int(dag_config.get("max_concurrent_dags", 16))
        cache_ttl_seconds = int(dag_config.get("cache_ttl_seconds", 5))
        dag_cache_size = int(
            dag_config.get("dag_cache_size", 5000)
        )  # 5000 entries as this is what our fetch_next_job uses
        self._topology_cache = DagTopologyCache(maxsize=dag_cache_size)

        heartbeat_config_dict = config.get("heartbeat", {})
        self.heartbeat_config = HeartbeatConfig.from_dict(heartbeat_config_dict)
        self.logger.info(f"Heartbeat configuration: {self.heartbeat_config}")

        self.max_concurrent_dags = max_concurrent_dags
        self._start_time = datetime.now(timezone.utc)

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

        self.logger.info(f"received message: {event_type} > {message}")
        if not isinstance(message, dict) or "job_id" not in message:
            self.logger.error(f"Invalid message format: {message}")
            return

        async with self._lock:
            try:
                job_id = message.get("job_id")
                status = JobStatus(event_type)
                work_item: WorkInfo = await self._loop.run_in_executor(
                    self._db_executor, self.get_job, job_id
                )
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
        ]

        # SQL files to be loaded
        sql_files = [
            "job_dependencies.sql",
            "fetch_next_job.sql",
            "create_indexes.sql",
            "create_constraints.sql",
            "resolve_dag_state.sql",
            "count_job_states.sql",
            "count_dag_states.sql",
            "refresh_job_priority.sql",
            "delete_dags_and_jobs.sql",
            "delete_failed_dags_and_jobs.sql",
            "delete_orphaned_jobs.sql",
            "jobs_with_unmet_dependencies.sql",
            # "notify_dag_state_change.sql", NOT USING CURRENTLY
            "purge_non_started_work.sql",
            "ready_jobs_view.sql",
            "refresh_dag_durations.sql",
            "refresh_job_durations.sql",
            "reset_active_dags_and_jobs.sql",
            "reset_all.sql",
            "reset_completed_dags_and_jobs.sql",
            "suspend_non_started_work.sql",
            "unsuspend_work.sql",
            "sync_job_dependencies.sql",
        ]

        commands.extend(
            [
                create_sql_from_file(
                    schema, os.path.join(__default_schema_dir__, fname)
                )
                for fname in sql_files
            ]
        )

        commands.extend(
            [
                create_sql_from_file(
                    schema, os.path.join(__default_psql_dir__, "cron_job_init.sql")
                )
            ]
        )

        query = ";\n".join(commands)

        locked_query = f"""
           BEGIN;
           SET LOCAL statement_timeout = '30s';
           SELECT pg_try_advisory_lock(1);
           {query};
           SELECT pg_advisory_unlock(1);
           COMMIT;
           """

        # Write query to temp file for review
        tmp_path = "/tmp/marie/psql"
        os.makedirs(tmp_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_file = os.path.join(tmp_path, f"locked_query_{timestamp}.sql")

        try:
            with open(query_file, 'w') as f:
                f.write(locked_query)
            self.logger.info(f"Wrote locked query to: {query_file}")
        except Exception as e:
            self.logger.error(f"Failed to write query to file: {e}")

        conn = None
        try:
            conn = self._get_connection()
            self._execute_sql_gracefully(locked_query, connection=conn)
        except (Exception, psycopg2.Error) as error:
            if isinstance(error, psycopg2.errors.DuplicateTable):
                self.logger.warning("Tables already exist, skipping creation.")
            else:
                self.logger.error(f"Error creating tables: {error}")
                raise RuntimeFailToStart(
                    f"Failed to create tables in schema '{schema}': {error}"
                )
        finally:
            self._close_connection(conn)

    async def wipe(self) -> None:
        """Clears the schedule storage."""
        schema = DEFAULT_SCHEMA
        query = f"""
           TRUNCATE {schema}.job, {schema}.archive
           """
        self.logger.info(f"Wiping all data in schema '{schema}'")
        conn = None
        try:
            conn = self._get_connection()
            self._execute_sql_gracefully(query, connection=conn)
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error wiping tables: {error}")
            raise RuntimeFailToStart(
                f"Failed to wipe tables in schema '{schema}': {error}"
            )
        finally:
            self._close_connection(conn)

    async def is_installed(self) -> bool:
        """check if the tables are installed"""
        schema = DEFAULT_SCHEMA
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            cursor = self._execute_sql_gracefully(
                version_table_exists(schema), return_cursor=True, connection=conn
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
            self._close_connection(conn)
        return False

    async def create_queue(self, queue_name: str) -> None:
        """Setup the queue for the scheduler."""
        self._execute_sql_gracefully(create_queue(DEFAULT_SCHEMA, queue_name, {}))

    async def _get_defined_queues(self) -> set[str]:
        """Setup the queue for the scheduler."""
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            cursor = self._execute_sql_gracefully(
                f"SELECT name FROM {DEFAULT_SCHEMA}.queue",
                return_cursor=True,
                connection=conn,
            )
            if cursor and cursor.rowcount > 0:
                result = cursor.fetchall()
                return {name[0] for name in result}
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error getting known queues: {error}")
            raise RuntimeFailToStart(
                f"Unable to find queues in schema '{DEFAULT_SCHEMA}': {error}"
            )
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

        return set()  # Return an empty set if no queues are defined

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

        defined_queues = await self._get_defined_queues()
        for work_queue in self.known_queues.difference(defined_queues):
            self.logger.info(f"Create queue: {work_queue}")
            await self.create_queue(work_queue)
            await self.create_queue(f"${work_queue}_dlq")

        self.running = True
        # self.sync_task = asyncio.create_task(self._sync())
        # self.monitoring_task = asyncio.create_task(self._monitor())
        self.monitoring_task = None

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(self.heartbeat_config)
        )
        self._poll_task = asyncio.create_task(self._poll())
        self._cluster_state_monitor_task = asyncio.create_task(
            self.__monitor_deployment_updates()
        )

        self._worker_tasks = [
            asyncio.create_task(self._process_submission_queue(worker_id))
            for worker_id in range(self.max_workers)
        ]

        # self._sync_dag_task = asyncio.create_task(self._sync_dag())
        await self.notify_event()

    async def _safe_count_states(
        self,
        count_method: Union[
            Callable[[], Dict[str, Any]], Callable[[], Awaitable[Dict[str, Any]]]
        ],
    ) -> Dict[str, Any]:
        """Safely call count methods with proper async handling."""
        try:
            if asyncio.iscoroutinefunction(count_method):
                return await count_method()
            else:
                return await self._loop.run_in_executor(self._db_executor, count_method)
        except Exception as e:
            self.logger.error(f"Error in count method: {e}")
            return {}

    async def _heartbeat_loop(self, config: HeartbeatConfig) -> None:
        """Periodic heartbeat with throughput, rolling averages, and trend analysis (global + per queue)."""

        self.logger.info(
            f"Heartbeat loop started: interval={config.interval}s, "
            f"window={config.window_minutes}m, recent={config.recent_window_minutes}m"
        )

        _seen_executors = set()
        _max_seen_executors = {}

        last_heartbeat_time = None
        last_completed_jobs = {}
        last_completed_dags = {}
        history = deque()  # (timestamp, jobs_per_queue, dags_per_queue)

        rolling_job_rates = deque(maxlen=config.trend_points)
        rolling_dag_rates = deque(maxlen=config.trend_points)

        rolling_job_rates_per_queue = defaultdict(
            lambda: deque(maxlen=config.trend_points)
        )
        rolling_dag_rates_per_queue = defaultdict(
            lambda: deque(maxlen=config.trend_points)
        )

        def _get_completed_per_queue(states: dict, key: str = "completed") -> dict:
            if not states or "queues" not in states:
                return {}
            return {qname: q.get(key, 0) for qname, q in states["queues"].items()}

        def _calc_throughput(curr: dict, prev: dict, minutes: float) -> dict:
            return {
                qname: (
                    max(0.0, (curr[qname] - prev.get(qname, 0)) / minutes)
                    if minutes > 0
                    else 0.0
                )
                for qname in curr
            }

        def _calc_diff(curr: dict, prev: dict) -> dict:
            return {qname: curr[qname] - prev.get(qname, 0) for qname in curr}

        def _trend(values: deque) -> str:
            """Return colored trend arrow (⬆️, ⬇️, ➡️)."""
            if not config.enable_trend_arrows or len(values) < 2:
                return ""
            first, last = values[0], values[-1]
            if last > first * 1.05:
                return f"{GREEN}⬆️{RESET}"
            elif last < first * 0.95:
                return f"{RED}⬇️{RESET}"
            return f"{YELLOW}➡️{RESET}"

        retry_count = 0

        while self.running:
            try:
                unix_now = time.time()
                queue_size = self._event_queue.qsize()
                active_dags = list(self.active_dags.keys())
                slot_info = None

                # Executor stats collection (configurable)
                if config.enable_executor_stats:
                    slot_info = available_slots_by_executor(ClusterState.deployments)
                    _seen_executors.update(slot_info.keys())

                    for executor, count in slot_info.items():
                        current_max = _max_seen_executors.get(executor, 0)
                        _max_seen_executors[executor] = max(current_max, count)

                dag_states = await self._safe_count_states(self.count_dag_states)
                job_states = await self._safe_count_states(self.count_job_states)

                current_completed_jobs = _get_completed_per_queue(
                    job_states, "completed"
                )
                current_completed_dags = _get_completed_per_queue(
                    dag_states, "completed"
                )

                total_completed_jobs = sum(current_completed_jobs.values())
                total_completed_dags = sum(current_completed_dags.values())

                history.append(
                    (
                        unix_now,
                        current_completed_jobs.copy(),
                        current_completed_dags.copy(),
                    )
                )

                # Clean up old history (beyond rolling window)
                cutoff = unix_now - config.window_minutes * 60
                while history and history[0][0] < cutoff:
                    history.popleft()

                # --- Recent Throughput ---
                recent_window_seconds = config.recent_window_minutes * 60.0
                recent_cutoff = unix_now - recent_window_seconds

                recent_history = [
                    entry for entry in history if entry[0] >= recent_cutoff
                ]

                if len(recent_history) >= 2:
                    # Use oldest and newest entries from recent window
                    t0, jobs0, dags0 = recent_history[0]
                    t1, jobs1, dags1 = recent_history[-1]
                    time_diff_minutes = (t1 - t0) / 60.0

                    if time_diff_minutes > 0:
                        jobs_per_queue_instant = _calc_throughput(
                            jobs1, jobs0, time_diff_minutes
                        )
                        dags_per_queue_instant = _calc_throughput(
                            dags1, dags0, time_diff_minutes
                        )
                        jobs_per_min_global_instant = sum(
                            jobs_per_queue_instant.values()
                        )
                        dags_per_min_global_instant = sum(
                            dags_per_queue_instant.values()
                        )
                    else:
                        jobs_per_queue_instant = {}
                        dags_per_queue_instant = {}
                        jobs_per_min_global_instant = dags_per_min_global_instant = 0.0
                elif last_heartbeat_time is not None:
                    # Fallback to heartbeat-based calculation if insufficient recent history
                    time_diff_minutes = (unix_now - last_heartbeat_time) / 60.0
                    jobs_per_queue_instant = _calc_throughput(
                        current_completed_jobs, last_completed_jobs, time_diff_minutes
                    )
                    dags_per_queue_instant = _calc_throughput(
                        current_completed_dags, last_completed_dags, time_diff_minutes
                    )
                    jobs_per_min_global_instant = sum(jobs_per_queue_instant.values())
                    dags_per_min_global_instant = sum(dags_per_queue_instant.values())
                else:
                    jobs_per_queue_instant = {}
                    dags_per_queue_instant = {}
                    jobs_per_min_global_instant = dags_per_min_global_instant = 0.0

                # --- Rolling throughput (full window) ---
                jobs_per_min_global_window = 0.0
                dags_per_min_global_window = 0.0
                jobs_per_queue_window = defaultdict(float)
                dags_per_queue_window = defaultdict(float)
                window_suffix = " (insufficient data)"

                if len(history) >= 2:
                    # Time-weighted rolling average calculation.
                    # This is more robust against bursty traffic than a simple start/end point calculation.
                    total_job_rate_x_time = 0.0
                    total_dag_rate_x_time = 0.0
                    total_time_diff = 0.0

                    per_queue_job_rate_x_time = defaultdict(float)
                    per_queue_dag_rate_x_time = defaultdict(float)

                    for i in range(1, len(history)):
                        t_prev, jobs_prev, dags_prev = history[i - 1]
                        t_curr, jobs_curr, dags_curr = history[i]

                        interval_minutes = (t_curr - t_prev) / 60.0
                        if interval_minutes <= 0:
                            continue

                        total_time_diff += interval_minutes

                        # Calculate global rate for this interval
                        interval_jobs_completed = sum(jobs_curr.values()) - sum(
                            jobs_prev.values()
                        )
                        interval_dags_completed = sum(dags_curr.values()) - sum(
                            dags_prev.values()
                        )

                        interval_job_rate = (
                            max(0, interval_jobs_completed) / interval_minutes
                        )
                        interval_dag_rate = (
                            max(0, interval_dags_completed) / interval_minutes
                        )

                        total_job_rate_x_time += interval_job_rate * interval_minutes
                        total_dag_rate_x_time += interval_dag_rate * interval_minutes

                        # Calculate per-queue rate for this interval
                        all_queues = set(jobs_curr.keys()) | set(jobs_prev.keys())
                        for q in all_queues:
                            q_jobs_completed = jobs_curr.get(q, 0) - jobs_prev.get(q, 0)
                            q_dags_completed = dags_curr.get(q, 0) - dags_prev.get(q, 0)
                            q_job_rate = max(0, q_jobs_completed) / interval_minutes
                            q_dag_rate = max(0, q_dags_completed) / interval_minutes
                            per_queue_job_rate_x_time[q] += (
                                q_job_rate * interval_minutes
                            )
                            per_queue_dag_rate_x_time[q] += (
                                q_dag_rate * interval_minutes
                            )

                    if total_time_diff > 0:
                        jobs_per_min_global_window = (
                            total_job_rate_x_time / total_time_diff
                        )
                        dags_per_min_global_window = (
                            total_dag_rate_x_time / total_time_diff
                        )
                        for q in per_queue_job_rate_x_time:
                            jobs_per_queue_window[q] = (
                                per_queue_job_rate_x_time[q] / total_time_diff
                            )
                        for q in per_queue_dag_rate_x_time:
                            dags_per_queue_window[q] = (
                                per_queue_dag_rate_x_time[q] / total_time_diff
                            )

                    actual_window_minutes = (history[-1][0] - history[0][0]) / 60.0
                    if actual_window_minutes < config.window_minutes * 0.9:
                        window_suffix = f" ({actual_window_minutes:.1f}m data)"
                    else:
                        window_suffix = ""
                else:
                    # Fallback for rolling if not enough history
                    jobs_per_queue_window = jobs_per_queue_instant.copy()
                    dags_per_queue_window = dags_per_queue_instant.copy()
                    jobs_per_min_global_window = jobs_per_min_global_instant
                    dags_per_min_global_window = dags_per_min_global_instant

                rolling_job_rates.append(jobs_per_min_global_window)
                rolling_dag_rates.append(dags_per_min_global_window)

                for q in set(current_completed_jobs) | set(current_completed_dags):
                    rolling_job_rates_per_queue[q].append(
                        jobs_per_queue_window.get(q, 0.0)
                    )
                    rolling_dag_rates_per_queue[q].append(
                        dags_per_queue_window.get(q, 0.0)
                    )

                # Snapshot
                last_heartbeat_time = unix_now
                last_completed_jobs = current_completed_jobs
                last_completed_dags = current_completed_dags

                # --- Logging with configuration ---
                self.logger.info("🔄  Scheduler Heartbeat")
                self.logger.info(f"  📦  Queue Size       : {queue_size}")
                self.logger.info(f"  🧠  Active DAGs      : {len(active_dags)}")
                # Global throughput + trend
                self.logger.info(f"  📈  Throughput: ")

                self.logger.info(
                    f"  • recent ({config.recent_window_minutes}m): {jobs_per_min_global_instant:.2f} jobs/min, "
                    f"{dags_per_min_global_instant:.2f} dags/min"
                )

                trend_job = _trend(rolling_job_rates)
                trend_dag = _trend(rolling_dag_rates)

                self.logger.info(
                    f"  • rolling (last {config.window_minutes}m{window_suffix}): "
                    f"{jobs_per_min_global_window:.2f} jobs/min {trend_job}, "
                    f"{dags_per_min_global_window:.2f} dags/min {trend_dag}"
                )
                self.logger.info(
                    f"  ✅  Totals            : {total_completed_jobs} jobs, {total_completed_dags} dags"
                )

                # Per-queue throughput + trend (configurable)
                if config.enable_per_queue_stats and (
                    jobs_per_queue_instant or dags_per_queue_instant
                ):
                    self.logger.info("  📊 Per-Queue Throughput:")
                    for qname in sorted(
                        set(current_completed_jobs) | set(current_completed_dags)
                    ):
                        jpm_i = jobs_per_queue_instant.get(qname, 0.0)
                        dpm_i = dags_per_queue_instant.get(qname, 0.0)
                        jpm_w = jobs_per_queue_window.get(qname, 0.0)
                        dpm_w = dags_per_queue_window.get(qname, 0.0)

                        jtot = current_completed_jobs.get(qname, 0)
                        dtot = current_completed_dags.get(qname, 0)

                        jtrend = _trend(rolling_job_rates_per_queue[qname])
                        dtrend = _trend(rolling_dag_rates_per_queue[qname])

                        self.logger.info(
                            f"   • {qname:<12} | Jobs: {jpm_i:.2f}/min ({config.recent_window_minutes}m), "
                            f"{jpm_w:.2f}/min ({config.window_minutes}m{window_suffix}) {jtrend}"
                        )
                        self.logger.info(
                            f"     {'':<12} | DAGs: {dpm_i:.2f}/min ({config.recent_window_minutes}m), "
                            f"{dpm_w:.2f}/min ({config.window_minutes}m{window_suffix}) {dtrend}"
                        )
                        self.logger.info(
                            f"     {'':<12} | Totals: {jtot} jobs, {dtot} dags"
                        )

                if config.log_active_dags and active_dags:
                    shown = ', '.join(active_dags[:5])
                    suffix = '...' if len(active_dags) > 5 else ''
                    self.logger.debug(f"     DAG IDs          : [{shown}{suffix}]")

                print_dag_state_summary(dag_states)
                print_job_state_summary(job_states)

                if config.enable_executor_stats:
                    print_slots_table(slot_info, _max_seen_executors)

                (
                    await self.diagnose_pool()
                    if asyncio.iscoroutinefunction(self.diagnose_pool)
                    else self.diagnose_pool()
                )

                await asyncio.sleep(config.interval)
                retry_count = 0

            except Exception as e:
                retry_count += 1
                self.logger.error(
                    f"Heartbeat loop error (attempt {retry_count}/{config.max_retries}): {e}"
                )

                if retry_count >= config.max_retries:
                    self.logger.critical(
                        f"Heartbeat loop failed {config.max_retries} times, stopping"
                    )
                    break

                backoff_time = config.error_backoff * (
                    2 ** (retry_count - 1)
                ) + random.uniform(0, 1)
                self.logger.warning(f"Retrying heartbeat in {backoff_time:.2f}s...")
                await asyncio.sleep(backoff_time)

    async def _poll(self):
        """
        Handles the job scheduling polling process for the database in serial mode.

        This coroutine function is responsible for managing
        ecutors. It includes mechanisms for
        handling job dependencies, activation and cleanup of DAGs (Directed Acyclic Graphs),
        and backoff strategies for managing idle periods or situations with no executor slots
        available.

        :return: None
        """
        self.logger.info("Starting database job scheduler (serial mode)")
        wait_time = INIT_POLL_PERIOD
        batch_size = 25000
        failures = 0
        idle_streak = 0

        last_deployments_timestamp = ClusterState.deployments_last_updated
        slots_by_executor = available_slots_by_executor(ClusterState.deployments).copy()
        recently_activated_dags = deque()
        ACTIVATION_TIMEOUT = 60  # Seconds to consider DAG recently activated

        def cleanup_recently_activated_dags():
            """Removes DAGs that were activated too long ago from the deque"""
            current_time = time.time()
            while (
                recently_activated_dags
                and recently_activated_dags[0][1] < current_time - ACTIVATION_TIMEOUT
            ):
                recently_activated_dags.popleft()

        dag_id = None  # this is the last dag_id that was processed
        max_concurrent_dags = self.max_concurrent_dags
        self.logger.info(f"Max concurrent DAGs set to: {max_concurrent_dags}")

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
                slots_by_executor = available_slots_by_executor(
                    ClusterState.deployments
                ).copy()

                # back off if no slots
                self.logger.info(
                    f"Available slots ({len(self.active_dags)}/{max_concurrent_dags}) : {slots_by_executor}"
                )
                if not any(slots_by_executor.values()):
                    self.logger.warning("No available executor slots. Backing off.")
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
                recently_activated_dag_ids = set(d for d, _ in recently_activated_dags)

                flat_jobs = self.execution_planner.plan(
                    flat_jobs,
                    slots_by_executor,
                    self.active_dags,
                    # recently_activated_dag_ids,
                )
                scheduled_any = False

                if not flat_jobs:
                    self.logger.debug(
                        "Slots available but no ready jobs; sleeping briefly before next poll."
                    )
                    await asyncio.sleep(SHORT_POLL_INTERVAL)
                    continue

                # Count jobs per executor
                jobs_by_executor = defaultdict(int)
                for entrypoint, _ in flat_jobs:
                    executor = entrypoint.split("://")[0]
                    jobs_by_executor[executor] += 1

                self.logger.info(f"Jobs candidates per executor: {jobs_by_executor}")
                for executor, count in jobs_by_executor.items():
                    self.logger.info(f"  {executor}: {count} jobs")

                await self.debug_work_plans(flat_jobs, records_by_queue)

                jobs_scheduled_this_cycle = defaultdict(int)
                enqueue_tasks = []

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

                    dag = await self.get_dag_by_id(dag_id)
                    if dag is None:
                        self._remove_dag_from_memory(dag_id, "DAG not found: {dag_id}")
                        raise ValueError(f"DAG not found: {dag_id}")

                    if dag_id not in self.active_dags:
                        await self.mark_as_active_dag(work_info)
                        self.active_dags[dag_id] = dag
                        recently_activated_dags.append((dag_id, time.time()))
                        self.logger.debug(f"DAG activated: {dag_id}")

                    node = self.get_node_from_dag(work_info.id, dag)

                    if self._is_noop_query_definition(node):
                        start_t = time.time()
                        # expensive operations
                        # sorted_nodes = topological_sort(dag)
                        # job_levels = compute_job_levels(sorted_nodes, dag)
                        sorted_nodes, job_levels = (
                            self._topology_cache.get_sorted_nodes_and_levels(
                                dag, dag_id
                            )
                        )
                        max_level = max(job_levels.values())
                        now = datetime.now()
                        # there is no need to put_status as the complete call will do it
                        # await self.put_status(
                        #     work_info.id, WorkState.COMPLETED, now, now
                        # )
                        await self.complete(work_info.id, work_info, {}, force=True)
                        if (
                            max_level == work_info.job_level
                        ):  # There is no need to resolve DAG if we are in NOOP and not at the END
                            await self.resolve_dag_status(work_info.id, work_info)
                        await self.notify_event()

                        continue

                    if slots_by_executor.get(executor, 0) <= 0:
                        self.logger.debug(
                            f"No slots for {executor}, delaying job {work_info.id}"
                        )
                        continue

                    # Reserve a slot and create a task for this job
                    slots_by_executor[executor] -= 1

                    async def __schedule_job(wi: WorkInfo):
                        return await self.enqueue(wi)

                    task = asyncio.create_task(__schedule_job(work_info))
                    enqueue_tasks.append(
                        {'task': task, 'work_info': work_info, 'executor': executor}
                    )

                #  enqueue tasks in parallel and process the results
                scheduled_any = False
                if enqueue_tasks:
                    task_list = [t['task'] for t in enqueue_tasks]
                    results = await asyncio.gather(*task_list, return_exceptions=True)

                    for i, result in enumerate(results):
                        work_info = enqueue_tasks[i]['work_info']
                        executor = enqueue_tasks[i]['executor']

                        if isinstance(result, Exception):
                            self.logger.error(
                                f"Failed to schedule job {work_info.id}: {result}"
                            )
                        elif result:  # result is the enqueue_id
                            jobs_scheduled_this_cycle[executor] += 1
                            self.logger.info(f"Job scheduled: {result} on {executor}")
                            scheduled_any = True
                        else:
                            self.logger.error(
                                f"Failed to enqueue job: {work_info.id} on {executor} - no result returned"
                            )

                if jobs_scheduled_this_cycle or len(flat_jobs) > 0:
                    self.logger.info("Scheduling summary for this cycle:")
                    all_executors = sorted(list(jobs_by_executor.keys()))

                    for executor in all_executors:
                        candidates = jobs_by_executor.get(executor, 0)
                        scheduled = jobs_scheduled_this_cycle.get(executor, 0)
                        remaining = candidates - scheduled
                        self.logger.info(
                            f"  - {executor}: {scheduled} scheduled / {candidates} candidates ({remaining} remaining)"
                        )

                    total_scheduled = sum(jobs_scheduled_this_cycle.values())
                    remaining_candidates = len(flat_jobs) - total_scheduled
                    self.logger.info(
                        f"Total scheduled: {total_scheduled}. Total unscheduled candidates: {remaining_candidates}."
                    )

                self.logger.debug(f"Remaining slots: {slots_by_executor}")
                # acknowledgments, we are waiting for the jobs to be scheduled on the executor and get response via ETCD
                if scheduled_any:
                    await self.notify_event()

                # if scheduled_any:
                #     try:
                #         await asyncio.wait_for(
                #             ClusterState.scheduled_event.wait(), timeout=1
                #         )  # THIS SHOULD BE SAME AS ETCD lease time
                #     except asyncio.TimeoutError:
                #         self.logger.warning("Timeout waiting for schedule confirmation")
                #     finally:
                #         ClusterState.scheduled_event.clear()
                #         await self.notify_event()

                idle_streak = 0 if scheduled_any else idle_streak + 1
                wait_time = adjust_backoff(wait_time, idle_streak, scheduled_any)
                failures = 0
            except Exception as e:
                self.logger.error("Poll loop exception", exc_info=True)
                failures += 1

                self._remove_dag_from_memory(dag_id, "poll loop exception ")

                if failures >= 5:
                    self.logger.warning("Too many failures — entering cooldown")
                    await asyncio.sleep(60)
                    failures = 0
                    # TODO : Fire ENGINE FAILURE NOTIFICATION

    async def debug_work_plans(
        self,
        flat_jobs: list[Tuple[str, WorkInfo]],
        records_by_queue: dict[str, list[Any]],
    ) -> None:
        """
        Debugs and writes work plans involving queues and job details to a file if the
        environment variable `MARIE_DEBUG_QUERY_PLAN` is set.

        :param flat_jobs: A collection of jobs and their associated work information
            represented as a list of tuples. Each tuple contains an entry point and
            its corresponding work details.
        :param records_by_queue: A dictionary mapping queue names to the list of records
            associated with each queue.
        :return: None if the debug operation completes. If the environment variable
            `MARIE_DEBUG_QUERY_PLAN` is not set, the function immediately returns without
            performing any operation.
        """

        if "MARIE_DEBUG_QUERY_PLAN" not in os.environ:
            return

        os.makedirs("/tmp/marie/plans", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file_path = f"/tmp/marie/plans/flat_jobs_plan_debug_{timestamp}.txt"
        try:
            with open(debug_file_path, 'w') as debug_file:
                debug_file.write("Nodes:\n")
                debug_file.write(json.dumps(ClusterState.deployments, indent=4))

                debug_file.write("\n\n")
                debug_file.write("Slots:\n")

                debug_file.write(json.dumps(self.get_available_slots(), indent=4))
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
                if False:
                    debug_file.write("\n\n")
                    debug_file.write(f"Queues:\n")
                    for queue_name, records in records_by_queue.items():
                        debug_file.write(f"Queue: {queue_name}\n")
                        for record in records:
                            debug_file.write(f"  Records:  {record},\n")
                    debug_file.write("\n\n")
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
            debug_data["queue_status"] = self.get_queue_status()
        except Exception as e:
            debug_data["queue_status_error"] = str(e)

        try:
            debug_data["job_state_counts"] = self.count_job_states()
        except Exception as e:
            debug_data["job_state_counts_error"] = str(e)

        try:
            debug_data["dag_state_counts"] = self.count_dag_states()
        except Exception as e:
            debug_data["dag_state_counts_error"] = str(e)

        return debug_data

    async def enqueue(self, work_info: WorkInfo) -> str | None:
        """
        Enqueues a work item for processing on the next available executor.

        :param work_info: The information about the work item to be processed.
        :return: The ID of the work item if successfully enqueued, None otherwise.
        """
        self.logger.debug(f"Enqueuing work item : {work_info.id}")
        confirmation_event = asyncio.Event()

        submission_id = work_info.id
        entrypoint = work_info.data.get("metadata", {}).get("on")
        if not entrypoint:
            raise ValueError("The entrypoint 'on' is not defined in metadata")

        await self.job_manager.job_info_client().delete_info(submission_id)

        try:
            returned_id = await self.job_manager.submit_job(
                entrypoint=entrypoint,
                submission_id=submission_id,
                metadata=work_info.data,
                confirmation_event=confirmation_event,
            )
            # wait for runner ack (keep this short—matches your lease)
            await asyncio.wait_for(confirmation_event.wait(), timeout=2)

            ok = await self.mark_as_active(work_info)
            if not ok:
                self.logger.error(f"Failed to mark ACTIVE in DB: {submission_id}")
                return None

            return returned_id

        except asyncio.TimeoutError:
            self.logger.error(
                f"Timeout waiting for submit confirmation: {submission_id}"
            )
            # ensure job is visible again
            await self.put_status(submission_id, WorkState.RETRY)
            return None
        except Exception as e:
            self.logger.error(f"Enqueue unexpected error for {submission_id}: {e}")
            await self.put_status(submission_id, WorkState.RETRY)
            return None

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
            self.logger.warning("No known queues, skipping fetching jobs.")
            return records_by_queue

        conn = None
        cursor = None

        try:
            conn = self._get_connection()
            fetch_query_def = fetch_next_job(DEFAULT_SCHEMA)
            for queue in self.known_queues:
                cursor = None
                try:
                    query = fetch_query_def(
                        name=queue,
                        batch_size=limit,
                        include_metadata=False,
                        priority=True,
                    )
                    # we can't use named cursors as it will throw an error
                    cursor = conn.cursor()
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
                finally:
                    self._close_cursor(cursor)

            conn.commit()
            return records_by_queue
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error fetching next job: {error}")
            conn.rollback()
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)
        return {}

    def get_job(self, job_id: str) -> Optional[WorkInfo]:
        """
        Get a job by its ID.
        :param job_id:
        """
        schema = DEFAULT_SCHEMA
        table = DEFAULT_JOB_TABLE
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
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
            conn.commit()
            if record:
                return self.record_to_work_info(record)
            return None
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error getting job: {error}")
            conn.rollback()
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    async def get_job_for_policy(self, work_info: WorkInfo) -> Optional[WorkInfo]:
        """
        Find a job by its name and data.
        :param work_info:
        """
        schema = DEFAULT_SCHEMA
        table = DEFAULT_JOB_TABLE
        ref_type = work_info.data.get("metadata", {}).get("ref_type", "")
        ref_id = work_info.data.get("metadata", {}).get("ref_id", "")
        cursor = None
        conn = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
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
            conn.commit()
            if record:
                return self.record_to_work_info(record)
            return None
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error getting job: {error}")
            conn.rollback()
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

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
            states = "','".join(WorkState.__members__.keys()).lower()

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
        cursor = None
        new_key_added = False

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
                job_inserted = cursor is not None and cursor.rowcount > 0
                if i == 0:
                    new_key_added = job_inserted

                self._close_cursor(cursor)

            connection.commit()
            return new_key_added, new_dag_key

        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error creating job: {error}")
            if connection:
                connection.rollback()
            raise ValueError(
                f"Job creation for submission_id {submission_id} failed. "
                f"Please check the logs for more information. {error}"
            )
        finally:
            self._close_cursor(cursor)
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
            self.logger.info(
                f'Refreshing job priorities...  : {self._submission_count}'
            )
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

        def db_call() -> bool:
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    query, return_cursor=True, connection=conn
                )
                return result[0] if cursor and (result := cursor.fetchone()) else False
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error updating {label}: {error}")
                return False
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

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
        :param job_id: The ID of the job.
        :param work_item: The work item to cancel.
        """
        conn = None
        try:
            conn = self._get_connection()
            self.logger.info(f"Cancelling job: {job_id}")
            self._execute_sql_gracefully(
                cancel_jobs(DEFAULT_SCHEMA, work_item.name, [job_id]), connection=conn
            )
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error handling job event: {error}")
        finally:
            self._close_connection(conn)

    async def resume_job(self, job_id: str) -> None:
        """
        Resume a job by its ID.
        :param job_id:
        """
        name = "extract"  # TODO this is a placeholder
        conn = None
        try:
            self.logger.info(f"Resuming job: {job_id}")
            conn = self._get_connection()
            self._execute_sql_gracefully(
                resume_jobs(DEFAULT_SCHEMA, name, [job_id]), connection=conn
            )
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error handling job event: {error}")
        finally:
            self._close_connection(conn)

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

        def db_call():
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

            conn = None
            try:
                conn = self._get_connection()
                self._execute_sql_gracefully(update_query, connection=conn)
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error handling job event: {error}")
            finally:
                self._close_connection(conn)

        await self._loop.run_in_executor(self._db_executor, db_call)

    async def maintenance(self):
        """
        Performs the maintenance process, including expiring, archiving, and purging.

        :return: None
        """
        try:
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

    def _count_states_generic(
        self, query_func, schema: str = None
    ) -> Dict[str, Dict[str, int]]:
        if schema is None:
            schema = DEFAULT_SCHEMA

        state_count_default = {key.lower(): 0 for key in WorkState.__members__.keys()}
        counts = []
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            cursor = self._execute_sql_gracefully(
                query_func(schema), return_cursor=True, connection=conn
            )
            counts = cursor.fetchall()
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error handling state count: {error}")
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

        states = {"queues": {}}
        for item in counts:
            name, state, size = item
            if name:
                if name not in states["queues"]:
                    states["queues"][name] = state_count_default.copy()
                queue = states["queues"][name]
                queue[state or "all"] = int(size)

        # Calculate the 'all' column as the sum of all state columns for each queue
        for queue in states["queues"].values():
            # Exclude the 'all' key itself from the sum
            queue["all"] = sum(v for k, v in queue.items() if k != "all")

        return states

    def count_job_states(self) -> Dict[str, Dict[str, int]]:
        """
        Fetch and count job states from the database.
        """
        return self._count_states_generic(count_job_states)

    def count_dag_states(self) -> Dict[str, Dict[str, int]]:
        """
        Fetch and count dag states from the database.
        """
        return self._count_states_generic(count_dag_states)

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
                conn = None
                cursor = None

                try:
                    conn = self._get_connection()
                    cursor = self._execute_sql_gracefully(
                        try_set_monitor_time(
                            DEFAULT_SCHEMA,
                            monitor_state_interval_seconds=int(MONITORING_POLL_PERIOD),
                        ),
                        return_cursor=True,
                        connection=conn,
                    )
                    monitored_on = cursor.fetchone()
                except (Exception, psycopg2.Error) as error:
                    self.logger.error(f"Error handling job event: {error}")
                finally:
                    self._close_cursor(cursor)
                    self._close_connection(conn)

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

        def db_call():
            def complete_jobs_wrapper(
                schema: str, name: str, ids: list, output: dict, _force: bool
            ):
                if _force:
                    return complete_jobs_by_id(schema, name, ids, output)
                else:
                    return complete_jobs(schema, name, ids, output)

            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    complete_jobs_wrapper(
                        DEFAULT_SCHEMA,
                        work_item.name,
                        [job_id],
                        {"on_complete": "done", **(output_metadata or {})},
                        force,
                    ),
                    return_cursor=True,
                    connection=conn,
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
                self._close_connection(conn)

        await self._loop.run_in_executor(self._db_executor, db_call)

    async def fail(
        self, job_id: str, work_item: WorkInfo, output_metadata: dict = None
    ):
        self.logger.info(f"Job failed : {job_id}")

        def db_call():
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    fail_jobs_by_id(
                        DEFAULT_SCHEMA,
                        work_item.name,
                        [job_id],
                        {"on_complete": "failed", **(output_metadata or {})},
                    ),
                    return_cursor=True,
                    connection=conn,
                )
                counts = cursor.fetchone()[0]
                if counts > 0:
                    self.logger.info(f"Completed failed job: {job_id}")
                else:
                    self.logger.error(f"Error completing failed job: {job_id}")
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error completing failed job: {error}")
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        await self._loop.run_in_executor(self._db_executor, db_call)

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
        await self._loop.run_in_executor(self._db_executor, self._blocking_sync_dag)

    def _blocking_sync_dag(self, interval: int = 30) -> None:
        """
        Validate that DAGs in memory still exist and are active in database
        """
        self.logger.info(f"Starting DAG sync polling (interval: {interval}s)")

        while self.running:
            cursor = None
            conn = None

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

                conn = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    query, memory_dag_ids, return_cursor=True, connection=conn
                )
                if not cursor:
                    self.logger.warning("No result from DAG validation query")
                    self._close_connection(conn)
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
                self._close_connection(conn)
            time.sleep(interval)
        self.logger.debug(f"DAG sync polling stopped")

    def _remove_dag_from_memory(self, dag_id: str, reason: str):
        """Centralized method to remove DAG from memory with logging"""
        if dag_id in self.active_dags:
            del self.active_dags[dag_id]
            self.logger.warning(f"Removed DAG {dag_id} from active_dags ({reason})")
        else:
            self.logger.debug(f"DAG {dag_id} not in active_dags ({reason})")

    async def notify_event(self) -> bool:
        if self._debounced_notify:
            return False
        self._debounced_notify = True
        try:
            self._event_queue.put_nowait("wake")
        except asyncio.QueueFull:
            pass
        return True

    def _resolve_dag_status_sync(self, dag_id: str) -> Optional[str]:
        """Synchronous helper to resolve DAG status in the database."""
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            resolve_query = (
                f"SELECT {DEFAULT_SCHEMA}.resolve_dag_state('{dag_id}'::uuid);"
            )
            cursor = self._execute_sql_gracefully(
                resolve_query, return_cursor=True, connection=conn
            )
            dag_state = result[0] if cursor and (result := cursor.fetchone()) else None
            return dag_state
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

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
        self.logger.debug(f"🔍 Resolving DAG status: {work_info.dag_id}")

        try:
            dag_state = await self._loop.run_in_executor(
                self._db_executor, self._resolve_dag_status_sync, work_info.dag_id
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

    async def get_dag_by_id(self, dag_id: str) -> QueryPlan | None:
        """
        Retrieves a DAG by its ID, using in-memory cache if available.
        Falls back to loading from db if missing.
        """
        # Return from cache if present
        if dag_id in self.active_dags:
            return self.active_dags[dag_id]

        def db_call() -> QueryPlan | None:
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    load_dag(DEFAULT_SCHEMA, dag_id),
                    return_cursor=True,
                    connection=conn,
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
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

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
                "message": f"Successfully recheduling summary for this cyset active DAGs, cleared {cleared_count} DAGs",
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


def query_plan_work_items(work_info: WorkInfo) -> tuple[QueryPlan, list[WorkInfo]]:
    """
    Generates a query plan and associated work items by processing the provided
    work information object. The method builds a directed acyclic graph (DAG) of
    work items and computes hierarchical job levels for its tasks.

    :param work_info: An object encapsulating details about the work items,
                      including metadata and configuration needed to plan the
                      execution.

    :return: A tuple consisting of:
             - A `QueryPlan` object representing the plan generated by the
               query planner.
             - A list of `WorkInfo` objects, each representing an individual
               task/node in the execution plan's DAG.
    """
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
            if query_planner_name not in PostgreSQLJobScheduler._mapper_warnings_shown:
                logger.warning(
                    f"No mapper configuration found for {query_planner_name}, "
                    "using default metadata."
                )
                PostgreSQLJobScheduler._mapper_warnings_shown.add(query_planner_name)

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
