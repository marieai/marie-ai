import asyncio
import math
import socket
import time
import traceback
import uuid
import uuid as _uuid
from asyncio import Queue
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import psycopg2

from marie.constants import (
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
    QueryPlan,
)
from marie.query_planner.builtin import register_all_known_planners
from marie.query_planner.model import QueryPlannersConf
from marie.scheduler.dag_topology_cache import DagTopologyCache
from marie.scheduler.fixtures import *
from marie.scheduler.global_execution_planner import GlobalPriorityExecutionPlanner
from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.job_scheduler import JobScheduler, JobSubmissionRequest
from marie.scheduler.memory_frontier import MemoryFrontier
from marie.scheduler.models import ExistingWorkPolicy, HeartbeatConfig, WorkInfo
from marie.scheduler.planner_util import (
    _is_noop_query_definition,
    debug_candidates_and_plan,
    get_node_from_dag,
    query_plan_work_items,
)
from marie.scheduler.plans import (
    cancel_jobs,
    complete_jobs,
    complete_jobs_by_id,
    create_queue,
    fail_jobs_by_id,
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
from marie.scheduler.repository import SchedulerRepository
from marie.scheduler.scheduler_heartbeat import SchedulerHeartbeat
from marie.scheduler.state import WorkState
from marie.scheduler.util import (
    adjust_backoff,
    available_slots_by_executor,
    convert_job_status_to_work_state,
)
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


# FIXME : Today we are tracking at the executor level, however that might not be the best
# approach. We might want to track at the deployment level (endpoint level) instead.
# this will allow us to track the status of the deployment and not just the executor.


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

        self.active_dags = {}
        self.job_manager = job_manager
        self._loop = get_or_reuse_loop()
        self._setup_event_subscriptions()
        self._setup_storage(config, connection_only=True)
        self._db = SchedulerRepository(config)

        self.execution_planner = GlobalPriorityExecutionPlanner()
        register_all_known_planners(
            QueryPlannersConf.from_dict(config.get("query_planners", {}))
        )

        dag_config = config.get("dag_manager", {})
        dag_strategy = dag_config.get("strategy", 'fixed')  # fixed or dynamic
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
        self.frontier = MemoryFrontier()

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
            work_item: WorkInfo = await self.get_job(job_id)

            if work_item is None:
                self.logger.error(f"WorkItem not found: {job_id}")
                return

            now = datetime.now()
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
            "lease/release_expired_leases.sql",  # reaper to run periodically
            "lease/release_lease.sql",
            "lease/activate_from_lease.sql",
            "lease/clear_all_leases.sql",
            "lease/hydrate_frontier.sql",
            "lease/hydrate_frontier_jobs.sql",
            "lease/lease_jobs_by_id.sql",
            "lease/reap_expired_leases.sql",
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

        # We need to display the status
        await self.hydrate_from_db()

        self.running = True
        # self.sync_task = asyncio.create_task(self._sync())
        # self.monitoring_task = asyncio.create_task(self._monitor())
        self.monitoring_task = None

        # self._heartbeat_task = asyncio.create_task(
        #     self._heartbeat_loop(self.heartbeat_config)
        # )

        await self.heartbeat.start()
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

                # current slots (will also be used to enforce per-executor caps DURING dispatch)
                slots_by_executor = available_slots_by_executor(
                    ClusterState.deployments
                ).copy()

                def slot_filter(wi: WorkInfo) -> bool:
                    ep = wi.data.get("metadata", {}).get("on", "")
                    exe = ep.split("://", 1)[0] if "://" in ep else ep
                    # allow NOOP/unknown
                    return (
                        True
                        if not exe or exe == "noop"
                        else (slots_by_executor.get(exe, 0) > 0)
                    )

                if not any(slots_by_executor.values()):
                    self.logger.debug("No available executor slots. Backing off.")
                    idle_streak += 1
                    wait_time = adjust_backoff(
                        wait_time,
                        idle_streak,
                        scheduled=False,
                        min_poll_period=MIN_POLL_PERIOD,
                    )
                    continue

                # FETCH READY CANDIDATES (executor-agnostic)
                # frontier should not filter by executors; let planner decide
                candidates_wi: list[WorkInfo] = await self.frontier.peek_ready(
                    batch_size,  # filter_fn=slot_filter
                )

                if not candidates_wi or len(candidates_wi) == 0:
                    self.logger.debug("No ready work in memory; short sleep.")
                    await asyncio.sleep(SHORT_POLL_INTERVAL)
                    idle_streak += 1
                    wait_time = adjust_backoff(
                        wait_time,
                        idle_streak,
                        scheduled=False,
                        min_poll_period=MIN_POLL_PERIOD,
                    )
                    continue

                # Build (entrypoint, wi) tuples for planner input
                planner_candidates: list[tuple[str, WorkInfo]] = []
                for wi in candidates_wi:
                    ep = wi.data.get("metadata", {}).get("on", "")
                    if not ep:
                        self.logger.error(f"Job without entrypoint 'on': {wi.id}")
                        continue
                    planner_candidates.append((ep, wi))

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
                    self.logger.warning(
                        "Planner returned no picks; short sleep. planner_candidates = {len(planner_candidates)}"
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

                # TAKE + SOFT-LEASE
                selected_ids = [wi.id for _, wi in planned]
                selected_wis: List[WorkInfo] = await self.frontier.take(selected_ids)

                taken = len(selected_wis)
                requested = len(selected_ids)
                if taken != requested:
                    taken_ids = {wi.id for wi in selected_wis}
                    missing = list(set(selected_ids) - taken_ids)
                    self.logger.warning(
                        f"Not all jobs taken from frontier: taken={taken} / selected={requested}; "
                        f"missing_ids={missing[:10]}{'...' if len(missing) > 10 else ''}"
                    )

                planned_by_id = {wi.id: (ep, wi) for ep, wi in planned}
                ids_by_job_name: dict[str, list[str]] = defaultdict(list)

                for wi in selected_wis:
                    ids_by_job_name[wi.name].append(wi.id)

                leased_ids: set[str] = set()
                for job_name, ids in ids_by_job_name.items():
                    try:
                        self.logger.debug(
                            f'_lease_jobs_db size = {job_name} : {len(ids)}'
                        )
                        got = await self._lease_jobs_db(job_name, ids)
                        leased_ids.update(got)
                        self.logger.debug(
                            f'_lease_jobs_db got = {job_name} : {len(got)}'
                        )
                        if len(got) < len(ids):
                            self.logger.warning(
                                f"DB lease shortfall for '{job_name}': got {len(got)}/{len(ids)}"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"DB lease failed for '{job_name}' ({len(ids)} ids): {e}"
                        )
                # put *everything* back
                if not leased_ids:
                    for wi in selected_wis:
                        await self.frontier.release_lease_local(wi.id)
                    self.logger.warning(
                        "No candidates could be leased in DB; backing off."
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

                for entrypoint, wi in leased_jobs:
                    dag_id = wi.dag_id
                    if (
                        dag_id not in self.active_dags
                        and len(self.active_dags) >= max_concurrent_dags
                    ):
                        self.logger.debug(
                            f"Max DAG limit reached ({len(self.active_dags)}/{max_concurrent_dags}). Skipping {wi.id}"
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

                    # NOOP vs normal
                    node = get_node_from_dag(wi.id, self.active_dags[dag_id])
                    if _is_noop_query_definition(node):
                        # Complete locally while we still hold the DB lease
                        try:
                            await self.complete(wi.id, wi, {}, force=True)

                            # Clear local soft-lease (don't requeue a completed job)
                            self.frontier.leased_until.pop(wi.id, None)
                            await self.frontier.on_job_completed(wi.id)
                            scheduled_any = True

                            sorted_nodes, job_levels = (
                                self._topology_cache.get_sorted_nodes_and_levels(
                                    self.active_dags[dag_id], dag_id
                                )
                            )
                            if job_levels.get(wi.id, -1) == max(job_levels.values()):
                                await self.resolve_dag_status(wi.id, wi)

                        except Exception as e:
                            self.logger.error(f"Completing NOOP {wi.id} failed: {e}")
                            await self._release_lease_db([wi.id])
                            await self.frontier.release_lease_local(wi.id)
                        continue

                    # Normal job: check slots then dispatch
                    exe = entrypoint.split("://", 1)[0]
                    if slots_by_executor.get(exe, 0) <= 0:
                        self.logger.debug(f"No slots for {exe}, delaying job {wi.id}")
                        await self._release_lease_db([wi.id])
                        await self.frontier.release_lease_local(wi.id)
                        continue

                    # reserve a slot and dispatch
                    slots_by_executor[exe] -= 1
                    enqueue_tasks.append(
                        {
                            "task": asyncio.create_task(
                                self._activate_and_enqueue_job(wi)
                            ),
                            "wi": wi,
                            "exe": exe,
                        }
                    )

                if enqueue_tasks:
                    results = await asyncio.gather(
                        *[t["task"] for t in enqueue_tasks], return_exceptions=True
                    )
                    for i, result in enumerate(results):
                        wi = enqueue_tasks[i]["wi"]
                        exe = enqueue_tasks[i]["exe"]

                        if isinstance(result, Exception) or not result:
                            # dispatch failed → release lease & requeue
                            self.logger.error(f"Dispatch failed for {wi.id}: {result}")
                            await self._release_lease_db([wi.id])
                            await self.frontier.release_lease_local(wi.id)
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
            debug_data["job_state_counts"] = self._db.count_job_states()
        except Exception as e:
            debug_data["job_state_counts_error"] = str(e)

        try:
            debug_data["dag_state_counts"] = self._db.count_dag_states()
        except Exception as e:
            debug_data["dag_state_counts_error"] = str(e)

        return debug_data

    async def enqueue(self, work_info: WorkInfo) -> bool:
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
        # Fast path, no lock
        if job_id in self._job_cache:
            # Move to end to signify it's recently used
            self._job_cache[job_id] = self._job_cache.pop(job_id)
            return self._job_cache[job_id]

        def db_call():
            """Synchronous database call to fetch the job."""
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
                          id, name, priority, state, retry_limit, start_after,
                          expire_in, data, retry_delay, retry_backoff, keep_until,
                          dag_id, job_level
                    FROM {schema}.{table}
                    WHERE id = %s
                    """,
                    (job_id,),
                )
                record = cursor.fetchone()
                conn.commit()
                if record:
                    return self.record_to_work_info(record)
                return None
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error getting job '{job_id}': {error}")
                if conn:
                    conn.rollback()
                return None
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        work_item = await self._loop.run_in_executor(self._db_executor, db_call)
        # Evict oldest if cache is over size
        if work_item:
            self._job_cache[job_id] = work_item
            if len(self._job_cache) > self._job_cache_max_size:
                self._job_cache.pop(next(iter(self._job_cache)))

        return work_item

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

    def _create_dag_and_jobs_sync(
        self,
        work_info: WorkInfo,
        submission_id: str,
        plan: QueryPlan,
        dag_nodes: list[WorkInfo],
    ) -> Tuple[bool, str]:
        """Synchronous method for blocking database operations
        It is important to run this in a thread to prevent blocking the main event loop.

        :param work_info: WorkInfo object containing job details
        :param submission_id: Unique identifier for the job submission
        :param plan: The query plan representing the DAG structure
        :param dag_nodes: List of WorkInfo objects representing individual jobs in the DAG
        :return: Tuple indicating if a new key was added and the new DAG key
        """
        dag_id = submission_id
        dag_name = f"{dag_id}_dag"
        work_info.dag_id = dag_id
        connection = None
        cursor = None
        new_key_added = False
        try:
            json_serialized_dag = plan.model_dump()
            connection = self._get_connection()
            cursor = self._execute_sql_gracefully(
                insert_dag(DEFAULT_SCHEMA, dag_id, dag_name, json_serialized_dag),
                connection=connection,
                return_cursor=True,
            )
            new_dag_key = (
                result[0] if cursor and (result := cursor.fetchone()) else None
            )
            self._close_cursor(cursor)

            for i, dag_work_info in enumerate(dag_nodes):
                cursor = self._execute_sql_gracefully(
                    insert_job(DEFAULT_SCHEMA, dag_work_info),
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
            if connection:
                connection.rollback()
            raise ValueError(f"Job creation failed for {submission_id}: {error}")
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

        # Build plan & nodes once (used by DB and memory)
        plan, dag_nodes = query_plan_work_items(work_info)

        # until we upgrade to pycog3 and convert to async we have to run in a thread to prevent blocking of the loop
        def _sync():
            for dag_work_info in dag_nodes:
                dag_work_info.dag_id = submission_id

            return self._create_dag_and_jobs_sync(
                work_info, submission_id, plan, dag_nodes
            )

        new_key_added, new_dag_key = await self._loop.run_in_executor(
            self._db_executor, _sync
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
        self.logger.debug(f"Marking as active : {work_info.id}")
        return await self._mark_active(
            "job", mark_as_active_jobs(DEFAULT_SCHEMA, work_info.name, [work_info.id])
        )

    async def mark_as_active_dag(self, work_info: WorkInfo) -> bool:
        return await self._mark_active(
            "dag", mark_as_active_dags(DEFAULT_SCHEMA, [work_info.dag_id])
        )

    async def _mark_active(self, label: str, query: str) -> bool:
        self.logger.debug(f"Marking as active : {label} ")

        def db_call() -> bool:
            cursor = None
            conn = None
            try:
                conn = self._get_connection()
                cursor = self._execute_sql_gracefully(
                    query, return_cursor=True, connection=conn
                )
                val = result[0] if cursor and (result := cursor.fetchone()) else False
                conn.commit()
                return val
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error updating {label}: {error}")
                return False
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        # There is no need to lock here as we are the entrypoint
        # async with self._status_update_lock:
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
        self.logger.info(f"Cancelling job: {job_id}")

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                self._execute_sql_gracefully(
                    cancel_jobs(DEFAULT_SCHEMA, work_item.name, [job_id]),
                    connection=conn,
                )
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error cancelling job: {error}")
            finally:
                self._close_connection(conn)

        async with self._status_update_lock[job_id]:
            await self._loop.run_in_executor(self._db_executor, db_call)

    async def resume_job(self, job_id: str) -> None:
        """
        Resume a job by its ID.
        :param job_id:
        """
        name = "extract"  # TODO this is a placeholder
        self.logger.info(f"Resuming job: {job_id}")

        def db_call():
            conn = None
            try:
                conn = self._get_connection()
                self._execute_sql_gracefully(
                    resume_jobs(DEFAULT_SCHEMA, name, [job_id]), connection=conn
                )
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error resuming job: {error}")
            finally:
                self._close_connection(conn)

        await self._loop.run_in_executor(self._db_executor, db_call)

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

        async with self._status_update_lock[job_id]:
            await self._loop.run_in_executor(self._db_executor, db_call)
            # self._job_cache.pop(job_id, None)

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
        """Expire jobs with expired leases."""
        self.logger.debug("Checking for expired job leases")

        def db_call():
            """Sync DB call to release expired leases."""
            conn = None
            released_count = 0
            try:
                conn = self._get_connection()
                query = "SELECT marie_scheduler.release_expired_leases()"
                result = self._execute_sql_gracefully(query, connection=conn)

                if result and isinstance(result, list) and len(result) > 0:
                    count = result[0][0]
                    if count:
                        released_count = count
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Failed to expire jobs: {error}", exc_info=True)
            finally:
                self._close_connection(conn)
            return released_count

        released_count = await self._loop.run_in_executor(self._db_executor, db_call)
        if released_count > 0:
            self.logger.info(f"Released expired job leases : {released_count}")
            await self.notify_event()

    async def archive(self):
        self.logger.info("Archiving jobs")

    async def purge(self):
        self.logger.info("Purging jobs")

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
                if False:
                    sql_probe = """
                                SELECT id, name, state, lease_owner, completed_on
                                FROM marie_scheduler.job
                                WHERE id = ANY (%s::uuid[]);
                                """
                    with conn, conn.cursor() as cur:
                        ids = [job_id]
                        cur.execute(sql_probe, (ids,))
                        self.logger.info(f"pre-complete probe rows={cur.fetchall()}")

                query = complete_jobs_wrapper(
                    DEFAULT_SCHEMA,
                    work_item.name,
                    [job_id],
                    {"on_complete": "done", **(output_metadata or {})},
                    force,
                )
                cursor = self._execute_sql_gracefully(
                    query,
                    return_cursor=True,
                    connection=conn,
                )
                counts = cursor.fetchone()[0]
                if counts > 0:
                    self.logger.debug(f"Completed job: {job_id} : {counts}")
                else:
                    self.logger.error(
                        f"Error completing job: {job_id} : {counts} \n {query}"
                    )
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error completing job: {error}")
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        async with self._status_update_lock[job_id]:
            await self._loop.run_in_executor(self._db_executor, db_call)
            # self._job_cache.pop(job_id, None) # invalidate cache

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

        async with self._status_update_lock[job_id]:
            await self._loop.run_in_executor(self._db_executor, db_call)
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

            def _sync(dag_id: str) -> Optional[str]:
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
                    dag_state = (
                        result[0] if cursor and (result := cursor.fetchone()) else None
                    )
                    return dag_state
                finally:
                    self._close_cursor(cursor)
                    self._close_connection(conn)

            dag_state = await self._loop.run_in_executor(
                self._db_executor, _sync, work_info.dag_id
            )

            self.logger.debug(f"Resolved DAG state: {dag_state}")
            if dag_state not in ("completed", "failed"):
                self.logger.debug(f"DAG is still in progress: {work_info.dag_id}")
                return False

            if work_info.dag_id in self.active_dags:
                del self.active_dags[work_info.dag_id]
                self.logger.debug(
                    f"Removed DAG from cache: {work_info.dag_id}, size = {len(self.active_dags)}"
                )

            self.logger.debug(
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
                return False

            status = "OK" if dag_state == "completed" else "FAILED"

            if status == "OK":
                fs = await self.frontier.finalize_dag(work_info.dag_id)

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
                    self.logger.debug(f"Loaded DAG from DB: {dag_id}")
                    return dag
                else:
                    self.logger.warning(f"DAG not found in DB: {dag_id}")
                    return None
            except Exception as e:
                self.logger.error(f"Error loading DAG {dag_id}: {e}")
                return None
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    def get_available_slots(self) -> dict[str, int]:
        return available_slots_by_executor(ClusterState.deployments)

    async def reset_active_dags(self):
        """
        Reset the active DAGs dictionary, clearing all currently tracked DAGs.
        This can be useful for debugging or when you need to force a fresh state.

        Returns:
            dict: Information about the reset operation including count of cleared DAGs
        """
        try:
            cleared_count = len(self.active_dags) if self.active_dags else 0
            cleared_dags = list(self.active_dags.keys()) if self.active_dags else []

            for dag_id in cleared_dags:
                await self.frontier.finalize_dag(dag_id)

            self.active_dags = {}

            self.logger.info(f"Reset active DAGs: cleared {cleared_count} DAGs")
            if cleared_dags:
                self.logger.debug(f"Cleared DAGs: {cleared_dags}")

            return {
                "success": True,
                "cleared_count": cleared_count,
                "cleared_dags": cleared_dags,
                "message": f"Successfully rescheduling summary for this cycle active DAGs, cleared {cleared_count} DAGs",
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

    async def _lease_jobs_db(self, job_name: str, ids: list[str]) -> set[str]:
        """
        Try to lease the given job ids for this scheduler instance in the DB.
        Returns the subset of ids that were successfully leased.
        """
        if not ids:
            return set()

        if not self.distributed_scheduler:
            return set(ids)

        norm_ids = list({str(i) for i in ids})

        def _lease_sync() -> set[str]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                sql = f"""
                    SELECT unnest({DEFAULT_SCHEMA}.lease_jobs_by_id(
                        %s::uuid[],          -- _ids
                        %s::interval,        -- _ttl
                        %s,                  -- _owner
                        %s                   -- _name
                    ))
                """
                ttl_interval = f"{int(self.lease_ttl_seconds)} seconds"
                params = (norm_ids, ttl_interval, self.lease_owner, job_name)
                cursor = self._execute_sql_gracefully(
                    sql, data=params, return_cursor=True, connection=conn
                )
                if not cursor:
                    return set()
                leased: set[str] = set()
                # tolerate either 1-column or 2-column return from the function
                for row in cursor.fetchall():
                    if len(row) >= 1 and row[0] is not None:
                        leased.add(str(row[0]))
                return leased
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _lease_sync)

    async def _activate_from_lease_db(self, ids: list[str]) -> set[str]:
        """
        Promote leased jobs to active in DB once dispatch is acknowledged.
        """
        if not ids:
            return set()

        if not self.distributed_scheduler:
            return set(ids)

        def _activate_sync() -> set[str]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                # Function signature: activate_from_lease(_ids uuid[], _run_owner text, _run_ttl interval)
                sql = (
                    f"SELECT unnest({DEFAULT_SCHEMA}.activate_from_lease("
                    f"%s::uuid[], %s, %s::interval)) AS id"
                )
                params = (
                    ids,  # -> %s::uuid[]
                    self.lease_owner,  # -> _run_owner text
                    f"{self.run_ttl_seconds} seconds",  # -> _run_ttl interval
                )
                cursor = self._execute_sql_gracefully(
                    sql, data=params, return_cursor=True, connection=conn
                )
                rows = cursor.fetchall() if cursor else []
                return {row[0] for row in rows}
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _activate_sync)

    async def _release_lease_db(self, ids: list[str]) -> set[str]:
        """
        Release DB leases for the given job ids if dispatch fails or needs retry.
        """
        if not ids:
            return set()

        if not self.distributed_scheduler:
            return set(ids)

        def _release_sync() -> set[str]:
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                # Cast to uuid[] and UNNEST the returned uuid[]
                sql = f"SELECT unnest({DEFAULT_SCHEMA}.release_lease(%s::uuid[])) AS id"
                params = (ids,)
                cursor = self._execute_sql_gracefully(
                    sql, data=params, return_cursor=True, connection=conn
                )
                rows = cursor.fetchall() if cursor else []
                return {str(row[0]) for row in rows}
            finally:
                self._close_cursor(cursor)
                self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, _release_sync)

    async def hydrate_from_db(
        self,
        dag_batch_size: int = 1000,
        itersize: int = 5000,
        log_every_seconds: float = 2.0,
    ) -> None:
        """
        Rebuild MemoryFrontier from DB in two phases with progress & timing logs:
          1) Stream DAGs that still have unfinished work (created/retry).
          2) In batches of DAG IDs, stream their unfinished jobs with already-filtered deps.
          3) Add once per DAG: self.frontier.add_dag(dag, nodes)
        """

        def _stream_dags():
            conn = self._get_connection()
            cur = None
            try:
                cur = conn.cursor(name="hydrate_frontier_dags")
                cur.itersize = itersize
                cur.execute(
                    "SELECT dag_id, serialized_dag FROM marie_scheduler.hydrate_frontier_dags()"
                )
                for dag_id, serialized_dag in cur:
                    yield dag_id, serialized_dag
                if cur and not cur.closed:
                    self._close_cursor(cur)
                cur = None  # so we don't try to close again in finally
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                if cur and not cur.closed:
                    self._close_cursor(cur)
                self._close_connection(conn)

        t0 = time.monotonic()
        self.logger.info("Hydrate: phase 1 (DAG discovery) started…")

        dag_rows = await self._loop.run_in_executor(
            self._db_executor, lambda: list(_stream_dags())
        )
        discover_elapsed = time.monotonic() - t0
        self.logger.info(
            f"Hydrate: phase 1 complete — discovered {len(dag_rows)} DAG(s) in {discover_elapsed:.2f}s "
            f"({(len(dag_rows) / discover_elapsed if discover_elapsed > 0 else 0):.1f} DAGs/sec)."
        )

        # Build map of dag_id -> QueryPlan
        dags: dict[str, QueryPlan] = {}
        dag_ids_ordered: list[str] = []
        parse_skipped = 0
        for dag_id, dag_def in dag_rows:
            if not dag_def:
                parse_skipped += 1
                self.logger.warning(
                    f"Hydrate: DAG {dag_id} has no serialized_dag; skipping."
                )
                continue
            try:
                dags[str(dag_id)] = QueryPlan(**dag_def)
                dag_ids_ordered.append(str(dag_id))
            except Exception as e:
                parse_skipped += 1
                self.logger.error(f"Hydrate: unable to parse DAG {dag_id}: {e}")

        if not dags:
            total_elapsed = time.monotonic() - t0
            self.logger.info(
                f"Hydrate: no DAGs to hydrate (skipped {parse_skipped}). Done in {total_elapsed:.2f}s."
            )
            await self.notify_event()
            return

        self.logger.info(
            f"Hydrate: {len(dags)} DAG(s) ready for job loading "
            f"(skipped {parse_skipped}, total discovered {len(dag_rows)})."
        )

        def _stream_jobs_for_batch(dag_ids_batch):
            conn = self._get_connection()
            cur = None
            try:
                dag_ids_text = [
                    str(x) for x in dag_ids_batch
                ]  # ensure strings so psycopg2 adapts them cleanly when casting to uuid[]
                cur = conn.cursor(name="hydrate_frontier_jobs")
                cur.itersize = itersize
                cur.execute(
                    "SELECT dag_id, job FROM marie_scheduler.hydrate_frontier_jobs((%s)::uuid[])",
                    (dag_ids_text,),
                )
                for row in cur:
                    yield row
                if cur and not cur.closed:
                    self._close_cursor(cur)
                cur = None
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                if cur and not cur.closed:
                    self._close_cursor(cur)
                self._close_connection(conn)

        def _chunks(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i : i + n]

        self.logger.info(
            f"Hydrate: phase 2 (job loading) — {len(dag_ids_ordered)} DAG(s), "
            f"batch size {dag_batch_size}, cursor itersize {itersize}."
        )

        buckets: dict[str, list[WorkInfo]] = defaultdict(list)

        # Progress counters
        total_dags = len(dag_ids_ordered)
        processed_dags = 0
        processed_jobs = 0
        last_log_t = time.monotonic()
        phase2_start = last_log_t

        # For batch-level logging
        batch_idx = 0
        for batch in _chunks(dag_ids_ordered, dag_batch_size):
            batch_idx += 1
            b_start = time.monotonic()

            rows = await self._loop.run_in_executor(
                self._db_executor, lambda: list(_stream_jobs_for_batch(batch))
            )

            for dag_id, j in rows:
                dag_id = str(dag_id)
                if dag_id not in dags:
                    continue
                try:
                    state_raw = j.get("state")
                    wi = WorkInfo(
                        id=str(j["id"]),
                        name=j["name"],
                        priority=j["priority"],
                        state=WorkState(state_raw) if state_raw else None,
                        retry_limit=j["retry_limit"],
                        start_after=j["start_after"],
                        expire_in_seconds=0,  # map INTERVAL to seconds
                        data=j["data"],
                        retry_delay=j["retry_delay"],
                        retry_backoff=j["retry_backoff"],
                        keep_until=j["keep_until"],
                        dag_id=dag_id,
                        job_level=j["job_level"],
                    )
                    deps = j.get("dependencies") or []
                    wi.dependencies = [str(d) for d in deps]
                    buckets[dag_id].append(wi)
                    processed_jobs += 1
                except Exception as e:
                    self.logger.error(
                        f"Hydrate: failed to build WorkInfo for DAG {dag_id}: {e}"
                    )

            processed_dags += len(batch)
            b_elapsed = time.monotonic() - b_start
            jobs_in_batch = len(rows)
            job_rate = jobs_in_batch / b_elapsed if b_elapsed > 0 else 0.0

            now = time.monotonic()
            if (now - last_log_t) >= log_every_seconds:
                overall_elapsed = now - phase2_start
                overall_rate = (
                    processed_jobs / overall_elapsed if overall_elapsed > 0 else 0.0
                )
                pct = (processed_dags / total_dags) * 100.0
                # ETA by DAG batches (rough) — avoids needing total jobs in advance
                remaining_dags = max(0, total_dags - processed_dags)
                dags_per_sec = (len(batch) / b_elapsed) if b_elapsed > 0 else 0.0
                eta_s = (
                    (remaining_dags / dags_per_sec) if dags_per_sec > 0 else math.inf
                )

                self.logger.info(
                    f"Hydrate: batch {batch_idx} — "
                    f"{len(batch)} DAG(s) in {b_elapsed:.2f}s, {jobs_in_batch} jobs "
                    f"({job_rate:.1f} jobs/s). "
                    f"Progress: {processed_dags}/{total_dags} DAGs ({pct:.1f}%), "
                    f"{processed_jobs} jobs total, "
                    f"overall {overall_rate:.1f} jobs/s, "
                    f"ETA ~ {eta_s:.1f}s."
                )
                last_log_t = now

        add_start = time.monotonic()
        added = 0
        total_pending_jobs = 0
        for dag_id, nodes in buckets.items():
            if not nodes:
                continue
            try:
                await self.frontier.add_dag(dags[dag_id], nodes)
                added += 1
                total_pending_jobs += len(nodes)
            except Exception as e:
                self.logger.error(f"Hydrate: frontier.add_dag failed for {dag_id}: {e}")
        add_elapsed = time.monotonic() - add_start

        total_elapsed = time.monotonic() - t0
        self.logger.info(
            f"Hydrate: complete — {added} DAG(s) added to frontier, "
            f"{total_pending_jobs} pending jobs. "
            f"Add phase {add_elapsed:.2f}s, total elapsed {total_elapsed:.2f}s."
        )

        snap = self.frontier.summary(detail=True, top_n=3)
        self.logger.info(
            "Frontier: dags=%s jobs=%s ready=%s leased=%s age(p50/p90/max)=%.1f/%.1f/%.1f",
            snap["totals"]["dags"],
            snap["totals"]["jobs"],
            snap["totals"]["ready"],
            snap["totals"]["leased"],
            snap["ready_age_seconds"]["p50"],
            snap["ready_age_seconds"]["p90"],
            snap["ready_age_seconds"]["max"],
        )

        await self.notify_event()

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
