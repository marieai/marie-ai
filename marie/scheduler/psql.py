import asyncio
import contextlib
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pprint import pprint
from typing import Any, Dict, Optional

import psycopg2

from marie.excepts import BadConfigSource
from marie.helper import get_or_reuse_loop
from marie.job.common import JobStatus
from marie.job.job_manager import JobManager
from marie.job.partition.job_partitioner import MarieJobPartitioner
from marie.job.pydantic_models import JobPartition
from marie.logging_core.logger import MarieLogger
from marie.logging_core.predefined import default_logger as logger
from marie.query_planner.planner import (
    plan_to_yaml,
    print_sorted_nodes,
    query_planner,
    topological_sort,
    visualize_query_plan_graph,
)
from marie.scheduler.fixtures import *
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
    insert_job,
    insert_version,
    mark_as_active_jobs,
    resume_jobs,
    to_timestamp_with_tz,
    try_set_monitor_time,
    version_table_exists,
)
from marie.scheduler.state import WorkState
from marie.scheduler.util import available_slots_by_executor, has_available_slot
from marie.serve.runtimes.servers.cluster_state import ClusterState
from marie.storage.database.postgres import PostgresqlMixin

INIT_POLL_PERIOD = 0.250  # 250ms
MAX_POLL_PERIOD = 16.0  # 16s

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


class PostgreSQLJobScheduler(PostgresqlMixin, JobScheduler):
    """A PostgreSQL-based job scheduler."""

    def __init__(self, config: Dict[str, Any], job_manager: JobManager):
        super().__init__()
        self.logger = MarieLogger(PostgreSQLJobScheduler.__name__)
        if job_manager is None:
            raise BadConfigSource("job_manager argument is required for JobScheduler")

        self._fetch_lock = asyncio.Lock()
        self._fetch_event = asyncio.Event()
        self._fetch_counter = 0

        self.known_queues = set(config.get("queue_names", []))
        self.running = False
        self.task = None
        self.sync_task = None
        self.monitoring_task = None

        self.aqueue: Optional[asyncio.Queue] = None

        lock_free = True
        self._lock = (
            contextlib.AsyncExitStack() if lock_free else asyncio.Lock()
        )  # Lock to prevent concurrent access to the database

        if self.known_queues is None or len(self.known_queues) == 0:
            raise BadConfigSource("Queue names are required for JobScheduler")

        self.logger.info(f"Queue names to monitor: {self.known_queues}")
        self.job_manager = job_manager
        self._loop = get_or_reuse_loop()
        self._setup_event_subscriptions()
        self._setup_storage(config, connection_only=True)

    async def handle_job_event(self, event_type: str, message: Any):
        """
        Handles a job event.

        :param event_type: The type of the event.
        :param message: The message associated with the event.
        """
        # print if the lock is acquired
        async with self._lock:
            self.logger.info(f"received message: {event_type} > {message}")
            job_id = message.get("job_id")
            status = JobStatus(event_type)
            work_item: WorkInfo = await self.get_job(job_id)

            if work_item is None:
                self.logger.error(f"WorkItem not found: {job_id}")
                return
            work_state = convert_job_status_to_work_state(status)

            if status == JobStatus.PENDING:
                self.logger.info(f"Job pending : {job_id}")
            elif status == JobStatus.SUCCEEDED:
                self.logger.info(f"Job succeeded : {job_id}")
                await self.complete(job_id, work_item)
            elif status == JobStatus.FAILED:
                self.logger.info(f"Job failed : {job_id}")
                await self.fail(job_id, work_item)
            elif status == JobStatus.RUNNING:
                self.logger.info(f"Job running : {job_id}")
                await self.put_status(job_id, work_state, datetime.now(), None)
            else:
                self.logger.error(f"Unhandled job status: {status}. Marking as FAILED.")
                await self.fail(job_id, work_item)  # Fail-safe

            if status.is_terminal():
                self.logger.info(f"Job {job_id} is in terminal state {status}")
                await self.notify_event()

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
                    self.connection.rollback()

    async def wipe(self) -> None:
        """Clears the schedule storage."""
        schema = DEFAULT_SCHEMA
        query = f"""
           TRUNCATE {schema}.job, {schema}.archive
           """
        with self:
            try:
                self._execute_sql_gracefully(query)
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error clearing tables: {error}")
                self.connection.rollback()

    async def is_installed(self) -> bool:
        """check if the tables are installed"""
        schema = DEFAULT_SCHEMA
        with self:
            try:
                cursor = self._execute_sql_gracefully(version_table_exists(schema))
                if cursor and cursor.rowcount > 0:
                    result = cursor.fetchone()
                    if result and result[0] is not None:
                        return True
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error clearing tables: {error}")
        return False

    async def create_queue(self, queue_name: str) -> None:
        """Setup the queue for the scheduler."""

        with self:
            try:
                self._execute_sql_gracefully(
                    create_queue(DEFAULT_SCHEMA, queue_name, {})
                )
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error setting up queue: {error}")
                self.connection.rollback()

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

        self.running = True
        # self.sync_task = asyncio.create_task(self._sync())
        self.task = asyncio.create_task(self._poll())
        # self.monitoring_task = asyncio.create_task(self._monitor())

    async def _poll(self):
        self.logger.info("Starting database job scheduler")
        wait_time = INIT_POLL_PERIOD
        batch_size = 10
        failures = 0
        last_deployments_timestamp = ClusterState.deployments_last_updated
        slots_by_entrypoint = available_slots_by_executor(ClusterState.deployments)

        while self.running:
            self.logger.info(
                f"Checking for new jobs, time = {wait_time}, events = {self._fetch_counter}, set = {self._fetch_event.is_set()}"
            )
            try:
                try:
                    await asyncio.wait_for(self._fetch_event.wait(), timeout=wait_time)
                    self._fetch_event.clear()
                    while self._fetch_counter > 0:
                        self._fetch_counter -= 1
                    self.logger.info(f"Processing job events: {self._fetch_counter}")
                    wait_time = INIT_POLL_PERIOD
                except asyncio.TimeoutError:
                    pass  # Proceed normally if timeout happens

                self.logger.info(f'----------------------------------------------')
                self.logger.info(f'----------------------------------------------')
                deployments_last_updated = ClusterState.deployments_last_updated
                current_deployments = ClusterState.deployments

                if deployments_last_updated != last_deployments_timestamp:
                    self.logger.info(
                        "Recalculated slots_by_entrypoint due to deployments change."
                    )
                    self.logger.info(
                        f"deployments_last_updated   : {deployments_last_updated}"
                    )
                    self.logger.info(
                        f"last_deployments_timestamp : {last_deployments_timestamp}"
                    )

                    last_deployments_timestamp = deployments_last_updated
                    slots_by_entrypoint = available_slots_by_executor(
                        current_deployments
                    )

                has_any_open_slots = any(
                    slots > 0 for slots in slots_by_entrypoint.values()
                )
                has_records = False
                self.logger.info(
                    f"Slots by entrypoint [{has_any_open_slots}]: {slots_by_entrypoint}"
                )

                if has_any_open_slots:
                    # TODO: Allow contronlling how many jobs to fetch per queue
                    records_by_queue = await self.get_work_items_by_queue(
                        limit=batch_size
                    )
                    records_by_entrypoint = defaultdict(list)

                    for queue, records in records_by_queue.items():
                        self.logger.info(f"Records for queue: {queue}")
                        for record in records:
                            work_info = self.record_to_work_info(record)
                            entrypoint = work_info.data.get("metadata", {}).get("on")
                            records_by_entrypoint[entrypoint].append(work_info)

                    for entrypoint, records in records_by_entrypoint.items():
                        print('------------')
                        print(
                            f'Work items for entrypoint: {entrypoint} : {len(records)}'
                        )

                        for work_info in records:
                            print(f'Work item : {work_info}')
                            has_records = True
                            submission_id = work_info.id
                            entrypoint = work_info.data.get("metadata", {}).get("on")
                            executor = entrypoint.split("://")[0]
                            # this can happen when the executor is not yet deployed or is being redeployed
                            if executor not in slots_by_entrypoint:
                                self.logger.warning(
                                    f"Executor `{executor}` not found in slots_by_entrypoint. Retrying job `{submission_id}` after short delay."
                                )
                                await asyncio.sleep(2)  # Retry after a short delay
                                continue

                            print('------------')
                            print(f'Work item : {work_info}')
                            print(
                                f'Work item entrypoint: {entrypoint} : slots : {slots_by_entrypoint[executor]}'
                            )
                            print(f'slots : {slots_by_entrypoint[executor]}')
                            if slots_by_entrypoint[executor] <= 0:
                                self.logger.info(
                                    f"No available slots for work, scheduling : {submission_id}, time = {wait_time}, on = {entrypoint}"
                                )
                                break
                            slots_by_entrypoint[executor] -= 1
                            print(
                                "ClusterState.deployments : ", ClusterState.deployments
                            )
                            print(
                                f'slots after decrement : {slots_by_entrypoint[executor]}'
                            )
                            # FIXME: WE NEED  MAKE SURE THAT THE CLUSTER STATE IS UPDATED BEFORE WE SCHEDULE THE JOB
                            await self.mark_as_active(work_info)
                            job_id = await self.enqueue(work_info)

                            if job_id is None:
                                self.logger.error(
                                    f"Error scheduling work item: {work_info.id}"
                                )
                            else:
                                self.logger.info(
                                    f"Work item scheduled with ID: {job_id}"
                                )

                            scheduler_timeout = 100
                            try:
                                await asyncio.wait_for(
                                    ClusterState.scheduled_event.wait(),
                                    timeout=scheduler_timeout,
                                )
                                ClusterState.scheduled_event.clear()
                                self.logger.info(
                                    f"Scheduled event received, resetting wait time."
                                )
                            except asyncio.TimeoutError:
                                pass  # Proceed normally if timeout happens
                wait_time = (
                    INIT_POLL_PERIOD
                    if has_records
                    else min(wait_time * 2, MAX_POLL_PERIOD)
                )
            except Exception as e:
                self.logger.error(f"Error fetching jobs: {e}")
                self.logger.error(traceback.format_exc())
                failures += 1
                if failures > 5:
                    self.logger.error(
                        f"Too many failures, entering cooldown (sleeping for 60s)"
                    )
                    await asyncio.sleep(60)
                    failures = 0

    async def stop(self, timeout: float = 2.0) -> None:
        self.logger.info("Stopping job scheduling agent")
        self.running = False

        if self.task is not None:
            try:
                await asyncio.wait_for(self.task, timeout)
            except asyncio.TimeoutError:
                self.logger.warning("Task did not complete in time and was cancelled")

        if self.monitoring_task is not None:
            try:
                await asyncio.wait_for(self.monitoring_task, timeout)
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Monitoring task did not complete in time and was cancelled"
                )

    def debug_info(self) -> str:
        print("Debugging info")

    async def enqueue(self, work_info: WorkInfo) -> str | None:
        """
        Enqueues a work item for processing on the next available executor.

        :param work_info: The information about the work item to be processed.
        :return: The ID of the work item if successfully enqueued, None otherwise.
        """
        self.logger.info(f"Enqueuing work item : {work_info.id}")
        submission_id = work_info.id
        entrypoint = work_info.data.get("metadata", {}).get("on")
        if not entrypoint:
            raise ValueError("The entrypoint 'on' is not defined in metadata")

        if False:
            if not has_available_slot(entrypoint, ClusterState.deployments):
                self.logger.info(
                    f"No available slots for work, scheduling : {submission_id}"
                )
                return None
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
                        # print({query})
                        # we can't use named cursors as it will throw an error
                        cursor = self.connection.cursor()
                        cursor.itersize = limit
                        cursor.execute(f"{query}")
                        records = [record for record in cursor]
                        if records:
                            records_by_queue[queue] = records
                            self.logger.info(
                                f"Fetched {len(records)} jobs from queue: {queue}"
                            )
                        else:
                            self.logger.info(f"No jobs found in queue: {queue}")

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
                          keep_until
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
                        keep_until
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
                    SELECT id,name, priority,state,retry_limit,start_after,expire_in,data,retry_delay,retry_backoff,keep_until
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

        return await self.__submit_job(work_info, overwrite)

    async def __submit_job(self, work_info: WorkInfo, overwrite: bool = True) -> str:
        """
        :param work_info:
        :param overwrite:
        :return:
        """
        new_key_added = False
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

        # splits = self.calculate_splits(work_info)
        topological_sorted_nodes = query_plan_work_items(work_info)

        with self:
            try:
                cursor = self._execute_sql_gracefully(
                    insert_job(DEFAULT_SCHEMA, work_info)
                )
                new_key_added = cursor is not None and cursor.rowcount > 0
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error creating job: {error}")
                self.connection.rollback()
                raise ValueError(
                    f"Job creation for submission_id {submission_id} failed. "
                    f"Please check the logs for more information. {error}"
                )

        if not new_key_added:
            raise ValueError(
                f"Job with submission_id {submission_id} already exists. "
                "Please use a different submission_id."
            )

        self.logger.info(f"Job NOTIFY START with ID: {submission_id}")
        await self.notify_event()
        return submission_id

    async def mark_as_active(self, work_info: WorkInfo) -> bool:
        print(f"Marking job as active : {work_info.id}")

        with self:
            try:
                cursor = self._execute_sql_gracefully(
                    mark_as_active_jobs(DEFAULT_SCHEMA, work_info.name, [work_info.id])
                )
                key_updated = cursor is not None and cursor.rowcount > 0
                return key_updated
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error updating job: {error}")
                self.connection.rollback()
                return False

    async def is_valid_submission(
        self, work_info: WorkInfo, policy: ExistingWorkPolicy
    ) -> bool:
        """
        :param work_info: Information about the work to be checked for validity.
        :type work_info: WorkInfo
        :param policy: Policy that dictates the rules for the work submission.
        :type policy: ExistingWorkPolicy
        :return: Returns True if the work submission is valid according to the policy, False otherwise.
        :rtype: bool
        """
        if policy == ExistingWorkPolicy.ALLOW_ALL:
            return True
        if policy == ExistingWorkPolicy.REJECT_ALL:
            return False
        if policy == ExistingWorkPolicy.ALLOW_DUPLICATE:
            return True
        if policy == ExistingWorkPolicy.REJECT_DUPLICATE:
            existing_job = await self.get_job_for_policy(work_info)
            return existing_job is None
        if policy == ExistingWorkPolicy.REPLACE:
            existing_job = await self.get_job_for_policy(work_info)
            if existing_job and existing_job.state and existing_job.state.is_terminal():
                return True
            else:
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

    async def count_states(self) -> Dict[str, Dict[str, int]]:
        """
        Fetch and count job states from the database.

        :return: A dictionary with queue names as keys and state counts as values.
        """
        state_count_default = {key.lower(): 0 for key in WorkState.__members__.keys()}
        counts = []

        with self:
            try:
                cursor = self._execute_sql_gracefully(count_states(DEFAULT_SCHEMA))
                counts = cursor.fetchall()
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error handling job event: {error}")

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
        )

    async def _monitor(self):
        wait_time = MONITORING_POLL_PERIOD
        while self.running:
            self.logger.debug(f"Polling jobs status : {wait_time}")
            await asyncio.sleep(wait_time)
            try:
                monitored_on = None
                try:
                    cursor = self._execute_sql_gracefully(
                        try_set_monitor_time(
                            DEFAULT_SCHEMA,
                            monitor_state_interval_seconds=int(MONITORING_POLL_PERIOD),
                        )
                    )
                    monitored_on = cursor.fetchone()
                except (Exception, psycopg2.Error) as error:
                    self.logger.error(f"Error handling job event: {error}")

                if monitored_on is None:
                    self.logger.error("Error setting monitor time")
                    continue

                states = await self.count_states()
                logger.info(f"job state: {states}")
                # TODO: emit event
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
        self.logger.info(f"Job completed : {job_id}, {work_item}")
        with self:

            def complete_jobs_wrapper(
                schema: str, name: str, ids: list, output: dict, _force: bool
            ):
                if _force:
                    return complete_jobs_by_id(schema, name, ids, output)
                else:
                    return complete_jobs(schema, name, ids, output)

            try:
                cursor = self._execute_sql_gracefully(
                    complete_jobs_wrapper(
                        DEFAULT_SCHEMA,
                        work_item.name,
                        [job_id],
                        {"on_complete": "done", **(output_metadata or {})},
                        force,
                    )
                )
                counts = cursor.fetchone()[0]
                if counts > 0:
                    self.logger.info(f"Completed job: {job_id} : {counts}")
                else:
                    self.logger.error(f"Error completing job: {job_id}")
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error completing job: {error}")

    async def fail(
        self, job_id: str, work_item: WorkInfo, output_metadata: dict = None
    ):
        self.logger.info(f"Job failed : {job_id}, {work_item}")
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
        wait_time = SYNC_POLL_PERIOD
        while self.running:
            self.logger.info(f"Syncing jobs status : {wait_time}")
            await asyncio.sleep(wait_time)
            job_info_client = self.job_manager.job_info_client()

            try:
                active_jobs = await self.list_jobs(
                    state=[WorkState.ACTIVE.value, WorkState.CREATED.value]
                )
                if active_jobs:
                    for job_id, work_item in active_jobs.items():
                        self.logger.info(f"Syncing job: {job_id}, {work_item}")
                        job_info = await job_info_client.get_info(job_id)
                        if job_info is None:
                            self.logger.error(f"Job not found: {job_id}")
                            continue

                        job_info_state = convert_job_status_to_work_state(
                            job_info.status
                        )
                        if (
                            job_info.status.is_terminal()
                            and work_item.state != job_info_state
                        ):
                            self.logger.info(
                                f"State mismatch for job {job_id}: "
                                f"WorkState={work_item.state}, JobInfoState={job_info_state}. "
                                f"Updating to JobInfoState."
                            )
                            # check that the job can be synced by checking the end time have been at least 60min
                            synchronize = False
                            remaining_time = None
                            min_sync_interval = 5
                            if job_info.end_time is not None:
                                timestamp_ms = (
                                    job_info.end_time
                                )  # Unix timestamp in milliseconds
                                timestamp_s = timestamp_ms / 1000
                                end_time = datetime.fromtimestamp(timestamp_s)
                                remaining_time = end_time - datetime.now()
                                if end_time < datetime.now() - timedelta(
                                    minutes=min_sync_interval
                                ):
                                    synchronize = True

                            if not synchronize:
                                self.logger.info(
                                    f"Job has not ended more than {min_sync_interval} minutes ago, skipping "
                                    f"synchronization.  {job_id}: {remaining_time.total_seconds()}"
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
                                    f"Unhandled terminal status: {job_info.status}"
                                )

            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error syncing jobs: {error}")
                traceback.print_exc()
                if self.connection:
                    self.connection.rollback()  # Ensure rollback on failure

    def calculate_splits(self, work_info: WorkInfo) -> list[JobPartition]:
        """
        Calculate the splits for a work item.
        This is used to split the work item into smaller parts for parallel processing.
        :param work_info:
        :return:
        """

        partitioner = MarieJobPartitioner(10)
        partitions = partitioner.partition(work_info)

        return partitions

    async def notify_event(self) -> bool:
        self.logger.info(f"notify_event : {self._fetch_counter}")
        async with self._fetch_lock:
            if self._fetch_counter > 0:
                return False

            self._fetch_counter += 1
            self._fetch_event.set()
            return True


def query_plan_work_items(work_info):
    import numpy as np

    pprint(f"Query planning : {work_info}")
    frames = []
    for i in range(3):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frames.append(frame)

    plan = query_planner(frames, layout="12345")

    pprint(plan.model_dump())
    visualize_query_plan_graph(plan)
    yml_str = plan_to_yaml(plan)
    pprint(yml_str)

    sorted_nodes = topological_sort(plan)
    print("Topologically sorted nodes:", sorted_nodes)
    print_sorted_nodes(sorted_nodes, plan)
