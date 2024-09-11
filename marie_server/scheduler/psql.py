import asyncio
import traceback
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import psycopg2

from marie.helper import get_or_reuse_loop
from marie.job.common import JobStatus
from marie.job.job_manager import JobManager
from marie.logging.logger import MarieLogger
from marie.logging.predefined import default_logger as logger
from marie.storage.database.postgres import PostgresqlMixin
from marie_server.scheduler.fixtures import *
from marie_server.scheduler.job_scheduler import JobScheduler
from marie_server.scheduler.models import WorkInfo
from marie_server.scheduler.plans import (
    count_states,
    create_queue,
    fetch_next_job,
    insert_job,
    to_timestamp_with_tz,
    version_table_exists,
)
from marie_server.scheduler.state import WorkState

INIT_POLL_PERIOD = 1.250  # 250ms
MAX_POLL_PERIOD = 16.0  # 16s

MONITORING_POLL_PERIOD = 5.0  # 5s

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
    else:
        raise ValueError(f"Unknown JobStatus: {job_status}")


class PostgreSQLJobScheduler(PostgresqlMixin, JobScheduler):
    """A PostgreSQL-based job scheduler."""

    def __init__(self, config: Dict[str, Any], job_manager: JobManager):
        super().__init__()
        self._reset_on_complete = False
        self.logger = MarieLogger(PostgreSQLJobScheduler.__name__)
        if job_manager is None:
            raise ValueError("Job manager is required for JobScheduler")

        self.running = False
        self.task = None
        self.monitoring_task = None

        lock_free = True
        self._lock = (
            asyncio.Lock() if lock_free else asyncio.Lock()
        )  # Lock to prevent concurrent access to the database

        self.job_manager = job_manager
        self._loop = get_or_reuse_loop()
        self._setup_storage(config, connection_only=True)
        self._setup_event_subscriptions()

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
            work_item = await self.get_job(job_id)
            if work_item is None:
                self.logger.error(f"WorkItem not found: {job_id}")
                return

            completed_on = None
            started_on = None

            if status == JobStatus.PENDING:
                self.logger.info(f"Job pending : {job_id}")
            elif status == JobStatus.SUCCEEDED:
                self.logger.info(f"Job succeeded : {job_id}")
                completed_on = datetime.now()
            elif status == JobStatus.FAILED:
                self.logger.info(f"Job failed : {job_id}")
            elif status == JobStatus.RUNNING:
                self.logger.info(f"Job running : {job_id}")
                started_on = datetime.now()
            else:
                self.logger.error(f"Unhandled status : {status}")

            work_state = convert_job_status_to_work_state(status)
            await self.put_status(job_id, work_state, started_on, completed_on)

            if status.is_terminal():
                self.logger.info(f"Job {job_id} is in terminal state {status}")
                self._reset_on_complete = True

    def create_tables(self, schema: str):
        """
        :param schema: The name of the schema where the tables will be created.
        :return: None
        """
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
                return cursor is not None and cursor.rowcount > 0
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error clearing tables: {error}")
                self.connection.rollback()
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

        # TODO : This is a placeholder
        queue = "extract"
        await self.create_queue(queue)
        await self.create_queue(f"${queue}_dlq")

        self.running = True
        self.task = asyncio.create_task(self._poll())
        self.monitoring_task = asyncio.create_task(self._monitor())

    async def _poll(self):
        self.logger.info("Starting database scheduler")
        wait_time = INIT_POLL_PERIOD
        sleep_chunk = 0.250

        while self.running:
            self.logger.info(f"Polling for new jobs : {wait_time}")
            elapsed_time = 0
            while elapsed_time < wait_time:
                await asyncio.sleep(sleep_chunk)
                elapsed_time += sleep_chunk
                if self._reset_on_complete:
                    self.logger.info(
                        f"Elapsed time : {elapsed_time} > {wait_time} : {self._reset_on_complete}"
                    )
                    wait_time = INIT_POLL_PERIOD
                    self._reset_on_complete = False
                    break

            has_records = False
            if not self.job_manager.has_available_slot():
                self.logger.debug(
                    f"No available slots for work, waiting for slots :{wait_time}"
                )
                has_records = False
            else:
                records = await self.get_work_items(
                    limit=self.job_manager.SLOTS_AVAILABLE
                )
                if records is not None:
                    for record in records:
                        has_records = True
                        work_item = self.record_to_work_info(record)
                        has_available_slots, job_id = await self.enqueue(work_item)
                        self.logger.info(f"Work item scheduled with ID: {job_id}")
                        if not has_available_slots:
                            self.logger.info(
                                f"No more available slots for work, waiting for slots :{wait_time}"
                            )
                            break
            wait_time = (
                INIT_POLL_PERIOD if has_records else min(wait_time * 2, MAX_POLL_PERIOD)
            )

    async def stop(self) -> None:
        self.logger.info("Stopping job scheduling agent")
        self.running = False

        if self.task is not None:
            await self.task
        if self.monitoring_task is not None:
            await self.monitoring_task

    def debug_info(self) -> str:
        print("Debugging info")

    async def enqueue(self, work_info: WorkInfo) -> tuple[bool, str]:
        """
        Enqueues a work item for processing on the next available executor.

        :param work_info: The information about the work item to be processed.
        :return: A tuple containing a boolean indicating whether the work item was successfully enqueued and the ID of the work item.
        """
        if not self.job_manager.has_available_slot():
            self.logger.info(
                f"No available slots for work, scheduling : {work_info.id}"
            )
            return False, None

        submission_id = work_info.id
        returned_id = await self.job_manager.submit_job(
            entrypoint="echo hello", submission_id=submission_id
        )
        return True, returned_id

    async def get_work_items(
        self,
        limit: int = 1,
        stop_event: asyncio.Event = None,
    ) -> List[Any]:
        """Get the Jobs from the PSQL database.

        :param limit: the maximal number records to get
        :param stop_event: an event to signal when to stop iterating over the records
        :return:
        """
        async with self._lock:
            with self:
                try:
                    fetch_query_def = fetch_next_job(DEFAULT_SCHEMA)
                    query = fetch_query_def(
                        name="extract",  # TODO this is a placeholder
                        batch_size=limit,
                        include_metadata=False,
                        priority=True,
                    )
                    # we can't use named cursors as it will throw an error
                    cursor = self.connection.cursor()
                    cursor.itersize = limit
                    cursor.execute(f"{query}")
                    records = [record for record in cursor]

                    return records
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

    async def list_jobs(self, state: Optional[str] = None) -> Dict[str, WorkInfo]:
        work_items = {}
        schema = DEFAULT_SCHEMA
        table = DEFAULT_JOB_TABLE
        states = "','".join(WorkState.__members__.keys())
        if state is not None:
            if state.upper() not in WorkState.__members__:
                raise ValueError(f"Invalid state: {state}")
            states = state
        states = states.lower()

        with self:
            try:
                cursor = self.connection.cursor("doc_iterator")
                cursor.itersize = 10000
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
                          on_complete
                    FROM {schema}.{table} 
                    WHERE state IN ('{states}')
                    """
                    # + (f" limit = {limit}" if limit > 0 else "")
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
        """
        new_key_added = False
        submission_id = work_info.id

        work_info.retry_limit = 2

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

        await self.enqueue(work_info)
        return submission_id

    def stop_job(self, job_id: str) -> bool:
        """Request a job to exit, fire and forget.
        Returns whether or not the job was running.
        """
        raise NotImplementedError

    async def delete_job(self, job_id: str):
        """Deletes the job with the given job_id."""
        ...

        raise NotImplementedError

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

    async def count_states(self):
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
            state = state or "all"
            queue[state] = int(size)

        return states

    def record_to_work_info(self, record):
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
                states = await self.count_states()
                logger.info(f"job state: {states}")
                # TODO: emit event
            except Exception as e:
                logger.error(f"Error monitoring jobs: {e}")
                traceback.print_exc()
                # TODO: emit error event
