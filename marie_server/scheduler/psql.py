import asyncio
import threading
import traceback
from typing import Any, AsyncGenerator, Dict, Optional

import psycopg2

from marie.helper import get_or_reuse_loop
from marie.logging.logger import MarieLogger
from marie.logging.predefined import default_logger as logger
from marie.storage.database.postgres import PostgresqlMixin
from marie_server.job.job_manager import JobManager
from marie_server.scheduler.fixtures import *
from marie_server.scheduler.job_scheduler import JobScheduler
from marie_server.scheduler.models import WorkInfo
from marie_server.scheduler.plans import insert_job
from marie_server.scheduler.state import States

INIT_POLL_PERIOD = 1.250  # 250ms
MAX_POLL_PERIOD = 16.0  # 16s

DEFAULT_SCHEMA = "marie_scheduler"
COMPLETION_JOB_PREFIX = f"__state__{States.COMPLETED.value}__"


class PostgreSQLJobScheduler(PostgresqlMixin, JobScheduler):
    """A PostgreSQL-based job scheduler."""

    def __init__(self, config: Dict[str, Any], job_manager: JobManager):
        super().__init__()
        self.logger = MarieLogger(PostgreSQLJobScheduler.__name__)
        self.running = False
        self.job_manager = job_manager
        self._loop = get_or_reuse_loop()
        self._setup_storage(config, connection_only=True)

    def create_tables(self, schema: str):
        """
        :param schema: The name of the schema where the tables will be created.
        :return: None
        """
        commands = [
            create_schema(schema),
            create_version_table(schema),
            create_job_state_enum(schema),
            create_job_table(schema),
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

    async def start(self) -> None:
        """
        Starts the job scheduling agent.

        :return: None
        """
        logger.info("Starting job scheduling agent")
        self.create_tables(DEFAULT_SCHEMA)
        self.running = True

        if False:

            def _run():
                try:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None

                    if loop is None:
                        asyncio.run(self.__poll())
                    else:
                        loop.run_until_complete(self.__poll())
                except Exception as e:
                    logger.error(f"Unable to setup job scheduler: {e}")
                    logger.error(traceback.format_exc())

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join()  # FOR TESTING PURPOSES ONLY

    async def __poll(self):
        print("Starting poller with psql")
        self.running = True
        wait_time = INIT_POLL_PERIOD

        while self.running:
            print(f"Polling for new jobs : {wait_time}")
            await asyncio.sleep(wait_time)
            document_iterator = self.get_work_items()
            has_records = False

            for record in document_iterator:
                has_records = True
                print("record", record)
                job_id = await self.schedule(record)
                self.logger.info(f"Work item scheduled with ID: {job_id}")
            wait_time = (
                INIT_POLL_PERIOD if has_records else min(wait_time * 2, MAX_POLL_PERIOD)
            )

    async def stop(self) -> None:
        self.running = False

    def debug_info(self) -> str:
        print("Debugging info")

    async def schedule(self, record: WorkInfo) -> str:
        """
        :param record:
        """
        print("scheduling : ", record)
        return "job_id"

    async def get_work_items(
        self,
        limit: int = 0,
    ) -> AsyncGenerator[Any, None]:
        """Get the Jobs from the PSQL database.

        :param limit: the maximal number records to get
        :return:
        """
        with self:
            try:
                cursor = self.connection.cursor("doc_iterator")
                cursor.itersize = 10000
                cursor.execute(
                    f"""
                    SELECT * FROM job_queue
                    """
                    # + (f" limit = {limit}" if limit > 0 else "")
                )
                for record in cursor:
                    print(record)
                    doc_id = record[0]
                    yield doc_id
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error importing snapshot: {error}")
                self.connection.rollback()
            self.connection.commit()

    async def get_job(self, job_id: str) -> Optional[WorkInfo]:
        """
        Get a job by its ID.
        :param job_id:
        """
        schema = DEFAULT_SCHEMA
        table = "job"

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
                          on_complete
                    FROM {schema}.{table}
                    WHERE id = '{job_id}'
                    """
                )
                record = cursor.fetchone()
                if record:
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
                        on_complete=record[11],
                    )
                return None
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error importing snapshot: {error}")
                self.connection.rollback()
            self.connection.commit()

    async def list_jobs(self) -> Dict[str, WorkInfo]:
        work_items = {}
        schema = DEFAULT_SCHEMA
        table = "job"

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
                    """
                    # + (f" limit = {limit}" if limit > 0 else "")
                )

                for record in cursor:
                    work_items[record[0]] = WorkInfo(
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
                        on_complete=record[11],
                    )
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error listing jobs: {error}")
                self.connection.rollback()
            self.connection.commit()
        return work_items

    async def submit_job(self, work_info: WorkInfo, overwrite: bool = True) -> bool:
        """
        Inserts a new work item into the scheduler.
        :param work_info: The work item to insert.
        :param overwrite: Whether to overwrite the work item if it already exists.
        :return:
        """
        insert_query = insert_job(DEFAULT_SCHEMA, work_info)
        with self:
            try:
                cursor = self._execute_sql_gracefully(insert_query)
                record = cursor.fetchone()
                print("record", record)
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Error inserting job: {error}")
                self.connection.rollback()
            self.connection.commit()
        return True

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
