import asyncio
import threading
import traceback
from enum import Enum
from typing import Any, Dict, Generator, List

import psycopg2

from marie.excepts import BadConfigSource
from marie.logging.logger import MarieLogger
from marie.logging.predefined import default_logger as logger
from marie.storage.database.postgres import PostgresqlMixin
from marie_server.scheduler.fixtures import *
from marie_server.scheduler.scheduler import Scheduler
from marie_server.scheduler.state import States

INIT_POLL_PERIOD = 1.250  # 250ms
MAX_POLL_PERIOD = 16.0  # 16s


DEFAULT_SCHEMA = "marie_scheduler"
COMPLETION_JOB_PREFIX = f"__state__{States.COMPLETED.value}__"


class PostgreSQLJobScheduler(PostgresqlMixin, Scheduler):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = MarieLogger("PostgreSQLJobScheduler")
        print("config", config)
        self.running = False
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
            create_index_singleton_on(schema),
            create_index_singleton_key_on(schema),
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
            self._execute_sql_gracefully(locked_query)

    def start_schedule(self) -> None:
        """
        Starts the job scheduling agent.

        :return: None
        """
        logger.info("Starting job scheduling agent")
        self.create_tables(DEFAULT_SCHEMA)

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
            document_iterator = self.get_document_iterator()
            has_records = False

            for record in document_iterator:
                has_records = True
                print("record", record)
                await self.schedule(record)

            wait_time = (
                INIT_POLL_PERIOD if has_records else min(wait_time * 2, MAX_POLL_PERIOD)
            )

    def stop_schedule(self) -> None:
        pass

    def debug_info(self) -> str:
        pass

    # async def get_records_for_run(self) -> List[Dict[str, Any]]:
    #     records = []
    #     records.append({"id": 1, "name": "test"})
    #     return records

    async def schedule(self, record):
        print("scheduling : ", record)

    def get_document_iterator(
        self,
        limit: int = 0,
    ) -> Generator[Any, None, None]:
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
