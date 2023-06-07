import asyncio
import threading
import traceback
from typing import Any, Dict, List, Generator

import psycopg2

from marie.excepts import BadConfigSource
from marie.logging.logger import MarieLogger
from marie.storage.database.postgres import PostgresqlMixin
from marie_server.scheduler.scheduler import Scheduler
from marie.logging.predefined import default_logger as logger

INIT_POLL_PERIOD = 1.250  # 250ms
MAX_POLL_PERIOD = 16.0  # 16s


class PostgreSQLJobScheduler(PostgresqlMixin, Scheduler):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = MarieLogger("PostgreSQLJobScheduler")
        print("config", config)
        self.running = False
        self._setup_storage(config)

    def start_schedule(self) -> None:
        logger.info("Starting job scheduling agent")

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

    def _create_table(self, table_name: str) -> None:
        """Create the table if it doesn't exist."""
        print("creating table : ", table_name)

        self._execute_sql_gracefully(
            f"""
             CREATE TABLE IF NOT EXISTS  queue (
                 id UUID PRIMARY KEY,
                 created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                 updated_at TIMESTAMP WITH TIME ZONE NOT NULL,

                 scheduled_for TIMESTAMP WITH TIME ZONE NOT NULL,
                 failed_attempts INT NOT NULL,
                 status INT NOT NULL,
                 message JSONB NOT NULL
             );

             CREATE INDEX index_queue_on_scheduled_for ON queue (scheduled_for);
             CREATE INDEX index_queue_on_status ON queue (status);
             """,
        )

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
