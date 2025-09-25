from typing import Any, Dict

import psycopg2

from marie.logging_core.logger import MarieLogger
from marie.scheduler.plans import (
    count_dag_states,
    count_job_states,
)
from marie.scheduler.state import WorkState
from marie.storage.database.postgres import PostgresqlMixin

DEFAULT_SCHEMA = "marie_scheduler"
DEFAULT_JOB_TABLE = "job"


class SchedulerRepository(PostgresqlMixin):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.logger = MarieLogger(SchedulerRepository.__name__)
        self._setup_storage(config, connection_only=True)

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
