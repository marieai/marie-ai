from typing import Any, Dict

from marie.job.common import JobInfoStorageClient
from marie.logging_core.logger import MarieLogger
from marie.storage.database.postgres import PostgresqlMixin


class SyncManager:
    """
    SyncManager is responsible for synchronizing the state of the scheduler with the state of the executor if they are out of sync.
    We will also publish events to the event publisher to notify the user of the state of the Job

    Example : If we restart the scheduler, we need to synchronize the state of the executor jobs that have completed while the scheduler was down.
    Downside to this is that floating executors will not be able to sync their state with the scheduler  during the time the scheduler is down.

    """

    def __init__(
        self,
        config: Dict[str, Any],
        job_info_client: JobInfoStorageClient,
        psql_mixin: PostgresqlMixin,
    ):
        self.config = config
        self.logger = MarieLogger(self.__class__.__name__)
        self.job_info_client = job_info_client
        self.psql_mixin = psql_mixin

        print("SyncManager init called")
        print(job_info_client)
        print(psql_mixin)

        self.run_sync()

    async def start(self) -> None:
        """
        Starts the job synchronization agent.

        :return: None
        """

        pass

    def run_sync(self):
        """
        Run the synchronization process.

        :return: None
        """
        print("Running sync")

        # Get all the jobs from the scheduler that are not in the TERMINAL state
