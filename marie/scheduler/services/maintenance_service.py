import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import psycopg2

from marie.logging_core.logger import MarieLogger
from marie.scheduler.repository import JobRepository


class MaintenanceService:
    """
    Service for performing periodic maintenance tasks on the scheduler.
    Handles expiring leases, archiving completed jobs, and purging old data.
    """

    def __init__(
        self,
        repository: JobRepository,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        notify_callback: Optional[callable] = None,
        maintenance_interval: int = 60,  # seconds
    ):
        """
        Initialize the maintenance service.

        :param repository: JobRepository for database operations
        :param loop: Event loop for async operations
        :param executor: Thread pool executor for blocking operations
        :param notify_callback: Callback function to trigger scheduler events
        :param maintenance_interval: How often to run maintenance (in seconds)
        """
        self.logger = MarieLogger(MaintenanceService.__name__)
        self.repository = repository
        self._loop = loop or asyncio.get_event_loop()
        self._executor = executor
        self._notify_callback = notify_callback
        self.maintenance_interval = maintenance_interval

        # Maintenance task
        self._maintenance_task: Optional[asyncio.Task] = None
        self._running = False

    # ==================== Maintenance Operations ====================

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
        """
        Expire jobs with expired leases.
        Releases leases that have timed out so jobs can be retried.
        """
        self.logger.debug("Checking for expired job leases")

        def db_call():
            """Sync DB call to release expired leases."""
            conn = None
            released_count = 0
            try:
                conn = self.repository._get_connection()
                query = "SELECT marie_scheduler.release_expired_leases()"
                result = self.repository._execute_sql_gracefully(query, connection=conn)

                if result and isinstance(result, list) and len(result) > 0:
                    count = result[0][0]
                    if count:
                        released_count = count
            except (Exception, psycopg2.Error) as error:
                self.logger.error(f"Failed to expire jobs: {error}", exc_info=True)
            finally:
                self.repository._close_connection(conn)
            return released_count

        released_count = await self._loop.run_in_executor(self._executor, db_call)
        if released_count > 0:
            self.logger.info(f"Released expired job leases: {released_count}")
            if self._notify_callback:
                await self._notify_callback()

    async def archive(self):
        """
        Archive completed jobs.
        Move completed jobs to archive table for historical tracking.
        """
        self.logger.debug("Archiving completed jobs")
        # TODO: Implement archival logic
        # - Move completed jobs older than X days to archive table
        # - Keep original IDs for reference
        # - Update archive timestamp

    async def purge(self):
        """
        Purge old archived jobs.
        Remove very old archived jobs to prevent database bloat.
        """
        self.logger.debug("Purging old archived jobs")
        # TODO: Implement purge logic
        # - Delete archived jobs older than retention period
        # - Respect configured retention policy
        # - Log purge statistics

    async def start(self):
        """
        Start the periodic maintenance task.
        """
        if self._maintenance_task:
            self.logger.warning("Maintenance service already running")
            return

        self._running = True
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.logger.info(
            f"Started MaintenanceService (interval: {self.maintenance_interval}s)"
        )

    async def stop(self):
        """
        Stop the periodic maintenance task.
        """
        self._running = False
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None
            self.logger.info("Stopped MaintenanceService")

    async def _maintenance_loop(self):
        """
        Periodic loop that runs maintenance tasks.
        """
        self.logger.info(
            f"Starting maintenance loop (interval: {self.maintenance_interval}s)"
        )

        while self._running:
            try:
                # Run maintenance tasks
                await self.maintenance()
            except Exception as e:
                self.logger.error(f"Error in maintenance loop: {e}")

            # Wait for next cycle
            await asyncio.sleep(self.maintenance_interval)

        self.logger.info("Maintenance loop stopped")

    async def run_now(self):
        """
        Manually trigger a maintenance run immediately.
        Useful for testing or forced cleanup.
        """
        self.logger.info("Running maintenance manually")
        await self.maintenance()

    async def expire_now(self):
        """
        Manually trigger lease expiration immediately.
        """
        self.logger.info("Running lease expiration manually")
        await self.expire()

    def set_interval(self, interval: int):
        """
        Update the maintenance interval.

        :param interval: New interval in seconds
        """
        old_interval = self.maintenance_interval
        self.maintenance_interval = interval
        self.logger.info(
            f"Updated maintenance interval: {old_interval}s -> {interval}s"
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get current maintenance configuration.

        :return: Configuration dictionary
        """
        return {
            "interval_seconds": self.maintenance_interval,
            "running": self._running,
            "has_task": self._maintenance_task is not None,
        }
