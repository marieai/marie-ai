import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

from marie.logging_core.logger import MarieLogger
from marie.scheduler.models import RecoveredRunLease
from marie.scheduler.repository import JobRepository


class MaintenanceService:
    """Recover expired scheduler leases and active run attempts."""

    def __init__(
        self,
        repository: JobRepository,
        notify_callback: Callable[[], Awaitable[bool]] | None = None,
        recovery_callback: Optional[
            Callable[[list[RecoveredRunLease]], Awaitable[None]]
        ] = None,
        maintenance_interval: int = 60,  # seconds
    ) -> None:
        """
        Initialize the maintenance service.

        :param repository: JobRepository for database operations
        :param notify_callback: Callback function to trigger scheduler events
        :param maintenance_interval: How often to run maintenance (in seconds)
        """
        self.logger = MarieLogger(MaintenanceService.__name__)
        self.repository = repository
        self._notify_callback = notify_callback
        self._recovery_callback = recovery_callback
        self.maintenance_interval = maintenance_interval

        # Maintenance task
        self._maintenance_task: Optional[asyncio.Task] = None
        self._running = False

    # ==================== Maintenance Operations ====================

    async def maintenance(self) -> None:
        """Run periodic lease recovery."""
        try:
            await self.expire()
        except Exception as e:
            self.logger.error(f"Error in maintenance: {e}")

    async def expire(self):
        """
        Expire jobs with expired leases.
        Releases leases that have timed out so jobs can be retried.
        """
        self.logger.debug("Checking for expired job leases")

        released_count = await self.repository.release_expired_leases()
        recovered = await self.repository.recover_expired_run_leases()
        recovered_retry_count = sum(
            1 for row in recovered if row.recovered_state == "retry"
        )
        recovered_failed_count = sum(
            1 for row in recovered if row.recovered_state == "failed"
        )

        if released_count > 0:
            self.logger.info(f"Released expired job leases: {released_count}")
        if recovered_retry_count > 0:
            self.logger.warning(
                f"Recovered expired active run leases to retry: {recovered_retry_count}"
            )
        if recovered_failed_count > 0:
            self.logger.warning(
                f"Recovered expired active run leases to failed: {recovered_failed_count}"
            )

        if recovered and self._recovery_callback:
            await self._recovery_callback(recovered)

        if released_count > 0 or recovered:
            if self._notify_callback:
                await self._notify_callback()

    async def archive(self) -> None:
        """Raise until job archival and its retention policy are implemented."""
        raise NotImplementedError("Job archiving is not implemented")

    async def purge(self) -> None:
        """Raise until archived-job purging and its retention policy are implemented."""
        raise NotImplementedError("Archived job purging is not implemented")

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
