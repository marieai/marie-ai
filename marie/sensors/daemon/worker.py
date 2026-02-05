"""
Sensor Worker Daemon

Background service that continuously evaluates active sensors and submits
jobs when sensor conditions are met.

Design:
1. Runs in dedicated background thread via server_runtime.py (LLMTrackingWorker pattern)
2. Uses NOTIFY for fast-path wake-up, polling as authoritative backup
3. ALWAYS records ticks (success, skipped, or failed)
4. Uses STARTED tick reservation for crash recovery
5. Idempotent job submission via run_key

Integration:
    Started by setup_sensor_worker() in marie/utils/server_runtime.py.
    Called from setup_server() in marie_server/__main__.py.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from marie.logging_core.logger import MarieLogger
from marie.sensors.config import SensorSettings
from marie.sensors.context import SensorEvaluationContext
from marie.sensors.exceptions import SensorEvaluationError
from marie.sensors.registry import SensorRegistry, register_all_sensors
from marie.sensors.state.psql_storage import PostgreSQLSensorStorage
from marie.sensors.types import (
    RunRequest,
    SensorResult,
    SensorStatus,
    SensorType,
    TickStatus,
)

if TYPE_CHECKING:
    from marie.scheduler.services import NotificationService

logger = MarieLogger("SensorWorker")


class SensorWorker:
    """
    Background worker that evaluates all active sensors on a continuous loop.

    Runs in a dedicated thread with its own event loop (via server_runtime.py).
    Uses PostgreSQL NOTIFY for fast-path wake-up, polling as backup.

    Lifecycle:
        1. Initialize with config
        2. set_storage() - set PostgreSQL storage
        3. start() - begins async evaluation loop
        4. stop() - graceful shutdown
    """

    def __init__(
        self,
        config: Dict[str, Any],
        job_scheduler=None,
        notification_service: Optional["NotificationService"] = None,
    ):
        """
        Initialize the sensor worker.

        :param config: Application configuration
        :param job_scheduler: Reference to job scheduler for job submission
        :param notification_service: Optional notification service for NOTIFY
        """
        self._config = config
        self._job_scheduler = job_scheduler
        self._notification_service = notification_service

        # Load settings
        self._settings = SensorSettings.from_config(config)
        self._settings.validate()

        # State
        self._storage: Optional[PostgreSQLSensorStorage] = None
        self._started = False
        self._daemon_task: Optional[asyncio.Task] = None

        # Events for coordination
        self._wake_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()

        # Registry for sensor evaluators
        self._registry = SensorRegistry.get_instance()

        # Resources shared across evaluations
        self._resources: Dict[str, Any] = {}

        # Statistics
        self._stats = {
            "cycles": 0,
            "sensors_evaluated": 0,
            "ticks_recorded": 0,
            "jobs_submitted": 0,
            "errors": 0,
        }

    def set_storage(self, storage: PostgreSQLSensorStorage) -> None:
        """Set the storage instance (called during initialization)."""
        self._storage = storage

    def set_job_scheduler(self, job_scheduler) -> None:
        """Set the job scheduler reference."""
        self._job_scheduler = job_scheduler

    def set_notification_service(
        self, notification_service: "NotificationService"
    ) -> None:
        """Set the notification service for NOTIFY integration."""
        self._notification_service = notification_service

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self) -> None:
        """
        Start the sensor worker.

        This creates an async task for the daemon loop. The task runs
        within the existing event loop (not asyncio.run()).
        """
        if self._started:
            logger.warning("SensorWorker is already running")
            return

        logger.info("Starting SensorWorker")

        # Register all sensor evaluators
        register_all_sensors()

        # Register NOTIFY handler if notification service is available
        if self._notification_service and self._settings.enable_notify_wake:
            self._notification_service.register_handler(
                channel="sensor_event",
                handler=self._handle_sensor_notification,
            )
            logger.info("Registered sensor_event notification handler")

        self._started = True
        self._shutdown_event.clear()

        # Start the daemon task
        self._daemon_task = asyncio.create_task(self._daemon_loop())
        logger.info(
            f"SensorWorker started (interval: {self._settings.daemon_interval_seconds}s)"
        )

    async def stop(self) -> None:
        """
        Stop the sensor worker gracefully.

        Signals shutdown and waits for the daemon task to complete.
        """
        if not self._started:
            return

        logger.info("Stopping SensorWorker")
        self._shutdown_event.set()

        if self._daemon_task and not self._daemon_task.done():
            self._daemon_task.cancel()
            try:
                await self._daemon_task
            except asyncio.CancelledError:
                pass

        # Unregister notification handler
        if self._notification_service:
            self._notification_service.unregister_handler("sensor_event")

        # Cleanup resources
        await self._cleanup_resources()

        self._started = False
        logger.info(f"SensorWorker stopped. Stats: {self._stats}")

    async def _cleanup_resources(self) -> None:
        """Clean up any resources (e.g., HTTP clients)."""
        if "http_client" in self._resources:
            try:
                await self._resources["http_client"].close()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")

    # =========================================================================
    # NOTIFICATION HANDLING
    # =========================================================================

    async def _handle_sensor_notification(self, payload: Dict[str, Any]) -> None:
        """
        Handle sensor_event notification from PostgreSQL.

        This is called when a new event is inserted into event_log.
        We simply set the wake event - the daemon loop will poll
        on the next iteration.

        Multiple notifications coalesce into a single wake-up.
        """
        logger.debug(f"Received sensor notification: {payload}")
        self._wake_event.set()

    # =========================================================================
    # DAEMON LOOP
    # =========================================================================

    async def _daemon_loop(self) -> None:
        """
        Main daemon loop that continuously evaluates sensors.

        The loop:
        1. Evaluates all active sensors
        2. Waits for: timeout OR wake notification OR shutdown
        3. Repeats

        Uses asyncio.wait() with FIRST_COMPLETED for efficient waiting.
        """
        logger.info("Sensor daemon loop started")

        # Resume any STARTED ticks from previous run (crash recovery)
        await self._resume_started_ticks()

        while not self._shutdown_event.is_set():
            cycle_start = time.perf_counter()

            try:
                await self._evaluate_all_sensors()
                self._stats["cycles"] += 1
            except Exception as e:
                logger.error(f"Error in daemon cycle: {e}", exc_info=True)
                self._stats["errors"] += 1

            # Calculate sleep time
            elapsed = time.perf_counter() - cycle_start
            sleep_time = max(0, self._settings.daemon_interval_seconds - elapsed)

            # Wait for timeout, wake, or shutdown
            self._wake_event.clear()
            try:
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(self._shutdown_event.wait()),
                        asyncio.create_task(self._wake_event.wait()),
                    ],
                    timeout=sleep_time,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks to avoid leaks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            except asyncio.CancelledError:
                break

        logger.info("Sensor daemon loop exiting")

    # =========================================================================
    # SENSOR EVALUATION
    # =========================================================================

    async def _evaluate_all_sensors(self) -> None:
        """Evaluate all active sensors."""
        if not self._storage:
            logger.warning("Storage not initialized, skipping evaluation")
            return

        sensors = await self._storage.get_active_sensors()

        if not sensors:
            logger.debug("No active sensors to evaluate")
            return

        logger.debug(f"Evaluating {len(sensors)} active sensors")

        for sensor in sensors[: self._settings.max_sensors_per_cycle]:
            if self._shutdown_event.is_set():
                break

            # Check minimum interval
            if not self._should_evaluate(sensor):
                continue

            try:
                await self._evaluate_sensor_with_tick(sensor)
                self._stats["sensors_evaluated"] += 1
            except Exception as e:
                logger.error(
                    f"Error evaluating sensor {sensor.get('name')}: {e}",
                    exc_info=True,
                )
                self._stats["errors"] += 1

    def _should_evaluate(self, sensor: Dict[str, Any]) -> bool:
        """Check if sensor should be evaluated based on minimum interval."""
        last_tick_at = sensor.get("last_tick_at")
        if not last_tick_at:
            return True

        min_interval = sensor.get("minimum_interval_seconds", 30)
        now = datetime.now(timezone.utc)

        if last_tick_at.tzinfo is None:
            last_tick_at = last_tick_at.replace(tzinfo=timezone.utc)

        elapsed = (now - last_tick_at).total_seconds()
        return elapsed >= min_interval

    async def _evaluate_sensor_with_tick(self, sensor: Dict[str, Any]) -> None:
        """
        Evaluate a single sensor and record a tick.

        This method ALWAYS records a tick (success, skipped, or failed).
        """
        tick_start = time.perf_counter()
        sensor_id = sensor.get("id")
        sensor_name = sensor.get("name", "unknown")
        sensor_type = SensorType(sensor.get("sensor_type"))

        status = TickStatus.FAILED
        cursor = None
        run_ids = []
        skip_reason = None
        error_message = None
        tick_id = None

        try:
            # Build evaluation context
            context = await self._build_context(sensor)

            # Get evaluator for this sensor type
            evaluator_class = self._registry.get_evaluator(sensor_type)
            evaluator = evaluator_class(sensor)

            # Create STARTED tick for crash recovery (if we'll submit jobs)
            # For now, we evaluate first to see if jobs will be submitted

            # Evaluate the sensor
            result = await evaluator.evaluate(context)

            # Process result
            if result.has_run_requests():
                # Create STARTED tick before job submission
                tick_id = await self._storage.create_tick(
                    sensor_id=sensor_id,
                    status=TickStatus.STARTED,
                    run_requests=[rr.to_dict() for rr in result.run_requests],
                    reserved_run_ids=[str(uuid4()) for _ in result.run_requests],
                )

                # Submit jobs
                for run_request in result.run_requests:
                    job_id = await self._submit_run_request(sensor, run_request)
                    if job_id:
                        run_ids.append(job_id)
                        self._stats["jobs_submitted"] += 1

                status = TickStatus.SUCCESS
            elif result.skip_reason:
                status = TickStatus.SKIPPED
                skip_reason = result.skip_reason.message
            else:
                status = TickStatus.SKIPPED
                skip_reason = "No run requests generated"

            cursor = result.cursor

        except SensorEvaluationError as e:
            logger.error(f"Sensor evaluation error for {sensor_name}: {e}")
            status = TickStatus.FAILED
            error_message = str(e)

        except Exception as e:
            logger.error(
                f"Unexpected error evaluating sensor {sensor_name}: {e}",
                exc_info=True,
            )
            status = TickStatus.FAILED
            error_message = f"Unexpected error: {e}"

        # Calculate duration
        duration_ms = int((time.perf_counter() - tick_start) * 1000)

        # ALWAYS record tick (success, skipped, or failed)
        await self._storage.record_tick(
            sensor_id=sensor_id,
            status=status,
            cursor=cursor,
            run_ids=run_ids,
            skip_reason=skip_reason,
            error_message=error_message,
            duration_ms=duration_ms,
            tick_id=tick_id,
        )
        self._stats["ticks_recorded"] += 1

        # Update cursor on success/skipped
        if cursor and status != TickStatus.FAILED:
            await self._storage.set_cursor(sensor_id, cursor)

    async def _build_context(self, sensor: Dict[str, Any]) -> SensorEvaluationContext:
        """Build evaluation context for a sensor."""
        sensor_id = sensor.get("id")
        sensor_type = SensorType(sensor.get("sensor_type"))
        cursor = await self._storage.get_cursor(sensor_id)

        context = SensorEvaluationContext(
            sensor_id=sensor_id,
            sensor_name=sensor.get("name", ""),
            sensor_type=sensor_type,
            cursor=cursor,
            last_tick_at=sensor.get("last_tick_at"),
            last_run_key=sensor.get("last_run_key"),
            config=sensor.get("config", {}),
            target_job_name=sensor.get("target_job_name"),
            target_dag_id=sensor.get("target_dag_id"),
            resources=self._resources,
            logger=MarieLogger(f"sensor.{sensor.get('name')}"),
        )

        # Load pending events for event-based sensors
        if sensor_type in (SensorType.WEBHOOK, SensorType.EVENT):
            external_id = sensor.get("external_id")
            context.pending_events = await self._storage.get_pending_events(
                sensor_external_id=external_id,
                cursor=cursor,
                limit=self._settings.max_events_per_tick,
            )

        # Load completed jobs for run_status sensors
        if sensor_type == SensorType.RUN_STATUS:
            context.completed_jobs = await self._load_completed_jobs(sensor, cursor)

        return context

    async def _load_completed_jobs(
        self, sensor: Dict[str, Any], cursor: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Load completed jobs for run_status sensor evaluation."""
        # This would query the job table for recently completed jobs
        # For now, return empty list - actual implementation depends on
        # how the job scheduler exposes this data
        if self._job_scheduler:
            # TODO: Implement job completion query
            pass
        return []

    async def _submit_run_request(
        self, sensor: Dict[str, Any], run_request: RunRequest
    ) -> Optional[str]:
        """
        Submit a job from a run request.

        Checks idempotency via run_key before submission.
        """
        sensor_id = sensor.get("id")
        sensor_name = sensor.get("name", "unknown")

        # Check idempotency
        if run_request.run_key:
            if await self._storage.has_run_key(sensor_id, run_request.run_key):
                logger.debug(f"Skipping duplicate run_key: {run_request.run_key}")
                return None

        # Determine target job name
        job_name = run_request.job_name or sensor.get("target_job_name")
        dag_id = run_request.dag_id or sensor.get("target_dag_id")

        if not job_name and not dag_id:
            logger.warning(f"Sensor {sensor_name} has no target job or DAG configured")
            return None

        # Submit job via scheduler
        if self._job_scheduler:
            try:
                from marie.scheduler.models import WorkInfo
                from marie.scheduler.state import WorkState

                work_info = WorkInfo(
                    name=job_name,
                    dag_id=dag_id,
                    priority=run_request.priority,
                    data={
                        "run_config": run_request.run_config,
                        "sensor_id": sensor_id,
                        "sensor_name": sensor_name,
                        "run_key": run_request.run_key,
                        **run_request.tags,
                    },
                    state=WorkState.CREATED,
                    retry_limit=3,
                    retry_delay=2,
                    retry_backoff=True,
                )

                # Use scheduler's submit method
                job_id = await self._job_scheduler.submit_job(work_info)

                # Record run_key for idempotency
                if run_request.run_key:
                    await self._storage.record_run_key(
                        sensor_id, run_request.run_key, job_id
                    )

                logger.info(
                    f"Sensor {sensor_name} triggered job {job_id} "
                    f"(run_key: {run_request.run_key})"
                )
                return job_id

            except Exception as e:
                logger.error(f"Failed to submit job from sensor {sensor_name}: {e}")
                raise

        else:
            logger.warning("No job scheduler available for job submission")
            return None

    # =========================================================================
    # CRASH RECOVERY
    # =========================================================================

    async def _resume_started_ticks(self) -> None:
        """
        Resume STARTED ticks from previous daemon run.

        This handles crash recovery by finding ticks that were started
        but never completed, and resuming job submission.
        """
        if not self._storage:
            return

        started_ticks = await self._storage.get_started_ticks(
            threshold_hours=self._settings.stuck_tick_threshold_hours
        )

        if not started_ticks:
            return

        logger.info(f"Found {len(started_ticks)} stuck STARTED ticks to recover")

        for tick in started_ticks:
            await self._recover_tick(tick)

    async def _recover_tick(self, tick: Dict[str, Any]) -> None:
        """
        Recover a single stuck STARTED tick.

        Attempts to resume job submission for any reserved runs
        that weren't submitted.
        """
        tick_id = tick.get("id")
        sensor_id = tick.get("sensor_id")
        sensor_name = tick.get("sensor_name", "unknown")

        logger.info(f"Recovering tick {tick_id} for sensor {sensor_name}")

        run_requests = tick.get("run_requests", [])
        reserved_run_ids = tick.get("reserved_run_ids", [])
        existing_run_ids = tick.get("run_ids", [])

        # Find runs that weren't submitted
        new_run_ids = list(existing_run_ids)

        for i, run_request_data in enumerate(run_requests):
            run_request = RunRequest.from_dict(run_request_data)

            # Check if this run was already submitted
            if run_request.run_key:
                if await self._storage.has_run_key(sensor_id, run_request.run_key):
                    continue

            # Get the reserved run ID
            reserved_id = reserved_run_ids[i] if i < len(reserved_run_ids) else None

            # Submit the job
            # Build minimal sensor dict for submission
            sensor = {
                "id": sensor_id,
                "name": sensor_name,
                "target_job_name": tick.get("target_job_name"),
                "target_dag_id": tick.get("target_dag_id"),
            }

            try:
                job_id = await self._submit_run_request(sensor, run_request)
                if job_id:
                    new_run_ids.append(job_id)
            except Exception as e:
                logger.error(f"Failed to recover run request: {e}")

        # Update the tick to SUCCESS or FAILED
        status = TickStatus.SUCCESS if new_run_ids else TickStatus.FAILED
        error_message = None if new_run_ids else "Recovery failed - no jobs submitted"

        await self._storage.record_tick(
            sensor_id=sensor_id,
            status=status,
            run_ids=new_run_ids,
            error_message=error_message,
            tick_id=tick_id,
        )

        logger.info(
            f"Recovered tick {tick_id}: status={status.value}, "
            f"jobs={len(new_run_ids)}"
        )

    # =========================================================================
    # STATUS & METRICS
    # =========================================================================

    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._started

    def get_stats(self) -> Dict[str, int]:
        """Get current statistics."""
        return self._stats.copy()
