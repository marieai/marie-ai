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
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

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
    from marie.scheduler.models import WorkInfo
    from marie.scheduler.services import NotificationService

logger = MarieLogger("SensorWorker")

_ENVELOPE_KEYS = (
    "project_id",
    "ref_id",
    "ref_type",
    "uri",
    "tenant_id",
    "index_id",
    "index_name",
    "source_id",
    'item_id',
    'kb_indexing_dag_id',
    'kb_attempt_id',
    'owner_type',
    'owner_id',
    "run_params",
    'storage_ref',
    'remote_source',
    'fabric_group_id',
    'organization_id',
    'workspace_id',
    'user_id',
    'input_ref',
    'upstream',
)


def sensor_targets_fabric(sensor: Dict[str, Any], fabric_group_id: str | None) -> bool:
    config = sensor.get('config') or {}
    target = config.get('fabric_group_id') or (config.get('storage_ref') or {}).get(
        'fabric_group_id'
    )
    return not target or target == fabric_group_id


def sensor_configuration_digest(sensor: dict[str, Any]) -> str:
    value = [
        sensor.get('config') or {},
        sensor.get('target_job_name'),
        sensor.get('target_dag_id'),
    ]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def build_work_info(
    run_request: RunRequest,
    sensor_id: str,
    sensor_name: str,
    job_name: str,
    dag_id: Optional[str],
    reserved_id: str | None = None,
) -> "WorkInfo":
    """Build a WorkInfo for a sensor-triggered job submission.

    Mirrors the defaults used by marie_gateway.handle_job_submit_command
    for start_after/expire_in_seconds/keep_until so sensor-submitted jobs
    satisfy the same scheduler contract as gateway-submitted jobs.
    """
    from marie.scheduler.models import WorkInfo
    from marie.scheduler.state import WorkState

    now = datetime.now(timezone.utc)
    item_execution_id = None
    if (run_request.run_config or {}).get('ref_type') == 'kb_item':
        from marie.kb.item_state import item_indexing_run

        _, item_execution_id = item_indexing_run(run_request.run_config)
    return WorkInfo(
        id=item_execution_id
        or (
            str(uuid5(NAMESPACE_URL, f"marie:sensor:{sensor_id}:{run_request.run_key}"))
            if run_request.run_key
            else (reserved_id or str(uuid4()))
        ),
        name=job_name,
        dag_id=dag_id,
        priority=run_request.priority,
        routing_fabric_group_id=(run_request.run_config or {}).get('fabric_group_id'),
        routing_request_source='workflow',
        data=build_job_data(run_request, sensor_id, sensor_name),
        state=WorkState.CREATED,
        retry_limit=3,
        retry_delay=2,
        retry_backoff=True,
        policy=(
            'replace'
            if job_name == 'kb_datasource_sync'
            else (
                'allow_duplicate'
                if run_request.tags.get('trigger') == 'run_status'
                else None
            )
        ),
        start_after=now,
        expire_in_seconds=0,
        keep_until=now + timedelta(days=2),
        soft_sla=now,
        hard_sla=now + timedelta(hours=4),
    )


def build_job_data(
    run_request: RunRequest, sensor_id: str, sensor_name: str
) -> Dict[str, Any]:
    """Job data payload: legacy run_config + canonical metadata envelope (spec A0.2)."""
    rc = run_request.run_config or {}
    metadata: Dict[str, Any] = {k: rc[k] for k in _ENVELOPE_KEYS if k in rc}
    metadata["planner"] = rc.get("planner", run_request.job_name)
    if "project_id" not in metadata and "tenant_id" in rc:
        metadata["project_id"] = rc["tenant_id"]
    submission_sensor_id = sensor_id
    if rc.get('ref_type') == 'kb_item':
        from marie.kb.item_state import item_indexing_run

        scope, _ = item_indexing_run(rc)
        submission_sensor_id = str(
            uuid5(NAMESPACE_URL, f'marie:kb-item:{scope.values()}')
        )
    return {
        "run_config": rc,
        "metadata": metadata,
        **run_request.tags,
        "sensor_id": submission_sensor_id,
        'source_sensor_id': sensor_id,
        "sensor_name": sensor_name,
        "run_key": run_request.run_key,
    }


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
        queue_config = config.get('llm_queue') or {}
        scheduler_config = config.get('scheduler') or {}
        identities = [
            config.get('fabric_group_id'),
            queue_config.get('fabric_group_id'),
            scheduler_config.get('fabric_group_id'),
            (queue_config.get('scheduler') or {}).get('fabric_group_id'),
            os.getenv('LLM_QUEUE_FABRIC_GROUP_ID'),
        ]
        self._fabric_group_id = None
        if any(identities):
            from marie.engine.llm_queue.config import resolve_fabric_id

            self._fabric_group_id = resolve_fabric_id(*identities)
            self._resources['fabric_group_id'] = self._fabric_group_id

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
        self._resources["postgres_pool"] = storage.pool

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

        sensors = [
            sensor
            for sensor in await self._storage.get_active_sensors()
            if sensor_targets_fabric(sensor, self._fabric_group_id)
        ]

        if not sensors:
            logger.debug("No active sensors to evaluate")
            return

        logger.debug(f"Evaluating {len(sensors)} active sensors")

        eligible = [sensor for sensor in sensors if self._should_evaluate(sensor)]
        limit = self._settings.max_sensors_per_cycle
        concurrency = self._settings.max_concurrent_evaluations

        async def evaluate(sensor: dict[str, Any]) -> None:
            if self._shutdown_event.is_set():
                return
            try:
                await self._evaluate_sensor_with_tick(sensor)
                self._stats["sensors_evaluated"] += 1
            except Exception:
                logger.exception(f"Error evaluating sensor {sensor.get('name')}")
                self._stats["errors"] += 1

        for offset in range(0, min(len(eligible), limit), concurrency):
            await asyncio.gather(
                *(
                    evaluate(sensor)
                    for sensor in eligible[offset : min(offset + concurrency, limit)]
                )
            )

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
        async with self._storage.sensor_lock(sensor['id']) as acquired:
            if not acquired:
                return
            current = await self._storage.get_sensor(sensor['id'])
            if (
                not current
                or current['status'] != 'active'
                or not sensor_targets_fabric(current, self._fabric_group_id)
            ):
                return
            if current.get('config', {}).get('kbSourceId') or current.get(
                'config', {}
            ).get('kbIndexId'):
                await self._evaluate_locked_sensor(current)
                return
            pending = await self._storage.get_pending_tick(current['id'])
            if pending:
                digest = (pending.get('trigger_payload') or {}).get(
                    'configuration_digest'
                )
                if digest and digest != sensor_configuration_digest(current):
                    # Reconcile accepted work without submitting stale configuration.
                    accepted_ids = []
                    for request_data, reserved_id in zip(
                        pending.get('run_requests') or [],
                        pending.get('reserved_run_ids') or [],
                        strict=True,
                    ):
                        request = RunRequest.from_dict(request_data)
                        receipt = (
                            await self._storage.get_run_key_job(
                                current['id'], request.run_key
                            )
                            if request.run_key
                            else None
                        )
                        if not receipt:
                            if not self._job_scheduler:
                                raise RuntimeError(
                                    'Scheduler unavailable to reconcile changed configuration'
                                )
                            state = await self._job_scheduler.get_sensor_job_state(
                                reserved_id, current['id'], request.run_key, None
                            )
                            if state is not None:
                                receipt = reserved_id
                                if request.run_key:
                                    await self._storage.record_run_key(
                                        current['id'], request.run_key, receipt
                                    )
                        if receipt:
                            accepted_ids.append(receipt)
                    await self._storage.record_tick(
                        sensor_id=current['id'],
                        status=TickStatus.SKIPPED,
                        tick_id=pending['id'],
                        skip_reason='Configuration changed before admission completed',
                        run_ids=accepted_ids,
                        expected_updated_at=current.get('updated_at'),
                    )
                    return
                await self._recover_tick(
                    {
                        **pending,
                        'sensor_name': current['name'],
                        'target_job_name': current.get('target_job_name'),
                        'target_dag_id': current.get('target_dag_id'),
                        'sensor_updated_at': current.get('updated_at'),
                    }
                )
                return
            await self._evaluate_locked_sensor(current)

    async def _evaluate_locked_sensor(self, sensor: Dict[str, Any]) -> None:
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
            if sensor.get('config', {}).get('kbSourceId') or sensor.get(
                'config', {}
            ).get('kbIndexId'):
                raise RuntimeError(
                    'Datasource execution is not ready; F10b target admission is required'
                )
            if sensor_type == SensorType.SCHEDULE and sensor.get('last_run_key'):
                previous_id = await self._storage.get_run_key_job(
                    sensor_id, sensor['last_run_key']
                )
                if previous_id:
                    if not self._job_scheduler:
                        raise RuntimeError(
                            'No job scheduler available for sensor admission'
                        )
                    state = await self._job_scheduler.get_sensor_job_state(
                        previous_id,
                        sensor_id,
                        sensor['last_run_key'],
                        None,
                    )
                    if state in ('created', 'retry', 'active'):
                        await self._storage.record_tick(
                            sensor_id=sensor_id,
                            status=TickStatus.SKIPPED,
                            skip_reason='Previous scheduled job is pending or running',
                        )
                        return
            # Build evaluation context
            context = await self._build_context(sensor)

            # Get evaluator for this sensor type (and subtype, if configured)
            subtype = sensor.get("config", {}).get("subtype")
            evaluator_class = self._registry.get_evaluator(sensor_type, subtype=subtype)
            evaluator = evaluator_class(sensor)

            # Create STARTED tick for crash recovery (if we'll submit jobs)
            # For now, we evaluate first to see if jobs will be submitted

            # Evaluate the sensor
            result = await evaluator.evaluate(context)
            for request in result.run_requests:
                target = (
                    sensor.get('config', {}).get('fabric_group_id')
                    or request.run_config.get('fabric_group_id')
                    or self._fabric_group_id
                )
                if target:
                    request.run_config['fabric_group_id'] = target

            # Process result
            if result.has_run_requests():
                # Create STARTED tick before job submission
                reserved = [
                    build_work_info(
                        rr,
                        sensor_id,
                        sensor_name,
                        rr.job_name or sensor.get("target_job_name"),
                        rr.dag_id or sensor.get("target_dag_id"),
                    ).id
                    for rr in result.run_requests
                ]
                tick_id = await self._storage.create_tick(
                    sensor_id=sensor_id,
                    status=TickStatus.STARTED,
                    run_requests=[rr.to_dict() for rr in result.run_requests],
                    reserved_run_ids=reserved,
                    cursor=result.cursor,
                    trigger_payload={
                        "configuration_digest": sensor_configuration_digest(sensor)
                    },
                )

                # Submit jobs
                for run_request, reserved_id in zip(
                    result.run_requests, reserved, strict=True
                ):
                    job_id = await self._submit_run_request(
                        sensor, run_request, reserved_id
                    )
                    if job_id:
                        run_ids.append(job_id)
                        self._stats["jobs_submitted"] += 1

                status = TickStatus.SUCCESS if run_ids else TickStatus.SKIPPED
                if not run_ids:
                    skip_reason = 'Items changed before admission'
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
            expected_updated_at=sensor.get("updated_at"),
        )
        self._stats["ticks_recorded"] += 1

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
            resources={**self._resources, "postgres_pool": self._storage},
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
        return await self._storage.get_completed_workflows(
            sensor, limit=self._settings.max_events_per_tick
        )

    async def _submit_run_request(
        self,
        sensor: Dict[str, Any],
        run_request: RunRequest,
        reserved_id: str | None = None,
    ) -> str | None:
        sensor_id = sensor['id']
        job_name = run_request.job_name or sensor.get('target_job_name')
        dag_id = run_request.dag_id or sensor.get('target_dag_id')
        if not job_name:
            raise ValueError('Sensor has no target job queue configured')
        item_store = None
        item_scope = None
        item_execution = None
        if (run_request.run_config or {}).get('ref_type') == 'kb_item':
            from marie.kb.item_state import KbItemStore, item_indexing_run

            item_scope, item_execution = item_indexing_run(run_request.run_config)
            item_store = KbItemStore(self._storage.pool)
            current = await item_store.get(item_scope)
            if (
                current is None
                or current['state'] == 'removed'
                or str(current['indexing_dag_id']) != item_execution
            ):
                return None
            if current.get('input_ref') != run_request.run_config.get('input_ref'):
                return None
            if str(current['indexed_dag_id']) == item_execution:
                return item_execution
            if current['state'] != 'pending':
                return None
        if run_request.run_key:
            receipt = await self._storage.get_run_key_job(
                sensor_id, run_request.run_key
            )
            if receipt:
                if item_store is not None:
                    state = (
                        await self._job_scheduler.get_sensor_job_state(
                            receipt,
                            build_job_data(
                                run_request, sensor_id, sensor.get('name', '')
                            )['sensor_id'],
                            run_request.run_key,
                            job_name,
                        )
                        if self._job_scheduler is not None
                        else None
                    )
                    if receipt != item_execution or state is None:
                        message = 'KB scheduler receipt is unavailable; retry the item explicitly'
                        await item_store.fail(item_scope, item_execution, message)
                        raise RuntimeError(message)
                    await item_store.mark_admitted(item_scope, item_execution)
                return receipt
        if not self._job_scheduler:
            raise RuntimeError('No job scheduler available for sensor admission')
        work_info = build_work_info(
            run_request,
            sensor_id,
            sensor.get('name', ''),
            job_name,
            dag_id,
            reserved_id,
        )
        job_id = await self._job_scheduler.submit_sensor_job(work_info)
        if job_id != work_info.id:
            raise RuntimeError('Scheduler did not confirm durable sensor admission')
        if run_request.run_key:
            await self._storage.record_run_key(sensor_id, run_request.run_key, job_id)
        if item_store is not None:
            await item_store.mark_admitted(item_scope, item_execution)
        return job_id

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
            sensor = await self._storage.get_sensor(tick['sensor_id'])
            if sensor and sensor_targets_fabric(sensor, self._fabric_group_id):
                await self._evaluate_sensor_with_tick(sensor)

    async def _recover_tick(self, tick: Dict[str, Any]) -> None:
        """
        Recover a single stuck STARTED tick.

        Attempts to resume job submission for any reserved runs
        that weren't submitted.
        """
        for request in tick.get('run_requests') or []:
            if not sensor_targets_fabric(
                {'config': request.get('run_config') or {}}, self._fabric_group_id
            ):
                return
        tick_id = tick.get("id")
        sensor_id = tick.get("sensor_id")
        sensor_name = tick.get("sensor_name", "unknown")

        logger.info(f"Recovering tick {tick_id} for sensor {sensor_name}")

        run_requests = tick.get("run_requests", [])
        reserved_run_ids = tick.get("reserved_run_ids", [])
        existing_run_ids = tick.get("run_ids", [])

        new_run_ids = list(existing_run_ids)
        errors = []
        for i, data in enumerate(run_requests):
            request = RunRequest.from_dict(data)
            reserved_id = reserved_run_ids[i] if i < len(reserved_run_ids) else None
            sensor = {
                'id': sensor_id,
                'name': sensor_name,
                'target_job_name': tick.get('target_job_name'),
                'target_dag_id': tick.get('target_dag_id'),
            }
            try:
                job_id = await self._submit_run_request(sensor, request, reserved_id)
                if job_id and job_id not in new_run_ids:
                    new_run_ids.append(job_id)
            except Exception as error:
                errors.append(str(error))
        status = (
            TickStatus.FAILED
            if errors or not run_requests
            else TickStatus.SUCCESS
            if new_run_ids
            else TickStatus.SKIPPED
        )
        await self._storage.record_tick(
            sensor_id=sensor_id,
            status=status,
            cursor=tick.get('cursor') if status == TickStatus.SUCCESS else None,
            run_ids=new_run_ids,
            error_message='; '.join(errors) or None,
            skip_reason='Items changed before admission'
            if status == TickStatus.SKIPPED
            else None,
            tick_id=tick_id,
            expected_updated_at=tick.get("sensor_updated_at"),
        )

        logger.info(
            f"Recovered tick {tick_id}: status={status.value}, jobs={len(new_run_ids)}"
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
