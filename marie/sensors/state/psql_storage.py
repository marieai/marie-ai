"""
PostgreSQL implementation of sensor state storage.

This implementation uses asyncpg for async database operations
and follows the patterns established in the scheduler repository.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from marie.logging_core.logger import MarieLogger
from marie.sensors.state.storage import SensorStateStorage
from marie.sensors.types import SensorStatus, SensorType, TickStatus

logger = MarieLogger("PostgreSQLSensorStorage")


class PostgreSQLSensorStorage(SensorStateStorage):
    """
    PostgreSQL storage implementation for sensor state.

    Uses connection pool from asyncpg for efficient async operations.
    All sensor data is stored in the marie_scheduler schema.
    """

    _instance: Optional["PostgreSQLSensorStorage"] = None

    def __init__(self, pool, schema: str = "marie_scheduler"):
        """
        Initialize the storage with a connection pool.

        :param pool: asyncpg connection pool
        :param schema: Database schema name
        """
        self._pool = pool
        self._schema = schema

    @classmethod
    def get_instance(cls) -> "PostgreSQLSensorStorage":
        """Get the singleton instance."""
        if cls._instance is None:
            raise RuntimeError(
                "PostgreSQLSensorStorage not initialized. " "Call initialize() first."
            )
        return cls._instance

    @classmethod
    def initialize(
        cls, pool, schema: str = "marie_scheduler"
    ) -> "PostgreSQLSensorStorage":
        """
        Initialize the singleton instance.

        :param pool: asyncpg connection pool
        :param schema: Database schema name
        :return: The storage instance
        """
        cls._instance = cls(pool, schema)
        return cls._instance

    # =========================================================================
    # SENSOR OPERATIONS
    # =========================================================================

    async def get_sensor(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get a sensor by its internal ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM {self._schema}.sensor
                WHERE id = $1
                """,
                UUID(sensor_id),
            )
            return self._row_to_dict(row)

    async def get_sensor_by_external_id(
        self, external_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a sensor by its external ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM {self._schema}.sensor
                WHERE external_id = $1
                """,
                UUID(external_id),
            )
            return self._row_to_dict(row)

    async def get_active_sensors(self) -> List[Dict[str, Any]]:
        """Get all active sensors for daemon evaluation."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._schema}.sensor
                WHERE status = 'active'
                ORDER BY last_tick_at ASC NULLS FIRST
                """
            )
            return [self._row_to_dict(row) for row in rows]

    async def get_active_sensors_by_type(
        self, sensor_type: SensorType
    ) -> List[Dict[str, Any]]:
        """Get active sensors of a specific type."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._schema}.sensor
                WHERE status = 'active' AND sensor_type = $1
                ORDER BY last_tick_at ASC NULLS FIRST
                """,
                sensor_type.value,
            )
            return [self._row_to_dict(row) for row in rows]

    async def create_sensor(self, sensor_data: Dict[str, Any]) -> str:
        """Create a new sensor."""
        async with self._pool.acquire() as conn:
            sensor_id = await conn.fetchval(
                f"""
                INSERT INTO {self._schema}.sensor (
                    external_id, name, sensor_type, config,
                    target_job_name, target_dag_id, status,
                    minimum_interval_seconds
                )
                VALUES ($1, $2, $3::marie_scheduler.sensor_type, $4, $5, $6,
                        $7::marie_scheduler.sensor_status, $8)
                RETURNING id
                """,
                UUID(sensor_data["external_id"]),
                sensor_data["name"],
                sensor_data["sensor_type"],
                json.dumps(sensor_data.get("config", {})),
                sensor_data.get("target_job_name"),
                (
                    UUID(sensor_data["target_dag_id"])
                    if sensor_data.get("target_dag_id")
                    else None
                ),
                sensor_data.get("status", "inactive"),
                sensor_data.get("minimum_interval_seconds", 30),
            )
            return str(sensor_id)

    async def update_sensor(
        self, sensor_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a sensor's configuration."""
        # Build dynamic update query
        set_clauses = []
        values = []
        param_idx = 1

        for key, value in updates.items():
            if key in (
                "name",
                "sensor_type",
                "target_job_name",
                "minimum_interval_seconds",
            ):
                set_clauses.append(f"{key} = ${param_idx}")
                values.append(value)
                param_idx += 1
            elif key == "config":
                set_clauses.append(f"config = ${param_idx}")
                values.append(json.dumps(value))
                param_idx += 1
            elif key == "target_dag_id":
                set_clauses.append(f"target_dag_id = ${param_idx}")
                values.append(UUID(value) if value else None)
                param_idx += 1

        if not set_clauses:
            return await self.get_sensor(sensor_id)

        set_clauses.append("updated_at = NOW()")
        values.append(UUID(sensor_id))

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE {self._schema}.sensor
                SET {', '.join(set_clauses)}
                WHERE id = ${param_idx}
                RETURNING *
                """,
                *values,
            )
            return self._row_to_dict(row)

    async def delete_sensor(self, sensor_id: str) -> bool:
        """Delete a sensor and all related data."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self._schema}.sensor
                WHERE id = $1
                """,
                UUID(sensor_id),
            )
            return result == "DELETE 1"

    async def update_sensor_status(self, sensor_id: str, status: SensorStatus) -> bool:
        """Update a sensor's operational status."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                UPDATE {self._schema}.sensor
                SET status = $2::marie_scheduler.sensor_status, updated_at = NOW()
                WHERE id = $1
                """,
                UUID(sensor_id),
                status.value,
            )
            return result == "UPDATE 1"

    # =========================================================================
    # CURSOR OPERATIONS
    # =========================================================================

    async def get_cursor(self, sensor_id: str) -> Optional[str]:
        """Get the current cursor for a sensor."""
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                f"""
                SELECT cursor FROM {self._schema}.sensor
                WHERE id = $1
                """,
                UUID(sensor_id),
            )

    async def set_cursor(self, sensor_id: str, cursor: str) -> None:
        """Set the cursor for a sensor."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {self._schema}.sensor
                SET cursor = $2, updated_at = NOW()
                WHERE id = $1
                """,
                UUID(sensor_id),
                cursor,
            )

    # =========================================================================
    # TICK OPERATIONS
    # =========================================================================

    async def create_tick(
        self,
        sensor_id: str,
        status: TickStatus,
        run_requests: Optional[List[Dict[str, Any]]] = None,
        reserved_run_ids: Optional[List[str]] = None,
    ) -> str:
        """Create a new tick record (for STARTED state)."""
        async with self._pool.acquire() as conn:
            tick_id = await conn.fetchval(
                f"""
                INSERT INTO {self._schema}.sensor_tick (
                    sensor_id, status, run_requests, reserved_run_ids
                )
                VALUES ($1, $2::marie_scheduler.tick_status, $3, $4)
                RETURNING id
                """,
                UUID(sensor_id),
                status.value,
                json.dumps(run_requests) if run_requests else None,
                [UUID(rid) for rid in reserved_run_ids] if reserved_run_ids else [],
            )
            return str(tick_id)

    async def record_tick(
        self,
        sensor_id: str,
        status: TickStatus,
        cursor: Optional[str] = None,
        run_ids: Optional[List[str]] = None,
        skip_reason: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        trigger_payload: Optional[Dict[str, Any]] = None,
        tick_id: Optional[str] = None,
    ) -> str:
        """
        Record a sensor tick and update sensor state atomically.

        This is the CRITICAL method that ensures:
        1. Tick is always recorded (success, skipped, or failed)
        2. Sensor state is updated atomically
        3. Failure tracking is maintained
        """
        now = datetime.now(timezone.utc)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                if tick_id:
                    # Update existing tick (STARTED -> final)
                    await conn.execute(
                        f"""
                        UPDATE {self._schema}.sensor_tick
                        SET status = $2::marie_scheduler.tick_status,
                            cursor = $3,
                            run_ids = $4,
                            skip_reason = $5,
                            error_message = $6,
                            completed_at = $7,
                            duration_ms = $8,
                            trigger_payload = $9
                        WHERE id = $1
                        """,
                        UUID(tick_id),
                        status.value,
                        cursor,
                        [UUID(rid) for rid in run_ids] if run_ids else [],
                        skip_reason,
                        error_message,
                        now,
                        duration_ms,
                        json.dumps(trigger_payload) if trigger_payload else None,
                    )
                    result_tick_id = tick_id
                else:
                    # Create new tick
                    result_tick_id = await conn.fetchval(
                        f"""
                        INSERT INTO {self._schema}.sensor_tick (
                            sensor_id, status, cursor, run_ids,
                            skip_reason, error_message, started_at,
                            completed_at, duration_ms, trigger_payload
                        )
                        VALUES ($1, $2::marie_scheduler.tick_status, $3, $4, $5, $6, $7, $7, $8, $9)
                        RETURNING id
                        """,
                        UUID(sensor_id),
                        status.value,
                        cursor,
                        [UUID(rid) for rid in run_ids] if run_ids else [],
                        skip_reason,
                        error_message,
                        now,
                        duration_ms,
                        json.dumps(trigger_payload) if trigger_payload else None,
                    )

                # Update sensor state atomically
                if status == TickStatus.SUCCESS:
                    # Success: reset failure_count, update last_tick_at
                    await conn.execute(
                        f"""
                        UPDATE {self._schema}.sensor
                        SET last_tick_at = $2,
                            failure_count = 0,
                            last_error = NULL,
                            updated_at = $2
                        WHERE id = $1
                        """,
                        UUID(sensor_id),
                        now,
                    )
                elif status == TickStatus.FAILED:
                    # Failure: increment failure_count, record error
                    await conn.execute(
                        f"""
                        UPDATE {self._schema}.sensor
                        SET last_tick_at = $2,
                            failure_count = failure_count + 1,
                            last_error = $3,
                            updated_at = $2
                        WHERE id = $1
                        """,
                        UUID(sensor_id),
                        now,
                        error_message,
                    )
                else:
                    # Skipped/Started: just update last_tick_at
                    await conn.execute(
                        f"""
                        UPDATE {self._schema}.sensor
                        SET last_tick_at = $2, updated_at = $2
                        WHERE id = $1
                        """,
                        UUID(sensor_id),
                        now,
                    )

                return str(result_tick_id)

    async def get_tick(self, tick_id: str) -> Optional[Dict[str, Any]]:
        """Get a tick by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM {self._schema}.sensor_tick
                WHERE id = $1
                """,
                UUID(tick_id),
            )
            return self._row_to_dict(row)

    async def get_ticks(
        self,
        sensor_id: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[TickStatus] = None,
    ) -> List[Dict[str, Any]]:
        """Get tick history for a sensor."""
        async with self._pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    f"""
                    SELECT * FROM {self._schema}.sensor_tick
                    WHERE sensor_id = $1 AND status = $2::marie_scheduler.tick_status
                    ORDER BY started_at DESC
                    LIMIT $3 OFFSET $4
                    """,
                    UUID(sensor_id),
                    status.value,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT * FROM {self._schema}.sensor_tick
                    WHERE sensor_id = $1
                    ORDER BY started_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    UUID(sensor_id),
                    limit,
                    offset,
                )
            return [self._row_to_dict(row) for row in rows]

    async def get_started_ticks(
        self, threshold_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get STARTED ticks older than threshold (for crash recovery)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT st.*, s.name as sensor_name, s.target_job_name, s.target_dag_id
                FROM {self._schema}.sensor_tick st
                JOIN {self._schema}.sensor s ON st.sensor_id = s.id
                WHERE st.status = 'started'
                  AND st.started_at < NOW() - ($1 || ' hours')::INTERVAL
                ORDER BY st.started_at ASC
                """,
                str(threshold_hours),
            )
            return [self._row_to_dict(row) for row in rows]

    # =========================================================================
    # RUN KEY OPERATIONS
    # =========================================================================

    async def has_run_key(self, sensor_id: str, run_key: str) -> bool:
        """Check if a run_key has already been processed."""
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                f"""
                SELECT EXISTS(
                    SELECT 1 FROM {self._schema}.sensor_run_key
                    WHERE sensor_id = $1 AND run_key = $2
                )
                """,
                UUID(sensor_id),
                run_key,
            )

    async def record_run_key(
        self, sensor_id: str, run_key: str, job_id: Optional[str] = None
    ) -> None:
        """Record a processed run_key for idempotency."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._schema}.sensor_run_key (sensor_id, run_key, job_id)
                VALUES ($1, $2, $3)
                ON CONFLICT (sensor_id, run_key) DO NOTHING
                """,
                UUID(sensor_id),
                run_key,
                UUID(job_id) if job_id else None,
            )

    # =========================================================================
    # EVENT LOG OPERATIONS
    # =========================================================================

    async def get_pending_events(
        self,
        sensor_external_id: Optional[str] = None,
        source: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get pending events from the event_log."""
        conditions = []
        values = []
        param_idx = 1

        if cursor:
            conditions.append(f"event_log_id > ${param_idx}")
            values.append(int(cursor))
            param_idx += 1

        if sensor_external_id:
            conditions.append(f"sensor_external_id = ${param_idx}")
            values.append(UUID(sensor_external_id))
            param_idx += 1

        if source:
            conditions.append(f"source = ${param_idx}")
            values.append(source)
            param_idx += 1

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        values.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self._schema}.event_log
                {where_clause}
                ORDER BY event_log_id ASC
                LIMIT ${param_idx}
                """,
                *values,
            )
            return [self._row_to_dict(row) for row in rows]

    async def insert_event(
        self,
        source: str,
        payload: Dict[str, Any],
        sensor_external_id: Optional[str] = None,
        sensor_type: Optional[SensorType] = None,
        routing_key: Optional[str] = None,
        event_key: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert an event into the event_log."""
        async with self._pool.acquire() as conn:
            event_id = await conn.fetchval(
                f"""
                INSERT INTO {self._schema}.event_log (
                    source, sensor_type, sensor_external_id,
                    routing_key, event_key, payload, headers
                )
                VALUES ($1, $2::marie_scheduler.sensor_type, $3, $4, $5, $6, $7)
                RETURNING event_id
                """,
                source,
                sensor_type.value if sensor_type else None,
                UUID(sensor_external_id) if sensor_external_id else None,
                routing_key,
                event_key,
                json.dumps(payload),
                json.dumps(headers or {}),
            )
            return str(event_id)

    # =========================================================================
    # WEBHOOK REGISTRATION
    # =========================================================================

    async def get_webhook_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """Get webhook registration by path."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT wr.*, s.external_id as sensor_external_id
                FROM {self._schema}.webhook_registration wr
                JOIN {self._schema}.sensor s ON wr.sensor_id = s.id
                WHERE wr.path = $1
                """,
                path,
            )
            return self._row_to_dict(row)

    async def create_webhook_registration(
        self,
        sensor_id: str,
        path: str,
        methods: List[str],
        auth_type: Optional[str] = None,
        auth_secret: Optional[str] = None,
    ) -> str:
        """Create a webhook registration."""
        async with self._pool.acquire() as conn:
            reg_id = await conn.fetchval(
                f"""
                INSERT INTO {self._schema}.webhook_registration (
                    sensor_id, path, methods, auth_type, auth_secret
                )
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                UUID(sensor_id),
                path,
                methods,
                auth_type,
                auth_secret,
            )
            return str(reg_id)

    # =========================================================================
    # MAINTENANCE OPERATIONS
    # =========================================================================

    async def cleanup_old_ticks(self, retention_days: int = 30) -> int:
        """Delete old ticks for retention."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                f"SELECT {self._schema}.cleanup_old_ticks($1)",
                retention_days,
            )
            return result or 0

    async def cleanup_old_events(self, retention_days: int = 14) -> int:
        """Delete old events for retention."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                f"SELECT {self._schema}.cleanup_old_events($1)",
                retention_days,
            )
            return result or 0

    async def cleanup_old_run_keys(self, retention_days: int = 30) -> int:
        """Delete old run keys for retention."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                f"SELECT {self._schema}.cleanup_old_run_keys($1)",
                retention_days,
            )
            return result or 0

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _row_to_dict(self, row) -> Optional[Dict[str, Any]]:
        """Convert an asyncpg Record to a dictionary."""
        if row is None:
            return None

        result = dict(row)

        # Convert UUIDs to strings
        for key, value in result.items():
            if isinstance(value, UUID):
                result[key] = str(value)
            elif key in (
                "config",
                "payload",
                "headers",
                "run_requests",
                "trigger_payload",
            ):
                # Parse JSON fields
                if isinstance(value, str):
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        pass
            elif key == "run_ids" and value:
                # Convert UUID array to string array
                result[key] = [str(v) for v in value]
            elif key == "reserved_run_ids" and value:
                result[key] = [str(v) for v in value]

        return result
