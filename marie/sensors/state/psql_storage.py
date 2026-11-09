"""
PostgreSQL implementation of sensor state storage.

This implementation uses psycopg 3 for async database operations
and follows the patterns established in the scheduler repository.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
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

    Uses a psycopg 3 async connection pool for efficient async operations.
    All sensor data is stored in the marie_scheduler schema.
    """

    _instance: Optional["PostgreSQLSensorStorage"] = None

    def __init__(self, pool, schema: str = "marie_scheduler"):
        """
        Initialize the storage with a connection pool.

        :param pool: psycopg 3 async connection pool
        :param schema: Database schema name
        """
        self._pool = pool
        self._schema = schema
        self._locked_connection: ContextVar[Any] = ContextVar(
            "sensor_connection", default=None
        )

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        connection = self._locked_connection.get()
        if connection is not None:
            yield connection
        else:
            async with self._pool.acquire() as connection:
                yield connection

    async def fetchrow(self, query: str, *args: Any) -> Any:
        """Run a sensor resource lookup on the evaluation's existing connection."""
        async with self._connection() as connection:
            return await connection.fetchrow(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """Read registered work on the evaluation's existing connection."""
        async with self._connection() as connection:
            return await connection.fetch(query, *args)

    @asynccontextmanager
    async def sensor_lock(self, sensor_id: str) -> AsyncIterator[bool]:
        """Serialize evaluation/recovery across workers without waiting on another worker."""
        async with self._pool.acquire() as connection:
            key = str(UUID(sensor_id))
            locked = await connection.fetchval(
                "SELECT pg_try_advisory_lock(hashtextextended($1, 72010))", key
            )
            if not locked:
                yield False
                return
            token = self._locked_connection.set(connection)
            try:
                yield True
            finally:
                self._locked_connection.reset(token)
                await connection.execute(
                    "SELECT pg_advisory_unlock(hashtextextended($1, 72010))", key
                )

    async def get_pending_tick(self, sensor_id: str) -> dict[str, Any] | None:
        async with self._connection() as conn:
            row = await conn.fetchrow(
                f"""SELECT * FROM {self._schema}.sensor_tick
                    WHERE sensor_id=$1 AND status IN ('started', 'failed')
                        AND run_requests IS NOT NULL
                        AND started_at > COALESCE((
                            SELECT max(started_at) FROM {self._schema}.sensor_tick
                            WHERE sensor_id=$1 AND status IN ('success', 'skipped')
                        ), '-infinity'::timestamptz)
                    ORDER BY started_at DESC LIMIT 1""",
                UUID(sensor_id),
            )
            return self._row_to_dict(row)

    async def repository_schedule_active(self, sensor: dict[str, Any]) -> bool:
        """Check the existing native source/schedule binding before admission."""
        config = sensor['config']
        run = config['run_config']
        async with self._connection() as connection:
            return bool(
                await connection.fetchval(
                    '''SELECT 1 FROM marie_studio.kb_source s
                   JOIN marie_studio.kb_index k ON k.id=s.index_id
                   JOIN marie_studio.tenant t ON t.id=k.tenant_id
                   JOIN marie_studio.git_repositories r ON r.id=(s.dataset_inputs->>'repositoryId')::uuid
                   WHERE s.id=$1 AND k.id=$2 AND t.id=$3 AND t.organization_id=$4 AND k.workspace_id=$5
                     AND s.trigger_config_id=$6 AND s.sync_mode='scheduled' AND s.source_type='repository'
                     AND s.status <> 'disabled' AND t.status='active' AND r.is_enabled AND r.git_enabled
                     AND NOT COALESCE((k.metadata->>'removalPending')::boolean,false)
                     AND NOT COALESCE((s.metadata->>'removalPending')::boolean,false)
                     AND (r.visibility='system' OR (r.visibility='organization' AND r.organization_id=t.organization_id))''',
                    UUID(config['kbSourceId']),
                    UUID(config['kbIndexId']),
                    UUID(run['tenant_id']),
                    UUID(run['organization_id']),
                    UUID(run['workspace_id']),
                    UUID(sensor['external_id']),
                )
            )

    async def get_run_key_job(self, sensor_id: str, run_key: str) -> str | None:
        async with self._connection() as conn:
            job = await conn.fetchval(
                f"SELECT job_id FROM {self._schema}.sensor_run_key WHERE sensor_id=$1 AND run_key=$2",
                UUID(sensor_id),
                run_key,
            )
            return str(job) if job else None

    async def get_completed_workflows(
        self, sensor: Dict[str, Any], *, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Read unacknowledged terminal DAG occurrences, including late commits."""
        from marie.constants import KV_NAMESPACE_JOB
        from marie.job.common import JOB_INFO_KEY_PREFIX
        from marie.sensors.definitions.run_status_sensor import RunStatusSensor

        if not 1 <= limit <= 1000:
            raise ValueError('Workflow completion limit must be between 1 and 1000')
        evaluator = RunStatusSensor(sensor)
        evaluator.validate_config()
        patterns = [
            pattern.replace('\\', '\\\\')
            .replace('%', r'\%')
            .replace('_', r'\_')
            .replace('*', '%')
            for pattern in evaluator.monitored_job_names
        ]
        scope = evaluator.source_scope
        kb_handoff = sensor.get('config', {}).get('subtype') == 'kb_workflow'
        destinations = (
            f"""
            AND EXISTS (
                SELECT 1 FROM {self._schema}.kb_item_state item
                WHERE item.input_ref->>'kind'='upstream' AND item.input_ref->>'dag_id'=t.id::text
                  AND item.state <> 'removed' AND (
                    item.input_ref->>'completion_id' IS DISTINCT FROM t.history_id::text OR
                    (item.state='pending' AND item.admitted_at IS NULL))
            )
        """
            if kb_handoff
            else ''
        )
        async with self._connection() as connection:
            rows = await connection.fetch(
                f"""
                WITH terminal AS (
                    SELECT DISTINCT ON (h.id, h.completed_on, h.state) h.*
                    FROM {self._schema}.dag_history h
                    JOIN {self._schema}.dag d ON d.id=h.id AND d.state=h.state
                        AND d.completed_on=h.completed_on
                    WHERE h.state=ANY($2::text[]) AND h.completed_on IS NOT NULL
                        AND h.history_created_on >= COALESCE($3::timestamptz, '-infinity')
                        AND ($4::text[] IS NULL OR h.project_id=ANY($4))
                        AND (cardinality($5::text[])=0 OR COALESCE(h.planner,h.submission_name,h.name) LIKE ANY($5))
                        AND (cardinality($6::uuid[])=0 OR h.id=ANY($6))
                    ORDER BY h.id,h.completed_on,h.state,h.history_id
                )
                SELECT t.id,t.id AS dag_id,t.history_id::text AS completion_id,
                    COALESCE(t.planner,t.submission_name,t.name) AS name,
                    t.state,t.project_id,t.completed_on AS completed_at,
                    root.data,COALESCE(result.outputs->0->'output','{{}}'::jsonb) AS output,
                    COALESCE(result.outputs,'[]'::jsonb) AS outputs
                FROM terminal t
                JOIN LATERAL (
                    SELECT j.data FROM {self._schema}.job j WHERE j.dag_id=t.id
                    ORDER BY j.job_level,j.id LIMIT 1
                ) root ON TRUE
                LEFT JOIN LATERAL (
                    SELECT jsonb_agg(jsonb_build_object(
                        'node_id',j.id,'endpoint',j.data->'metadata'->>'on',
                        'output',COALESCE((kv.value->>'runtime_env_json')::jsonb->'output',j.output)
                    ) ORDER BY j.job_level DESC,j.id) AS outputs
                    FROM {self._schema}.job j
                    LEFT JOIN {self._schema}.kv_store_worker kv
                        ON kv.namespace=$11 AND kv.key=$12 || j.id::text AND NOT kv.is_deleted
                        AND kv.value->>'run_attempt_id'=j.run_attempt_id::text
                    WHERE j.dag_id=t.id
                        AND left(COALESCE(j.data->'metadata'->>'on',''),4) <> 'noop'
                        AND COALESCE((kv.value->>'runtime_env_json')::jsonb->'output',j.output) IS NOT NULL
                ) result ON TRUE
                WHERE NOT EXISTS (
                    SELECT 1 FROM {self._schema}.sensor_run_key r
                    WHERE r.sensor_id=$1 AND r.run_key='run_status:' || $1::text || ':' || t.history_id::text
                )
                    AND root.data->>'sensor_id' IS DISTINCT FROM $1::text
                    AND ($7::text IS NULL OR root.data->'metadata'->>'organization_id'=$7
                        OR (root.data->'metadata'->>'organization_id' IS NULL AND $4::text[] IS NOT NULL))
                    AND ($8::text IS NULL OR root.data->'metadata'->>'workspace_id'=$8
                        OR (root.data->'metadata'->>'workspace_id' IS NULL AND $4::text[] IS NOT NULL))
                    AND ($9::text IS NULL OR root.data->'metadata'->>'fabric_group_id'=$9)
                    {destinations}
                ORDER BY t.history_id LIMIT $10
                """,
                UUID(sensor['id']),
                sorted(evaluator.normalized_statuses),
                None if kb_handoff else sensor.get('created_at'),
                scope.get('project_ids'),
                patterns,
                [UUID(value) for value in evaluator.monitored_dag_ids],
                scope.get('organization_id'),
                scope.get('workspace_id'),
                scope.get('fabric_group_id'),
                limit,
                KV_NAMESPACE_JOB.decode(),
                JOB_INFO_KEY_PREFIX,
            )
        return [self._row_to_dict(row) for row in rows]

    @property
    def pool(self) -> Any:
        return self._pool

    @classmethod
    def get_instance(cls) -> "PostgreSQLSensorStorage":
        """Get the singleton instance."""
        if cls._instance is None:
            raise RuntimeError(
                "PostgreSQLSensorStorage not initialized. Call initialize() first."
            )
        return cls._instance

    @classmethod
    def initialize(
        cls, pool, schema: str = "marie_scheduler"
    ) -> "PostgreSQLSensorStorage":
        """
        Initialize the singleton instance.

        :param pool: psycopg 3 async connection pool
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
        async with self._connection() as conn:
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
        async with self._connection() as conn:
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
        async with self._connection() as conn:
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
        async with self._connection() as conn:
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
        async with self._connection() as conn:
            sensor_id = await conn.fetchval(
                f"""
                INSERT INTO {self._schema}.sensor (
                    external_id, name, sensor_type, config,
                    target_job_name, target_dag_id, status,
                    minimum_interval_seconds
                )
                VALUES ($1, $2, $3::marie_scheduler.trigger_type, $4, $5, $6,
                        $7::marie_scheduler.trigger_status, $8)
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
        reset_conditions = []
        values = []
        param_idx = 1

        for key, value in updates.items():
            if key in (
                "name",
                "sensor_type",
                "target_job_name",
                "minimum_interval_seconds",
            ):
                if key == 'target_job_name':
                    reset_conditions.append(
                        f"target_job_name IS DISTINCT FROM ${param_idx}::text"
                    )
                set_clauses.append(f"{key} = ${param_idx}")
                values.append(value)
                param_idx += 1
            elif key == "config":
                reset_conditions.append(f"config IS DISTINCT FROM ${param_idx}::jsonb")
                set_clauses.append(f"config = ${param_idx}")
                values.append(json.dumps(value))
                param_idx += 1
            elif key == "target_dag_id":
                reset_conditions.append(
                    f"target_dag_id IS DISTINCT FROM ${param_idx}::uuid"
                )
                set_clauses.append(f"target_dag_id = ${param_idx}")
                values.append(UUID(value) if value else None)
                param_idx += 1

        if not set_clauses:
            return await self.get_sensor(sensor_id)

        if reset_conditions:
            set_clauses.append(
                "cursor = CASE WHEN sensor_type='schedule' AND ("
                + ' OR '.join(reset_conditions)
                + ") THEN NOW()::text ELSE cursor END"
            )
        set_clauses.append("updated_at = NOW()")
        values.append(UUID(sensor_id))

        async with self._connection() as conn:
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
        async with self._connection() as conn:
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
        async with self._connection() as conn:
            result = await conn.execute(
                f"""
                UPDATE {self._schema}.sensor
                SET status = $2::marie_scheduler.trigger_status,
                    cursor = CASE WHEN sensor_type='schedule' AND cursor IS NULL THEN NOW()::text ELSE cursor END,
                    updated_at = NOW()
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
        async with self._connection() as conn:
            return await conn.fetchval(
                f"""
                SELECT cursor FROM {self._schema}.sensor
                WHERE id = $1
                """,
                UUID(sensor_id),
            )

    async def set_cursor(self, sensor_id: str, cursor: str) -> None:
        """Set the cursor for a sensor."""
        async with self._connection() as conn:
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
        cursor: str | None = None,
        trigger_payload: dict[str, Any] | None = None,
    ) -> str:
        """Create a new tick record (for STARTED state)."""
        async with self._connection() as conn:
            tick_id = await conn.fetchval(
                f"""
                INSERT INTO {self._schema}.sensor_tick (
                    sensor_id, status, run_requests, reserved_run_ids, cursor, trigger_payload
                )
                VALUES ($1, $2::marie_scheduler.tick_status, $3, $4, $5, $6)
                RETURNING id
                """,
                UUID(sensor_id),
                status.value,
                json.dumps(run_requests) if run_requests else None,
                [UUID(rid) for rid in reserved_run_ids] if reserved_run_ids else [],
                cursor,
                json.dumps(trigger_payload) if trigger_payload else None,
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
        expected_updated_at: datetime | None = None,
    ) -> str:
        """
        Record a sensor tick and update sensor state atomically.

        This is the CRITICAL method that ensures:
        1. Tick is always recorded (success, skipped, or failed)
        2. Sensor state is updated atomically
        3. Failure tracking is maintained
        """
        now = datetime.now(timezone.utc)

        async with self._connection() as conn:
            async with conn.transaction():
                current_updated_at = await conn.fetchval(
                    f"SELECT updated_at FROM {self._schema}.sensor WHERE id=$1 FOR UPDATE",
                    UUID(sensor_id),
                )
                if (
                    expected_updated_at is not None
                    and current_updated_at != expected_updated_at
                ):
                    status = TickStatus.SKIPPED
                    skip_reason = 'Configuration changed while evaluating; admission receipts retained'
                    cursor = None
                if tick_id:
                    # Update existing tick (STARTED -> final)
                    await conn.execute(
                        f"""
                        UPDATE {self._schema}.sensor_tick
                        SET status = $2::marie_scheduler.tick_status,
                            cursor = COALESCE($3, cursor),
                            run_ids = $4,
                            skip_reason = $5,
                            error_message = $6,
                            completed_at = $7,
                            duration_ms = $8,
                            trigger_payload = COALESCE($9, trigger_payload)
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

                if cursor is not None and status in (
                    TickStatus.SUCCESS,
                    TickStatus.SKIPPED,
                ):
                    await conn.execute(
                        f"UPDATE {self._schema}.sensor SET cursor=$2 WHERE id=$1",
                        UUID(sensor_id),
                        cursor,
                    )
                if status in (TickStatus.SUCCESS, TickStatus.SKIPPED) and run_ids:
                    await conn.execute(
                        f"""UPDATE {self._schema}.sensor SET last_run_key=(
                            SELECT run_key FROM {self._schema}.sensor_run_key
                            WHERE sensor_id=$1 AND job_id=$2 ORDER BY created_at DESC LIMIT 1
                        ) WHERE id=$1""",
                        UUID(sensor_id),
                        UUID(run_ids[-1]),
                    )
                return str(result_tick_id)

    async def get_tick(self, tick_id: str) -> Optional[Dict[str, Any]]:
        """Get a tick by ID."""
        async with self._connection() as conn:
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
        async with self._connection() as conn:
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
        async with self._connection() as conn:
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
        async with self._connection() as conn:
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
        async with self._connection() as conn:
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

        async with self._connection() as conn:
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
        async with self._connection() as conn:
            event_id = await conn.fetchval(
                f"""
                INSERT INTO {self._schema}.event_log (
                    source, sensor_type, sensor_external_id,
                    routing_key, event_key, payload, headers
                )
                VALUES ($1, $2::marie_scheduler.trigger_type, $3, $4, $5, $6, $7)
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
        async with self._connection() as conn:
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
        async with self._connection() as conn:
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
        async with self._connection() as conn:
            result = await conn.fetchval(
                f"SELECT {self._schema}.cleanup_old_ticks($1)",
                retention_days,
            )
            return result or 0

    async def cleanup_old_events(self, retention_days: int = 14) -> int:
        """Delete old events for retention."""
        async with self._connection() as conn:
            result = await conn.fetchval(
                f"SELECT {self._schema}.cleanup_old_events($1)",
                retention_days,
            )
            return result or 0

    async def cleanup_old_run_keys(self, retention_days: int = 30) -> int:
        """Retain completion receipts while their source history can be replayed."""
        async with self._connection() as conn:
            result = await conn.fetchval(
                f"SELECT {self._schema}.cleanup_old_run_keys($1)",
                retention_days,
            )
            return result or 0

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _row_to_dict(self, row) -> Optional[Dict[str, Any]]:
        """Convert a database row to a dictionary."""
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
