"""
Abstract storage interface for sensor state.

This interface defines the contract for sensor state persistence.
Implementations handle actual database operations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from marie.sensors.types import SensorStatus, SensorType, TickStatus


class SensorStateStorage(ABC):
    """
    Abstract base class for sensor state storage.

    Implementations must provide methods for:
    - Sensor CRUD operations
    - Tick recording and history
    - Run key tracking for idempotency
    - Event log queries
    - Webhook registration
    """

    # =========================================================================
    # SENSOR OPERATIONS
    # =========================================================================

    @abstractmethod
    async def get_sensor(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a sensor by its internal ID.

        :param sensor_id: UUID of the sensor
        :return: Sensor data dict or None if not found
        """
        raise NotImplementedError

    @abstractmethod
    async def get_sensor_by_external_id(
        self, external_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a sensor by its external ID (trigger_config.id from marie_studio).

        :param external_id: External UUID from marie_studio
        :return: Sensor data dict or None if not found
        """
        raise NotImplementedError

    @abstractmethod
    async def get_active_sensors(self) -> List[Dict[str, Any]]:
        """
        Get all active sensors for daemon evaluation.

        :return: List of sensor data dicts ordered by last_tick_at ASC
        """
        raise NotImplementedError

    @abstractmethod
    async def get_active_sensors_by_type(
        self, sensor_type: SensorType
    ) -> List[Dict[str, Any]]:
        """
        Get active sensors of a specific type.

        :param sensor_type: Type of sensors to retrieve
        :return: List of sensor data dicts
        """
        raise NotImplementedError

    @abstractmethod
    async def create_sensor(self, sensor_data: Dict[str, Any]) -> str:
        """
        Create a new sensor.

        :param sensor_data: Sensor attributes
        :return: ID of created sensor
        """
        raise NotImplementedError

    @abstractmethod
    async def update_sensor(
        self, sensor_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update a sensor's configuration.

        :param sensor_id: UUID of the sensor
        :param updates: Fields to update
        :return: Updated sensor data or None if not found
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_sensor(self, sensor_id: str) -> bool:
        """
        Delete a sensor and all related data.

        :param sensor_id: UUID of the sensor
        :return: True if deleted, False if not found
        """
        raise NotImplementedError

    @abstractmethod
    async def update_sensor_status(self, sensor_id: str, status: SensorStatus) -> bool:
        """
        Update a sensor's operational status.

        :param sensor_id: UUID of the sensor
        :param status: New status
        :return: True if updated, False if not found
        """
        raise NotImplementedError

    # =========================================================================
    # CURSOR OPERATIONS
    # =========================================================================

    @abstractmethod
    async def get_cursor(self, sensor_id: str) -> Optional[str]:
        """
        Get the current cursor for a sensor.

        :param sensor_id: UUID of the sensor
        :return: Cursor value or None
        """
        raise NotImplementedError

    @abstractmethod
    async def set_cursor(self, sensor_id: str, cursor: str) -> None:
        """
        Set the cursor for a sensor.

        :param sensor_id: UUID of the sensor
        :param cursor: New cursor value
        """
        raise NotImplementedError

    # =========================================================================
    # TICK OPERATIONS
    # =========================================================================

    @abstractmethod
    async def create_tick(
        self,
        sensor_id: str,
        status: TickStatus,
        run_requests: Optional[List[Dict[str, Any]]] = None,
        reserved_run_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new tick record (for STARTED state before job submission).

        :param sensor_id: UUID of the sensor
        :param status: Initial tick status (typically STARTED)
        :param run_requests: Serialized run requests for crash recovery
        :param reserved_run_ids: Pre-reserved job IDs
        :return: ID of created tick
        """
        raise NotImplementedError

    @abstractmethod
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

        This method MUST:
        1. Create or update sensor_tick row
        2. Update sensor.last_tick_at
        3. Reset/increment failure_count based on status

        :param sensor_id: UUID of the sensor
        :param status: Final tick status
        :param cursor: Cursor after evaluation
        :param run_ids: Jobs that were submitted
        :param skip_reason: Reason for skipping (if SKIPPED)
        :param error_message: Error message (if FAILED)
        :param duration_ms: Evaluation duration in milliseconds
        :param trigger_payload: Debug payload
        :param tick_id: Existing tick ID to update (for STARTED -> final)
        :return: ID of the tick
        """
        raise NotImplementedError

    @abstractmethod
    async def get_tick(self, tick_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a tick by ID.

        :param tick_id: UUID of the tick
        :return: Tick data or None
        """
        raise NotImplementedError

    @abstractmethod
    async def get_ticks(
        self,
        sensor_id: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[TickStatus] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get tick history for a sensor.

        :param sensor_id: UUID of the sensor
        :param limit: Maximum rows to return
        :param offset: Rows to skip
        :param status: Filter by status (optional)
        :return: List of tick data dicts
        """
        raise NotImplementedError

    @abstractmethod
    async def get_started_ticks(
        self, threshold_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get STARTED ticks older than threshold (for crash recovery).

        :param threshold_hours: Hours to consider a tick stuck
        :return: List of stuck tick data
        """
        raise NotImplementedError

    # =========================================================================
    # RUN KEY OPERATIONS (Idempotency)
    # =========================================================================

    @abstractmethod
    async def has_run_key(self, sensor_id: str, run_key: str) -> bool:
        """
        Check if a run_key has already been processed.

        :param sensor_id: UUID of the sensor
        :param run_key: The run key to check
        :return: True if already processed
        """
        raise NotImplementedError

    @abstractmethod
    async def record_run_key(
        self, sensor_id: str, run_key: str, job_id: Optional[str] = None
    ) -> None:
        """
        Record a processed run_key for idempotency.

        :param sensor_id: UUID of the sensor
        :param run_key: The run key to record
        :param job_id: Associated job ID (optional)
        """
        raise NotImplementedError

    # =========================================================================
    # EVENT LOG OPERATIONS
    # =========================================================================

    @abstractmethod
    async def get_pending_events(
        self,
        sensor_external_id: Optional[str] = None,
        source: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get pending events from the event_log.

        :param sensor_external_id: Filter by target sensor
        :param source: Filter by event source
        :param cursor: Start after this event_log_id
        :param limit: Maximum events to return
        :return: List of event data dicts
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        Insert an event into the event_log.

        :param source: Event source (e.g., "webhook", "rabbitmq")
        :param payload: Event payload
        :param sensor_external_id: Target sensor
        :param sensor_type: Routing hint
        :param routing_key: Routing key
        :param event_key: Stable event key for deduplication
        :param headers: Event headers
        :return: event_id of inserted event
        """
        raise NotImplementedError

    # =========================================================================
    # WEBHOOK REGISTRATION
    # =========================================================================

    @abstractmethod
    async def get_webhook_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Get webhook registration by path.

        :param path: Webhook path (e.g., "/webhooks/github-push")
        :return: Webhook registration data or None
        """
        raise NotImplementedError

    @abstractmethod
    async def create_webhook_registration(
        self,
        sensor_id: str,
        path: str,
        methods: List[str],
        auth_type: Optional[str] = None,
        auth_secret: Optional[str] = None,
    ) -> str:
        """
        Create a webhook registration.

        :param sensor_id: UUID of the associated sensor
        :param path: Webhook path
        :param methods: Allowed HTTP methods
        :param auth_type: Authentication type
        :param auth_secret: Authentication secret
        :return: ID of created registration
        """
        raise NotImplementedError

    # =========================================================================
    # MAINTENANCE OPERATIONS
    # =========================================================================

    @abstractmethod
    async def cleanup_old_ticks(self, retention_days: int = 30) -> int:
        """
        Delete old ticks for retention.

        :param retention_days: Days to retain
        :return: Number of deleted rows
        """
        raise NotImplementedError

    @abstractmethod
    async def cleanup_old_events(self, retention_days: int = 14) -> int:
        """
        Delete old events for retention.

        :param retention_days: Days to retain
        :return: Number of deleted rows
        """
        raise NotImplementedError

    @abstractmethod
    async def cleanup_old_run_keys(self, retention_days: int = 30) -> int:
        """
        Delete old run keys for retention.

        :param retention_days: Days to retain
        :return: Number of deleted rows
        """
        raise NotImplementedError
