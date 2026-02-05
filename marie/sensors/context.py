"""
Sensor evaluation context.

The SensorEvaluationContext provides all information needed by a sensor
to evaluate whether it should fire. This includes identity, state from
previous evaluations, and type-specific data (e.g., webhook payload).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from marie.sensors.types import SensorType


@dataclass
class SensorEvaluationContext:
    """
    Context provided to sensor evaluation functions.

    This dataclass contains all the information a sensor needs to make
    an evaluation decision. Different sensor types use different subsets
    of these fields.
    """

    # Identity
    sensor_id: str
    sensor_name: str
    sensor_type: SensorType

    # State from previous evaluation
    cursor: Optional[str] = None  # User-managed state (e.g., event_log_id)
    last_tick_at: Optional[datetime] = None  # When sensor was last evaluated
    last_run_key: Optional[str] = None  # Most recent run key

    # Sensor configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Target information
    target_job_name: Optional[str] = None
    target_dag_id: Optional[str] = None

    # Resources (e.g., HTTP clients, database connections)
    resources: Dict[str, Any] = field(default_factory=dict)

    # Logger instance
    logger: Any = None  # MarieLogger, but avoid circular import

    # =========================================================================
    # Webhook-specific fields
    # =========================================================================
    request_body: Optional[Dict[str, Any]] = None
    request_headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)

    # =========================================================================
    # Event-specific fields (for message queue sensors)
    # =========================================================================
    event_type: Optional[str] = None
    event_payload: Optional[Dict[str, Any]] = None
    message_id: Optional[str] = None

    # =========================================================================
    # Event log batch (for event_log-based sensors)
    # =========================================================================
    pending_events: List[Dict[str, Any]] = field(default_factory=list)

    # =========================================================================
    # Run status sensor fields
    # =========================================================================
    completed_jobs: List[Dict[str, Any]] = field(default_factory=list)

    def update_cursor(self, new_cursor: str) -> None:
        """
        Update cursor for next evaluation.

        Note: This only updates the context's cursor field.
        The actual persistence happens in the daemon after evaluation.
        """
        self.cursor = new_cursor

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the sensor configuration."""
        return self.config.get(key, default)

    def log_debug(self, message: str) -> None:
        """Log a debug message if logger is available."""
        if self.logger:
            self.logger.debug(f"[{self.sensor_name}] {message}")

    def log_info(self, message: str) -> None:
        """Log an info message if logger is available."""
        if self.logger:
            self.logger.info(f"[{self.sensor_name}] {message}")

    def log_warning(self, message: str) -> None:
        """Log a warning message if logger is available."""
        if self.logger:
            self.logger.warning(f"[{self.sensor_name}] {message}")

    def log_error(self, message: str) -> None:
        """Log an error message if logger is available."""
        if self.logger:
            self.logger.error(f"[{self.sensor_name}] {message}")
