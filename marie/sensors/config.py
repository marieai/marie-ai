"""
Sensor system configuration.

Settings for the sensor daemon and related components.
Can be loaded from environment variables or YAML configuration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SensorSettings:
    """
    Configuration settings for the sensor system.

    These settings control daemon behavior, retention policies,
    and various thresholds.
    """

    # =========================================================================
    # Daemon Settings
    # =========================================================================

    # Interval between daemon polling cycles (seconds)
    daemon_interval_seconds: float = 5.0

    # Maximum number of sensors to evaluate per cycle
    max_sensors_per_cycle: int = 100

    # Maximum concurrent sensor evaluations
    max_concurrent_evaluations: int = 10

    # =========================================================================
    # Retention Settings
    # =========================================================================

    # How long to keep sensor_tick rows (days)
    retention_days_ticks: int = 30

    # How long to keep event_log rows (days)
    retention_days_events: int = 14

    # How long to keep sensor_run_key rows (days)
    retention_days_run_keys: int = 30

    # =========================================================================
    # Recovery Settings
    # =========================================================================

    # Threshold for marking STARTED ticks as stuck/abandoned (hours)
    stuck_tick_threshold_hours: int = 24

    # =========================================================================
    # Event Processing Settings
    # =========================================================================

    # Time bucket for generating event_key when source doesn't provide one (seconds)
    event_key_bucket_seconds: int = 60

    # Maximum events to process per tick (for event_log sensors)
    max_events_per_tick: int = 100

    # =========================================================================
    # Notification Settings
    # =========================================================================

    # Enable PostgreSQL NOTIFY fast-path wake-up
    enable_notify_wake: bool = True

    # Debounce window for coalescing notifications (milliseconds)
    notify_debounce_ms: int = 100

    # =========================================================================
    # Database Settings
    # =========================================================================

    # Schema name for sensor tables
    schema: str = "marie_scheduler"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SensorSettings":
        """
        Create SensorSettings from a dictionary.

        Only recognized keys are used; unknown keys are ignored.
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SensorSettings":
        """
        Create SensorSettings from the main application config.

        Looks for settings under the 'sensors' key.
        """
        sensor_config = config.get("sensors", {})
        return cls.from_dict(sensor_config)

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises ValueError if any setting is invalid.
        """
        if self.daemon_interval_seconds <= 0:
            raise ValueError("daemon_interval_seconds must be positive")

        if self.max_sensors_per_cycle <= 0:
            raise ValueError("max_sensors_per_cycle must be positive")

        if self.max_concurrent_evaluations <= 0:
            raise ValueError("max_concurrent_evaluations must be positive")

        if self.retention_days_ticks <= 0:
            raise ValueError("retention_days_ticks must be positive")

        if self.retention_days_events <= 0:
            raise ValueError("retention_days_events must be positive")

        if self.retention_days_run_keys <= 0:
            raise ValueError("retention_days_run_keys must be positive")

        if self.stuck_tick_threshold_hours <= 0:
            raise ValueError("stuck_tick_threshold_hours must be positive")

        if self.event_key_bucket_seconds <= 0:
            raise ValueError("event_key_bucket_seconds must be positive")

        if self.max_events_per_tick <= 0:
            raise ValueError("max_events_per_tick must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "daemon_interval_seconds": self.daemon_interval_seconds,
            "max_sensors_per_cycle": self.max_sensors_per_cycle,
            "max_concurrent_evaluations": self.max_concurrent_evaluations,
            "retention_days_ticks": self.retention_days_ticks,
            "retention_days_events": self.retention_days_events,
            "retention_days_run_keys": self.retention_days_run_keys,
            "stuck_tick_threshold_hours": self.stuck_tick_threshold_hours,
            "event_key_bucket_seconds": self.event_key_bucket_seconds,
            "max_events_per_tick": self.max_events_per_tick,
            "enable_notify_wake": self.enable_notify_wake,
            "notify_debounce_ms": self.notify_debounce_ms,
            "schema": self.schema,
        }
