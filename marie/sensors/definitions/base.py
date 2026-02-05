"""
Base sensor class.

All sensor implementations extend this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.types import SensorResult, SensorType


class BaseSensor(ABC):
    """
    Abstract base class for sensor implementations.

    Each sensor type (schedule, webhook, polling, etc.) implements
    this interface. The evaluate() method is called by the daemon
    on each tick to determine if the sensor should fire.
    """

    # Sensor type this evaluator handles (set by subclasses)
    sensor_type: SensorType

    def __init__(self, sensor_data: Dict[str, Any]):
        """
        Initialize the sensor with data from the database.

        :param sensor_data: Dictionary containing sensor row data
        """
        self.sensor_id: str = sensor_data.get("id")
        self.external_id: str = sensor_data.get("external_id")
        self.name: str = sensor_data.get("name", "")
        self.config: Dict[str, Any] = sensor_data.get("config", {})
        self.target_job_name: Optional[str] = sensor_data.get("target_job_name")
        self.target_dag_id: Optional[str] = sensor_data.get("target_dag_id")
        self.minimum_interval_seconds: int = sensor_data.get(
            "minimum_interval_seconds", 30
        )

    @abstractmethod
    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        """
        Evaluate the sensor and return the result.

        This method is called by the daemon on each tick. Implementations
        should:
        1. Check if the sensor's conditions are met
        2. Return SensorResult.fire() with run requests if conditions are met
        3. Return SensorResult.skip() with a reason if conditions are not met

        The method should be idempotent and safe to call repeatedly.

        :param context: Evaluation context with sensor state and resources
        :return: SensorResult indicating whether to fire or skip
        """
        raise NotImplementedError

    def validate_config(self) -> None:
        """
        Validate sensor configuration.

        Override in subclasses to add type-specific validation.
        Called during sensor creation/update.

        :raises SensorConfigError: If configuration is invalid
        """
        pass

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the sensor configuration."""
        return self.config.get(key, default)

    def get_required_config(self, key: str) -> Any:
        """
        Get a required configuration value.

        :raises SensorConfigError: If key is missing
        """
        from marie.sensors.exceptions import SensorConfigError

        if key not in self.config:
            raise SensorConfigError(f"Missing required config key: {key}", field=key)
        return self.config[key]

    def build_run_key(self, *parts: str) -> str:
        """
        Build a stable run key from parts.

        The run key is used for idempotency - if a sensor has already
        fired with this key, the job submission is skipped.

        :param parts: String parts to join into the key
        :return: Stable run key string
        """
        return ":".join(str(p) for p in parts if p)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.sensor_id}, name={self.name})"
