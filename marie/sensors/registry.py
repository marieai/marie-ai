"""
Sensor registry for managing sensor evaluators.

The registry maps sensor types to their evaluator implementations,
allowing the daemon to dynamically select the appropriate evaluator
for each sensor.
"""

from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

from marie.sensors.exceptions import SensorRegistryError
from marie.sensors.types import SensorType

if TYPE_CHECKING:
    from marie.sensors.definitions.base import BaseSensor


class SensorRegistry:
    """
    Registry for sensor evaluator classes.

    This singleton registry maps SensorType enum values to their
    corresponding evaluator implementations. Evaluators are registered
    at startup and retrieved during daemon evaluation loops.
    """

    _instance: Optional["SensorRegistry"] = None
    _evaluators: Dict[SensorType, Type["BaseSensor"]]

    def __new__(cls) -> "SensorRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._evaluators = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "SensorRegistry":
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self, sensor_type: SensorType, evaluator_class: Type["BaseSensor"]
    ) -> None:
        """
        Register an evaluator class for a sensor type.

        :param sensor_type: The sensor type to register
        :param evaluator_class: The evaluator class implementing BaseSensor
        """
        self._evaluators[sensor_type] = evaluator_class

    def get_evaluator(self, sensor_type: SensorType) -> Type["BaseSensor"]:
        """
        Get the evaluator class for a sensor type.

        :param sensor_type: The sensor type to look up
        :return: The evaluator class
        :raises SensorRegistryError: If no evaluator is registered for this type
        """
        if sensor_type not in self._evaluators:
            raise SensorRegistryError(
                f"No evaluator registered for sensor type: {sensor_type.value}"
            )
        return self._evaluators[sensor_type]

    def has_evaluator(self, sensor_type: SensorType) -> bool:
        """Check if an evaluator is registered for a sensor type."""
        return sensor_type in self._evaluators

    def get_registered_types(self) -> list[SensorType]:
        """Get list of sensor types with registered evaluators."""
        return list(self._evaluators.keys())

    def clear(self) -> None:
        """Clear all registered evaluators. Primarily for testing."""
        self._evaluators.clear()


def register_sensor(
    sensor_type: SensorType,
) -> Callable[[Type["BaseSensor"]], Type["BaseSensor"]]:
    """
    Decorator to register a sensor evaluator class.

    Usage:
        @register_sensor(SensorType.SCHEDULE)
        class ScheduleSensor(BaseSensor):
            ...
    """

    def decorator(cls: Type["BaseSensor"]) -> Type["BaseSensor"]:
        registry = SensorRegistry.get_instance()
        registry.register(sensor_type, cls)
        return cls

    return decorator


def register_all_sensors() -> None:
    """
    Register all built-in sensor evaluators.

    This function imports all sensor definition modules to trigger
    their @register_sensor decorators.
    """
    # Import modules to trigger registration
    from marie.sensors.definitions import (
        event_sensor,
        manual_sensor,
        polling_sensor,
        run_status_sensor,
        schedule_sensor,
        webhook_sensor,
    )
    from marie.sensors.definitions.data_sink import s3_sensor

    # Verify all expected types are registered
    registry = SensorRegistry.get_instance()
    expected_types = [
        SensorType.MANUAL,
        SensorType.SCHEDULE,
        SensorType.WEBHOOK,
        SensorType.POLLING,
        SensorType.EVENT,
        SensorType.RUN_STATUS,
        SensorType.DATA_SINK,
    ]

    missing = [t for t in expected_types if not registry.has_evaluator(t)]
    if missing:
        raise SensorRegistryError(
            f"Failed to register evaluators for: {[t.value for t in missing]}"
        )
