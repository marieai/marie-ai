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
    _evaluators: Dict[tuple, Type["BaseSensor"]]

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
        self,
        sensor_type: SensorType,
        evaluator_class: Type["BaseSensor"],
        subtype: Optional[str] = None,
    ) -> None:
        """
        Register an evaluator class for a sensor type (and optional subtype).

        Sensor types with a single evaluator (e.g. SCHEDULE, WEBHOOK) omit
        subtype. Sensor types that host multiple evaluators (e.g. DATA_SINK)
        register one per subtype so they can coexist.

        :param sensor_type: The sensor type to register
        :param evaluator_class: The evaluator class implementing BaseSensor
        :param subtype: Optional subtype distinguishing evaluators that share
            the same sensor_type
        """
        self._evaluators[(sensor_type, subtype)] = evaluator_class

    def get_evaluator(
        self, sensor_type: SensorType, subtype: Optional[str] = None
    ) -> Type["BaseSensor"]:
        """
        Get the evaluator class for a sensor type (and optional subtype).

        Resolution order: exact (sensor_type, subtype) match, then
        (sensor_type, None), then (DATA_SINK, "s3") for back-compat with
        sensors registered before subtypes existed.

        :param sensor_type: The sensor type to look up
        :param subtype: Optional subtype to disambiguate multiple evaluators
        :return: The evaluator class
        :raises SensorRegistryError: If no evaluator is registered for this
            (type, subtype) combination
        """
        for key in ((sensor_type, subtype), (sensor_type, None), (sensor_type, "s3")):
            if key in self._evaluators:
                return self._evaluators[key]
        raise SensorRegistryError(
            f"No evaluator registered for sensor type: {sensor_type.value}"
            + (f" subtype: {subtype}" if subtype else "")
        )

    def has_evaluator(
        self, sensor_type: SensorType, subtype: Optional[str] = None
    ) -> bool:
        """Check if an evaluator is registered for a sensor type (and optional subtype)."""
        return (sensor_type, subtype) in self._evaluators

    def get_registered_types(self) -> list[SensorType]:
        """Get list of sensor types with registered evaluators."""
        return list({key[0] for key in self._evaluators.keys()})

    def clear(self) -> None:
        """Clear all registered evaluators. Primarily for testing."""
        self._evaluators.clear()


def register_sensor(
    sensor_type: SensorType,
    subtype: Optional[str] = None,
) -> Callable[[Type["BaseSensor"]], Type["BaseSensor"]]:
    """
    Decorator to register a sensor evaluator class.

    Usage:
        @register_sensor(SensorType.SCHEDULE)
        class ScheduleSensor(BaseSensor):
            ...

        @register_sensor(SensorType.DATA_SINK, subtype="s3")
        class S3DataSinkSensor(BaseSensor):
            ...
    """

    def decorator(cls: Type["BaseSensor"]) -> Type["BaseSensor"]:
        registry = SensorRegistry.get_instance()
        registry.register(sensor_type, cls, subtype=subtype)
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
        kb_document_sensor,
        manual_sensor,
        polling_sensor,
        run_status_sensor,
        schedule_sensor,
        submission_document_sensor,
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

    registered_types = set(registry.get_registered_types())
    missing = [t for t in expected_types if t not in registered_types]
    if missing:
        raise SensorRegistryError(
            f"Failed to register evaluators for: {[t.value for t in missing]}"
        )
