"""
Sensor-specific exceptions.

These exceptions provide clear error handling for sensor operations.
"""


class SensorError(Exception):
    """Base exception for sensor-related errors."""

    pass


class SensorEvaluationError(SensorError):
    """
    Error during sensor evaluation.

    This exception is raised when a sensor's evaluate() method fails.
    The daemon will record this as a FAILED tick.
    """

    def __init__(self, message: str, sensor_id: str = None, cause: Exception = None):
        self.sensor_id = sensor_id
        self.cause = cause
        super().__init__(message)


class SensorConfigError(SensorError):
    """
    Invalid sensor configuration.

    Raised when sensor configuration is missing required fields
    or contains invalid values.
    """

    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(message)


class SensorNotFoundError(SensorError):
    """
    Sensor not found.

    Raised when attempting to access a sensor that doesn't exist.
    """

    def __init__(self, sensor_id: str = None, external_id: str = None):
        self.sensor_id = sensor_id
        self.external_id = external_id
        if sensor_id:
            message = f"Sensor not found: id={sensor_id}"
        elif external_id:
            message = f"Sensor not found: external_id={external_id}"
        else:
            message = "Sensor not found"
        super().__init__(message)


class SensorRegistryError(SensorError):
    """
    Error in sensor registry operations.

    Raised when sensor type is unknown or evaluator is not registered.
    """

    pass


class RunKeyDuplicateError(SensorError):
    """
    Duplicate run key detected.

    This is not necessarily an error - it indicates the sensor
    is attempting to fire with a run_key that has already been processed.
    The daemon handles this gracefully by skipping the duplicate.
    """

    def __init__(self, sensor_id: str, run_key: str):
        self.sensor_id = sensor_id
        self.run_key = run_key
        super().__init__(f"Duplicate run_key: sensor={sensor_id}, key={run_key}")


class EventLogCursorError(SensorError):
    """
    Error with event_log cursor.

    Raised when cursor is invalid or points to a non-existent position.
    """

    def __init__(self, message: str, cursor: str = None):
        self.cursor = cursor
        super().__init__(message)


class WebhookAuthError(SensorError):
    """
    Webhook authentication failed.

    Raised when webhook request fails authentication validation.
    """

    def __init__(self, message: str, path: str = None):
        self.path = path
        super().__init__(message)
