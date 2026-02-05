"""
Manual sensor implementation.

Manual sensors are click-to-run triggers for testing and one-off executions.
They always fire when evaluated, making them useful for development and debugging.
"""

from datetime import datetime, timezone
from typing import Any, Dict

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.base import BaseSensor
from marie.sensors.registry import register_sensor
from marie.sensors.types import SensorResult, SensorType


@register_sensor(SensorType.MANUAL)
class ManualSensor(BaseSensor):
    """
    Manual trigger sensor.

    This sensor is designed for manual testing and one-off job triggering.
    It always fires when evaluated, using a timestamp-based run key for
    idempotency within a short window.

    Configuration:
        None required (manual sensors have no configuration)

    Usage:
        Manual sensors are typically invoked via the /test endpoint
        rather than the daemon polling loop.
    """

    sensor_type = SensorType.MANUAL

    def __init__(self, sensor_data: Dict[str, Any]):
        super().__init__(sensor_data)

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        """
        Evaluate the manual sensor.

        Manual sensors always fire when evaluated, creating a job
        with a timestamp-based run key.
        """
        context.log_info("Manual trigger activated")

        # Generate a timestamp-based run key
        # This provides idempotency within a 1-second window
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        run_key = self.build_run_key("manual", self.sensor_id, timestamp)

        # Build run config from any provided payload
        run_config = {}
        if context.request_body:
            run_config["payload"] = context.request_body

        return SensorResult.fire(
            run_key=run_key,
            job_name=self.target_job_name,
            dag_id=self.target_dag_id,
            run_config=run_config,
            tags={"trigger": "manual", "sensor_id": self.sensor_id},
        )

    def validate_config(self) -> None:
        """Manual sensors have no required configuration."""
        pass
