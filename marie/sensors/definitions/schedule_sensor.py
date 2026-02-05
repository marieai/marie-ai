"""
Schedule sensor implementation.

Schedule sensors fire based on cron expressions, providing time-based
job triggering similar to traditional cron jobs.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.base import BaseSensor
from marie.sensors.exceptions import SensorConfigError
from marie.sensors.registry import register_sensor
from marie.sensors.types import SensorResult, SensorType

try:
    from croniter import croniter
except ImportError:
    croniter = None


@register_sensor(SensorType.SCHEDULE)
class ScheduleSensor(BaseSensor):
    """
    Cron-based schedule sensor.

    This sensor evaluates a cron expression and fires when the scheduled
    time has passed since the last evaluation.

    Configuration:
        cron: str - Cron expression (required)
            Example: "0 * * * *" (every hour)
        timezone: str - Timezone for cron evaluation (default: "UTC")

    Run Key:
        schedule:{sensor_id}:{scheduled_timestamp}

    This ensures each scheduled execution fires exactly once.
    """

    sensor_type = SensorType.SCHEDULE

    def __init__(self, sensor_data: Dict[str, Any]):
        super().__init__(sensor_data)
        self.cron_expr: str = self.get_config_value("cron", "")
        self.cron_timezone: str = self.get_config_value("timezone", "UTC")

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        """
        Evaluate the schedule sensor.

        Checks if the current time is past the next scheduled execution
        time based on the cron expression.
        """
        if croniter is None:
            return SensorResult.skip(
                "croniter library not installed",
                cursor=context.cursor,
            )

        if not self.cron_expr:
            return SensorResult.skip(
                "No cron expression configured",
                cursor=context.cursor,
            )

        now = datetime.now(timezone.utc)

        # Determine the reference time for cron calculation
        # Use cursor (last scheduled time) or last_tick_at or sensor creation
        reference_time = self._get_reference_time(context)

        # Calculate next scheduled time after reference
        cron = croniter(self.cron_expr, reference_time)
        next_scheduled = cron.get_next(datetime)

        # Make timezone-aware if needed
        if next_scheduled.tzinfo is None:
            next_scheduled = next_scheduled.replace(tzinfo=timezone.utc)

        # Check if we've passed the scheduled time
        if now < next_scheduled:
            return SensorResult.skip(
                f"Next scheduled at {next_scheduled.isoformat()}",
                cursor=context.cursor,
            )

        context.log_info(f"Schedule triggered at {next_scheduled.isoformat()}")

        # Generate run key based on scheduled time
        scheduled_str = next_scheduled.strftime("%Y%m%d%H%M")
        run_key = self.build_run_key("schedule", self.sensor_id, scheduled_str)

        # Update cursor to the scheduled time
        new_cursor = next_scheduled.isoformat()

        return SensorResult.fire(
            run_key=run_key,
            job_name=self.target_job_name,
            dag_id=self.target_dag_id,
            run_config={
                "scheduled_at": next_scheduled.isoformat(),
                "cron": self.cron_expr,
            },
            tags={
                "trigger": "schedule",
                "sensor_id": self.sensor_id,
                "scheduled_at": scheduled_str,
            },
            cursor=new_cursor,
        )

    def _get_reference_time(self, context: SensorEvaluationContext) -> datetime:
        """
        Get the reference time for cron calculation.

        Priority:
        1. Cursor (last scheduled execution time)
        2. Last tick time
        3. Current time minus minimum interval
        """
        # Try to parse cursor as ISO datetime
        if context.cursor:
            try:
                cursor_time = datetime.fromisoformat(context.cursor)
                if cursor_time.tzinfo is None:
                    cursor_time = cursor_time.replace(tzinfo=timezone.utc)
                return cursor_time
            except (ValueError, TypeError):
                pass

        # Fall back to last_tick_at
        if context.last_tick_at:
            if context.last_tick_at.tzinfo is None:
                return context.last_tick_at.replace(tzinfo=timezone.utc)
            return context.last_tick_at

        # Default: start from minimum_interval ago
        from datetime import timedelta

        return datetime.now(timezone.utc) - timedelta(
            seconds=self.minimum_interval_seconds
        )

    def validate_config(self) -> None:
        """Validate schedule sensor configuration."""
        if not self.cron_expr:
            raise SensorConfigError(
                "Schedule sensor requires 'cron' expression", field="cron"
            )

        if croniter is None:
            raise SensorConfigError(
                "croniter library required for schedule sensors. "
                "Install with: pip install croniter"
            )

        # Validate cron expression
        try:
            croniter(self.cron_expr)
        except (KeyError, ValueError) as e:
            raise SensorConfigError(
                f"Invalid cron expression '{self.cron_expr}': {e}", field="cron"
            )
