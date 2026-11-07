"""Cron schedules evaluated in local wall time with UTC occurrence identities."""

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions.base import BaseSensor
from marie.sensors.exceptions import SensorConfigError
from marie.sensors.registry import register_sensor
from marie.sensors.types import SensorResult, SensorType


def schedule_occurrence(
    expression: str, zone: ZoneInfo, reference: datetime, *, previous: bool = False
) -> datetime:
    """Find a real wall-time occurrence; skip gaps and use the first DST fold."""
    wall = reference.astimezone(zone).replace(tzinfo=None)
    if previous:
        first_fold = wall.replace(tzinfo=zone, fold=0).astimezone(timezone.utc)
        wall += reference - first_fold
    iterator = croniter(expression, wall)
    while True:
        candidate = (
            iterator.get_prev(datetime) if previous else iterator.get_next(datetime)
        )
        utc = candidate.replace(tzinfo=zone, fold=0).astimezone(timezone.utc)
        if utc.astimezone(zone).replace(tzinfo=None) != candidate:
            continue
        if previous and utc <= reference or not previous and utc > reference:
            return utc


@register_sensor(SensorType.SCHEDULE)
class ScheduleSensor(BaseSensor):
    """Emit at most the latest due occurrence, never acquire datasource content."""

    sensor_type = SensorType.SCHEDULE

    def __init__(self, sensor_data: dict[str, Any]) -> None:
        super().__init__(sensor_data)
        self.cron_expr = self.get_config_value('cron', '')
        self.cron_timezone = self.get_config_value('timezone', 'UTC')

    def validate_config(self) -> None:
        if not isinstance(self.cron_expr, str) or len(self.cron_expr.split()) != 5:
            raise SensorConfigError(
                'Schedule requires a five-field cron expression', field='cron'
            )
        try:
            croniter(self.cron_expr)
        except (KeyError, ValueError) as error:
            raise SensorConfigError('Invalid cron expression', field='cron') from error
        try:
            ZoneInfo(self.cron_timezone)
        except (ZoneInfoNotFoundError, ValueError, TypeError) as error:
            raise SensorConfigError(
                'Invalid IANA timezone', field='timezone'
            ) from error

    async def evaluate(self, context: SensorEvaluationContext) -> SensorResult:
        self.validate_config()
        now = datetime.now(timezone.utc)
        zone = ZoneInfo(self.cron_timezone)
        if context.cursor:
            reference = datetime.fromisoformat(context.cursor)
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=timezone.utc)
        else:
            return SensorResult.skip(
                f'Next scheduled at {schedule_occurrence(self.cron_expr, zone, now).isoformat()}',
                cursor=now.isoformat(),
            )

        latest = schedule_occurrence(
            self.cron_expr, zone, now + timedelta(microseconds=1), previous=True
        )
        if latest <= reference:
            return SensorResult.skip(
                f'Next scheduled at {schedule_occurrence(self.cron_expr, zone, now).isoformat()}',
                cursor=context.cursor,
            )
        return SensorResult.fire(
            run_key=self.build_run_key('schedule', self.sensor_id, latest.isoformat()),
            job_name=self.target_job_name,
            dag_id=self.target_dag_id,
            run_config={
                **self.get_config_value('run_config', {}),
                'scheduled_at': latest.isoformat(),
                'cron': self.cron_expr,
            },
            tags={'trigger': 'schedule'},
            cursor=latest.isoformat(),
        )
