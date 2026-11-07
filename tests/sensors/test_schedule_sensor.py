from datetime import datetime, timezone

import pytest

from marie.sensors.context import SensorEvaluationContext
from marie.sensors.definitions import schedule_sensor as module
from marie.sensors.definitions.schedule_sensor import ScheduleSensor
from marie.sensors.exceptions import SensorConfigError
from marie.sensors.types import SensorType


def frozen(monkeypatch, value):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    monkeypatch.setattr(module, 'datetime', Clock)


def setup(cron='0 2 * * 6', tz='America/Chicago', cursor=None):
    sensor = ScheduleSensor(
        {
            'id': 'source-a',
            'name': 'weekend',
            'config': {'cron': cron, 'timezone': tz},
            'target_job_name': 'connector',
        }
    )
    context = SensorEvaluationContext(
        'source-a', 'weekend', SensorType.SCHEDULE, cursor=cursor
    )
    return sensor, context


async def test_first_activation_starts_in_future(monkeypatch):
    now = datetime(2026, 11, 7, 8, 0, 5, tzinfo=timezone.utc)
    frozen(monkeypatch, now)
    sensor, context = setup()
    result = await sensor.evaluate(context)
    assert not result.run_requests
    assert result.cursor == now.isoformat()


async def test_weekend_local_time_and_coalesced_backlog(monkeypatch):
    frozen(monkeypatch, datetime(2026, 11, 22, 12, tzinfo=timezone.utc))
    sensor, context = setup(cursor='2026-10-01T00:00:00+00:00')
    result = await sensor.evaluate(context)
    assert result.cursor == '2026-11-21T08:00:00+00:00'
    assert len(result.run_requests) == 1
    context.cursor = result.cursor
    assert not (await sensor.evaluate(context)).run_requests


async def test_spring_gap_skips_missing_local_time(monkeypatch):
    frozen(monkeypatch, datetime(2026, 3, 8, 9, tzinfo=timezone.utc))
    sensor, context = setup('30 2 * * *', cursor='2026-03-07T08:30:00+00:00')
    assert not (await sensor.evaluate(context)).run_requests


async def test_fall_repeat_fires_once(monkeypatch):
    sensor, context = setup('30 1 * * *', cursor='2026-10-31T06:30:00+00:00')
    frozen(monkeypatch, datetime(2026, 11, 1, 6, 31, tzinfo=timezone.utc))
    first = await sensor.evaluate(context)
    assert first.cursor == '2026-11-01T06:30:00+00:00'
    context.cursor = first.cursor
    frozen(monkeypatch, datetime(2026, 11, 1, 7, 31, tzinfo=timezone.utc))
    assert not (await sensor.evaluate(context)).run_requests


async def test_fall_catchup_during_second_hour_before_scheduled_minute(monkeypatch):
    frozen(monkeypatch, datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc))
    sensor, context = setup('30 1 * * *', cursor='2026-10-31T06:30:00+00:00')
    assert (await sensor.evaluate(context)).cursor == '2026-11-01T06:30:00+00:00'


@pytest.mark.parametrize(
    'cron,tz', [('bad', 'UTC'), ('* * * * *', 'Mars/Nowhere'), ('0 * * * * *', 'UTC')]
)
def test_invalid_configuration(cron, tz):
    with pytest.raises(SensorConfigError):
        sensor, _ = setup(cron, tz)
        sensor.validate_config()
