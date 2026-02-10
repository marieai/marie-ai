"""
Sensor definition classes.

Each sensor type has its own implementation that extends BaseSensor.
"""

from marie.sensors.definitions.base import BaseSensor
from marie.sensors.definitions.event_sensor import EventSensor
from marie.sensors.definitions.manual_sensor import ManualSensor
from marie.sensors.definitions.polling_sensor import PollingSensor
from marie.sensors.definitions.run_status_sensor import RunStatusSensor
from marie.sensors.definitions.schedule_sensor import ScheduleSensor
from marie.sensors.definitions.webhook_sensor import WebhookSensor

__all__ = [
    "BaseSensor",
    "ManualSensor",
    "ScheduleSensor",
    "WebhookSensor",
    "PollingSensor",
    "EventSensor",
    "RunStatusSensor",
]
