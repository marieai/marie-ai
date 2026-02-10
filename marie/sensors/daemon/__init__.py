"""
Sensor daemon components.

The daemon continuously evaluates active sensors and submits jobs
when sensor conditions are met.
"""

from marie.sensors.daemon.worker import SensorWorker

__all__ = ["SensorWorker"]
