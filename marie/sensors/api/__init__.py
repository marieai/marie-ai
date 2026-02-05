"""
REST API endpoints for sensor management.
"""

from marie.sensors.api.rest import router as sensors_router
from marie.sensors.api.webhook_receiver import router as webhook_router

__all__ = ["sensors_router", "webhook_router"]
