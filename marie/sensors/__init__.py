"""
Marie Sensor/Trigger System

This module implements a Dagster-aligned sensor system for event-driven job triggering.
Sensors evaluate conditions and create jobs when their criteria are met.

Architecture:
- All sensors are evaluated by a background daemon (SensorWorker)
- Event sources (webhooks, message queues) write to a durable event_log
- Sensors poll the event_log using cursor-based pagination
- Idempotency is enforced via run_key deduplication

Key Components:
- types: Core enums and data classes (SensorType, SensorResult, RunRequest)
- context: SensorEvaluationContext for evaluation functions
- definitions: Sensor implementation classes (schedule, webhook, polling, etc.)
- daemon: SensorWorker background service
- state: Storage interfaces and PostgreSQL implementation
- api: REST endpoints for sensor management

Related:
- analysis/sensor-trigger-system-design.md
- analysis/sensor-trigger-implementation-checklist.md
"""

from marie.sensors.config import SensorSettings
from marie.sensors.context import SensorEvaluationContext
from marie.sensors.exceptions import (
    SensorConfigError,
    SensorError,
    SensorEvaluationError,
    SensorNotFoundError,
)
from marie.sensors.types import (
    RunRequest,
    SensorResult,
    SensorStatus,
    SensorType,
    SkipReason,
    TickStatus,
)

__all__ = [
    # Types
    "SensorType",
    "SensorStatus",
    "TickStatus",
    "RunRequest",
    "SkipReason",
    "SensorResult",
    # Context
    "SensorEvaluationContext",
    # Config
    "SensorSettings",
    # Exceptions
    "SensorError",
    "SensorEvaluationError",
    "SensorConfigError",
    "SensorNotFoundError",
]
