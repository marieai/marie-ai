"""
Core types for the sensor/trigger system.

These types map directly to the PostgreSQL enums and provide
the data structures used throughout the sensor evaluation pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SensorType(str, Enum):
    """
    Sensor types matching n8n trigger patterns.

    Maps to marie_scheduler.sensor_type enum in PostgreSQL.
    """

    MANUAL = "manual"  # Click-to-run for testing
    SCHEDULE = "schedule"  # Cron/time-based
    WEBHOOK = "webhook"  # HTTP endpoint (ingest to event_log)
    POLLING = "polling"  # External API polling
    EVENT = "event"  # Message queue (RabbitMQ/Kafka, ingest to event_log)
    RUN_STATUS = "run_status"  # Job completion monitoring
    ASSET = "asset"  # Asset materialization


class SensorStatus(str, Enum):
    """
    Sensor operational status.

    Maps to marie_scheduler.sensor_status enum in PostgreSQL.
    """

    ACTIVE = "active"  # Running, evaluating
    INACTIVE = "inactive"  # Stopped, not evaluating
    PAUSED = "paused"  # Temporarily paused
    ERROR = "error"  # Failed, needs attention


class TickStatus(str, Enum):
    """
    Status of a sensor evaluation tick.

    Maps to marie_scheduler.tick_status enum in PostgreSQL.

    The STARTED status enables two-phase commit for crash recovery:
    1. Create tick in STARTED state with reserved_run_ids
    2. Submit jobs
    3. Update tick to SUCCESS/FAILED

    On restart, STARTED ticks are resumed.
    """

    STARTED = "started"  # In progress (for crash recovery)
    SUCCESS = "success"  # Evaluated successfully, may have fired jobs
    SKIPPED = "skipped"  # Evaluated, no action needed (e.g., cron not due)
    FAILED = "failed"  # Evaluation error

    def is_terminal(self) -> bool:
        """Check if this status is terminal (evaluation complete)."""
        return self in (TickStatus.SUCCESS, TickStatus.SKIPPED, TickStatus.FAILED)


@dataclass
class RunRequest:
    """
    Request to submit a job when a sensor fires.

    The run_key provides idempotency - if a run_key has already been
    processed by this sensor, the job submission is skipped.
    """

    run_key: Optional[str] = None  # Idempotency key
    job_name: Optional[str] = None  # Target job/queue name
    dag_id: Optional[str] = None  # Target DAG
    run_config: Dict[str, Any] = field(default_factory=dict)  # Job configuration
    tags: Dict[str, str] = field(default_factory=dict)  # Job tags
    priority: int = 1  # Job priority (higher = more urgent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_key": self.run_key,
            "job_name": self.job_name,
            "dag_id": self.dag_id,
            "run_config": self.run_config,
            "tags": self.tags,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRequest":
        """Create from dictionary."""
        return cls(
            run_key=data.get("run_key"),
            job_name=data.get("job_name"),
            dag_id=data.get("dag_id"),
            run_config=data.get("run_config", {}),
            tags=data.get("tags", {}),
            priority=data.get("priority", 1),
        )


@dataclass
class SkipReason:
    """Reason for skipping sensor evaluation."""

    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {"message": self.message}


@dataclass
class SensorResult:
    """
    Result of a sensor evaluation.

    A sensor evaluation can result in:
    - One or more RunRequests (jobs to submit)
    - A SkipReason (no action needed this tick)
    - Both (fire some jobs, but also note a partial skip)

    The cursor is optional state that persists between evaluations.
    For event_log-based sensors, this is typically the last processed event_log_id.
    """

    run_requests: List[RunRequest] = field(default_factory=list)
    skip_reason: Optional[SkipReason] = None
    cursor: Optional[str] = None  # State for next evaluation

    @staticmethod
    def skip(message: str, cursor: Optional[str] = None) -> "SensorResult":
        """Create a result indicating the sensor should skip this evaluation."""
        return SensorResult(skip_reason=SkipReason(message), cursor=cursor)

    @staticmethod
    def fire(
        run_key: str,
        job_name: Optional[str] = None,
        dag_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        priority: int = 1,
        cursor: Optional[str] = None,
    ) -> "SensorResult":
        """Create a result with a single RunRequest."""
        return SensorResult(
            run_requests=[
                RunRequest(
                    run_key=run_key,
                    job_name=job_name,
                    dag_id=dag_id,
                    run_config=run_config or {},
                    tags=tags or {},
                    priority=priority,
                )
            ],
            cursor=cursor,
        )

    @staticmethod
    def fire_multiple(
        run_requests: List[RunRequest], cursor: Optional[str] = None
    ) -> "SensorResult":
        """Create a result with multiple RunRequests."""
        return SensorResult(run_requests=run_requests, cursor=cursor)

    def has_run_requests(self) -> bool:
        """Check if this result contains any run requests."""
        return len(self.run_requests) > 0

    def was_skipped(self) -> bool:
        """Check if this result indicates a skip."""
        return self.skip_reason is not None and not self.has_run_requests()

    def to_tick_status(self) -> TickStatus:
        """Determine the appropriate tick status for this result."""
        if self.has_run_requests():
            return TickStatus.SUCCESS
        elif self.skip_reason is not None:
            return TickStatus.SKIPPED
        else:
            # No run requests and no explicit skip = implicit skip
            return TickStatus.SKIPPED
