"""
LLM Tracking Types - Data models for LLM observability.

Ported from Langfuse TypeScript implementation to Python.
These types mirror the Langfuse data model for traces, observations, and scores.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4


class ObservationType(str, Enum):
    """Types of observations that can be recorded."""

    SPAN = "SPAN"
    GENERATION = "GENERATION"
    EVENT = "EVENT"


class ObservationLevel(str, Enum):
    """Severity levels for observations."""

    DEBUG = "DEBUG"
    DEFAULT = "DEFAULT"
    WARNING = "WARNING"
    ERROR = "ERROR"


class EventType(str, Enum):
    """Types of events that can be published to the queue."""

    TRACE_CREATE = "trace-create"
    TRACE_UPDATE = "trace-update"
    SPAN_CREATE = "span-create"
    SPAN_UPDATE = "span-update"
    GENERATION_CREATE = "generation-create"
    GENERATION_UPDATE = "generation-update"
    EVENT_CREATE = "event-create"
    SCORE_CREATE = "score-create"


class ScoreDataType(str, Enum):
    """Data types for score values."""

    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    BOOLEAN = "BOOLEAN"


@dataclass
class Usage:
    """Token usage information from LLM calls."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    # Additional usage details (e.g., cached_tokens, reasoning_tokens)
    usage_details: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        if self.input_tokens is not None:
            result["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            result["output_tokens"] = self.output_tokens
        if self.total_tokens is not None:
            result["total_tokens"] = self.total_tokens
        if self.usage_details:
            result["usage_details"] = self.usage_details
        return result


@dataclass
class Cost:
    """Cost information for LLM calls."""

    input_cost: Optional[Decimal] = None
    output_cost: Optional[Decimal] = None
    total_cost: Optional[Decimal] = None
    # Additional cost details (e.g., cached_cost)
    cost_details: Dict[str, Decimal] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        if self.input_cost is not None:
            result["input_cost"] = float(self.input_cost)
        if self.output_cost is not None:
            result["output_cost"] = float(self.output_cost)
        if self.total_cost is not None:
            result["total_cost"] = float(self.total_cost)
        if self.cost_details:
            result["cost_details"] = {k: float(v) for k, v in self.cost_details.items()}
        return result


@dataclass
class Trace:
    """
    Root container for a single LLM interaction/request.

    A trace represents a complete interaction, potentially containing multiple
    observations (spans, generations, events).
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    project_id: str = "default"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    release: Optional[str] = None
    version: Optional[str] = None
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "input": json.dumps(self.input) if self.input is not None else None,
            "output": json.dumps(self.output) if self.output is not None else None,
            "metadata": json.dumps(self.metadata) if self.metadata else "{}",
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "release": self.release,
            "version": self.version,
            "is_deleted": 1 if self.is_deleted else 0,
        }


@dataclass
class Observation:
    """
    A single observation within a trace.

    Observations can be:
    - SPAN: A logical unit of work (e.g., function call)
    - GENERATION: An LLM call with token usage
    - EVENT: A point-in-time event (no duration)
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    trace_id: str = ""
    project_id: str = "default"
    parent_observation_id: Optional[str] = None
    type: ObservationType = ObservationType.SPAN
    name: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    completion_start_time: Optional[datetime] = None  # Time to first token

    # Model info (for GENERATION type)
    model: Optional[str] = None
    model_parameters: Optional[Dict[str, Any]] = None

    # Usage & Cost
    usage: Optional[Usage] = None
    cost: Optional[Cost] = None

    # Content
    input: Optional[Any] = None
    output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Status
    level: ObservationLevel = ObservationLevel.DEFAULT
    status_message: Optional[str] = None

    # Versioning
    version: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "trace_id": self.trace_id,
            "project_id": self.project_id,
            "parent_observation_id": self.parent_observation_id,
            "type": self.type.value,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "completion_start_time": (
                self.completion_start_time.isoformat()
                if self.completion_start_time
                else None
            ),
            "model": self.model,
            "model_parameters": (
                json.dumps(self.model_parameters) if self.model_parameters else None
            ),
            "input": json.dumps(self.input) if self.input is not None else None,
            "output": json.dumps(self.output) if self.output is not None else None,
            "metadata": json.dumps(self.metadata) if self.metadata else "{}",
            "level": self.level.value,
            "status_message": self.status_message,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_deleted": 1 if self.is_deleted else 0,
        }

        # Add usage fields
        if self.usage:
            result["input_tokens"] = self.usage.input_tokens
            result["output_tokens"] = self.usage.output_tokens
            result["total_tokens"] = self.usage.total_tokens
            result["usage_details"] = json.dumps(self.usage.usage_details)

        # Add cost fields
        if self.cost:
            result["input_cost"] = (
                float(self.cost.input_cost) if self.cost.input_cost else None
            )
            result["output_cost"] = (
                float(self.cost.output_cost) if self.cost.output_cost else None
            )
            result["total_cost"] = (
                float(self.cost.total_cost) if self.cost.total_cost else None
            )
            result["cost_details"] = (
                json.dumps({k: float(v) for k, v in self.cost.cost_details.items()})
                if self.cost.cost_details
                else "{}"
            )

        return result


@dataclass
class Score:
    """
    Evaluation score for a trace or observation.

    Scores can be numeric, categorical, or boolean values used for
    quality evaluation and feedback.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    trace_id: str = ""
    observation_id: Optional[str] = None
    project_id: str = "default"
    name: str = ""
    value: Union[float, str, bool] = 0.0
    data_type: ScoreDataType = ScoreDataType.NUMERIC
    source: str = "API"  # API, EVAL, ANNOTATION
    comment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "observation_id": self.observation_id,
            "project_id": self.project_id,
            "name": self.name,
            "value": self.value,
            "data_type": self.data_type.value,
            "source": self.source,
            "comment": self.comment,
            "metadata": json.dumps(self.metadata) if self.metadata else "{}",
            "timestamp": self.timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_deleted": 1 if self.is_deleted else 0,
        }


@dataclass
class RawEvent:
    """
    Raw event stored in Postgres before processing.

    This represents the initial capture of an LLM event before it's
    normalized and written to ClickHouse.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    trace_id: str = ""
    event_type: EventType = EventType.TRACE_CREATE
    s3_key: Optional[str] = None  # If payload stored in S3
    payload: Optional[Dict[str, Any]] = None  # If payload is small, stored inline
    status: Literal["pending", "processed", "failed"] = "pending"
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "event_type": self.event_type.value,
            "s3_key": self.s3_key,
            "payload": json.dumps(self.payload) if self.payload else None,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "processed_at": (
                self.processed_at.isoformat() if self.processed_at else None
            ),
        }


@dataclass
class QueueMessage:
    """
    Message published to RabbitMQ for async processing.

    This is a lightweight message that references the raw event in Postgres.
    """

    event_id: str
    event_type: EventType
    trace_id: str
    project_id: str = "default"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "trace_id": self.trace_id,
            "project_id": self.project_id,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueMessage":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            trace_id=data["trace_id"],
            project_id=data.get("project_id", "default"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
