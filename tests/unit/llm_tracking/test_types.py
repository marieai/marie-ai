"""
Tests for LLM tracking data types and serialization.

Tests Trace, Observation, Score, RawEvent, QueueMessage, Usage, and Cost classes.
"""

import json
from datetime import datetime
from decimal import Decimal

import pytest

from marie.llm_tracking.types import (
    Cost,
    EventType,
    Observation,
    ObservationLevel,
    ObservationType,
    QueueMessage,
    RawEvent,
    Score,
    ScoreDataType,
    Trace,
    Usage,
)


class TestTraceToDict:
    """Test Trace serializes all fields correctly."""

    def test_trace_to_dict_minimal(self):
        """Test Trace with minimal fields serializes correctly."""
        trace = Trace(
            id="test-trace-id",
            name="test-trace",
            project_id="test-project",
        )
        result = trace.to_dict()

        assert result["id"] == "test-trace-id"
        assert result["name"] == "test-trace"
        assert result["project_id"] == "test-project"
        assert result["user_id"] is None
        assert result["session_id"] is None
        assert result["input"] is None
        assert result["output"] is None
        assert result["metadata"] == "{}"
        assert result["tags"] == []
        assert result["is_deleted"] == 0

    def test_trace_to_dict_full(self):
        """Test Trace with all fields serializes correctly."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        trace = Trace(
            id="test-trace-id",
            name="test-trace",
            project_id="test-project",
            user_id="user-123",
            session_id="session-456",
            input={"prompt": "Hello"},
            output={"response": "World"},
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
            timestamp=timestamp,
            created_at=timestamp,
            updated_at=timestamp,
            release="v1.0.0",
            version="1",
            is_deleted=False,
        )
        result = trace.to_dict()

        assert result["id"] == "test-trace-id"
        assert result["user_id"] == "user-123"
        assert result["session_id"] == "session-456"
        assert json.loads(result["input"]) == {"prompt": "Hello"}
        assert json.loads(result["output"]) == {"response": "World"}
        assert json.loads(result["metadata"]) == {"key": "value"}
        assert result["tags"] == ["tag1", "tag2"]
        assert result["release"] == "v1.0.0"
        assert result["version"] == "1"
        assert result["is_deleted"] == 0

    def test_trace_is_deleted_serializes_to_1(self):
        """Test is_deleted=True serializes to 1."""
        trace = Trace(is_deleted=True)
        result = trace.to_dict()
        assert result["is_deleted"] == 1


class TestObservationToDict:
    """Test Observation serializes all fields correctly."""

    def test_observation_to_dict_minimal(self):
        """Test Observation with minimal fields serializes correctly."""
        obs = Observation(
            id="obs-id",
            trace_id="trace-id",
            type=ObservationType.SPAN,
            name="test-span",
        )
        result = obs.to_dict()

        assert result["id"] == "obs-id"
        assert result["trace_id"] == "trace-id"
        assert result["type"] == "SPAN"
        assert result["name"] == "test-span"
        assert result["parent_observation_id"] is None
        assert result["model"] is None
        assert result["level"] == "DEFAULT"

    def test_observation_to_dict_generation_with_usage(self):
        """Test GENERATION observation with usage serializes correctly."""
        usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        cost = Cost(
            input_cost=Decimal("0.001"),
            output_cost=Decimal("0.002"),
            total_cost=Decimal("0.003"),
        )
        obs = Observation(
            id="gen-id",
            trace_id="trace-id",
            type=ObservationType.GENERATION,
            name="llm-call",
            model="gpt-4",
            model_parameters={"temperature": 0.7},
            usage=usage,
            cost=cost,
        )
        result = obs.to_dict()

        assert result["type"] == "GENERATION"
        assert result["model"] == "gpt-4"
        assert json.loads(result["model_parameters"]) == {"temperature": 0.7}
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["input_cost"] == 0.001
        assert result["output_cost"] == 0.002
        assert result["total_cost"] == 0.003

    def test_observation_levels(self):
        """Test all observation levels serialize correctly."""
        for level in ObservationLevel:
            obs = Observation(level=level)
            result = obs.to_dict()
            assert result["level"] == level.value


class TestScoreToDict:
    """Test Score serializes all fields correctly."""

    def test_score_to_dict_numeric(self):
        """Test numeric Score serializes correctly."""
        score = Score(
            id="score-id",
            trace_id="trace-id",
            name="quality",
            value=0.95,
            data_type=ScoreDataType.NUMERIC,
            source="API",
        )
        result = score.to_dict()

        assert result["id"] == "score-id"
        assert result["trace_id"] == "trace-id"
        assert result["name"] == "quality"
        assert result["value"] == 0.95
        assert result["data_type"] == "NUMERIC"
        assert result["source"] == "API"

    def test_score_to_dict_categorical(self):
        """Test categorical Score serializes correctly."""
        score = Score(
            id="score-id",
            trace_id="trace-id",
            name="category",
            value="good",
            data_type=ScoreDataType.CATEGORICAL,
        )
        result = score.to_dict()

        assert result["value"] == "good"
        assert result["data_type"] == "CATEGORICAL"

    def test_score_to_dict_boolean(self):
        """Test boolean Score serializes correctly."""
        score = Score(
            id="score-id",
            trace_id="trace-id",
            name="passed",
            value=True,
            data_type=ScoreDataType.BOOLEAN,
        )
        result = score.to_dict()

        assert result["value"] is True
        assert result["data_type"] == "BOOLEAN"

    def test_score_with_observation_id(self):
        """Test Score with observation_id serializes correctly."""
        score = Score(
            trace_id="trace-id",
            observation_id="obs-id",
            name="test",
            value=1.0,
        )
        result = score.to_dict()

        assert result["observation_id"] == "obs-id"


class TestRawEventFromDict:
    """Test RawEvent serialization and deserialization."""

    def test_raw_event_to_dict(self):
        """Test RawEvent serializes all fields correctly."""
        event = RawEvent(
            id="event-id",
            trace_id="trace-id",
            event_type=EventType.GENERATION_CREATE,
            s3_key="llm-events/2024/01/15/trace-id/gen_event-id.json.gz",
            model_name="gpt-4",
            model_provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            duration_ms=1500,
            time_to_first_token_ms=200,
            cost_usd=Decimal("0.005"),
            user_id="user-123",
            session_id="session-456",
            tags=["tag1", "tag2"],
            status="pending",
        )
        result = event.to_dict()

        assert result["id"] == "event-id"
        assert result["trace_id"] == "trace-id"
        assert result["event_type"] == "generation-create"
        assert result["s3_key"] == "llm-events/2024/01/15/trace-id/gen_event-id.json.gz"
        assert result["model_name"] == "gpt-4"
        assert result["model_provider"] == "openai"
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["duration_ms"] == 1500
        assert result["time_to_first_token_ms"] == 200
        assert result["cost_usd"] == 0.005
        assert result["user_id"] == "user-123"
        assert result["session_id"] == "session-456"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["status"] == "pending"

    def test_raw_event_status_values(self):
        """Test all RawEvent status values."""
        for status in ["pending", "processed", "failed"]:
            event = RawEvent(status=status)
            result = event.to_dict()
            assert result["status"] == status


class TestQueueMessageRoundTrip:
    """Test QueueMessage serializes and deserializes."""

    def test_queue_message_to_dict(self):
        """Test QueueMessage serializes correctly."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        msg = QueueMessage(
            event_id="event-id",
            event_type=EventType.TRACE_CREATE,
            trace_id="trace-id",
            project_id="test-project",
            timestamp=timestamp,
        )
        result = msg.to_dict()

        assert result["event_id"] == "event-id"
        assert result["event_type"] == "trace-create"
        assert result["trace_id"] == "trace-id"
        assert result["project_id"] == "test-project"
        assert result["timestamp"] == "2024-01-15T10:30:00"

    def test_queue_message_from_dict(self):
        """Test QueueMessage deserializes correctly."""
        data = {
            "event_id": "event-id",
            "event_type": "trace-create",
            "trace_id": "trace-id",
            "project_id": "test-project",
            "timestamp": "2024-01-15T10:30:00",
        }
        msg = QueueMessage.from_dict(data)

        assert msg.event_id == "event-id"
        assert msg.event_type == EventType.TRACE_CREATE
        assert msg.trace_id == "trace-id"
        assert msg.project_id == "test-project"
        assert msg.timestamp == datetime(2024, 1, 15, 10, 30, 0)

    def test_queue_message_round_trip(self):
        """Test QueueMessage round trip (to_dict -> from_dict)."""
        original = QueueMessage(
            event_id="event-123",
            event_type=EventType.GENERATION_CREATE,
            trace_id="trace-456",
            project_id="my-project",
        )
        data = original.to_dict()
        restored = QueueMessage.from_dict(data)

        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type
        assert restored.trace_id == original.trace_id
        assert restored.project_id == original.project_id

    def test_queue_message_default_project_id(self):
        """Test QueueMessage uses default project_id if not in dict."""
        data = {
            "event_id": "event-id",
            "event_type": "trace-create",
            "trace_id": "trace-id",
            "timestamp": "2024-01-15T10:30:00",
        }
        msg = QueueMessage.from_dict(data)

        assert msg.project_id == "default"


class TestUsageCalculation:
    """Test Usage total_tokens calculation."""

    def test_usage_to_dict_all_fields(self):
        """Test Usage with all fields serializes correctly."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            usage_details={"cached_tokens": 20, "reasoning_tokens": 10},
        )
        result = usage.to_dict()

        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["usage_details"] == {"cached_tokens": 20, "reasoning_tokens": 10}

    def test_usage_to_dict_minimal(self):
        """Test Usage with minimal fields serializes correctly."""
        usage = Usage()
        result = usage.to_dict()

        assert result == {}

    def test_usage_to_dict_partial(self):
        """Test Usage with partial fields serializes only provided fields."""
        usage = Usage(input_tokens=100)
        result = usage.to_dict()

        assert result == {"input_tokens": 100}
        assert "output_tokens" not in result
        assert "total_tokens" not in result


class TestCostCalculation:
    """Test Cost total_cost calculation."""

    def test_cost_to_dict_all_fields(self):
        """Test Cost with all fields serializes correctly."""
        cost = Cost(
            input_cost=Decimal("0.001"),
            output_cost=Decimal("0.002"),
            total_cost=Decimal("0.003"),
            cost_details={"cached_cost": Decimal("0.0005")},
        )
        result = cost.to_dict()

        assert result["input_cost"] == 0.001
        assert result["output_cost"] == 0.002
        assert result["total_cost"] == 0.003
        assert result["cost_details"] == {"cached_cost": 0.0005}

    def test_cost_to_dict_minimal(self):
        """Test Cost with minimal fields serializes correctly."""
        cost = Cost()
        result = cost.to_dict()

        assert result == {}

    def test_cost_decimal_to_float_conversion(self):
        """Test Decimal values are converted to float in serialization."""
        cost = Cost(total_cost=Decimal("0.123456789"))
        result = cost.to_dict()

        assert isinstance(result["total_cost"], float)
        assert result["total_cost"] == 0.123456789


class TestEventTypes:
    """Test EventType enum values."""

    def test_all_event_types_exist(self):
        """Test all expected event types exist."""
        expected_types = [
            "trace-create",
            "trace-update",
            "span-create",
            "span-update",
            "generation-create",
            "generation-update",
            "event-create",
            "score-create",
        ]
        actual_values = [e.value for e in EventType]

        for expected in expected_types:
            assert expected in actual_values, f"Missing event type: {expected}"

    def test_event_type_from_string(self):
        """Test EventType can be created from string value."""
        assert EventType("trace-create") == EventType.TRACE_CREATE
        assert EventType("generation-create") == EventType.GENERATION_CREATE


class TestObservationTypes:
    """Test ObservationType enum values."""

    def test_all_observation_types_exist(self):
        """Test all observation types exist."""
        assert ObservationType.SPAN.value == "SPAN"
        assert ObservationType.GENERATION.value == "GENERATION"
        assert ObservationType.EVENT.value == "EVENT"


class TestDefaultIds:
    """Test default ID generation."""

    def test_trace_generates_uuid(self):
        """Test Trace generates UUID by default."""
        trace = Trace()
        assert len(trace.id) == 36  # UUID format
        assert "-" in trace.id

    def test_observation_generates_uuid(self):
        """Test Observation generates UUID by default."""
        obs = Observation()
        assert len(obs.id) == 36

    def test_score_generates_uuid(self):
        """Test Score generates UUID by default."""
        score = Score()
        assert len(score.id) == 36

    def test_raw_event_generates_uuid(self):
        """Test RawEvent generates UUID by default."""
        event = RawEvent()
        assert len(event.id) == 36
