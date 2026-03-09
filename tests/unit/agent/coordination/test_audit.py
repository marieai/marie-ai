"""Unit tests for AuditLogger."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from marie.agent.coordination.audit import (
    AuditEvent,
    AuditEventType,
    InMemoryAuditLogger,
    StructuredAuditLogger,
    create_audit_logger,
)


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_all_event_types_defined(self):
        """Ensure all expected event types exist."""
        assert AuditEventType.MESSAGE_SENT == "message_sent"
        assert AuditEventType.MESSAGE_RECEIVED == "message_received"
        assert AuditEventType.AGENT_STARTED == "agent_started"
        assert AuditEventType.AGENT_COMPLETED == "agent_completed"
        assert AuditEventType.AGENT_FAILED == "agent_failed"
        assert AuditEventType.WORKFLOW_STARTED == "workflow_started"
        assert AuditEventType.WORKFLOW_COMPLETED == "workflow_completed"
        assert AuditEventType.WORKFLOW_FAILED == "workflow_failed"
        assert AuditEventType.CHECKPOINT_SAVED == "checkpoint_saved"
        assert AuditEventType.CHECKPOINT_RESTORED == "checkpoint_restored"
        assert AuditEventType.ROUTING_DECISION == "routing_decision"


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_event_creation_minimal(self):
        """Test creating event with minimal fields."""
        event = AuditEvent(
            event_type=AuditEventType.AGENT_STARTED,
            workflow_id="wf-123",
        )
        assert event.event_type == AuditEventType.AGENT_STARTED
        assert event.workflow_id == "wf-123"
        assert event.timestamp is not None
        assert event.agent_name is None
        assert event.message_id is None
        assert event.details == {}
        assert event.trace_id is None

    def test_event_creation_full(self):
        """Test creating event with all fields."""
        now = datetime.now(timezone.utc)
        event = AuditEvent(
            event_type=AuditEventType.AGENT_COMPLETED,
            workflow_id="wf-456",
            timestamp=now,
            agent_name="executor",
            message_id="msg-789",
            details={"duration_ms": 150.5},
            trace_id="trace-abc",
            span_id="span-def",
            parent_span_id="span-parent",
        )
        assert event.event_type == AuditEventType.AGENT_COMPLETED
        assert event.workflow_id == "wf-456"
        assert event.timestamp == now
        assert event.agent_name == "executor"
        assert event.message_id == "msg-789"
        assert event.details["duration_ms"] == 150.5
        assert event.trace_id == "trace-abc"
        assert event.span_id == "span-def"
        assert event.parent_span_id == "span-parent"

    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = AuditEvent(
            event_type=AuditEventType.WORKFLOW_STARTED,
            workflow_id="wf-dict",
            agent_name="planner",
            details={"goal": "Test"},
        )
        data = event.to_dict()

        assert data["event_type"] == "workflow_started"
        assert data["workflow_id"] == "wf-dict"
        assert data["agent_name"] == "planner"
        assert data["details"]["goal"] == "Test"
        assert "timestamp" in data


class TestInMemoryAuditLogger:
    """Tests for InMemoryAuditLogger."""

    @pytest.fixture
    def logger(self):
        """Fresh audit logger for testing."""
        return InMemoryAuditLogger()

    @pytest.mark.asyncio
    async def test_log_event(self, logger):
        """Test logging a single event."""
        event = AuditEvent(
            event_type=AuditEventType.AGENT_STARTED,
            workflow_id="wf-test",
            agent_name="planner",
        )
        await logger.log(event)

        events = await logger.query()
        assert len(events) == 1
        assert events[0] == event

    @pytest.mark.asyncio
    async def test_log_multiple_events(self, logger):
        """Test logging multiple events."""
        for i in range(5):
            await logger.log(AuditEvent(
                event_type=AuditEventType.AGENT_STARTED,
                workflow_id=f"wf-{i}",
                agent_name=f"agent_{i}",
            ))

        events = await logger.query()
        assert len(events) == 5

    @pytest.mark.asyncio
    async def test_query_by_workflow_id(self, logger):
        """Test filtering events by workflow ID."""
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-a", agent_name="a"))
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-b", agent_name="b"))
        await logger.log(AuditEvent(AuditEventType.AGENT_COMPLETED, "wf-a", agent_name="a"))

        wf_a_events = await logger.query(workflow_id="wf-a")
        assert len(wf_a_events) == 2
        assert all(e.workflow_id == "wf-a" for e in wf_a_events)

    @pytest.mark.asyncio
    async def test_query_by_agent_name(self, logger):
        """Test filtering events by agent name."""
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-1", agent_name="planner"))
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-1", agent_name="executor"))
        await logger.log(AuditEvent(AuditEventType.AGENT_COMPLETED, "wf-1", agent_name="planner"))

        planner_events = await logger.query(agent_name="planner")
        assert len(planner_events) == 2
        assert all(e.agent_name == "planner" for e in planner_events)

    @pytest.mark.asyncio
    async def test_query_by_event_type(self, logger):
        """Test filtering events by type."""
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-1"))
        await logger.log(AuditEvent(AuditEventType.AGENT_COMPLETED, "wf-1"))
        await logger.log(AuditEvent(AuditEventType.AGENT_FAILED, "wf-1"))

        started_events = await logger.query(event_type=AuditEventType.AGENT_STARTED)
        assert len(started_events) == 1
        assert started_events[0].event_type == AuditEventType.AGENT_STARTED

    @pytest.mark.asyncio
    async def test_query_by_time_range(self, logger):
        """Test filtering events by time range."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-1", timestamp=past))
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-2", timestamp=now))
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-3", timestamp=future))

        # Events from now onwards
        recent = await logger.query(start_time=now)
        assert len(recent) == 2

        # Events before future
        before_future = await logger.query(end_time=now)
        assert len(before_future) == 2

    @pytest.mark.asyncio
    async def test_query_with_limit(self, logger):
        """Test limiting query results."""
        for i in range(10):
            await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, f"wf-{i}"))

        limited = await logger.query(limit=5)
        assert len(limited) == 5

    @pytest.mark.asyncio
    async def test_max_events_limit(self, logger):
        """Test circular buffer behavior."""
        small_logger = InMemoryAuditLogger(max_events=5)

        for i in range(10):
            await small_logger.log(AuditEvent(AuditEventType.AGENT_STARTED, f"wf-{i}"))

        events = await small_logger.query()
        assert len(events) == 5
        # Should have the last 5 events
        assert events[0].workflow_id == "wf-5"
        assert events[-1].workflow_id == "wf-9"

    @pytest.mark.asyncio
    async def test_clear(self, logger):
        """Test clearing all events."""
        for i in range(5):
            await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, f"wf-{i}"))

        logger.clear()
        events = await logger.query()
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_combined_filters(self, logger):
        """Test combining multiple filters."""
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-a", agent_name="planner"))
        await logger.log(AuditEvent(AuditEventType.AGENT_COMPLETED, "wf-a", agent_name="planner"))
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-a", agent_name="executor"))
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-b", agent_name="planner"))

        # wf-a AND planner
        results = await logger.query(
            workflow_id="wf-a",
            agent_name="planner",
        )
        assert len(results) == 2

        # wf-a AND planner AND STARTED
        results = await logger.query(
            workflow_id="wf-a",
            agent_name="planner",
            event_type=AuditEventType.AGENT_STARTED,
        )
        assert len(results) == 1


class TestStructuredAuditLogger:
    """Tests for StructuredAuditLogger."""

    @pytest.fixture
    def logger(self):
        """Structured audit logger for testing."""
        return StructuredAuditLogger()

    @pytest.mark.asyncio
    async def test_log_event(self, logger):
        """Test logging event to structured logger."""
        event = AuditEvent(
            event_type=AuditEventType.WORKFLOW_STARTED,
            workflow_id="wf-structured",
            details={"goal": "Test structured logging"},
        )
        await logger.log(event)

        # Should be queryable
        events = await logger.query(workflow_id="wf-structured")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_query_functionality(self, logger):
        """Test that query works same as in-memory."""
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-1", agent_name="a"))
        await logger.log(AuditEvent(AuditEventType.AGENT_STARTED, "wf-2", agent_name="b"))

        results = await logger.query(agent_name="a")
        assert len(results) == 1


class TestCreateAuditLogger:
    """Tests for audit logger factory."""

    def test_create_enabled_logger(self):
        """Test creating enabled logger."""
        logger = create_audit_logger(enabled=True)
        assert isinstance(logger, StructuredAuditLogger)

    def test_create_disabled_logger(self):
        """Test creating disabled logger."""
        logger = create_audit_logger(enabled=False)
        assert isinstance(logger, InMemoryAuditLogger)
        # Disabled logger has max_events=0
        assert logger._max_events == 0
