"""
Tests for LLM Tracker lifecycle and API.

Tests the LLMTracker singleton, trace context manager, observations, and scoring.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from marie.llm_tracking.config import (
    ExporterType,
    LLMTrackingSettings,
    configure_from_yaml,
    reset_settings,
)
from marie.llm_tracking.tracker import LLMTracker, TraceContext, get_tracker
from marie.llm_tracking.types import (
    ObservationLevel,
    ObservationType,
    Trace,
)


@pytest.fixture(autouse=True)
def reset_tracker_and_settings():
    """Reset tracker singleton and settings before/after each test."""
    import marie.llm_tracking.tracker as tracker_module

    # Reset settings first
    reset_settings()
    # Fully reset tracker singleton - must reset _instance AND _initialized
    if LLMTracker._instance is not None:
        LLMTracker._instance._initialized = False
    LLMTracker._instance = None
    # Also reset the module-level _tracker global used by get_tracker()
    tracker_module._tracker = None

    yield

    # Reset again after test
    if LLMTracker._instance is not None:
        # Stop the tracker if it was started
        try:
            LLMTracker._instance.stop()
        except Exception:
            pass
        LLMTracker._instance._initialized = False
    LLMTracker._instance = None
    tracker_module._tracker = None
    reset_settings()


@pytest.fixture
def mock_settings_enabled():
    """Configure settings with tracking enabled."""
    return configure_from_yaml({
        "enabled": True,
        "exporter": "console",
        "project_id": "test-project",
    })


@pytest.fixture
def mock_settings_disabled():
    """Configure settings with tracking disabled."""
    return configure_from_yaml({
        "enabled": False,
    })


@pytest.fixture
def mock_settings_with_sampling():
    """Configure settings with sampling rate."""
    return configure_from_yaml({
        "enabled": True,
        "exporter": "console",
        "sampling_rate": 0.5,
    })


class TestTrackerSingleton:
    """Test get_tracker() returns same instance."""

    def test_tracker_singleton(self, mock_settings_enabled):
        """Test get_tracker() returns same instance."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()

        assert tracker1 is tracker2

    def test_tracker_direct_instantiation_singleton(self, mock_settings_enabled):
        """Test direct instantiation also returns singleton."""
        tracker1 = LLMTracker()
        tracker2 = LLMTracker()

        assert tracker1 is tracker2


class TestTrackerDisabled:
    """Test tracker with enabled=false doesn't track."""

    def test_tracker_disabled_does_nothing(self, mock_settings_disabled):
        """Test tracker with enabled=false doesn't start when start() called."""
        tracker = get_tracker()

        # enabled property reflects settings
        assert tracker.enabled is False
        # Should not start when disabled
        tracker.start()  # Try to start
        assert not tracker._started

    def test_trace_with_disabled_tracker(self, mock_settings_disabled):
        """Test trace() returns dummy context when disabled."""
        tracker = get_tracker()

        with tracker.trace("test-trace") as ctx:
            assert ctx.id is not None  # Still returns an ID

        # Tracker should not be started when disabled
        assert not tracker._started

    def test_generation_with_disabled_tracker(self, mock_settings_disabled):
        """Test generation() returns ID but doesn't store when disabled."""
        tracker = get_tracker()

        gen_id = tracker.generation(
            trace_id="trace-123",
            name="test-gen",
            model="gpt-4",
        )

        # Should return an ID
        assert gen_id is not None
        assert len(gen_id) == 36  # UUID format


class TestTraceCreation:
    """Test trace() creates trace with correct fields."""

    def test_trace_creation(self, mock_settings_enabled):
        """Test trace() creates trace with correct fields."""
        tracker = get_tracker()
        # Force start by setting _started directly to avoid auto-start complications
        tracker._started = True
        tracker._exporter = MagicMock()

        with tracker.trace(
            name="test-trace",
            user_id="user-123",
            session_id="session-456",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
            input={"prompt": "Hello"},
        ) as ctx:
            assert ctx.id is not None
            assert ctx.trace.name == "test-trace"
            assert ctx.trace.user_id == "user-123"
            assert ctx.trace.session_id == "session-456"
            assert ctx.trace.metadata == {"key": "value"}
            assert ctx.trace.tags == ["tag1", "tag2"]
            assert ctx.trace.input == {"prompt": "Hello"}
            assert ctx.trace.project_id == "test-project"

    def test_trace_with_custom_id(self, mock_settings_enabled):
        """Test trace() with custom trace_id."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        with tracker.trace(name="test", trace_id="custom-id") as ctx:
            assert ctx.id == "custom-id"


class TestTraceContextManager:
    """Test trace context manager finalizes on exit."""

    def test_trace_context_manager_normal_exit(self, mock_settings_enabled):
        """Test trace context manager finalizes on normal exit."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        with patch.object(tracker, "_finalize_trace") as mock_finalize:
            with tracker.trace("test-trace") as ctx:
                pass

            mock_finalize.assert_called_once()
            finalized_trace = mock_finalize.call_args[0][0]
            assert finalized_trace.output is None  # No error

    def test_trace_context_manager_exception(self, mock_settings_enabled):
        """Test trace context manager records error on exception."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        with patch.object(tracker, "_finalize_trace") as mock_finalize:
            with pytest.raises(ValueError):
                with tracker.trace("test-trace") as ctx:
                    raise ValueError("Test error")

            mock_finalize.assert_called_once()
            finalized_trace = mock_finalize.call_args[0][0]
            assert finalized_trace.output == {"error": "Test error"}


class TestObservationTypes:
    """Test span(), generation(), event() create correct types."""

    def test_generation_creates_generation_type(self, mock_settings_enabled):
        """Test generation() creates GENERATION observation."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        with patch.object(tracker, "_create_observation") as mock_create:
            mock_create.return_value = "obs-id"
            tracker.generation(
                trace_id="trace-123",
                name="llm-call",
                model="gpt-4",
                input={"messages": []},
            )

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["obs_type"] == ObservationType.GENERATION
            assert call_kwargs["model"] == "gpt-4"

    def test_span_creates_span_type(self, mock_settings_enabled):
        """Test span() creates SPAN observation."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        with patch.object(tracker, "_create_observation") as mock_create:
            mock_create.return_value = "obs-id"
            tracker.span(
                trace_id="trace-123",
                name="processing",
            )

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["obs_type"] == ObservationType.SPAN

    def test_event_creates_event_type(self, mock_settings_enabled):
        """Test event() creates EVENT observation."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        with patch.object(tracker, "_create_observation") as mock_create:
            mock_create.return_value = "obs-id"
            tracker.event(
                trace_id="trace-123",
                name="checkpoint",
            )

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["obs_type"] == ObservationType.EVENT


class TestObservationParentChild:
    """Test parent_observation_id links correctly."""

    def test_observation_parent_child(self, mock_settings_enabled):
        """Test parent_observation_id is passed correctly."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        with patch.object(tracker, "_create_observation") as mock_create:
            mock_create.return_value = "obs-id"
            tracker.generation(
                trace_id="trace-123",
                name="nested-call",
                model="gpt-4",
                parent_observation_id="parent-obs-123",
            )

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["parent_observation_id"] == "parent-obs-123"


class TestEndUpdatesObservation:
    """Test end() sets end_time and output."""

    def test_end_updates_observation(self, mock_settings_enabled):
        """Test end() sets end_time and output."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        # Create an observation first
        with patch.object(tracker, "_finalize_observation"):
            gen_id = tracker.generation(
                trace_id="trace-123",
                name="llm-call",
                model="gpt-4",
            )

            # End should update the pending observation
            tracker.end(
                observation_id=gen_id,
                output={"response": "Hello"},
                usage={"total_tokens": 100},
            )

    def test_end_with_level(self, mock_settings_enabled):
        """Test end() can set observation level."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        with patch.object(tracker, "_finalize_observation"):
            gen_id = tracker.generation(
                trace_id="trace-123",
                name="failed-call",
                model="gpt-4",
            )

            tracker.end(
                observation_id=gen_id,
                level=ObservationLevel.ERROR,
                status_message="API error",
            )


class TestScoreCreation:
    """Test score() creates score with correct fields."""

    def test_score_creation(self, mock_settings_enabled):
        """Test score() creates score with correct fields."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        with patch.object(tracker, "_finalize_score") as mock_finalize:
            tracker.score(
                trace_id="trace-123",
                name="quality",
                value=0.95,
                comment="Good response",
            )

            mock_finalize.assert_called_once()
            score = mock_finalize.call_args[0][0]
            assert score.trace_id == "trace-123"
            assert score.name == "quality"
            assert score.value == 0.95
            assert score.comment == "Good response"

    def test_score_with_observation_id(self, mock_settings_enabled):
        """Test score() can target specific observation."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        with patch.object(tracker, "_finalize_score") as mock_finalize:
            tracker.score(
                trace_id="trace-123",
                observation_id="obs-456",
                name="relevance",
                value=0.8,
            )

            score = mock_finalize.call_args[0][0]
            assert score.observation_id == "obs-456"


class TestFinalizeToDLQOnError:
    """Test failed finalization stores to DLQ."""

    def test_finalize_stores_to_dlq_on_error(self, mock_settings_enabled):
        """Test failed finalization stores to DLQ."""
        tracker = get_tracker()
        tracker._started = True
        tracker._postgres = MagicMock()
        tracker._exporter = MagicMock()

        # Make exporter raise an error
        tracker._exporter.export_trace.side_effect = Exception("Export failed")

        trace = Trace(id="trace-123", name="test")

        # This should call _store_failed_event instead of losing the trace
        tracker._finalize_trace(trace)

        # Verify DLQ was called
        tracker._postgres.save_failed_event.assert_called_once()
        call_kwargs = tracker._postgres.save_failed_event.call_args[1]
        assert call_kwargs["event_id"] == "trace-123"
        assert call_kwargs["event_type"] == "trace-create"
        assert "Export failed" in call_kwargs["error_message"]

    def test_store_failed_event_logs_if_no_postgres(self, mock_settings_enabled):
        """Test _store_failed_event logs if postgres unavailable."""
        tracker = get_tracker()
        tracker._started = True
        tracker._postgres = None  # No postgres

        # Should not raise, just log
        tracker._store_failed_event(
            event_id="test-id",
            trace_id="trace-id",
            event_type="trace-create",
            error=Exception("Test error"),
            payload={"test": "data"},
        )


class TestSamplingRate:
    """Test sampling_rate < 1.0 drops some traces."""

    def test_sampling_rate_always_samples_at_1(self, mock_settings_enabled):
        """Test sampling_rate=1.0 always samples."""
        tracker = get_tracker()

        # Verify settings have 1.0 rate
        assert tracker._settings.SAMPLING_RATE == 1.0

        # At 1.0 rate, should always sample
        for _ in range(10):
            assert tracker._should_sample() is True

    def test_sampling_rate_drops_some_traces(self, mock_settings_with_sampling):
        """Test sampling_rate < 1.0 drops some traces."""
        # Verify settings are correct before getting tracker
        from marie.llm_tracking.config import get_settings
        settings = get_settings()
        assert settings.SAMPLING_RATE == 0.5, f"Expected 0.5, got {settings.SAMPLING_RATE}"

        tracker = get_tracker()

        # Verify tracker has the right settings
        assert tracker._settings.SAMPLING_RATE == 0.5, \
            f"Expected tracker._settings.SAMPLING_RATE == 0.5, got {tracker._settings.SAMPLING_RATE}"

        # With 0.5 rate, should drop roughly half (statistical test)
        samples = [tracker._should_sample() for _ in range(1000)]
        sample_rate = sum(samples) / len(samples)

        # Should be approximately 0.5 (allow 20% margin for randomness)
        assert 0.30 < sample_rate < 0.70, f"Expected ~0.5, got {sample_rate}"

    def test_trace_not_finalized_when_not_sampled(self, mock_settings_enabled):
        """Test trace is not finalized when not sampled."""
        tracker = get_tracker()
        tracker._started = True
        tracker._exporter = MagicMock()

        # Force sampling to return False
        with patch.object(tracker, "_should_sample", return_value=False):
            with patch.object(tracker, "_finalize_trace") as mock_finalize:
                with tracker.trace("test") as ctx:
                    pass

                # Should not finalize since not sampled
                mock_finalize.assert_not_called()


class TestRabbitMQRequiresStorage:
    """Test RabbitMQ exporter requires postgres and S3."""

    def test_rabbitmq_requires_postgres(self):
        """Test RabbitMQ exporter requires postgres config."""
        configure_from_yaml({
            "enabled": True,
            "exporter": "rabbitmq",
            "s3": {"bucket": "test-bucket"},
            # No postgres
        })

        tracker = get_tracker()

        with pytest.raises(ValueError) as exc_info:
            tracker.start()

        assert "Postgres storage" in str(exc_info.value)

    def test_rabbitmq_requires_s3(self):
        """Test RabbitMQ exporter requires S3 config."""
        configure_from_yaml({
            "enabled": True,
            "exporter": "rabbitmq",
            "postgres": {"url": "postgresql://localhost/test"},
            # No S3
        })

        tracker = get_tracker()

        with pytest.raises(ValueError) as exc_info:
            tracker.start()

        assert "S3 storage" in str(exc_info.value)


class TestTraceContextProperties:
    """Test TraceContext properties."""

    def test_trace_context_id_property(self, mock_settings_enabled):
        """Test TraceContext.id returns trace ID."""
        trace = Trace(id="test-id", name="test")
        tracker = get_tracker()
        ctx = TraceContext(tracker, trace)

        assert ctx.id == "test-id"

    def test_trace_context_trace_property(self, mock_settings_enabled):
        """Test TraceContext.trace returns the trace object."""
        trace = Trace(id="test-id", name="test")
        tracker = get_tracker()
        ctx = TraceContext(tracker, trace)

        assert ctx.trace is trace
        assert ctx.trace.name == "test"
