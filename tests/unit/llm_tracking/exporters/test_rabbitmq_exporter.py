"""
Tests for RabbitMQ exporter.

Tests exporter lifecycle, error handling, and message publishing.
"""

from unittest.mock import MagicMock, patch

import pytest

from marie.llm_tracking.config import configure_from_yaml, reset_settings
from marie.llm_tracking.exporters.rabbitmq import RabbitMQExporter
from marie.llm_tracking.types import (
    EventType,
    Observation,
    ObservationType,
    Score,
    Trace,
)


@pytest.fixture(autouse=True)
def reset_settings_between_tests():
    """Reset settings before and after each test."""
    reset_settings()
    configure_from_yaml({
        "enabled": True,
        "exporter": "rabbitmq",
        "rabbitmq": {
            "url": "amqp://guest:guest@localhost:5672/",
            "exchange": "test-exchange",
            "routing_key": "test.key",
        },
    })
    yield
    reset_settings()


@pytest.fixture
def mock_pika_client():
    """Create mock BlockingPikaClient."""
    # BlockingPikaClient is imported inside start() from marie.messaging.rabbitmq.client
    with patch("marie.messaging.rabbitmq.client.BlockingPikaClient") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


class TestExporterNotStartedRaises:
    """Test publish raises if not started."""

    def test_exporter_not_started_raises(self):
        """Test publish raises RuntimeError if not started."""
        exporter = RabbitMQExporter()
        # Don't call start()

        trace = Trace(id="trace-123", name="test")

        with pytest.raises(RuntimeError) as exc_info:
            exporter.export_trace(trace)

        assert "not started" in str(exc_info.value)

    def test_exporter_not_started_raises_for_observation(self):
        """Test observation export raises if not started."""
        exporter = RabbitMQExporter()

        obs = Observation(
            id="obs-123",
            trace_id="trace-123",
            type=ObservationType.GENERATION,
        )

        with pytest.raises(RuntimeError) as exc_info:
            exporter.export_observation(obs)

        assert "not started" in str(exc_info.value)

    def test_exporter_not_started_raises_for_score(self):
        """Test score export raises if not started."""
        exporter = RabbitMQExporter()

        score = Score(
            id="score-123",
            trace_id="trace-123",
            name="quality",
            value=0.95,
        )

        with pytest.raises(RuntimeError) as exc_info:
            exporter.export_score(score)

        assert "not started" in str(exc_info.value)


class TestExporterPublishFailureRaises:
    """Test publish failure re-raises exception."""

    def test_exporter_publish_failure_raises(self, mock_pika_client):
        """Test publish failure re-raises exception."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        # Make publish raise an error
        mock_pika_client.publish_message.side_effect = Exception("Connection lost")

        trace = Trace(id="trace-123", name="test")

        with pytest.raises(Exception) as exc_info:
            exporter.export_trace(trace)

        assert "Connection lost" in str(exc_info.value)


class TestExportTracePublishesMessage:
    """Test export_trace publishes correct message."""

    def test_export_trace_publishes_message(self, mock_pika_client):
        """Test export_trace publishes correct message format."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        trace = Trace(
            id="trace-123",
            name="test-trace",
            project_id="test-project",
        )

        exporter.export_trace(trace)

        mock_pika_client.publish_message.assert_called_once()
        call_kwargs = mock_pika_client.publish_message.call_args[1]

        assert call_kwargs["exchange"] == "test-exchange"
        assert call_kwargs["routing_key"] == "test.key"

        message = call_kwargs["message"]
        assert message["event_id"] == "trace-123"
        assert message["event_type"] == "trace-create"
        assert message["trace_id"] == "trace-123"
        assert message["project_id"] == "test-project"


class TestExportObservationPublishesMessage:
    """Test export_observation publishes correct message."""

    def test_export_generation_publishes_message(self, mock_pika_client):
        """Test generation observation publishes with correct event type."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        obs = Observation(
            id="obs-123",
            trace_id="trace-456",
            project_id="test-project",
            type=ObservationType.GENERATION,
            name="llm-call",
        )

        exporter.export_observation(obs)

        message = mock_pika_client.publish_message.call_args[1]["message"]
        assert message["event_type"] == "generation-create"
        assert message["event_id"] == "obs-123"
        assert message["trace_id"] == "trace-456"

    def test_export_span_publishes_message(self, mock_pika_client):
        """Test span observation publishes with correct event type."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        obs = Observation(
            id="obs-123",
            trace_id="trace-456",
            type=ObservationType.SPAN,
        )

        exporter.export_observation(obs)

        message = mock_pika_client.publish_message.call_args[1]["message"]
        assert message["event_type"] == "span-create"

    def test_export_event_publishes_message(self, mock_pika_client):
        """Test event observation publishes with correct event type."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        obs = Observation(
            id="obs-123",
            trace_id="trace-456",
            type=ObservationType.EVENT,
        )

        exporter.export_observation(obs)

        message = mock_pika_client.publish_message.call_args[1]["message"]
        assert message["event_type"] == "event-create"


class TestQueueMessageFormat:
    """Test published message has correct format."""

    def test_queue_message_format(self, mock_pika_client):
        """Test queue message format matches QueueMessage spec."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        trace = Trace(
            id="trace-123",
            name="test",
            project_id="my-project",
        )

        exporter.export_trace(trace)

        message = mock_pika_client.publish_message.call_args[1]["message"]

        # Verify required fields
        assert "event_id" in message
        assert "event_type" in message
        assert "trace_id" in message
        assert "project_id" in message
        assert "timestamp" in message

        # Verify types
        assert isinstance(message["event_id"], str)
        assert isinstance(message["event_type"], str)
        assert isinstance(message["trace_id"], str)
        assert isinstance(message["project_id"], str)
        assert isinstance(message["timestamp"], str)


class TestExporterStartStop:
    """Test exporter lifecycle."""

    def test_exporter_start_creates_exchange(self, mock_pika_client):
        """Test start() declares exchange."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        mock_pika_client.exchange_declare.assert_called_once()
        call_kwargs = mock_pika_client.exchange_declare.call_args[1]
        assert call_kwargs["exchange"] == "test-exchange"
        assert call_kwargs["durable"] is True

    def test_exporter_stop_closes_connection(self, mock_pika_client):
        """Test stop() closes connection."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        exporter.stop()

        mock_pika_client.close.assert_called_once()

    def test_exporter_stop_handles_close_error(self, mock_pika_client):
        """Test stop() handles close error gracefully."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        mock_pika_client.close.side_effect = Exception("Close failed")

        # Should not raise
        exporter.stop()


class TestUrlParsing:
    """Test RabbitMQ URL parsing."""

    def test_parse_url_with_vhost(self):
        """Test URL parsing extracts vhost correctly."""
        exporter = RabbitMQExporter(
            rabbitmq_url="amqp://user:pass@localhost:5672/myvhost"
        )
        config = exporter._parse_url()

        assert config["hostname"] == "localhost"
        assert config["port"] == 5672
        assert config["username"] == "user"
        assert config["password"] == "pass"
        assert config["vhost"] == "myvhost"

    def test_parse_url_default_vhost(self):
        """Test URL parsing with default vhost."""
        exporter = RabbitMQExporter(
            rabbitmq_url="amqp://user:pass@localhost:5672/"
        )
        config = exporter._parse_url()

        assert config["vhost"] == "/"

    def test_parse_url_with_tls(self):
        """Test URL parsing with amqps scheme."""
        exporter = RabbitMQExporter(
            rabbitmq_url="amqps://user:pass@localhost:5671/vhost"
        )
        config = exporter._parse_url()

        assert config["tls"] is True

    def test_parse_url_encoded_vhost(self):
        """Test URL parsing with URL-encoded vhost."""
        exporter = RabbitMQExporter(
            rabbitmq_url="amqp://user:pass@localhost:5672/my%2Fvhost"
        )
        config = exporter._parse_url()

        # Should URL-decode the vhost
        assert config["vhost"] == "my/vhost"


class TestExportMultiple:
    """Test batch export methods."""

    def test_export_traces_batch(self, mock_pika_client):
        """Test export_traces exports multiple traces."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        traces = [
            Trace(id=f"trace-{i}", name=f"test-{i}")
            for i in range(3)
        ]

        exporter.export_traces(traces)

        assert mock_pika_client.publish_message.call_count == 3

    def test_export_observations_batch(self, mock_pika_client):
        """Test export_observations exports multiple observations."""
        exporter = RabbitMQExporter()

        with patch(
            "pika.exchange_type.ExchangeType"
        ):
            exporter.start()

        observations = [
            Observation(
                id=f"obs-{i}",
                trace_id="trace-123",
                type=ObservationType.GENERATION,
            )
            for i in range(3)
        ]

        exporter.export_observations(observations)

        assert mock_pika_client.publish_message.call_count == 3
