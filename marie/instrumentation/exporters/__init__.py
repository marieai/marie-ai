"""
LLM Tracking Exporters - Export events to various destinations.

Available exporters:
- ConsoleExporter: Development/debugging (logs to stdout)
- RabbitMQExporter: Production (publishes to RabbitMQ for async processing)
"""

from marie.instrumentation.exporters.base import BaseExporter
from marie.instrumentation.exporters.console import ConsoleExporter

__all__ = ["BaseExporter", "ConsoleExporter"]


# Lazy import for RabbitMQExporter to avoid dependency issues
def get_rabbitmq_exporter():
    """Get RabbitMQExporter (lazy import)."""
    from marie.instrumentation.exporters.rabbitmq import RabbitMQExporter

    return RabbitMQExporter
