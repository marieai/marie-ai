"""
OTEL Toast Handler - sends events to OpenTelemetry Collector.

Events are sent as OTEL logs which flow through the collector pipeline:
  Application -> OTLP -> OTel Collector -> ClickHouse -> HyperDX UI

This replaces/supplements the PostgreSQL event_tracking approach with
a standard observability pipeline using ClickHouse for time-series storage.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LogRecord
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.trace import INVALID_SPAN_CONTEXT

from marie.messaging.events import EventMessage
from marie.messaging.toast_handler import ToastHandler

logger = logging.getLogger(__name__)


class OTELToastHandler(ToastHandler):
    """
    Toast handler that publishes events to OpenTelemetry Collector as logs.

    Events are structured as OTEL log records with attributes for:
    - event.type: The event type (e.g., ENGINE_EVENT, RUN_SUCCESS)
    - job.id: The job identifier
    - job.tag: The job tag for categorization
    - status: Event status (INFO, WARN, ERROR)
    - event.source: Event source URI (e.g., gateway://scheduler)
    - api_key: API key (first 10 chars for privacy)
    - payload.*: Flattened event payload attributes

    Configuration:
        config:
            endpoint: "http://localhost:4317"  # OTLP gRPC endpoint
            service_name: "marie-ai"
            queue:
                maxsize: 4096
                drop_if_full: false
            batch:
                max_export_batch_size: 512
                schedule_delay_millis: 5000
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.config = config or {}

        # OTLP configuration
        self._endpoint = self.config.get("endpoint", "http://localhost:4317")
        self._service_name = self.config.get("service_name", "marie-ai")
        self._insecure = self.config.get("insecure", True)

        # Queue configuration
        q_cfg = self.config.get("queue", {})
        self._queue_maxsize: int = int(q_cfg.get("maxsize", 4096))
        self._drop_if_full: bool = bool(q_cfg.get("drop_if_full", False))
        self._enqueue_timeout_s: float = float(q_cfg.get("enqueue_timeout_s", 0.0))

        # Batch configuration
        b_cfg = self.config.get("batch", {})
        self._max_export_batch_size = int(b_cfg.get("max_export_batch_size", 512))
        self._schedule_delay_millis = int(b_cfg.get("schedule_delay_millis", 5000))

        # Initialize OTEL logger
        self._logger_provider: Optional[LoggerProvider] = None
        self._otel_logger = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._setup_otel_logger()

        logger.info(
            f"OTELToastHandler initialized: endpoint={self._endpoint}, "
            f"service={self._service_name}"
        )

        # Try to start worker if event loop is already running
        try:
            loop = asyncio.get_running_loop()
            self._worker_task = loop.create_task(self._worker())
            logger.info("OTELToastHandler worker started in __init__")
        except RuntimeError:
            # No running loop yet; worker will start on first notify()
            pass

    def _setup_otel_logger(self) -> None:
        """Initialize OTEL logger provider and exporter."""
        exporter = OTLPLogExporter(
            endpoint=self._endpoint,
            insecure=self._insecure,
        )

        self._logger_provider = LoggerProvider()
        self._logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                exporter,
                max_export_batch_size=self._max_export_batch_size,
                schedule_delay_millis=self._schedule_delay_millis,
            )
        )

        self._otel_logger = self._logger_provider.get_logger(
            self._service_name,
            version="1.0.0",
        )

    def get_supported_events(self) -> List[str]:
        """Returns list of supported event patterns."""
        return ["*"]

    @property
    def priority(self) -> int:
        """Handler priority."""
        return 1

    async def notify(self, notification: EventMessage, **kwargs: Any) -> bool:
        """Enqueue event for async processing."""
        self._check_kwargs(kwargs)

        # Start worker if constructed before loop was running
        if self._worker_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._worker_task = loop.create_task(self._worker())
                logger.info("OTELToastHandler worker started lazily in notify()")
            except RuntimeError:
                logger.warning("notify() called without a running loop")
                raise

        try:
            if self._enqueue_timeout_s > 0:
                await asyncio.wait_for(
                    self._queue.put(notification),
                    timeout=self._enqueue_timeout_s,
                )
            else:
                self._queue.put_nowait(notification)
            return True
        except asyncio.QueueFull:
            if self._drop_if_full:
                logger.warning(f"OTEL queue full, dropping event: {notification.id}")
                return False
            raise
        except asyncio.TimeoutError:
            logger.warning(f"OTEL queue timeout, dropping event: {notification.id}")
            return False

    async def start(self) -> None:
        """Start background worker."""
        self._shutdown_event.clear()
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("OTELToastHandler worker started")

    async def _worker(self) -> None:
        """Process queue and emit OTEL logs."""
        while not self._shutdown_event.is_set():
            try:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                self._emit_log(item)
                self._queue.task_done()

            except Exception as e:
                logger.exception(f"OTELToastHandler worker error: {e}")

    def _emit_log(self, msg: EventMessage) -> None:
        """Emit an OTEL log record for the event."""
        try:
            # Map status to severity
            severity_map = {
                "INFO": 9,
                "WARN": 13,
                "WARNING": 13,
                "ERROR": 17,
                "FATAL": 21,
            }
            severity_number = severity_map.get(msg.status.upper(), 9)
            severity_text = msg.status.upper()

            # Build log attributes
            attributes: Dict[str, Any] = {
                "event.id": msg.id,
                "event.type": msg.event,
                "event.source": msg.source,
                "job.id": msg.jobid,
                "job.tag": msg.jobtag,
                "status": msg.status,
                "api_key": msg.api_key[:10] if msg.api_key else "",
                "service.name": self._service_name,
            }

            # Flatten payload attributes
            if isinstance(msg.payload, dict):
                for key, value in msg.payload.items():
                    if isinstance(value, (str, int, float, bool)):
                        attributes[f"payload.{key}"] = value
                    else:
                        attributes[f"payload.{key}"] = self._safe_json_dumps(value)
            else:
                attributes["payload"] = self._safe_json_dumps(msg.payload)

            # Create and emit log record
            log_record = LogRecord(
                timestamp=msg.timestamp * 1_000_000,  # ms to ns
                observed_timestamp=int(time.time_ns()),
                trace_id=INVALID_SPAN_CONTEXT.trace_id,
                span_id=INVALID_SPAN_CONTEXT.span_id,
                trace_flags=INVALID_SPAN_CONTEXT.trace_flags,
                severity_text=severity_text,
                severity_number=severity_number,
                body=f"[{msg.event}] {msg.jobtag}: {self._get_message_summary(msg)}",
                attributes=attributes,
            )
            self._otel_logger.emit(log_record)

        except Exception as e:
            logger.error(f"Failed to emit OTEL log for event {msg.id}: {e}")

    def _get_message_summary(self, msg: EventMessage) -> str:
        """Extract a summary message from the payload."""
        if isinstance(msg.payload, dict):
            for key in ("message", "msg", "description", "error", "reason"):
                if key in msg.payload:
                    return str(msg.payload[key])[:200]
            for value in msg.payload.values():
                if isinstance(value, str) and len(value) > 5:
                    return value[:200]
        return str(msg.payload)[:200] if msg.payload else ""

    def _safe_json_dumps(self, obj: Any) -> str:
        """Safely serialize object to JSON."""
        try:
            return json.dumps(obj, default=str)
        except (TypeError, ValueError):
            return str(obj)

    async def close(self, drain: bool = True, timeout: float = 5.0) -> None:
        """Graceful shutdown."""
        self._shutdown_event.set()

        if drain and not self._queue.empty():
            logger.info(f"Draining {self._queue.qsize()} events...")
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Drain timeout, {self._queue.qsize()} events dropped")

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._logger_provider:
            self._logger_provider.shutdown()

        logger.info("OTELToastHandler closed")
