"""
LLM Tracker - Main entry point for LLM observability.

The LLMTracker is a singleton class that manages traces and observations
for LLM calls. It integrates with the exporter and storage systems.

Usage:
    from marie.llm_tracking import get_tracker

    tracker = get_tracker()

    # Create a trace for a request
    with tracker.trace("my-request") as trace:
        # Start a generation observation
        gen_id = tracker.generation(
            trace_id=trace.id,
            name="openai_completion",
            model="gpt-4",
            input=messages,
        )

        # Make the API call...
        response = openai.chat.completions.create(...)

        # End the observation with output
        tracker.end(gen_id, output=response.content, usage=response.usage)
"""

import atexit
import logging
import random
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from marie.llm_tracking.config import ExporterType, get_settings
from marie.llm_tracking.exporters.base import BaseExporter
from marie.llm_tracking.exporters.console import ConsoleExporter
from marie.llm_tracking.token_counter import (
    count_tokens_with_fallback,
    extract_usage_from_response,
)
from marie.llm_tracking.types import (
    Cost,
    EventType,
    Observation,
    ObservationLevel,
    ObservationType,
    RawEvent,
    Score,
    ScoreDataType,
    Trace,
    Usage,
)

logger = logging.getLogger(__name__)


class TraceContext:
    """Context manager for traces."""

    def __init__(self, tracker: "LLMTracker", trace: Trace):
        self._tracker = tracker
        self._trace = trace

    @property
    def id(self) -> str:
        return self._trace.id

    @property
    def trace(self) -> Trace:
        return self._trace

    def __enter__(self) -> "TraceContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # Error occurred
            self._trace.output = {"error": str(exc_val)}
        self._tracker._finalize_trace(self._trace)
        return None


class LLMTracker:
    """
    Main LLM tracking class - singleton.

    Manages traces, observations, and scores for LLM calls.
    Integrates with exporters (console, RabbitMQ) and storage (Postgres, S3).
    """

    _instance: Optional["LLMTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "LLMTracker":
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize the tracker."""
        if getattr(self, "_initialized", False):
            return

        self._settings = get_settings()
        self._exporter: Optional[BaseExporter] = None
        self._postgres: Optional[Any] = None
        self._s3: Optional[Any] = None
        self._started = False

        # In-memory observation tracking
        self._pending_observations: Dict[str, Observation] = {}
        self._observations_lock = threading.Lock()

        self._initialized = True

    @property
    def enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self._settings.ENABLED

    def start(self) -> None:
        """
        Initialize the tracker and its components.

        Called automatically on first use, but can be called explicitly.
        """
        if self._started:
            return

        if not self._settings.ENABLED:
            logger.info("LLM tracking is disabled")
            return

        # Guard: If using RabbitMQ exporter, storage MUST be configured
        # RabbitMQ messages only contain event_id - worker needs Postgres/S3
        if self._settings.EXPORTER == ExporterType.RABBITMQ:
            if not self._settings.POSTGRES_URL:
                raise ValueError(
                    "RabbitMQ exporter requires Postgres storage. "
                    "Configure postgres.url in llm_tracking config."
                )
            if not self._settings.S3_BUCKET:
                raise ValueError(
                    "RabbitMQ exporter requires S3 storage. "
                    "Configure s3.bucket (or use shared storage.s3) in llm_tracking config."
                )

        try:
            # Initialize exporter
            self._exporter = self._create_exporter()
            self._exporter.start()

            # Initialize storage if configured
            if self._settings.storage_enabled:
                self._init_storage()

            self._started = True
            logger.info(
                f"LLM tracker started: exporter={self._settings.EXPORTER.value}, "
                f"storage={'enabled' if self._settings.storage_enabled else 'disabled'}"
            )

            # Register shutdown hook
            atexit.register(self.stop)

        except Exception as e:
            logger.error(f"Failed to start LLM tracker: {e}")
            self._started = False

    def _create_exporter(self) -> BaseExporter:
        """Create the configured exporter."""
        if self._settings.EXPORTER == ExporterType.CONSOLE:
            return ConsoleExporter(
                verbose=self._settings.DEBUG,
            )
        elif self._settings.EXPORTER == ExporterType.RABBITMQ:
            from marie.llm_tracking.exporters.rabbitmq import RabbitMQExporter

            return RabbitMQExporter()
        else:
            raise ValueError(f"Unknown exporter type: {self._settings.EXPORTER}")

    def _init_storage(self) -> None:
        """Initialize S3 and Postgres storage.

        S3 is required - all payloads are stored there.
        PostgreSQL stores metadata only.
        """
        # S3 is required - all payloads go to S3
        if not self._settings.S3_BUCKET:
            raise ValueError(
                "S3 bucket not configured. All payloads are stored in S3. "
                "Set MARIE_LLM_TRACKING_S3_BUCKET environment variable."
            )

        try:
            from marie.llm_tracking.storage.s3 import S3Storage

            self._s3 = S3Storage()
            self._s3.start()
        except Exception as e:
            logger.error(f"Failed to initialize S3 storage: {e}")
            raise

        # PostgreSQL stores metadata only
        try:
            from marie.llm_tracking.storage.postgres import PostgresStorage

            self._postgres = PostgresStorage()
            self._postgres.start()
        except Exception as e:
            logger.warning(f"Postgres storage not available: {e}")

    def stop(self) -> None:
        """Shutdown the tracker and flush pending data."""
        if not self._started:
            return

        logger.info("Stopping LLM tracker...")

        if self._exporter:
            try:
                self._exporter.flush()
                self._exporter.stop()
            except Exception as e:
                logger.error(f"Error stopping exporter: {e}")

        if self._postgres:
            try:
                self._postgres.stop()
            except Exception as e:
                logger.error(f"Error stopping Postgres storage: {e}")

        if self._s3:
            try:
                self._s3.stop()
            except Exception as e:
                logger.error(f"Error stopping S3 storage: {e}")

        self._started = False
        logger.info("LLM tracker stopped")

    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        if self._settings.SAMPLING_RATE >= 1.0:
            return True
        return random.random() < self._settings.SAMPLING_RATE

    def _ensure_started(self) -> bool:
        """Ensure tracker is started, return False if tracking is disabled."""
        if not self._settings.ENABLED:
            return False
        if not self._started:
            self.start()
        return self._started

    # ========== Trace API ==========

    @contextmanager
    def trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        input: Optional[Any] = None,
        trace_id: Optional[str] = None,
    ):
        """
        Create a trace context manager.

        Args:
            name: Trace name
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            tags: Tags for filtering
            input: Input data
            trace_id: Optional custom trace ID

        Yields:
            TraceContext with trace ID and trace object
        """
        if not self._ensure_started() or not self._should_sample():
            # Return a dummy context
            dummy = Trace(id=trace_id or str(uuid4()), name=name)
            yield TraceContext(self, dummy)
            return

        trace = Trace(
            id=trace_id or str(uuid4()),
            name=name,
            project_id=self._settings.PROJECT_ID,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            tags=tags or [],
            input=input,
        )

        ctx = TraceContext(self, trace)
        exc_info = (None, None, None)
        try:
            yield ctx
        except Exception as e:
            exc_info = (type(e), e, e.__traceback__)
            raise
        finally:
            # Finalize the trace
            ctx.__exit__(*exc_info)

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        input: Optional[Any] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """
        Create a trace without context manager.

        Returns trace_id for use with observations.
        Call update_trace() to finalize.
        """
        if not self._ensure_started() or not self._should_sample():
            return trace_id or str(uuid4())

        trace = Trace(
            id=trace_id or str(uuid4()),
            name=name,
            project_id=self._settings.PROJECT_ID,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            tags=tags or [],
            input=input,
        )

        self._finalize_trace(trace)
        return trace.id

    def _store_failed_event(
        self,
        event_id: Optional[str],
        trace_id: Optional[str],
        event_type: str,
        error: Exception,
        payload: Dict[str, Any],
    ) -> None:
        """Store a failed event to the DLQ for later retry/investigation."""
        import traceback

        if not self._postgres:
            # If postgres isn't available, we can't save to DLQ - log and return
            logger.error(
                f"Cannot save to DLQ (postgres unavailable): "
                f"event_id={event_id}, type={event_type}, error={error}"
            )
            return

        try:
            self._postgres.save_failed_event(
                event_id=event_id,
                trace_id=trace_id,
                event_type=event_type,
                error_message=str(error),
                payload=payload,
                error_type=type(error).__name__,
                stack_trace=traceback.format_exc(),
            )
        except Exception as dlq_error:
            # Last resort - log everything we can
            logger.critical(
                f"CRITICAL: Failed to save to DLQ, event data may be lost! "
                f"event_id={event_id}, trace_id={trace_id}, type={event_type}, "
                f"original_error={error}, dlq_error={dlq_error}"
            )

    def _finalize_trace(self, trace: Trace) -> None:
        """Export and store a trace."""
        if not self._started:
            return

        trace.updated_at = datetime.utcnow()

        try:
            # Store raw event if storage is enabled
            if self._postgres:
                self._store_trace_event(trace)

            # Export to exporter
            if self._exporter:
                self._exporter.export_trace(trace)

        except Exception as e:
            logger.exception(f"Failed to finalize trace {trace.id}")
            # Store to DLQ instead of silently losing the event
            self._store_failed_event(
                event_id=trace.id,
                trace_id=trace.id,
                event_type="trace-create",
                error=e,
                payload=trace.to_dict(),
            )

    def _store_trace_event(self, trace: Trace) -> None:
        """Store trace: payload to S3, metadata to PostgreSQL."""
        if not self._s3:
            logger.error("S3 not initialized - cannot store trace payload")
            return

        payload = trace.to_dict()

        # 1. Always save payload to S3
        s3_key = self._s3.save_payload(
            payload=payload,
            trace_id=trace.id,
            event_id=trace.id,
            event_type="trace",
        )

        # 2. Save metadata to PostgreSQL
        if self._postgres:
            event = RawEvent(
                id=trace.id,
                trace_id=trace.id,
                event_type=EventType.TRACE_CREATE,
                s3_key=s3_key,
                # Metadata from trace
                user_id=trace.user_id,
                session_id=trace.session_id,
                tags=trace.tags,
            )
            self._postgres.save_event(event)

    # ========== Observation API ==========

    def generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        parent_observation_id: Optional[str] = None,
        observation_id: Optional[str] = None,
    ) -> str:
        """
        Start a generation observation (LLM call).

        Args:
            trace_id: Parent trace ID
            name: Observation name
            model: Model name (e.g., "gpt-4")
            input: Input messages/prompt
            metadata: Additional metadata
            model_parameters: Model parameters (temperature, etc.)
            parent_observation_id: Parent observation ID for nesting
            observation_id: Optional custom observation ID

        Returns:
            Observation ID
        """
        return self._create_observation(
            trace_id=trace_id,
            name=name,
            obs_type=ObservationType.GENERATION,
            model=model,
            input=input,
            metadata=metadata,
            model_parameters=model_parameters,
            parent_observation_id=parent_observation_id,
            observation_id=observation_id,
        )

    def span(
        self,
        trace_id: str,
        name: str,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_observation_id: Optional[str] = None,
        observation_id: Optional[str] = None,
    ) -> str:
        """
        Start a span observation (logical unit of work).

        Args:
            trace_id: Parent trace ID
            name: Observation name
            input: Input data
            metadata: Additional metadata
            parent_observation_id: Parent observation ID for nesting
            observation_id: Optional custom observation ID

        Returns:
            Observation ID
        """
        return self._create_observation(
            trace_id=trace_id,
            name=name,
            obs_type=ObservationType.SPAN,
            input=input,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            observation_id=observation_id,
        )

    def event(
        self,
        trace_id: str,
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: ObservationLevel = ObservationLevel.DEFAULT,
        parent_observation_id: Optional[str] = None,
        observation_id: Optional[str] = None,
    ) -> str:
        """
        Create a point-in-time event observation.

        Events don't have duration - they're immediately finalized.

        Args:
            trace_id: Parent trace ID
            name: Event name
            input: Input data
            output: Output data
            metadata: Additional metadata
            level: Severity level
            parent_observation_id: Parent observation ID
            observation_id: Optional custom observation ID

        Returns:
            Observation ID
        """
        obs_id = self._create_observation(
            trace_id=trace_id,
            name=name,
            obs_type=ObservationType.EVENT,
            input=input,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            observation_id=observation_id,
            level=level,
        )

        # Events are immediately finalized
        self.end(obs_id, output=output)
        return obs_id

    def _create_observation(
        self,
        trace_id: str,
        name: str,
        obs_type: ObservationType,
        model: Optional[str] = None,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        parent_observation_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        level: ObservationLevel = ObservationLevel.DEFAULT,
    ) -> str:
        """Internal method to create an observation."""
        if not self._ensure_started():
            return observation_id or str(uuid4())

        obs = Observation(
            id=observation_id or str(uuid4()),
            trace_id=trace_id,
            project_id=self._settings.PROJECT_ID,
            parent_observation_id=parent_observation_id,
            type=obs_type,
            name=name,
            model=model,
            model_parameters=model_parameters,
            input=input,
            metadata=metadata or {},
            level=level,
        )

        # Store in pending observations
        with self._observations_lock:
            self._pending_observations[obs.id] = obs

        return obs.id

    def end(
        self,
        observation_id: str,
        output: Optional[Any] = None,
        usage: Optional[Union[Dict, Usage, Any]] = None,
        cost: Optional[Union[Dict, Cost]] = None,
        status_message: Optional[str] = None,
        level: Optional[ObservationLevel] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        End an observation with output and usage data.

        Args:
            observation_id: Observation ID from generation/span
            output: Output data
            usage: Token usage (from API or dict)
            cost: Cost information
            status_message: Optional status message
            level: Override observation level
            metadata: Additional metadata to merge
        """
        with self._observations_lock:
            obs = self._pending_observations.pop(observation_id, None)

        if obs is None:
            logger.warning(f"Observation not found: {observation_id}")
            return

        obs.end_time = datetime.utcnow()
        obs.output = output
        obs.updated_at = datetime.utcnow()

        if status_message:
            obs.status_message = status_message
        if level:
            obs.level = level
        if metadata:
            obs.metadata.update(metadata)

        # Process usage
        if usage is not None:
            if isinstance(usage, Usage):
                obs.usage = usage
            elif isinstance(usage, dict):
                obs.usage = Usage(
                    input_tokens=usage.get("input_tokens")
                    or usage.get("prompt_tokens"),
                    output_tokens=usage.get("output_tokens")
                    or usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                )
            else:
                # Try to extract from API response object
                obs.usage = extract_usage_from_response(usage, obs.model or "")

        # Fall back to token counting if usage not complete
        if obs.usage is None or (
            obs.usage.input_tokens is None and obs.usage.output_tokens is None
        ):
            obs.usage = count_tokens_with_fallback(
                input_data=obs.input,
                output_data=obs.output,
                api_usage=obs.usage,
                model=obs.model or "",
            )

        # Process cost
        if cost is not None:
            if isinstance(cost, Cost):
                obs.cost = cost
            elif isinstance(cost, dict):
                obs.cost = Cost(
                    input_cost=cost.get("input_cost"),
                    output_cost=cost.get("output_cost"),
                    total_cost=cost.get("total_cost"),
                )

        self._finalize_observation(obs)

    def error(
        self,
        observation_id: str,
        error: Union[Exception, str],
        level: ObservationLevel = ObservationLevel.ERROR,
    ) -> None:
        """
        End an observation with an error.

        Args:
            observation_id: Observation ID
            error: Exception or error message
            level: Observation level (default: ERROR)
        """
        error_msg = str(error) if isinstance(error, Exception) else error
        self.end(
            observation_id=observation_id,
            output={"error": error_msg},
            status_message=error_msg,
            level=level,
        )

    def _finalize_observation(self, obs: Observation) -> None:
        """Export and store an observation."""
        if not self._started:
            return

        try:
            # Store raw event if storage is enabled
            if self._postgres:
                self._store_observation_event(obs)

            # Export to exporter
            if self._exporter:
                self._exporter.export_observation(obs)

        except Exception as e:
            logger.exception(f"Failed to finalize observation {obs.id}")
            # Store to DLQ instead of silently losing the event
            event_type = f"{obs.type.value.lower()}-create"
            self._store_failed_event(
                event_id=obs.id,
                trace_id=obs.trace_id,
                event_type=event_type,
                error=e,
                payload=obs.to_dict(),
            )

    def _store_observation_event(self, obs: Observation) -> None:
        """Store observation: payload to S3, metadata to PostgreSQL."""
        if not self._s3:
            logger.error("S3 not initialized - cannot store observation payload")
            return

        payload = obs.to_dict()

        # 1. Always save payload to S3
        s3_key = self._s3.save_payload(
            payload=payload,
            trace_id=obs.trace_id,
            event_id=obs.id,
            event_type=obs.type.value.lower(),
        )

        # 2. Determine event type
        event_type = EventType.GENERATION_CREATE
        if obs.type == ObservationType.SPAN:
            event_type = EventType.SPAN_CREATE
        elif obs.type == ObservationType.EVENT:
            event_type = EventType.EVENT_CREATE

        # 3. Extract metadata for PostgreSQL
        # Calculate duration in ms
        duration_ms = None
        if obs.end_time and obs.start_time:
            delta = obs.end_time - obs.start_time
            duration_ms = int(delta.total_seconds() * 1000)

        # Calculate time to first token
        time_to_first_token_ms = None
        if obs.completion_start_time and obs.start_time:
            delta = obs.completion_start_time - obs.start_time
            time_to_first_token_ms = int(delta.total_seconds() * 1000)

        # Extract cost
        cost_usd = None
        if obs.cost and obs.cost.total_cost:
            cost_usd = obs.cost.total_cost

        # 4. Save metadata to PostgreSQL
        if self._postgres:
            event = RawEvent(
                id=obs.id,
                trace_id=obs.trace_id,
                event_type=event_type,
                s3_key=s3_key,
                # Model info
                model_name=obs.model,
                model_provider=obs.metadata.get("provider") if obs.metadata else None,
                # Token metrics
                prompt_tokens=obs.usage.input_tokens if obs.usage else None,
                completion_tokens=obs.usage.output_tokens if obs.usage else None,
                total_tokens=obs.usage.total_tokens if obs.usage else None,
                # Performance metrics
                duration_ms=duration_ms,
                time_to_first_token_ms=time_to_first_token_ms,
                # Cost
                cost_usd=cost_usd,
            )
            self._postgres.save_event(event)

    # ========== Score API ==========

    def score(
        self,
        trace_id: str,
        name: str,
        value: Union[float, str, bool],
        observation_id: Optional[str] = None,
        data_type: Optional[ScoreDataType] = None,
        source: str = "API",
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        score_id: Optional[str] = None,
    ) -> str:
        """
        Create a score for a trace or observation.

        Args:
            trace_id: Trace ID
            name: Score name
            value: Score value
            observation_id: Optional observation ID
            data_type: Score data type (auto-detected if not provided)
            source: Score source (API, EVAL, ANNOTATION)
            comment: Optional comment
            metadata: Additional metadata
            score_id: Optional custom score ID

        Returns:
            Score ID
        """
        if not self._ensure_started():
            return score_id or str(uuid4())

        # Auto-detect data type
        if data_type is None:
            if isinstance(value, bool):
                data_type = ScoreDataType.BOOLEAN
            elif isinstance(value, (int, float)):
                data_type = ScoreDataType.NUMERIC
            else:
                data_type = ScoreDataType.CATEGORICAL

        score = Score(
            id=score_id or str(uuid4()),
            trace_id=trace_id,
            observation_id=observation_id,
            project_id=self._settings.PROJECT_ID,
            name=name,
            value=value,
            data_type=data_type,
            source=source,
            comment=comment,
            metadata=metadata or {},
        )

        self._finalize_score(score)
        return score.id

    def _finalize_score(self, score: Score) -> None:
        """Export and store a score."""
        if not self._started:
            return

        try:
            if self._exporter:
                self._exporter.export_score(score)
        except Exception as e:
            logger.exception(f"Failed to finalize score {score.id}")
            # Store to DLQ instead of silently losing the event
            self._store_failed_event(
                event_id=score.id,
                trace_id=score.trace_id,
                event_type="score-create",
                error=e,
                payload=score.to_dict(),
            )

    # ========== Utility methods ==========

    def flush(self) -> None:
        """Flush any pending events to the exporter."""
        if self._exporter:
            self._exporter.flush()

    def update_trace(
        self,
        trace_id: str,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Update a trace with additional data.

        This method persists to S3/Postgres before publishing to RabbitMQ,
        ensuring updates are durable and can be processed by the worker.

        Args:
            trace_id: Trace ID
            output: Output data
            metadata: Additional metadata to merge
            tags: Additional tags
        """
        if not self._ensure_started():
            return

        # Create trace object for update (name required for storage)
        trace = Trace(
            id=trace_id,
            name=f"trace_update_{trace_id}",
            project_id=self._settings.PROJECT_ID,
            output=output,
            metadata=metadata or {},
            tags=tags or [],
        )

        # Use _finalize_trace which:
        # 1. Stores to S3 and Postgres via _store_trace_event()
        # 2. Exports to RabbitMQ via export_trace()
        self._finalize_trace(trace)


# Singleton access
_tracker: Optional[LLMTracker] = None


def get_tracker() -> LLMTracker:
    """Get the singleton LLM tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = LLMTracker()
    return _tracker
