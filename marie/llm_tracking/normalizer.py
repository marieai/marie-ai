"""
Event Normalizer - Normalize LLM events for ClickHouse storage.

Handles:
- Token counting (API usage + tiktoken fallback)
- Cost calculation based on model pricing
- Payload normalization and validation
- Merge/overwrite logic for updates
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

from marie.llm_tracking.token_counter import count_tokens_with_fallback
from marie.llm_tracking.types import (
    Cost,
    EventType,
    Observation,
    ObservationType,
    RawEvent,
    Score,
    Trace,
    Usage,
)

logger = logging.getLogger(__name__)


# Default model pricing (per 1K tokens)
# These are approximate prices - should be configured per deployment
DEFAULT_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Anthropic models
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    # Qwen models (approximate)
    "qwen": {"input": 0.001, "output": 0.002},
    "qwen2": {"input": 0.001, "output": 0.002},
    # Default for unknown models
    "default": {"input": 0.001, "output": 0.002},
}


@dataclass
class NormalizedTrace:
    """Normalized trace ready for ClickHouse insert."""

    trace: Trace
    is_update: bool = False


@dataclass
class NormalizedObservation:
    """Normalized observation ready for ClickHouse insert."""

    observation: Observation
    is_update: bool = False


@dataclass
class NormalizedScore:
    """Normalized score ready for ClickHouse insert."""

    score: Score


class EventNormalizer:
    """
    Normalizes LLM tracking events for ClickHouse storage.

    Handles:
    - Token counting when not provided
    - Cost calculation based on model pricing
    - Payload validation and sanitization
    - Merge logic for update events
    """

    def __init__(
        self,
        model_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        enable_token_counting: bool = True,
    ):
        """
        Initialize the normalizer.

        Args:
            model_pricing: Custom model pricing dict (model_name -> {input: price, output: price})
            enable_token_counting: Whether to count tokens when not provided
        """
        self._pricing = model_pricing or DEFAULT_MODEL_PRICING
        self._enable_token_counting = enable_token_counting

    def normalize_trace(
        self,
        raw_event: RawEvent,
        payload: Optional[Dict[str, Any]] = None,
    ) -> NormalizedTrace:
        """
        Normalize a trace event.

        Args:
            raw_event: The raw event from Postgres
            payload: The full payload (from inline or S3)

        Returns:
            NormalizedTrace ready for ClickHouse
        """
        data = payload or raw_event.payload or {}

        trace = Trace(
            id=data.get("id", raw_event.id),
            name=data.get("name", ""),
            project_id=data.get("project_id", "default"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            input=self._parse_json_field(data.get("input")),
            output=self._parse_json_field(data.get("output")),
            metadata=self._parse_json_field(data.get("metadata")) or {},
            tags=data.get("tags", []),
            timestamp=self._parse_datetime(data.get("timestamp")),
            created_at=self._parse_datetime(data.get("created_at")),
            updated_at=datetime.utcnow(),
            release=data.get("release"),
            version=data.get("version"),
            is_deleted=bool(data.get("is_deleted", False)),
        )

        is_update = raw_event.event_type == EventType.TRACE_UPDATE

        return NormalizedTrace(trace=trace, is_update=is_update)

    def normalize_observation(
        self,
        raw_event: RawEvent,
        payload: Optional[Dict[str, Any]] = None,
    ) -> NormalizedObservation:
        """
        Normalize an observation event with token counting and cost calculation.

        Args:
            raw_event: The raw event from Postgres
            payload: The full payload (from inline or S3)

        Returns:
            NormalizedObservation ready for ClickHouse
        """
        data = payload or raw_event.payload or {}

        # Parse observation type
        obs_type_str = data.get("type", "SPAN")
        try:
            obs_type = ObservationType(obs_type_str)
        except ValueError:
            obs_type = ObservationType.SPAN

        # Parse usage if provided
        usage = None
        if "usage" in data and data["usage"]:
            usage_data = data["usage"]
            if isinstance(usage_data, dict):
                usage = Usage(
                    input_tokens=usage_data.get("input_tokens"),
                    output_tokens=usage_data.get("output_tokens"),
                    total_tokens=usage_data.get("total_tokens"),
                    usage_details=usage_data.get("usage_details", {}),
                )
        elif any(k in data for k in ["input_tokens", "output_tokens", "total_tokens"]):
            usage = Usage(
                input_tokens=data.get("input_tokens"),
                output_tokens=data.get("output_tokens"),
                total_tokens=data.get("total_tokens"),
                usage_details=self._parse_json_field(data.get("usage_details")) or {},
            )

        # Parse cost if provided
        cost = None
        if "cost" in data and data["cost"]:
            cost_data = data["cost"]
            if isinstance(cost_data, dict):
                cost = Cost(
                    input_cost=(
                        Decimal(str(cost_data.get("input_cost", 0)))
                        if cost_data.get("input_cost")
                        else None
                    ),
                    output_cost=(
                        Decimal(str(cost_data.get("output_cost", 0)))
                        if cost_data.get("output_cost")
                        else None
                    ),
                    total_cost=(
                        Decimal(str(cost_data.get("total_cost", 0)))
                        if cost_data.get("total_cost")
                        else None
                    ),
                    cost_details={
                        k: Decimal(str(v))
                        for k, v in cost_data.get("cost_details", {}).items()
                    },
                )
        elif any(k in data for k in ["input_cost", "output_cost", "total_cost"]):
            cost = Cost(
                input_cost=(
                    Decimal(str(data.get("input_cost", 0)))
                    if data.get("input_cost")
                    else None
                ),
                output_cost=(
                    Decimal(str(data.get("output_cost", 0)))
                    if data.get("output_cost")
                    else None
                ),
                total_cost=(
                    Decimal(str(data.get("total_cost", 0)))
                    if data.get("total_cost")
                    else None
                ),
            )

        # Get model name for token counting and cost calculation
        model = data.get("model")

        # Create observation
        observation = Observation(
            id=data.get("id", raw_event.id),
            trace_id=data.get("trace_id", raw_event.trace_id),
            project_id=data.get("project_id", "default"),
            parent_observation_id=data.get("parent_observation_id"),
            type=obs_type,
            name=data.get("name", ""),
            start_time=self._parse_datetime(data.get("start_time")),
            end_time=self._parse_datetime(data.get("end_time")),
            completion_start_time=self._parse_datetime(
                data.get("completion_start_time")
            ),
            model=model,
            model_parameters=self._parse_json_field(data.get("model_parameters")),
            usage=usage,
            cost=cost,
            input=self._parse_json_field(data.get("input")),
            output=self._parse_json_field(data.get("output")),
            metadata=self._parse_json_field(data.get("metadata")) or {},
            level=data.get("level", "DEFAULT"),
            status_message=data.get("status_message"),
            version=data.get("version"),
            created_at=self._parse_datetime(data.get("created_at")),
            updated_at=datetime.utcnow(),
            is_deleted=bool(data.get("is_deleted", False)),
        )

        # Token counting if not provided and enabled
        if (
            self._enable_token_counting
            and obs_type == ObservationType.GENERATION
            and (observation.usage is None or observation.usage.input_tokens is None)
        ):
            calculated_usage = count_tokens_with_fallback(
                input_data=observation.input,
                output_data=observation.output,
                api_usage=observation.usage,
                model=model or "",
            )
            if calculated_usage:
                observation.usage = calculated_usage

        # Cost calculation if not provided
        if observation.usage and (
            observation.cost is None or observation.cost.total_cost is None
        ):
            observation.cost = self._calculate_cost(
                model=model,
                usage=observation.usage,
            )

        is_update = raw_event.event_type in [
            EventType.SPAN_UPDATE,
            EventType.GENERATION_UPDATE,
        ]

        return NormalizedObservation(observation=observation, is_update=is_update)

    def normalize_score(
        self,
        raw_event: RawEvent,
        payload: Optional[Dict[str, Any]] = None,
    ) -> NormalizedScore:
        """
        Normalize a score event.

        Args:
            raw_event: The raw event from Postgres
            payload: The full payload (from inline or S3)

        Returns:
            NormalizedScore ready for ClickHouse
        """
        data = payload or raw_event.payload or {}

        score = Score(
            id=data.get("id", raw_event.id),
            trace_id=data.get("trace_id", raw_event.trace_id),
            observation_id=data.get("observation_id"),
            project_id=data.get("project_id", "default"),
            name=data.get("name", ""),
            value=data.get("value", 0.0),
            data_type=data.get("data_type", "NUMERIC"),
            source=data.get("source", "API"),
            comment=data.get("comment"),
            metadata=self._parse_json_field(data.get("metadata")) or {},
            timestamp=self._parse_datetime(data.get("timestamp")),
            created_at=self._parse_datetime(data.get("created_at")),
            updated_at=datetime.utcnow(),
            is_deleted=bool(data.get("is_deleted", False)),
        )

        return NormalizedScore(score=score)

    def _calculate_cost(
        self,
        model: Optional[str],
        usage: Usage,
    ) -> Cost:
        """
        Calculate cost based on model pricing and token usage.

        Args:
            model: Model name
            usage: Token usage

        Returns:
            Calculated Cost object
        """
        # Find pricing for model (try exact match, then prefix match, then default)
        pricing = self._pricing.get("default", {"input": 0.001, "output": 0.002})

        if model:
            model_lower = model.lower()
            if model_lower in self._pricing:
                pricing = self._pricing[model_lower]
            else:
                # Try prefix matching
                for model_prefix, model_pricing in self._pricing.items():
                    if model_lower.startswith(model_prefix):
                        pricing = model_pricing
                        break

        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0

        # Calculate costs (pricing is per 1K tokens)
        input_cost = (
            Decimal(str(input_tokens))
            * Decimal(str(pricing["input"]))
            / Decimal("1000")
        )
        output_cost = (
            Decimal(str(output_tokens))
            * Decimal(str(pricing["output"]))
            / Decimal("1000")
        )
        total_cost = input_cost + output_cost

        return Cost(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
        )

    def _parse_json_field(self, value: Any) -> Any:
        """Parse a JSON field that might be a string or already parsed."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def _parse_datetime(self, value: Any) -> datetime:
        """Parse a datetime from various formats."""
        if value is None:
            return datetime.utcnow()
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return datetime.utcnow()
        return datetime.utcnow()

    def merge_observations(
        self,
        existing: Observation,
        update: Observation,
    ) -> Observation:
        """
        Merge an update into an existing observation.

        Langfuse-style merge: non-null update values overwrite existing,
        except for immutable fields (id, trace_id, project_id, start_time).
        Metadata is merged (update keys overwrite existing keys).

        Args:
            existing: The existing observation
            update: The update to apply

        Returns:
            Merged observation
        """
        # Immutable fields - always keep from existing
        merged = Observation(
            id=existing.id,
            trace_id=existing.trace_id,
            project_id=existing.project_id,
            start_time=existing.start_time,
            created_at=existing.created_at,
        )

        # Overwritable fields - use update if not None
        merged.parent_observation_id = (
            update.parent_observation_id or existing.parent_observation_id
        )
        merged.type = (
            update.type if update.type != ObservationType.SPAN else existing.type
        )
        merged.name = update.name or existing.name
        merged.end_time = update.end_time or existing.end_time
        merged.completion_start_time = (
            update.completion_start_time or existing.completion_start_time
        )
        merged.model = update.model or existing.model
        merged.model_parameters = update.model_parameters or existing.model_parameters
        merged.input = update.input if update.input is not None else existing.input
        merged.output = update.output if update.output is not None else existing.output
        merged.level = update.level if update.level else existing.level
        merged.status_message = update.status_message or existing.status_message
        merged.version = update.version or existing.version
        merged.is_deleted = update.is_deleted or existing.is_deleted

        # Merge usage
        if update.usage:
            if existing.usage:
                merged.usage = Usage(
                    input_tokens=update.usage.input_tokens
                    or existing.usage.input_tokens,
                    output_tokens=update.usage.output_tokens
                    or existing.usage.output_tokens,
                    total_tokens=update.usage.total_tokens
                    or existing.usage.total_tokens,
                    usage_details={
                        **existing.usage.usage_details,
                        **update.usage.usage_details,
                    },
                )
            else:
                merged.usage = update.usage
        else:
            merged.usage = existing.usage

        # Merge cost
        if update.cost:
            if existing.cost:
                merged.cost = Cost(
                    input_cost=update.cost.input_cost or existing.cost.input_cost,
                    output_cost=update.cost.output_cost or existing.cost.output_cost,
                    total_cost=update.cost.total_cost or existing.cost.total_cost,
                    cost_details={
                        **existing.cost.cost_details,
                        **update.cost.cost_details,
                    },
                )
            else:
                merged.cost = update.cost
        else:
            merged.cost = existing.cost

        # Merge metadata
        merged.metadata = {**existing.metadata, **update.metadata}

        merged.updated_at = datetime.utcnow()

        return merged

    def merge_traces(
        self,
        existing: Trace,
        update: Trace,
    ) -> Trace:
        """
        Merge an update into an existing trace.

        Args:
            existing: The existing trace
            update: The update to apply

        Returns:
            Merged trace
        """
        # Immutable fields
        merged = Trace(
            id=existing.id,
            project_id=existing.project_id,
            timestamp=existing.timestamp,
            created_at=existing.created_at,
        )

        # Overwritable fields
        merged.name = update.name or existing.name
        merged.user_id = update.user_id or existing.user_id
        merged.session_id = update.session_id or existing.session_id
        merged.input = update.input if update.input is not None else existing.input
        merged.output = update.output if update.output is not None else existing.output
        merged.release = update.release or existing.release
        merged.version = update.version or existing.version
        merged.is_deleted = update.is_deleted or existing.is_deleted

        # Merge metadata
        merged.metadata = {**existing.metadata, **update.metadata}

        # Merge tags (unique, sorted)
        merged.tags = sorted(set(existing.tags + update.tags))

        merged.updated_at = datetime.utcnow()

        return merged


# Singleton instance
_normalizer: Optional[EventNormalizer] = None


def get_normalizer() -> EventNormalizer:
    """Get the singleton normalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = EventNormalizer()
    return _normalizer
