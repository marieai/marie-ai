from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marie.instrumentation.types import Observation, Usage

from openinference.instrumentation import (
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

# Marie-specific vendor patterns (extends OI's built-in list)
_MODEL_VENDOR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(gpt-|o1-|o3-|dall-e|text-)"), "openai"),
    (re.compile(r"^claude-"), "anthropic"),
    (re.compile(r"^(qwen|Qwen)"), "qwen"),
    (re.compile(r"^(gemini-|gemma-)"), "google"),
    (re.compile(r"^(mistral|mixtral|codestral)"), "mistralai"),
    (re.compile(r"^command"), "cohere"),
    (re.compile(r"^deepseek"), "deepseek"),
]


def infer_llm_system(model_name: str) -> str:
    """Infer LLM vendor from model name. Extends OI's built-in enum with Marie patterns."""
    for pattern, vendor in _MODEL_VENDOR_PATTERNS:
        if pattern.match(model_name):
            return vendor
    return model_name.split("-")[0].split("/")[-1].lower()


def infer_span_kind(
    obs_type: str, name: str, metadata: dict | None = None
) -> OpenInferenceSpanKindValues:
    """Map Marie observation type + context to OI span kind."""
    if obs_type == "GENERATION":
        return OpenInferenceSpanKindValues.LLM
    if obs_type == "SPAN":
        name_lower = name.lower()
        if any(kw in name_lower for kw in ("agent", "react", "planner")):
            return OpenInferenceSpanKindValues.AGENT
        if any(kw in name_lower for kw in ("tool", "search", "retrieve")):
            return OpenInferenceSpanKindValues.TOOL
        if any(kw in name_lower for kw in ("embed",)):
            return OpenInferenceSpanKindValues.EMBEDDING
        if any(kw in name_lower for kw in ("retriev", "fetch", "lookup")):
            return OpenInferenceSpanKindValues.RETRIEVER
    return OpenInferenceSpanKindValues.CHAIN


def observation_to_span_attributes(observation: "Observation") -> dict[str, Any]:
    """
    Build OI attribute dict from a completed Observation.

    Delegates standard attribute building to openinference.instrumentation helpers.
    Adds Marie-specific attributes (project_id, observation_type, cost).
    """
    attrs: dict[str, Any] = {}

    # Span kind
    kind = infer_span_kind(observation.type, observation.name, observation.metadata)
    attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] = kind.value

    # LLM-specific attributes
    if kind == OpenInferenceSpanKindValues.LLM and observation.model:
        llm_system = infer_llm_system(observation.model)
        attrs.update(
            get_llm_attributes(
                system=llm_system,
                model_name=observation.model,
                input_messages=_extract_messages(observation.input, "input"),
                output_messages=_extract_messages(observation.output, "output"),
                token_count=_to_oi_token_count(observation.usage),
                invocation_parameters=observation.model_parameters,
            )
        )

    # Input/output (for non-LLM spans, or as fallback)
    if observation.input is not None and kind != OpenInferenceSpanKindValues.LLM:
        attrs.update(get_input_attributes(observation.input))
    if observation.output is not None and kind != OpenInferenceSpanKindValues.LLM:
        attrs.update(get_output_attributes(observation.output))

    # Marie-specific attributes
    # project_id is a top-level field on Observation, NOT inside metadata
    if observation.project_id:
        attrs["marie.project_id"] = observation.project_id
    attrs["marie.observation_type"] = observation.type

    # Cost (OI semconv)
    # Marie Cost fields: input_cost, output_cost, total_cost (Decimal — convert to float for OTel)
    if observation.cost:
        if observation.cost.total_cost is not None:
            attrs[SpanAttributes.LLM_COST_TOTAL] = float(observation.cost.total_cost)
        if observation.cost.input_cost is not None:
            attrs[SpanAttributes.LLM_COST_PROMPT] = float(observation.cost.input_cost)
        if observation.cost.output_cost is not None:
            attrs[SpanAttributes.LLM_COST_COMPLETION] = float(
                observation.cost.output_cost
            )

    return attrs


def _extract_messages(data: Any, direction: str) -> list[dict] | None:
    """Extract message list from input/output data for OI message attributes."""
    if isinstance(data, list):
        if all(isinstance(m, dict) and "role" in m for m in data):
            return data
    if isinstance(data, dict) and "messages" in data:
        return data["messages"]
    return None


def _to_oi_token_count(usage: "Usage | None") -> dict | None:
    """
    Convert Marie Usage to OI TokenCount TypedDict.

    Marie Usage fields: input_tokens, output_tokens, total_tokens
    OI TokenCount keys:  prompt, completion, total
    """
    if usage is None:
        return None
    result = {}
    if usage.input_tokens is not None:
        result["prompt"] = usage.input_tokens
    if usage.output_tokens is not None:
        result["completion"] = usage.output_tokens
    if usage.total_tokens is not None:
        result["total"] = usage.total_tokens
    return result or None
