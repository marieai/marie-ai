"""Guardrail registry for discovery and instantiation.

This module provides the guardrail registry and resolution functions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from marie.agent.guardrails.base import Guardrail
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.agent.guardrails.registry")

# Global registry mapping "{phase}:{name}" to guardrail class
GUARDRAIL_REGISTRY: Dict[str, Type[Guardrail]] = {}


def register_guardrail(name: str, phase: str):
    """Decorator to register a guardrail class.

    Sets the class attributes `name` and `phase`, and registers
    the class in GUARDRAIL_REGISTRY under the key "{phase}:{name}".

    Args:
        name: Guardrail name (e.g., "pii", "prompt_injection")
        phase: Execution phase ("before", "after", or "tool_call")

    Example:
        ```python
        @register_guardrail("pii", "before")
        class PIIBeforeGuardrail(Guardrail):
            async def evaluate(self, content: Any, context: dict) -> GuardrailResult: ...
        ```
    """
    valid_phases = ("before", "after", "tool_call")
    if phase not in valid_phases:
        raise ValueError(f"Invalid phase '{phase}'. Must be one of: {valid_phases}")

    def decorator(cls: Type[Guardrail]) -> Type[Guardrail]:
        cls.name = name
        cls.phase = phase
        key = f"{phase}:{name}"
        GUARDRAIL_REGISTRY[key] = cls
        logger.debug(f"Registered guardrail: {key}")
        return cls

    return decorator


def resolve_guardrails_for_phase(
    phase: str,
    entries: List[Dict[str, Any]],
) -> List[Guardrail]:
    """Instantiate guardrails for a specific phase from config entries.

    Args:
        phase: Execution phase ("before", "after", or "tool_call")
        entries: List of config entries, each containing:
            - type: Guardrail type name (e.g., "pii", "prompt_injection")
            - config: Optional guardrail-specific configuration dict

    Returns:
        List of instantiated Guardrail objects

    Raises:
        ValueError: If a guardrail type is not found in the registry

    Example:
        ```python
        entries = [
            {"type": "pii", "config": {"check_email": True}},
            {"type": "prompt_injection"},
        ]
        guardrails = resolve_guardrails_for_phase("before", entries)
        ```
    """
    guardrails: List[Guardrail] = []

    for entry in entries:
        guardrail_type = entry.get("type")
        if not guardrail_type:
            logger.warning(f"Skipping guardrail entry without type: {entry}")
            continue

        key = f"{phase}:{guardrail_type}"
        cls = GUARDRAIL_REGISTRY.get(key)

        if cls is None:
            available = [k for k in GUARDRAIL_REGISTRY if k.startswith(f"{phase}:")]
            raise ValueError(
                f"Unknown guardrail '{guardrail_type}' for phase '{phase}'. "
                f"Available: {available}"
            )

        # Instantiate with config
        config_dict = entry.get("config", {})
        config = cls.config_class(**config_dict)
        guardrails.append(cls(config=config))
        logger.debug(f"Resolved guardrail: {key}")

    return guardrails


def get_available_guardrails() -> Dict[str, List[str]]:
    """Get all available guardrails grouped by phase.

    Returns:
        Dict mapping phase names to lists of guardrail names
    """
    by_phase: Dict[str, List[str]] = {
        "before": [],
        "after": [],
        "tool_call": [],
    }

    for key in GUARDRAIL_REGISTRY:
        phase, name = key.split(":", 1)
        if phase in by_phase:
            by_phase[phase].append(name)

    return by_phase
