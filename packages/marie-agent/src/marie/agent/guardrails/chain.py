"""Guardrail chain for executing multiple guardrails in sequence.

This module provides the GuardrailChain class that runs guardrails
in priority order and short-circuits on BLOCK or ESCALATE.
"""

from __future__ import annotations

import logging
from typing import Any, List

from pydantic import BaseModel, Field

from marie.agent.guardrails.base import Guardrail
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult

logger = logging.getLogger("marie.agent.guardrails.chain")


class GuardrailChainResult(BaseModel):
    """Result from running a guardrail chain.

    Attributes:
        action: Final action (worst case from all guardrails)
        results: Individual results from each guardrail
        final_content: Content after all modifications applied
    """

    action: GuardrailAction = Field(
        default=GuardrailAction.ALLOW,
        description="Final action to take",
    )
    results: List[GuardrailResult] = Field(
        default_factory=list,
        description="Results from each guardrail in the chain",
    )
    final_content: Any = Field(
        default=None,
        description="Content after modifications (if any)",
    )


class GuardrailChain:
    """Execute multiple guardrails in priority order.

    Guardrails are sorted by priority (highest first) and executed
    sequentially. The chain short-circuits on BLOCK or ESCALATE,
    returning immediately without running remaining guardrails.

    For MODIFY actions, the modified content is passed to subsequent
    guardrails in the chain.

    Example:
        ```python
        chain = GuardrailChain(
            [
                PIIGuardrail(config=PIIConfig(priority=200)),
                ContentFilterGuardrail(config=ContentFilterConfig(priority=100)),
            ]
        )

        result = await chain.run("user input", {"phase": "before"})
        if result.action == GuardrailAction.BLOCK:
            return error_response(result.results[-1].message)
        elif result.action == GuardrailAction.MODIFY:
            processed_input = result.final_content
        ```
    """

    def __init__(self, guardrails: List[Guardrail]):
        """Initialize the guardrail chain.

        Args:
            guardrails: List of guardrails to execute. Will be sorted
                       by priority (highest first) and filtered to
                       only enabled guardrails.
        """
        # Filter disabled guardrails and sort by priority (highest first)
        self._guardrails = sorted(
            [g for g in guardrails if g.config.enabled],
            key=lambda g: g.config.priority,
            reverse=True,
        )

    @property
    def is_empty(self) -> bool:
        """Check if the chain has no guardrails."""
        return len(self._guardrails) == 0

    @property
    def guardrails(self) -> List[Guardrail]:
        """Get the list of guardrails in execution order."""
        return self._guardrails

    async def run(self, content: Any, context: dict) -> GuardrailChainResult:
        """Run all guardrails in the chain.

        Args:
            content: Content to evaluate
            context: Execution context

        Returns:
            GuardrailChainResult with final action and all results
        """
        results: List[GuardrailResult] = []
        current = content

        for guard in self._guardrails:
            try:
                result = await guard.evaluate(current, context)
                result.guardrail_name = guard.name
                results.append(result)

                logger.debug(
                    f"Guardrail {guard.name} returned {result.action.value} "
                    f"(score={result.score:.2f})"
                )

                # Short-circuit on BLOCK
                if result.action == GuardrailAction.BLOCK:
                    logger.warning(
                        f"Guardrail {guard.name} blocked content: {result.message}"
                    )
                    return GuardrailChainResult(
                        action=GuardrailAction.BLOCK,
                        results=results,
                        final_content=current,
                    )

                # Short-circuit on ESCALATE
                if result.action == GuardrailAction.ESCALATE:
                    logger.info(
                        f"Guardrail {guard.name} escalated for review: {result.message}"
                    )
                    return GuardrailChainResult(
                        action=GuardrailAction.ESCALATE,
                        results=results,
                        final_content=current,
                    )

                # Propagate MODIFY to next guardrail
                if (
                    result.action == GuardrailAction.MODIFY
                    and result.modified_content is not None
                ):
                    logger.debug(f"Guardrail {guard.name} modified content")
                    current = result.modified_content

            except Exception as e:
                logger.error(f"Guardrail {guard.name} failed: {e}")
                # On error, add a failed result and continue
                # This allows the chain to be resilient to individual failures
                results.append(
                    GuardrailResult(
                        action=GuardrailAction.ALLOW,
                        score=0.0,
                        message=f"Guardrail error: {e}",
                        guardrail_name=guard.name,
                        metadata={"error": str(e)},
                    )
                )

        # Determine final action
        # If any guardrail modified content, return MODIFY
        final_action = GuardrailAction.ALLOW
        if current != content:
            final_action = GuardrailAction.MODIFY

        return GuardrailChainResult(
            action=final_action,
            results=results,
            final_content=current,
        )
