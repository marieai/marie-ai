"""Guardrail evaluation result types.

This module defines the result types returned by guardrail evaluations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class GuardrailAction(str, Enum):
    """Action to take based on guardrail evaluation.

    Attributes:
        ALLOW: Content passes, proceed normally
        BLOCK: Content blocked, return error response
        MODIFY: Content modified, use modified_content
        ESCALATE: Requires human review
    """

    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    ESCALATE = "escalate"


class GuardrailResult(BaseModel):
    """Result from a single guardrail evaluation.

    Attributes:
        action: The action to take (allow, block, modify, escalate)
        score: Confidence score (0.0 to 1.0)
        message: Human-readable message explaining the result
        modified_content: Modified content if action is MODIFY
        guardrail_name: Name of the guardrail that produced this result
        metadata: Additional metadata (MUST contain only summary data:
                  counts, type names, pattern identifiers. NEVER raw
                  content, matched text, or tool args.)
    """

    action: GuardrailAction = Field(
        default=GuardrailAction.ALLOW,
        description="Action to take based on evaluation",
    )
    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the evaluation",
    )
    message: Optional[str] = Field(
        default=None,
        description="Human-readable message explaining the result",
    )
    modified_content: Optional[Any] = Field(
        default=None,
        description="Modified content if action is MODIFY",
    )
    guardrail_name: str = Field(
        default="",
        description="Name of the guardrail that produced this result",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Summary metadata (counts, types, identifiers only)",
    )

    def is_blocking(self) -> bool:
        """Check if this result blocks execution."""
        return self.action in (GuardrailAction.BLOCK, GuardrailAction.ESCALATE)

    def is_modifying(self) -> bool:
        """Check if this result modifies content."""
        return (
            self.action == GuardrailAction.MODIFY and self.modified_content is not None
        )
