"""Prompt injection detection guardrail.

Detects common prompt injection patterns in user input.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from pydantic import Field

from marie.agent.guardrails.base import Guardrail, GuardrailConfig
from marie.agent.guardrails.registry import register_guardrail
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult

# Common prompt injection patterns
DEFAULT_INJECTION_PATTERNS: List[Tuple[str, str, float]] = [
    # Direct instruction overrides
    (
        r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        "instruction_override",
        0.9,
    ),
    (
        r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        "instruction_override",
        0.9,
    ),
    (
        r"(?i)forget\s+(all\s+)?(previous|prior|your)\s+(instructions?|prompts?|rules?)",
        "instruction_override",
        0.9,
    ),
    (
        r"(?i)do\s+not\s+follow\s+(your|the)\s+(instructions?|rules?|guidelines?)",
        "instruction_override",
        0.9,
    ),
    # Role manipulation
    (r"(?i)you\s+are\s+now\s+(?:a|an|the)\s+", "role_manipulation", 0.8),
    (r"(?i)pretend\s+(?:to\s+be|you\s+are)\s+", "role_manipulation", 0.8),
    (r"(?i)act\s+as\s+(?:if\s+you\s+are\s+)?(?:a|an|the)\s+", "role_manipulation", 0.7),
    (r"(?i)roleplay\s+as\s+", "role_manipulation", 0.8),
    (r"(?i)from\s+now\s+on[,\s]+(?:you\s+are|act\s+as)", "role_manipulation", 0.8),
    # System prompt extraction
    (
        r"(?i)(?:show|reveal|display|print|output|tell)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?|rules?)",
        "prompt_extraction",
        0.9,
    ),
    (
        r"(?i)what\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instructions?|rules?)",
        "prompt_extraction",
        0.8,
    ),
    (
        r"(?i)repeat\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)",
        "prompt_extraction",
        0.9,
    ),
    # Delimiter injection
    (r"<\|(?:im_start|im_end|system|user|assistant)\|>", "delimiter_injection", 0.95),
    (r"\[(?:INST|/INST|SYS|/SYS)\]", "delimiter_injection", 0.95),
    (r"<<SYS>>|<</SYS>>", "delimiter_injection", 0.95),
    # Jailbreak patterns
    (r"(?i)(?:DAN|do\s+anything\s+now)\s+mode", "jailbreak", 0.9),
    (r"(?i)developer\s+mode\s+(?:enabled|activated|on)", "jailbreak", 0.9),
    (
        r"(?i)(?:enable|activate|turn\s+on)\s+(?:developer|debug|admin)\s+mode",
        "jailbreak",
        0.9,
    ),
    (
        r"(?i)bypass\s+(?:safety|content|ethical)\s+(?:filters?|guidelines?|restrictions?)",
        "jailbreak",
        0.95,
    ),
    # Hypothetical framing
    (r"(?i)hypothetically[,\s]+(?:if|what\s+if)", "hypothetical_framing", 0.5),
    (
        r"(?i)for\s+(?:educational|research|academic)\s+purposes?\s+only",
        "hypothetical_framing",
        0.5,
    ),
    (
        r"(?i)in\s+a\s+(?:fictional|hypothetical|theoretical)\s+scenario",
        "hypothetical_framing",
        0.5,
    ),
    # Encoded/obfuscated instructions
    (
        r"(?i)decode\s+(?:the\s+following|this)\s+(?:base64|hex|binary)",
        "encoded_instruction",
        0.7,
    ),
    (
        r"(?i)execute\s+(?:the\s+following|this)\s+(?:code|command|script)",
        "code_execution",
        0.8,
    ),
]


class PromptInjectionConfig(GuardrailConfig):
    """Configuration for prompt injection detection.

    Attributes:
        threshold: Minimum score to trigger detection (0.0-1.0)
        block_on_detect: Whether to block or just flag detections
        custom_patterns: Additional patterns to check
    """

    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum score to trigger detection",
    )
    block_on_detect: bool = Field(
        default=True,
        description="Block requests on detection (vs just flagging)",
    )
    custom_patterns: List[Tuple[str, str, float]] = Field(
        default_factory=list,
        description="Additional (pattern, name, score) tuples",
    )


@register_guardrail("prompt_injection", "before")
class PromptInjectionGuardrail(Guardrail):
    """Detects prompt injection attempts in user input.

    Scans for common patterns used in prompt injection attacks:
    - Instruction overrides ("ignore previous instructions")
    - Role manipulation ("you are now a...")
    - System prompt extraction attempts
    - Delimiter injection
    - Jailbreak patterns
    - Hypothetical framing

    Example:
        ```yaml
        guardrails:
          before:
            - type: prompt_injection
              config:
                threshold: 0.7
                block_on_detect: true
        ```
    """

    name = "prompt_injection"
    phase = "before"
    config_class = PromptInjectionConfig

    def __init__(self, config: PromptInjectionConfig = None):
        super().__init__(config or PromptInjectionConfig())
        self._patterns: List[Tuple[re.Pattern, str, float]] = []

        # Compile default patterns
        for pattern, name, score in DEFAULT_INJECTION_PATTERNS:
            self._patterns.append((re.compile(pattern), name, score))

        # Add custom patterns
        config = self.config
        if isinstance(config, PromptInjectionConfig):
            for pattern, name, score in config.custom_patterns:
                self._patterns.append((re.compile(pattern), name, score))

    async def evaluate(self, content: Any, context: dict) -> GuardrailResult:
        """Evaluate content for prompt injection patterns.

        Args:
            content: Input text to scan
            context: Execution context

        Returns:
            GuardrailResult with BLOCK if injection detected, ALLOW otherwise
        """
        if not isinstance(content, str):
            return GuardrailResult(action=GuardrailAction.ALLOW)

        config = self.config
        if not isinstance(config, PromptInjectionConfig):
            config = PromptInjectionConfig()

        detections: List[Tuple[str, float]] = []
        max_score = 0.0

        for pattern, name, score in self._patterns:
            if pattern.search(content):
                detections.append((name, score))
                max_score = max(max_score, score)

        if not detections:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                score=0.0,
            )

        # Check against threshold
        if max_score >= config.threshold:
            if config.block_on_detect:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    score=max_score,
                    message="Potential prompt injection detected",
                    metadata={
                        "detection_count": len(detections),
                        "patterns_matched": [d[0] for d in detections],
                        "max_score": max_score,
                    },
                )
            else:
                # Flag but allow
                return GuardrailResult(
                    action=GuardrailAction.ALLOW,
                    score=max_score,
                    message="Potential prompt injection detected (flagged)",
                    metadata={
                        "detection_count": len(detections),
                        "patterns_matched": [d[0] for d in detections],
                        "flagged": True,
                    },
                )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            score=max_score,
            metadata={
                "detection_count": len(detections),
                "below_threshold": True,
            },
        )
