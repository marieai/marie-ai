"""Input length validation guardrail.

Validates input length against character and token limits.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from marie.agent.guardrails.base import Guardrail, GuardrailConfig
from marie.agent.guardrails.registry import register_guardrail
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult


class InputLengthConfig(GuardrailConfig):
    """Configuration for input length validation.

    Attributes:
        max_chars: Maximum character count (0 = unlimited)
        max_tokens: Maximum token count estimate (0 = unlimited)
        min_chars: Minimum character count
        chars_per_token: Estimated characters per token for approximation
        block_message: Message when length limit exceeded
        truncate: Truncate instead of blocking (for max limits)
    """

    max_chars: int = Field(
        default=10000,
        ge=0,
        description="Maximum characters (0 = unlimited)",
    )
    max_tokens: int = Field(
        default=0,
        ge=0,
        description="Maximum tokens estimate (0 = unlimited)",
    )
    min_chars: int = Field(
        default=0,
        ge=0,
        description="Minimum characters required",
    )
    chars_per_token: float = Field(
        default=4.0,
        gt=0,
        description="Characters per token for estimation",
    )
    block_message: str = Field(
        default="Input exceeds maximum length",
        description="Message when length limit exceeded",
    )
    truncate: bool = Field(
        default=False,
        description="Truncate instead of blocking",
    )


def _estimate_tokens(text: str, chars_per_token: float) -> int:
    """Estimate token count from character count.

    This is a rough approximation. For precise counting,
    use a proper tokenizer.
    """
    return int(len(text) / chars_per_token)


@register_guardrail("input_length", "before")
class InputLengthGuardrail(Guardrail):
    """Validate input length against character and token limits.

    Checks input against:
    - Maximum character count
    - Maximum estimated token count
    - Minimum character count

    Can optionally truncate content instead of blocking.

    Example:
        ```yaml
        guardrails:
          before:
            - type: input_length
              config:
                max_chars: 10000
                max_tokens: 2500
                min_chars: 1
                truncate: false
        ```
    """

    name = "input_length"
    phase = "before"
    config_class = InputLengthConfig

    def __init__(self, config: InputLengthConfig = None):
        super().__init__(config or InputLengthConfig())

    async def evaluate(self, content: Any, context: dict) -> GuardrailResult:
        """Validate input length.

        Args:
            content: Input text to validate
            context: Execution context

        Returns:
            GuardrailResult with appropriate action
        """
        if not isinstance(content, str):
            return GuardrailResult(action=GuardrailAction.ALLOW)

        config = self.config
        if not isinstance(config, InputLengthConfig):
            config = InputLengthConfig()

        char_count = len(content)
        token_estimate = _estimate_tokens(content, config.chars_per_token)

        # Check minimum length
        if config.min_chars > 0 and char_count < config.min_chars:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                score=1.0,
                message=f"Input too short (minimum {config.min_chars} characters required)",
                metadata={
                    "char_count": char_count,
                    "min_chars": config.min_chars,
                },
            )

        # Check character limit
        exceeds_chars = config.max_chars > 0 and char_count > config.max_chars
        exceeds_tokens = config.max_tokens > 0 and token_estimate > config.max_tokens

        if not exceeds_chars and not exceeds_tokens:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                score=0.0,
                metadata={
                    "char_count": char_count,
                    "token_estimate": token_estimate,
                },
            )

        # Determine truncation target
        truncate_to: Optional[int] = None
        if exceeds_chars:
            truncate_to = config.max_chars
        if exceeds_tokens:
            token_char_limit = int(config.max_tokens * config.chars_per_token)
            if truncate_to is None or token_char_limit < truncate_to:
                truncate_to = token_char_limit

        # Truncate if configured
        if config.truncate and truncate_to:
            truncated = content[:truncate_to]
            # Try to truncate at word boundary
            if len(truncated) > 100:
                last_space = truncated.rfind(' ', truncate_to - 100, truncate_to)
                if last_space > 0:
                    truncated = truncated[:last_space]

            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                score=0.5,
                message=f"Input truncated from {char_count} to {len(truncated)} characters",
                modified_content=truncated,
                metadata={
                    "original_chars": char_count,
                    "truncated_chars": len(truncated),
                    "original_tokens": token_estimate,
                    "truncated_tokens": _estimate_tokens(
                        truncated, config.chars_per_token
                    ),
                },
            )

        # Block
        reason_parts = []
        if exceeds_chars:
            reason_parts.append(f"{char_count} chars (max {config.max_chars})")
        if exceeds_tokens:
            reason_parts.append(f"~{token_estimate} tokens (max {config.max_tokens})")

        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            score=1.0,
            message=f"{config.block_message}: {', '.join(reason_parts)}",
            metadata={
                "char_count": char_count,
                "token_estimate": token_estimate,
                "max_chars": config.max_chars,
                "max_tokens": config.max_tokens,
            },
        )
