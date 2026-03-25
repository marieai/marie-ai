"""Rate limiting guardrail.

Implements token bucket rate limiting per user/agent.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from pydantic import Field

from marie.agent.guardrails.base import Guardrail, GuardrailConfig
from marie.agent.guardrails.registry import register_guardrail
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult


class RateLimitConfig(GuardrailConfig):
    """Configuration for rate limiting.

    Attributes:
        requests_per_minute: Maximum requests per minute per key
        burst_size: Maximum burst size (token bucket capacity)
        key_type: Rate limit key type ('user', 'agent', 'conversation', 'global')
        block_message: Message when rate limit exceeded
    """

    requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum requests per minute",
    )
    burst_size: int = Field(
        default=10,
        ge=1,
        description="Maximum burst size",
    )
    key_type: str = Field(
        default="user",
        description="Rate limit key: 'user', 'agent', 'conversation', 'global'",
    )
    block_message: str = Field(
        default="Rate limit exceeded. Please wait before sending more requests.",
        description="Message when rate limit exceeded",
    )


class TokenBucket:
    """Token bucket rate limiter.

    Implements the token bucket algorithm for rate limiting.
    Tokens are added at a fixed rate and consumed on each request.
    """

    def __init__(self, rate: float, capacity: int):
        """Initialize the token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient
        """
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now

        # Add tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def available(self) -> float:
        """Get current available tokens."""
        now = time.monotonic()
        elapsed = now - self.last_update
        return min(self.capacity, self.tokens + elapsed * self.rate)


# Global storage for rate limit buckets
_rate_limit_buckets: Dict[str, TokenBucket] = {}


def _get_bucket(key: str, config: RateLimitConfig) -> TokenBucket:
    """Get or create a token bucket for the given key."""
    if key not in _rate_limit_buckets:
        rate = config.requests_per_minute / 60.0  # Convert to per-second
        _rate_limit_buckets[key] = TokenBucket(rate, config.burst_size)
    return _rate_limit_buckets[key]


def _get_rate_limit_key(context: dict, key_type: str) -> str:
    """Generate rate limit key from context."""
    if key_type == "global":
        return "global"
    elif key_type == "agent":
        return f"agent:{context.get('agent_name', 'default')}"
    elif key_type == "conversation":
        return f"conversation:{context.get('conversation_id', 'default')}"
    else:  # user
        return f"user:{context.get('user_id', 'anonymous')}"


@register_guardrail("rate_limit", "before")
class RateLimitGuardrail(Guardrail):
    """Rate limit requests using token bucket algorithm.

    Limits requests per minute based on user, agent, conversation,
    or global scope. Uses token bucket algorithm to allow bursts
    while maintaining average rate.

    Example:
        ```yaml
        guardrails:
          before:
            - type: rate_limit
              config:
                requests_per_minute: 60
                burst_size: 10
                key_type: user
        ```
    """

    name = "rate_limit"
    phase = "before"
    config_class = RateLimitConfig

    def __init__(self, config: RateLimitConfig = None):
        super().__init__(config or RateLimitConfig())

    async def evaluate(self, content: Any, context: dict) -> GuardrailResult:
        """Check rate limit for the request.

        Args:
            content: Request content (not used for rate limiting)
            context: Execution context with user_id, agent_name, etc.

        Returns:
            GuardrailResult with BLOCK if rate limit exceeded
        """
        config = self.config
        if not isinstance(config, RateLimitConfig):
            config = RateLimitConfig()

        # Get rate limit key
        key = _get_rate_limit_key(context, config.key_type)

        # Get or create bucket
        bucket = _get_bucket(key, config)

        # Try to consume a token
        if bucket.consume(1):
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                score=0.0,
                metadata={
                    "rate_limit_key": config.key_type,
                    "tokens_remaining": int(bucket.available()),
                },
            )

        # Rate limit exceeded
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            score=1.0,
            message=config.block_message,
            metadata={
                "rate_limit_key": config.key_type,
                "tokens_remaining": 0,
                "retry_after_seconds": int(1.0 / bucket.rate),
            },
        )


def reset_rate_limits() -> None:
    """Reset all rate limit buckets (useful for testing)."""
    global _rate_limit_buckets
    _rate_limit_buckets = {}
