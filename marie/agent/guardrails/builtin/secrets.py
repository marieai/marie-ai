"""Secrets detection guardrail.

Detects and redacts leaked credentials in agent output.
Reuses patterns from middleware/secrets_detection.py.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from pydantic import Field

from marie.agent.guardrails.base import Guardrail, GuardrailConfig
from marie.agent.guardrails.registry import register_guardrail
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult

# Common secret patterns (from middleware/secrets_detection.py)
DEFAULT_SECRET_PATTERNS: List[Tuple[str, str, str]] = [
    # API Keys
    (r"sk-[a-zA-Z0-9]{20,}", "openai_api_key", "[OPENAI_KEY REDACTED]"),
    (r"sk-ant-[a-zA-Z0-9-]{40,}", "anthropic_api_key", "[ANTHROPIC_KEY REDACTED]"),
    (r"AIza[a-zA-Z0-9_-]{35}", "google_api_key", "[GOOGLE_KEY REDACTED]"),
    (r"AKIA[A-Z0-9]{16}", "aws_access_key", "[AWS_KEY REDACTED]"),
    (r"ghp_[a-zA-Z0-9]{36}", "github_pat", "[GITHUB_TOKEN REDACTED]"),
    (r"gho_[a-zA-Z0-9]{36}", "github_oauth", "[GITHUB_TOKEN REDACTED]"),
    (r"glpat-[a-zA-Z0-9_-]{20}", "gitlab_pat", "[GITLAB_TOKEN REDACTED]"),
    (r"xox[baprs]-[a-zA-Z0-9-]{10,}", "slack_token", "[SLACK_TOKEN REDACTED]"),
    (r"sq0[a-z]{3}-[a-zA-Z0-9-]{22,}", "square_token", "[SQUARE_TOKEN REDACTED]"),
    (r"stripe[_-]?[a-z]+[_-][a-zA-Z0-9]{24,}", "stripe_key", "[STRIPE_KEY REDACTED]"),
    # Generic patterns
    (
        r"(?i)api[_-]?key['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{20,}['\"]?",
        "api_key_assignment",
        "[API_KEY REDACTED]",
    ),
    (
        r"(?i)password['\"]?\s*[:=]\s*['\"]?[^\s'\"]{8,}['\"]?",
        "password_assignment",
        "[PASSWORD REDACTED]",
    ),
    (
        r"(?i)secret['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{16,}['\"]?",
        "secret_assignment",
        "[SECRET REDACTED]",
    ),
    (
        r"(?i)token['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{20,}['\"]?",
        "token_assignment",
        "[TOKEN REDACTED]",
    ),
    (
        r"(?i)auth[_-]?token['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{20,}['\"]?",
        "auth_token_assignment",
        "[AUTH_TOKEN REDACTED]",
    ),
    # Connection strings
    (r"(?i)postgres://[^\s]+", "postgres_connection", "[POSTGRES_URL REDACTED]"),
    (r"(?i)mysql://[^\s]+", "mysql_connection", "[MYSQL_URL REDACTED]"),
    (r"(?i)mongodb\+srv://[^\s]+", "mongodb_connection", "[MONGODB_URL REDACTED]"),
    (r"(?i)redis://[^\s]+", "redis_connection", "[REDIS_URL REDACTED]"),
    (r"(?i)amqp://[^\s]+", "amqp_connection", "[AMQP_URL REDACTED]"),
    # Private keys
    (
        r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        "private_key_header",
        "[PRIVATE_KEY REDACTED]",
    ),
    # JWT tokens
    (
        r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
        "jwt_token",
        "[JWT_TOKEN REDACTED]",
    ),
    # Bearer tokens
    (
        r"(?i)bearer\s+[a-zA-Z0-9_-]{20,}",
        "bearer_token",
        "[BEARER_TOKEN REDACTED]",
    ),
]


class SecretsConfig(GuardrailConfig):
    """Configuration for secrets detection.

    Attributes:
        redact: Replace secrets with placeholders
        block_on_secrets: Block output containing secrets (vs redacting)
        additional_patterns: Additional (pattern, name, replacement) tuples
    """

    redact: bool = Field(
        default=True,
        description="Redact detected secrets",
    )
    block_on_secrets: bool = Field(
        default=False,
        description="Block output with secrets (vs redacting)",
    )
    additional_patterns: List[Tuple[str, str, str]] = Field(
        default_factory=list,
        description="Additional (pattern, name, replacement) tuples",
    )


def _compile_patterns(
    additional: List[Tuple[str, str, str]],
) -> List[Tuple[re.Pattern, str, str]]:
    """Compile secret detection patterns."""
    patterns = []
    for pattern, name, replacement in DEFAULT_SECRET_PATTERNS + additional:
        try:
            patterns.append((re.compile(pattern), name, replacement))
        except re.error:
            pass
    return patterns


def _detect_secrets(
    content: str, patterns: List[Tuple[re.Pattern, str, str]]
) -> List[Tuple[str, int, int, str]]:
    """Detect secrets in content.

    Returns:
        List of (type, start, end, replacement) tuples
    """
    findings = []
    for pattern, secret_type, replacement in patterns:
        for match in pattern.finditer(content):
            findings.append((secret_type, match.start(), match.end(), replacement))
    return findings


def _redact_secrets(content: str, findings: List[Tuple[str, int, int, str]]) -> str:
    """Redact secrets in content."""
    if not findings:
        return content

    # Sort by position (reverse) to redact from end to start
    findings = sorted(findings, key=lambda x: x[1], reverse=True)

    for _, start, end, replacement in findings:
        content = content[:start] + replacement + content[end:]

    return content


@register_guardrail("secrets", "after")
class SecretsAfterGuardrail(Guardrail):
    """Detect and redact leaked secrets in agent output.

    Scans output for common secret patterns including:
    - API keys (OpenAI, Anthropic, Google, AWS, GitHub, etc.)
    - Connection strings (PostgreSQL, MySQL, MongoDB, Redis)
    - Tokens (JWT, Bearer, OAuth)
    - Generic secrets (password=, api_key=, etc.)
    - Private key headers

    Example:
        ```yaml
        guardrails:
          after:
            - type: secrets
              config:
                redact: true
                block_on_secrets: false
        ```
    """

    name = "secrets"
    phase = "after"
    config_class = SecretsConfig

    def __init__(self, config: SecretsConfig = None):
        super().__init__(config or SecretsConfig())
        self._patterns: List[Tuple[re.Pattern, str, str]] = []
        self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        """Initialize compiled patterns."""
        config = self.config
        additional = []
        if isinstance(config, SecretsConfig):
            additional = config.additional_patterns
        self._patterns = _compile_patterns(additional)

    async def evaluate(self, content: Any, context: dict) -> GuardrailResult:
        """Evaluate output for leaked secrets.

        Args:
            content: Output text to scan
            context: Execution context

        Returns:
            GuardrailResult with MODIFY if secrets found and redaction enabled
        """
        if not isinstance(content, str):
            return GuardrailResult(action=GuardrailAction.ALLOW)

        config = self.config
        if not isinstance(config, SecretsConfig):
            config = SecretsConfig()

        findings = _detect_secrets(content, self._patterns)

        if not findings:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                score=0.0,
            )

        # Count by type
        type_counts = {}
        for secret_type, _, _, _ in findings:
            type_counts[secret_type] = type_counts.get(secret_type, 0) + 1

        # Block if configured
        if config.block_on_secrets:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                score=1.0,
                message="Output contains leaked secrets",
                metadata={
                    "secret_count": len(findings),
                    "secret_types": list(type_counts.keys()),
                },
            )

        # Redact secrets
        if config.redact:
            redacted = _redact_secrets(content, findings)
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                score=0.9,
                message="Secrets redacted from output",
                modified_content=redacted,
                metadata={
                    "secret_count": len(findings),
                    "secret_types": list(type_counts.keys()),
                    "type_counts": type_counts,
                },
            )

        # Flag but allow (shouldn't happen with default config)
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            score=0.9,
            message="Secrets detected in output (flagged)",
            metadata={
                "secret_count": len(findings),
                "secret_types": list(type_counts.keys()),
                "flagged": True,
            },
        )
