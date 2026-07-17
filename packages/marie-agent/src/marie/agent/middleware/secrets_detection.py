"""Secrets detection middleware for redacting leaked credentials.

Detects and optionally redacts sensitive information like API keys,
passwords, and other credentials in event data.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from marie.agent.middleware.protocol import BaseMiddleware

if TYPE_CHECKING:
    from marie.agent.emitter import Emitter

logger = logging.getLogger("marie.agent.middleware.secrets_detection")


# Common secret patterns
DEFAULT_SECRET_PATTERNS: List[Tuple[str, str]] = [
    # API Keys
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API Key"),
    (r"sk-ant-[a-zA-Z0-9-]{40,}", "Anthropic API Key"),
    (r"AIza[a-zA-Z0-9_-]{35}", "Google API Key"),
    (r"AKIA[A-Z0-9]{16}", "AWS Access Key ID"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub Personal Access Token"),
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth Token"),
    (r"glpat-[a-zA-Z0-9_-]{20}", "GitLab Personal Access Token"),
    # Generic patterns
    (
        r"(?i)api[_-]?key['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{20,}['\"]?",
        "API Key Assignment",
    ),
    (r"(?i)password['\"]?\s*[:=]\s*['\"]?[^\s'\"]{8,}['\"]?", "Password Assignment"),
    (r"(?i)secret['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{16,}['\"]?", "Secret Assignment"),
    (r"(?i)token['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{20,}['\"]?", "Token Assignment"),
    # Connection strings
    (r"(?i)postgres://[^\s]+", "PostgreSQL Connection String"),
    (r"(?i)mysql://[^\s]+", "MySQL Connection String"),
    (r"(?i)mongodb\+srv://[^\s]+", "MongoDB Connection String"),
    (r"(?i)redis://[^\s]+", "Redis Connection String"),
    # Private keys
    (r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "Private Key Header"),
]


class SecretsDetectionError(Exception):
    """Raised when secrets are detected in strict mode."""

    def __init__(self, message: str, secret_type: str):
        self.secret_type = secret_type
        super().__init__(message)


class SecretsDetectionMiddleware(BaseMiddleware):
    """Middleware that detects and redacts leaked credentials.

    Scans event data for common secret patterns and can:
    - Log a warning (default)
    - Raise an exception to block the operation (strict mode)
    - Redact secrets in-place (redact mode)
    """

    def __init__(
        self,
        patterns: Optional[List[Tuple[str, str]]] = None,
        strict: bool = False,
        redact: bool = True,
        redact_replacement: str = "[REDACTED]",
        on_detect: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Initialize secrets detection middleware.

        Args:
            patterns: List of (regex, description) tuples for secret detection
            strict: If True, raise exception on detection; otherwise log warning
            redact: If True, redact secrets in event data
            redact_replacement: Replacement string for redacted content
            on_detect: Optional callback when secret detected (secret_type, content)
        """
        super().__init__(name="SecretsDetectionMiddleware", priority=100)
        self._patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(p), desc) for p, desc in (patterns or DEFAULT_SECRET_PATTERNS)
        ]
        self._strict = strict
        self._redact = redact
        self._redact_replacement = redact_replacement
        self._on_detect = on_detect

    def bind(self, emitter: "Emitter") -> None:
        """Bind secrets detection to emitter events."""
        # Scan all events with high priority (blocking if strict)
        self._listener_ids.append(
            emitter.on(
                "*",
                self._scan_event,
                priority=self.priority,
                is_blocking=self._strict,
            )
        )

    def _detect_secrets(self, content: str) -> List[Tuple[str, str, int, int]]:
        """Detect secrets in content.

        Returns:
            List of (description, matched_text, start, end) tuples
        """
        findings: List[Tuple[str, str, int, int]] = []

        for pattern, description in self._patterns:
            for match in pattern.finditer(content):
                findings.append(
                    (description, match.group(), match.start(), match.end())
                )

        return findings

    def _redact_content(self, content: str) -> str:
        """Redact all detected secrets in content."""
        findings = self._detect_secrets(content)

        if not findings:
            return content

        # Sort by position (reverse) to redact from end to start
        findings.sort(key=lambda x: x[2], reverse=True)

        for _, _, start, end in findings:
            content = content[:start] + self._redact_replacement + content[end:]

        return content

    def _handle_detection(self, description: str, matched: str) -> None:
        """Handle a secret detection."""
        # Truncate matched text for logging (don't log full secrets)
        truncated = matched[:10] + "..." if len(matched) > 10 else matched

        if self._on_detect:
            self._on_detect(description, truncated)

        if self._strict:
            raise SecretsDetectionError(
                f"Secret detected: {description}",
                secret_type=description,
            )
        else:
            logger.warning(f"Secret detected: {description} (starts with: {truncated})")

    def _scan_value(self, value: Any) -> Any:
        """Scan and optionally redact a value."""
        if isinstance(value, str):
            findings = self._detect_secrets(value)
            for description, matched, _, _ in findings:
                self._handle_detection(description, matched)

            if self._redact and findings:
                return self._redact_content(value)
            return value

        elif isinstance(value, dict):
            return {k: self._scan_value(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._scan_value(v) for v in value]

        return value

    def _scan_event(self, data: Dict[str, Any]) -> None:
        """Scan event data for secrets."""
        # Scan all string values in the event data
        for key, value in list(data.items()):
            scanned = self._scan_value(value)
            if self._redact and scanned != value:
                data[key] = scanned
