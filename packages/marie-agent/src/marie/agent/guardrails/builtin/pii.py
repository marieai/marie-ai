"""PII (Personally Identifiable Information) detection guardrails.

Provides both before and after phase guardrails for detecting
and optionally redacting PII in content.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from pydantic import Field

from marie.agent.guardrails.base import Guardrail, GuardrailConfig
from marie.agent.guardrails.registry import register_guardrail
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult

# PII detection patterns: (pattern, type_name, replacement)
DEFAULT_PII_PATTERNS: List[Tuple[str, str, str]] = [
    # SSN (US Social Security Number)
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn", "[SSN REDACTED]"),
    (r"(?i)\b\d{9}\b(?=.*(?:ssn|social))", "ssn", "[SSN REDACTED]"),
    # Credit card numbers (major card types)
    (
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        "credit_card",
        "[CARD REDACTED]",
    ),
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "credit_card", "[CARD REDACTED]"),
    # Email addresses
    (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "email",
        "[EMAIL REDACTED]",
    ),
    # US phone numbers
    (
        r"\b(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        "phone",
        "[PHONE REDACTED]",
    ),
    # IP addresses
    (
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        "ip_address",
        "[IP REDACTED]",
    ),
    # Date of birth patterns
    (
        r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b",
        "date_of_birth",
        "[DOB REDACTED]",
    ),
    # Driver's license (generic pattern)
    (
        r"(?i)\b(?:DL|driver'?s?\s*license)[:\s#]*[A-Z0-9]{5,15}\b",
        "drivers_license",
        "[DL REDACTED]",
    ),
    # Passport numbers (generic)
    (r"(?i)\bpassport[:\s#]*[A-Z0-9]{6,12}\b", "passport", "[PASSPORT REDACTED]"),
    # Bank account numbers (generic)
    (r"(?i)\b(?:account|acct)[:\s#]*\d{8,17}\b", "bank_account", "[ACCOUNT REDACTED]"),
    # Routing numbers
    (r"(?i)\b(?:routing|aba)[:\s#]*\d{9}\b", "routing_number", "[ROUTING REDACTED]"),
]


class PIIConfig(GuardrailConfig):
    """Configuration for PII detection.

    Attributes:
        check_ssn: Detect Social Security Numbers
        check_credit_card: Detect credit card numbers
        check_email: Detect email addresses
        check_phone: Detect phone numbers
        check_ip: Detect IP addresses
        check_dob: Detect dates of birth
        check_ids: Detect driver's licenses, passports
        check_financial: Detect bank accounts, routing numbers
        redact: Replace PII with placeholders (vs just detecting)
        block_on_pii: Block content with PII (vs allowing with redaction)
    """

    check_ssn: bool = Field(default=True, description="Detect SSNs")
    check_credit_card: bool = Field(default=True, description="Detect credit cards")
    check_email: bool = Field(default=True, description="Detect emails")
    check_phone: bool = Field(default=True, description="Detect phone numbers")
    check_ip: bool = Field(default=False, description="Detect IP addresses")
    check_dob: bool = Field(default=True, description="Detect dates of birth")
    check_ids: bool = Field(default=True, description="Detect ID numbers")
    check_financial: bool = Field(default=True, description="Detect financial data")
    redact: bool = Field(default=True, description="Redact detected PII")
    block_on_pii: bool = Field(default=False, description="Block on PII detection")


def _get_enabled_patterns(config: PIIConfig) -> List[Tuple[re.Pattern, str, str]]:
    """Get compiled patterns based on config settings."""
    patterns = []
    type_to_flag = {
        "ssn": config.check_ssn,
        "credit_card": config.check_credit_card,
        "email": config.check_email,
        "phone": config.check_phone,
        "ip_address": config.check_ip,
        "date_of_birth": config.check_dob,
        "drivers_license": config.check_ids,
        "passport": config.check_ids,
        "bank_account": config.check_financial,
        "routing_number": config.check_financial,
    }

    for pattern, pii_type, replacement in DEFAULT_PII_PATTERNS:
        if type_to_flag.get(pii_type, True):
            patterns.append((re.compile(pattern), pii_type, replacement))

    return patterns


def _detect_pii(
    content: str, patterns: List[Tuple[re.Pattern, str, str]]
) -> List[Tuple[str, int, int, str]]:
    """Detect PII in content.

    Returns:
        List of (type, start, end, replacement) tuples
    """
    findings = []
    for pattern, pii_type, replacement in patterns:
        for match in pattern.finditer(content):
            findings.append((pii_type, match.start(), match.end(), replacement))
    return findings


def _redact_pii(content: str, findings: List[Tuple[str, int, int, str]]) -> str:
    """Redact PII in content based on findings."""
    if not findings:
        return content

    # Sort by position (reverse) to redact from end to start
    findings = sorted(findings, key=lambda x: x[1], reverse=True)

    for _, start, end, replacement in findings:
        content = content[:start] + replacement + content[end:]

    return content


@register_guardrail("pii", "before")
class PIIBeforeGuardrail(Guardrail):
    """Detect and optionally redact PII in user input.

    Scans input for common PII patterns including:
    - Social Security Numbers
    - Credit card numbers
    - Email addresses
    - Phone numbers
    - ID numbers (driver's license, passport)
    - Financial data (bank accounts, routing numbers)

    Can be configured to:
    - Block input containing PII
    - Redact PII before processing
    - Flag but allow PII

    Example:
        ```yaml
        guardrails:
          before:
            - type: pii
              config:
                check_email: true
                check_phone: true
                redact: true
                block_on_pii: false
        ```
    """

    name = "pii"
    phase = "before"
    config_class = PIIConfig

    def __init__(self, config: PIIConfig = None):
        super().__init__(config or PIIConfig())

    async def evaluate(self, content: Any, context: dict) -> GuardrailResult:
        """Evaluate input for PII.

        Args:
            content: Input text to scan
            context: Execution context

        Returns:
            GuardrailResult with appropriate action
        """
        if not isinstance(content, str):
            return GuardrailResult(action=GuardrailAction.ALLOW)

        config = self.config
        if not isinstance(config, PIIConfig):
            config = PIIConfig()

        patterns = _get_enabled_patterns(config)
        findings = _detect_pii(content, patterns)

        if not findings:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                score=0.0,
            )

        # Count by type
        type_counts = {}
        for pii_type, _, _, _ in findings:
            type_counts[pii_type] = type_counts.get(pii_type, 0) + 1

        # Block if configured
        if config.block_on_pii:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                score=1.0,
                message="PII detected in input",
                metadata={
                    "pii_count": len(findings),
                    "pii_types": list(type_counts.keys()),
                    "type_counts": type_counts,
                },
            )

        # Redact if configured
        if config.redact:
            redacted = _redact_pii(content, findings)
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                score=0.8,
                message="PII redacted from input",
                modified_content=redacted,
                metadata={
                    "pii_count": len(findings),
                    "pii_types": list(type_counts.keys()),
                    "type_counts": type_counts,
                },
            )

        # Flag but allow
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            score=0.8,
            message="PII detected in input (flagged)",
            metadata={
                "pii_count": len(findings),
                "pii_types": list(type_counts.keys()),
                "type_counts": type_counts,
                "flagged": True,
            },
        )


@register_guardrail("pii", "after")
class PIIAfterGuardrail(Guardrail):
    """Detect and redact PII in agent output.

    Scans agent responses for PII and redacts it before
    returning to the user. This prevents the agent from
    accidentally leaking sensitive information.

    Example:
        ```yaml
        guardrails:
          after:
            - type: pii
              config:
                redact: true
        ```
    """

    name = "pii"
    phase = "after"
    config_class = PIIConfig

    def __init__(self, config: PIIConfig = None):
        super().__init__(config or PIIConfig())

    async def evaluate(self, content: Any, context: dict) -> GuardrailResult:
        """Evaluate output for PII.

        Args:
            content: Output text to scan
            context: Execution context

        Returns:
            GuardrailResult with MODIFY if PII found and redaction enabled
        """
        if not isinstance(content, str):
            return GuardrailResult(action=GuardrailAction.ALLOW)

        config = self.config
        if not isinstance(config, PIIConfig):
            config = PIIConfig()

        patterns = _get_enabled_patterns(config)
        findings = _detect_pii(content, patterns)

        if not findings:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                score=0.0,
            )

        # Count by type
        type_counts = {}
        for pii_type, _, _, _ in findings:
            type_counts[pii_type] = type_counts.get(pii_type, 0) + 1

        # Always redact in after phase (output should never contain PII)
        if config.redact:
            redacted = _redact_pii(content, findings)
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                score=0.9,
                message="PII redacted from output",
                modified_content=redacted,
                metadata={
                    "pii_count": len(findings),
                    "pii_types": list(type_counts.keys()),
                    "type_counts": type_counts,
                },
            )

        # If not redacting, block output with PII
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            score=1.0,
            message="Output contains PII",
            metadata={
                "pii_count": len(findings),
                "pii_types": list(type_counts.keys()),
            },
        )
