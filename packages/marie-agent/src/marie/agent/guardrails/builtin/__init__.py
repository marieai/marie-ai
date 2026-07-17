"""Built-in guardrails for the Marie agent framework.

This package provides commonly-used guardrails that are registered
automatically when the guardrails package is imported.

Available guardrails by phase:

Before (input validation):
    - prompt_injection: Detect prompt injection attempts
    - pii: Detect/redact PII in input
    - content_filter: Filter banned words/patterns
    - rate_limit: Rate limiting per user/agent
    - input_length: Validate input length

After (output validation):
    - pii: Detect/redact PII in output
    - secrets: Detect/redact leaked credentials

Tool-call (tool access control):
    - tool_scope: Control which tools can execute
"""

# Import all built-in guardrails to register them
from marie.agent.guardrails.builtin.content_filter import ContentFilterGuardrail
from marie.agent.guardrails.builtin.input_length import InputLengthGuardrail
from marie.agent.guardrails.builtin.pii import PIIAfterGuardrail, PIIBeforeGuardrail
from marie.agent.guardrails.builtin.prompt_injection import PromptInjectionGuardrail
from marie.agent.guardrails.builtin.rate_limit import RateLimitGuardrail
from marie.agent.guardrails.builtin.secrets import SecretsAfterGuardrail
from marie.agent.guardrails.builtin.tool_scope import ToolScopeGuardrail

__all__ = [
    # Before phase
    "PromptInjectionGuardrail",
    "PIIBeforeGuardrail",
    "ContentFilterGuardrail",
    "RateLimitGuardrail",
    "InputLengthGuardrail",
    # After phase
    "PIIAfterGuardrail",
    "SecretsAfterGuardrail",
    # Tool-call phase
    "ToolScopeGuardrail",
]
