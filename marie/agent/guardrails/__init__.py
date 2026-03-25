"""Agent guardrails framework.

This package provides a guardrails system for validating and filtering
agent inputs and outputs at the executor level.

Key Components:
    - GuardrailAction: Enum of actions (ALLOW, BLOCK, MODIFY, ESCALATE)
    - GuardrailResult: Result from a single guardrail evaluation
    - Guardrail: Abstract base class for guardrails
    - GuardrailConfig: Base configuration for guardrails
    - GuardrailChain: Executes multiple guardrails in priority order
    - GuardedTool: Tool wrapper for tool-call guardrails
    - register_guardrail: Decorator to register guardrail classes
    - resolve_guardrails_for_phase: Instantiate guardrails from config

Phases:
    - before: Run before agent processes input
    - after: Run after agent generates output
    - tool_call: Run before a tool is executed

Example:
    ```python
    from marie.agent.guardrails import (
        Guardrail,
        GuardrailAction,
        GuardrailResult,
        register_guardrail,
    )


    @register_guardrail("my_filter", "before")
    class MyFilterGuardrail(Guardrail):
        async def evaluate(self, content: str, context: dict) -> GuardrailResult:
            if "blocked_word" in content.lower():
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    message="Content contains blocked word",
                )
            return GuardrailResult(action=GuardrailAction.ALLOW)
    ```

Configuration (YAML):
    ```yaml
    agent:
      name: my_agent
      guardrails:
        before:
          - type: prompt_injection
          - type: pii
            config:
              check_email: true
        after:
          - type: pii
          - type: secrets
        tool_call:
          - type: tool_scope
            config:
              allowed: [search, calculator]
    ```
"""

# Import builtin guardrails to register them
from marie.agent.guardrails import builtin as _builtin  # noqa: F401
from marie.agent.guardrails.base import Guardrail, GuardrailConfig
from marie.agent.guardrails.chain import GuardrailChain, GuardrailChainResult
from marie.agent.guardrails.guarded_tool import GuardedTool
from marie.agent.guardrails.registry import (
    GUARDRAIL_REGISTRY,
    get_available_guardrails,
    register_guardrail,
    resolve_guardrails_for_phase,
)
from marie.agent.guardrails.result import GuardrailAction, GuardrailResult

__all__ = [
    # Core types
    "GuardrailAction",
    "GuardrailResult",
    "Guardrail",
    "GuardrailConfig",
    "GuardrailChain",
    "GuardrailChainResult",
    "GuardedTool",
    # Registry
    "GUARDRAIL_REGISTRY",
    "register_guardrail",
    "resolve_guardrails_for_phase",
    "get_available_guardrails",
]
