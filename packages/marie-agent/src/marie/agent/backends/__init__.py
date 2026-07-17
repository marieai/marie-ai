"""Provider-independent backend contracts."""

from marie.agent.backends.base import (
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendConfig,
    CompositeBackend,
    ToolCallRecord,
)

__all__ = [
    "AgentBackend",
    "AgentResult",
    "AgentStatus",
    "BackendConfig",
    "CompositeBackend",
    "ToolCallRecord",
]
