"""Backend implementations for Marie agent framework."""

from marie.agent.backends.autogen_backend import (
    AutoGenAgentBackend,
    AutoGenBackendConfig,
    SingleAgentBackend,
    create_coding_backend,
    create_research_backend,
)
from marie.agent.backends.base import (
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendConfig,
    CompositeBackend,
    ToolCallRecord,
)
from marie.agent.backends.haystack_backend import (
    HaystackAgentBackend,
    HaystackBackendConfig,
    SimpleHaystackBackend,
)
from marie.agent.backends.openai_backend import (
    OpenAIAgentBackend,
    OpenAIBackendConfig,
)

__all__ = [
    # Base classes
    "AgentBackend",
    "AgentResult",
    "AgentStatus",
    "BackendConfig",
    "CompositeBackend",
    "ToolCallRecord",
    # OpenAI-compatible backend
    "OpenAIAgentBackend",
    "OpenAIBackendConfig",
    # Haystack backend
    "HaystackAgentBackend",
    "HaystackBackendConfig",
    "SimpleHaystackBackend",
    # AutoGen backend
    "AutoGenAgentBackend",
    "AutoGenBackendConfig",
    "SingleAgentBackend",
    "create_research_backend",
    "create_coding_backend",
]
