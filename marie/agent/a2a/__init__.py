"""A2A (Agent-to-Agent) protocol integration for Marie AI.

This package provides bidirectional A2A protocol support using the official
a2a-sdk package:

- Server: Expose Marie agents via A2A protocol
- Client: Call external A2A agents from Marie

Modules:
    sdk_executor: SDK-based AgentExecutor for Marie agents
    agent_card: AgentCard generation from Marie agents
    task: Task state mapping between Marie and A2A
    client: A2AClient for calling external agents
    discovery: Agent discovery with caching
    executor: A2AExecutor for exposing Marie agents
    errors: A2A-specific exception classes

SDK Types (re-exported for convenience):
    From a2a.types: AgentCard, AgentSkill, AgentCapabilities, Message,
                    Task, TaskState, TaskStatus, TextPart, etc.
"""

# Re-export commonly used SDK types
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

# Marie A2A components
from marie.agent.a2a.agent_card import (
    AgentCardBuilder,
    agent_card_from_agent,
    agent_card_from_config,
)
from marie.agent.a2a.client import A2AClient
from marie.agent.a2a.discovery import A2AAgentDiscovery, AgentRegistry, CachedAgent
from marie.agent.a2a.errors import (
    A2AClientError,
    A2AError,
    A2AErrorCode,
    A2AProtocolError,
    A2AServerError,
    AgentDiscoveryError,
    ContentTypeNotSupportedError,
    InvalidParamsError,
    InvalidRequestError,
    MethodNotFoundError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from marie.agent.a2a.executor import A2AExecutor
from marie.agent.a2a.sdk_executor import MarieA2AExecutor
from marie.agent.a2a.task import TaskStateMapper

__all__ = [
    # Executors
    "A2AExecutor",
    "MarieA2AExecutor",
    # Client
    "A2AClient",
    "A2AAgentDiscovery",
    "AgentRegistry",
    "CachedAgent",
    # Card builders
    "AgentCardBuilder",
    "agent_card_from_agent",
    "agent_card_from_config",
    # State mapping
    "TaskStateMapper",
    # Errors
    "A2AError",
    "A2AErrorCode",
    "A2AClientError",
    "A2AServerError",
    "A2AProtocolError",
    "TaskNotFoundError",
    "TaskNotCancelableError",
    "InvalidRequestError",
    "InvalidParamsError",
    "MethodNotFoundError",
    "UnsupportedOperationError",
    "ContentTypeNotSupportedError",
    "AgentDiscoveryError",
    # SDK types (re-exported)
    "AgentCard",
    "AgentSkill",
    "AgentCapabilities",
    "AgentProvider",
    "Message",
    "Task",
    "TaskState",
    "TaskStatus",
    "Part",
    "TextPart",
    "Artifact",
    "Role",
]
