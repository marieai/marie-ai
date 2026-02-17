"""A2A (Agent-to-Agent) protocol integration for Marie AI.

This package provides bidirectional A2A protocol support:
- Server: Expose Marie agents via A2A protocol
- Client: Call external A2A agents from Marie

Modules:
    types: Pydantic models for A2A protocol types
    constants: Protocol constants and error codes
    errors: A2A-specific exception classes
    jsonrpc: JSON-RPC 2.0 handler and dispatcher
    agent_card: AgentCard generation from Marie agents
    task: Task state machine and status mapping
    client: A2AClient for calling external agents
    discovery: Agent discovery with caching
    executor: A2AExecutor for exposing Marie agents
    streaming: SSE stream manager
"""

from marie.agent.a2a.agent_card import AgentCardBuilder
from marie.agent.a2a.client import A2AClient
from marie.agent.a2a.constants import A2AMethod, TaskState
from marie.agent.a2a.discovery import A2AAgentDiscovery
from marie.agent.a2a.errors import (
    A2AClientError,
    A2AError,
    A2AProtocolError,
    A2AServerError,
)
from marie.agent.a2a.executor import A2AExecutor
from marie.agent.a2a.task import TaskStateMapper

__all__ = [
    # Client
    "A2AClient",
    "A2AAgentDiscovery",
    # Server
    "A2AExecutor",
    "AgentCardBuilder",
    # Types and utilities
    "TaskState",
    "TaskStateMapper",
    "A2AMethod",
    # Errors
    "A2AError",
    "A2AClientError",
    "A2AServerError",
    "A2AProtocolError",
]
