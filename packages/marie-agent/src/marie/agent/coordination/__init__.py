"""Agent coordination module for multi-agent orchestration.

This module provides coordination patterns for running multiple agents
together, including parallel (fan-out), sequential (chain), and workflow topologies.
"""

from marie.agent.coordination.adapters import coordination_result_to_agent_result
from marie.agent.coordination.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    InMemoryAuditLogger,
    StructuredAuditLogger,
    create_audit_logger,
)
from marie.agent.coordination.chain import ChainCoordinator
from marie.agent.coordination.checkpoint import (
    CheckpointStore,
    InMemoryCheckpointStore,
)
from marie.agent.coordination.config import CoordinationConfig, MergeStrategy, Topology
from marie.agent.coordination.context import AgentExecutionContext
from marie.agent.coordination.execution import (
    execute_agent_async,
    execute_agent_with_timeout,
)
from marie.agent.coordination.fan_out import FanOutCoordinator
from marie.agent.coordination.message import (
    AgentMessage,
    AgentMessageType,
    ReservedReceiver,
    create_error_message,
    create_result_message,
    create_task_message,
)
from marie.agent.coordination.state import (
    AgentWorkflowState,
    AgentWorkflowStatus,
    create_workflow_state,
)
from marie.agent.coordination.topology import (
    AgentResult,
    BaseCoordinator,
    CoordinationResult,
    CoordinatorFactory,
)
from marie.agent.coordination.visualization import (
    AgentGridCell,
    AgentStatusGrid,
    ExecutionGraph,
    GraphEdge,
    GraphNode,
    MetricPoint,
    MetricSeries,
    NodeStatus,
    NodeType,
    SystemDashboardData,
    TimelineMarker,
    TimelineSegment,
    WorkflowDashboardData,
    WorkflowMetrics,
    WorkflowTimeline,
    build_workflow_dashboard,
)
from marie.agent.coordination.workflow import (
    MessageDrivenRoutingPolicy,
    RoutingPolicy,
    SequentialRoutingPolicy,
    WorkflowCoordinator,
)

# Register WorkflowCoordinator with factory
CoordinatorFactory.register("workflow", WorkflowCoordinator)

__all__ = [
    # Result types
    "AgentResult",
    "CoordinationResult",
    # Coordinators
    "BaseCoordinator",
    "ChainCoordinator",
    "CoordinatorFactory",
    "FanOutCoordinator",
    "WorkflowCoordinator",
    # Configuration
    "CoordinationConfig",
    "MergeStrategy",
    "Topology",
    # Execution utilities
    "AgentExecutionContext",
    "coordination_result_to_agent_result",
    "execute_agent_async",
    "execute_agent_with_timeout",
    # Message types
    "AgentMessage",
    "AgentMessageType",
    "ReservedReceiver",
    "create_error_message",
    "create_result_message",
    "create_task_message",
    # Workflow state
    "AgentWorkflowState",
    "AgentWorkflowStatus",
    "create_workflow_state",
    # Routing policies
    "MessageDrivenRoutingPolicy",
    "RoutingPolicy",
    "SequentialRoutingPolicy",
    # Checkpoint store
    "CheckpointStore",
    "InMemoryCheckpointStore",
    # Audit logging
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "InMemoryAuditLogger",
    "StructuredAuditLogger",
    "create_audit_logger",
    # Visualization models
    "AgentGridCell",
    "AgentStatusGrid",
    "ExecutionGraph",
    "GraphEdge",
    "GraphNode",
    "MetricPoint",
    "MetricSeries",
    "NodeStatus",
    "NodeType",
    "SystemDashboardData",
    "TimelineMarker",
    "TimelineSegment",
    "WorkflowDashboardData",
    "WorkflowMetrics",
    "WorkflowTimeline",
    "build_workflow_dashboard",
]
