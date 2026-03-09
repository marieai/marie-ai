"""Visualization models for workflow coordination dashboards.

This module provides data structures optimized for frontend visualization
of multi-agent workflows, including execution graphs, timelines, and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from marie.agent.coordination.state import AgentWorkflowState, AgentWorkflowStatus

# -------------------------------------------------------------------------
# Execution Graph Models (for DAG/Network Visualization)
# -------------------------------------------------------------------------


class NodeType(str, Enum):
    """Types of nodes in the execution graph."""

    AGENT = "agent"
    COORDINATOR = "coordinator"
    START = "start"
    END = "end"


class NodeStatus(str, Enum):
    """Status of a node in the execution graph."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GraphNode(BaseModel):
    """A node in the workflow execution graph.

    Designed for use with graph visualization libraries like
    React Flow, D3.js, or Cytoscape.
    """

    id: str
    label: str
    node_type: NodeType = NodeType.AGENT
    status: NodeStatus = NodeStatus.IDLE
    x: Optional[float] = None  # Position hint
    y: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """An edge in the workflow execution graph.

    Represents message flow between agents.
    """

    id: str
    source: str
    target: str
    label: Optional[str] = None
    edge_type: str = "message"  # message, control, error
    animated: bool = False  # For active edges
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionGraph(BaseModel):
    """Complete execution graph for visualization.

    Use with React Flow or similar libraries:
    ```typescript
    const { nodes, edges } = executionGraph;
    return <ReactFlow nodes={nodes} edges={edges} />;
    ```
    """

    workflow_id: str
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    layout: str = "dagre"  # dagre, elk, force
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_workflow_state(
        cls,
        state: AgentWorkflowState,
        agent_names: Optional[List[str]] = None,
    ) -> "ExecutionGraph":
        """Create execution graph from workflow state.

        Args:
            state: Current workflow state
            agent_names: Optional list of all agent names

        Returns:
            ExecutionGraph ready for visualization
        """
        nodes = []
        edges = []
        edge_count = 0

        # Start node
        nodes.append(
            GraphNode(
                id="__start__",
                label="Start",
                node_type=NodeType.START,
                status=NodeStatus.COMPLETED,
            )
        )

        # Agent nodes
        all_agents = set(agent_names or [])
        all_agents.update(state.step_history)

        for agent_name in all_agents:
            if agent_name in state.step_history:
                status = NodeStatus.COMPLETED
            elif agent_name == state.active_agent:
                status = NodeStatus.RUNNING
            else:
                status = NodeStatus.IDLE

            nodes.append(
                GraphNode(
                    id=agent_name,
                    label=agent_name,
                    node_type=NodeType.AGENT,
                    status=status,
                )
            )

        # End node
        end_status = NodeStatus.COMPLETED if state.is_terminal() else NodeStatus.IDLE
        nodes.append(
            GraphNode(
                id="__end__",
                label="End",
                node_type=NodeType.END,
                status=end_status,
            )
        )

        # Edges from communication history
        for sender, receiver, msg_type in state.communication_edges:
            edge_count += 1
            # Map coordinator to start node
            source = "__start__" if sender == "coordinator" else sender
            # Map __end__ and __coord__ to end node
            target = "__end__" if receiver in ("__end__", "__coord__") else receiver

            edges.append(
                GraphEdge(
                    id=f"e{edge_count}",
                    source=source,
                    target=target,
                    label=msg_type,
                    edge_type=msg_type,
                    animated=receiver == state.active_agent,
                )
            )

        return cls(
            workflow_id=state.workflow_id,
            nodes=nodes,
            edges=edges,
        )


# -------------------------------------------------------------------------
# Timeline Models (for Gantt/Timeline Visualization)
# -------------------------------------------------------------------------


class TimelineSegment(BaseModel):
    """A segment in the workflow timeline.

    Represents an agent's execution period.
    """

    id: str
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: NodeStatus = NodeStatus.COMPLETED
    output_preview: Optional[str] = None  # First N chars of output
    error: Optional[str] = None


class TimelineMarker(BaseModel):
    """A marker/event on the timeline."""

    id: str
    timestamp: datetime
    label: str
    marker_type: str = "event"  # event, error, checkpoint
    details: Optional[str] = None


class WorkflowTimeline(BaseModel):
    """Timeline visualization data for a workflow.

    Use with timeline libraries like vis-timeline or react-chrono.
    """

    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    segments: List[TimelineSegment] = Field(default_factory=list)
    markers: List[TimelineMarker] = Field(default_factory=list)
    status: AgentWorkflowStatus = AgentWorkflowStatus.PENDING


# -------------------------------------------------------------------------
# Agent Grid Models (for Status Dashboard)
# -------------------------------------------------------------------------


class AgentGridCell(BaseModel):
    """Status cell for a single agent in the grid view."""

    agent_name: str
    status: NodeStatus = NodeStatus.IDLE
    execution_count: int = 0
    total_duration_ms: float = 0.0
    last_execution: Optional[datetime] = None
    last_output: Optional[str] = None
    error_count: int = 0
    last_error: Optional[str] = None
    health_score: float = 1.0  # 0.0 to 1.0


class AgentStatusGrid(BaseModel):
    """Grid view of all agent statuses.

    Use for dashboard overview with agent cards.
    """

    workflow_id: str
    agents: List[AgentGridCell] = Field(default_factory=list)
    active_agent: Optional[str] = None
    total_agents: int = 0
    running_count: int = 0
    completed_count: int = 0
    failed_count: int = 0


# -------------------------------------------------------------------------
# Metrics Models (for Charts/Gauges)
# -------------------------------------------------------------------------


class MetricPoint(BaseModel):
    """A single metric data point."""

    timestamp: datetime
    value: float
    label: Optional[str] = None


class MetricSeries(BaseModel):
    """A series of metric points for charting."""

    name: str
    unit: str = ""  # ms, count, percent, etc.
    points: List[MetricPoint] = Field(default_factory=list)
    color: Optional[str] = None  # Hex color for chart


class WorkflowMetrics(BaseModel):
    """Aggregated metrics for workflow performance visualization.

    Use with chart libraries like Chart.js, Recharts, or Plotly.
    """

    workflow_id: str
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Summary metrics
    total_duration_ms: float = 0.0
    total_steps: int = 0
    success_rate: float = 1.0
    retry_count: int = 0

    # Per-agent metrics
    agent_durations: Dict[str, float] = Field(default_factory=dict)
    agent_call_counts: Dict[str, int] = Field(default_factory=dict)
    agent_error_counts: Dict[str, int] = Field(default_factory=dict)

    # Time series (for trend charts)
    duration_series: Optional[MetricSeries] = None
    throughput_series: Optional[MetricSeries] = None


# -------------------------------------------------------------------------
# Dashboard Aggregate Models
# -------------------------------------------------------------------------


class WorkflowDashboardData(BaseModel):
    """Complete dashboard data for a single workflow.

    Aggregates all visualization models for a full dashboard view.
    """

    workflow_id: str
    status: AgentWorkflowStatus
    goal: Optional[str] = None

    # Sub-components
    graph: Optional[ExecutionGraph] = None
    timeline: Optional[WorkflowTimeline] = None
    agent_grid: Optional[AgentStatusGrid] = None
    metrics: Optional[WorkflowMetrics] = None

    # Quick stats
    total_agents: int = 0
    completed_agents: int = 0
    running_agents: int = 0
    failed_agents: int = 0
    total_duration_ms: float = 0.0

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SystemDashboardData(BaseModel):
    """System-wide dashboard data across all workflows.

    For overview dashboards showing all active workflows.
    """

    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Active workflows summary
    active_workflows: int = 0
    completed_today: int = 0
    failed_today: int = 0
    total_agents_running: int = 0

    # Recent workflows
    recent_workflows: List[WorkflowDashboardData] = Field(default_factory=list)

    # System health
    avg_duration_ms: float = 0.0
    success_rate_24h: float = 1.0
    error_rate_24h: float = 0.0


# -------------------------------------------------------------------------
# Builder Functions
# -------------------------------------------------------------------------


def build_workflow_dashboard(
    state: AgentWorkflowState,
    agent_names: Optional[List[str]] = None,
    include_graph: bool = True,
    include_timeline: bool = True,
    include_metrics: bool = True,
) -> WorkflowDashboardData:
    """Build complete dashboard data from workflow state.

    Args:
        state: Current workflow state
        agent_names: List of all available agent names
        include_graph: Include execution graph
        include_timeline: Include timeline data
        include_metrics: Include metrics

    Returns:
        WorkflowDashboardData ready for frontend
    """
    all_agents = set(agent_names or [])
    all_agents.update(state.step_history)

    # Count agent statuses
    completed = len(state.step_history)
    running = 1 if state.active_agent else 0
    failed = len([e for e in state.errors if "failed" in e.lower()])

    dashboard = WorkflowDashboardData(
        workflow_id=state.workflow_id,
        status=state.status,
        goal=state.goal,
        total_agents=len(all_agents),
        completed_agents=completed,
        running_agents=running,
        failed_agents=failed,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )

    if include_graph:
        dashboard.graph = ExecutionGraph.from_workflow_state(state, list(all_agents))

    if include_timeline:
        # Build basic timeline from step history
        segments = []
        for i, agent_name in enumerate(state.step_history):
            segments.append(
                TimelineSegment(
                    id=f"seg-{i}",
                    agent_name=agent_name,
                    start_time=state.created_at,  # Simplified
                    status=NodeStatus.COMPLETED,
                )
            )

        dashboard.timeline = WorkflowTimeline(
            workflow_id=state.workflow_id,
            start_time=state.created_at,
            end_time=state.updated_at if state.is_terminal() else None,
            segments=segments,
            status=state.status,
        )

    if include_metrics:
        # Build basic metrics
        agent_counts = {}
        for agent_name in state.step_history:
            agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1

        dashboard.metrics = WorkflowMetrics(
            workflow_id=state.workflow_id,
            total_steps=state.step,
            agent_call_counts=agent_counts,
            success_rate=1.0 if state.status == AgentWorkflowStatus.COMPLETED else 0.5,
        )

    return dashboard
