"""Unit tests for visualization models."""

from __future__ import annotations

from datetime import datetime, timezone

from marie.agent.coordination.state import AgentWorkflowStatus, create_workflow_state
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
    TimelineSegment,
    WorkflowDashboardData,
    WorkflowMetrics,
    WorkflowTimeline,
    build_workflow_dashboard,
)


class TestGraphNode:
    """Tests for GraphNode model."""

    def test_node_creation_minimal(self):
        """Test creating node with minimal fields."""
        node = GraphNode(id="agent1", label="Agent 1")
        assert node.id == "agent1"
        assert node.label == "Agent 1"
        assert node.node_type == NodeType.AGENT
        assert node.status == NodeStatus.IDLE

    def test_node_creation_full(self):
        """Test creating node with all fields."""
        node = GraphNode(
            id="start",
            label="Start",
            node_type=NodeType.START,
            status=NodeStatus.COMPLETED,
            x=100.0,
            y=200.0,
            metadata={"custom": "value"},
        )
        assert node.node_type == NodeType.START
        assert node.x == 100.0
        assert node.metadata["custom"] == "value"


class TestGraphEdge:
    """Tests for GraphEdge model."""

    def test_edge_creation_minimal(self):
        """Test creating edge with minimal fields."""
        edge = GraphEdge(id="e1", source="a", target="b")
        assert edge.id == "e1"
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.edge_type == "message"
        assert edge.animated is False

    def test_edge_creation_animated(self):
        """Test creating animated edge."""
        edge = GraphEdge(
            id="e2",
            source="a",
            target="b",
            label="task",
            animated=True,
        )
        assert edge.animated is True
        assert edge.label == "task"


class TestExecutionGraph:
    """Tests for ExecutionGraph model."""

    def test_graph_creation_empty(self):
        """Test creating empty graph."""
        graph = ExecutionGraph(workflow_id="wf-123")
        assert graph.workflow_id == "wf-123"
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.layout == "dagre"

    def test_graph_from_workflow_state(self):
        """Test creating graph from workflow state."""
        state = create_workflow_state("Test goal", "wf-test")
        state.step_history = ["planner", "executor"]

        graph = ExecutionGraph.from_workflow_state(
            state, ["planner", "executor", "validator"]
        )

        assert graph.workflow_id == "wf-test"
        # Should have start, 3 agents, and end nodes
        assert len(graph.nodes) >= 4

        # Check node types
        node_ids = [n.id for n in graph.nodes]
        assert "__start__" in node_ids
        assert "__end__" in node_ids
        assert "planner" in node_ids

    def test_graph_node_status_completed(self):
        """Test that completed agents have correct status."""
        state = create_workflow_state("Test", "wf-test")
        state.step_history = ["agent1"]

        graph = ExecutionGraph.from_workflow_state(state, ["agent1", "agent2"])

        agent1_node = next(n for n in graph.nodes if n.id == "agent1")
        agent2_node = next(n for n in graph.nodes if n.id == "agent2")

        assert agent1_node.status == NodeStatus.COMPLETED
        assert agent2_node.status == NodeStatus.IDLE


class TestTimelineSegment:
    """Tests for TimelineSegment model."""

    def test_segment_creation(self):
        """Test creating timeline segment."""
        now = datetime.now(timezone.utc)
        segment = TimelineSegment(
            id="seg-1",
            agent_name="planner",
            start_time=now,
            status=NodeStatus.COMPLETED,
            duration_ms=150.5,
        )
        assert segment.agent_name == "planner"
        assert segment.duration_ms == 150.5


class TestWorkflowTimeline:
    """Tests for WorkflowTimeline model."""

    def test_timeline_creation(self):
        """Test creating workflow timeline."""
        now = datetime.now(timezone.utc)
        timeline = WorkflowTimeline(
            workflow_id="wf-123",
            start_time=now,
            status=AgentWorkflowStatus.RUNNING,
        )
        assert timeline.workflow_id == "wf-123"
        assert len(timeline.segments) == 0
        assert len(timeline.markers) == 0


class TestAgentGridCell:
    """Tests for AgentGridCell model."""

    def test_cell_creation_defaults(self):
        """Test creating cell with defaults."""
        cell = AgentGridCell(agent_name="test_agent")
        assert cell.agent_name == "test_agent"
        assert cell.status == NodeStatus.IDLE
        assert cell.execution_count == 0
        assert cell.health_score == 1.0

    def test_cell_creation_with_stats(self):
        """Test creating cell with execution stats."""
        cell = AgentGridCell(
            agent_name="busy_agent",
            status=NodeStatus.COMPLETED,
            execution_count=5,
            total_duration_ms=1500.0,
            error_count=1,
        )
        assert cell.execution_count == 5
        assert cell.error_count == 1


class TestAgentStatusGrid:
    """Tests for AgentStatusGrid model."""

    def test_grid_creation_empty(self):
        """Test creating empty grid."""
        grid = AgentStatusGrid(workflow_id="wf-123")
        assert grid.workflow_id == "wf-123"
        assert len(grid.agents) == 0
        assert grid.total_agents == 0

    def test_grid_with_agents(self):
        """Test grid with agent cells."""
        grid = AgentStatusGrid(
            workflow_id="wf-123",
            agents=[
                AgentGridCell(agent_name="a1", status=NodeStatus.COMPLETED),
                AgentGridCell(agent_name="a2", status=NodeStatus.RUNNING),
                AgentGridCell(agent_name="a3", status=NodeStatus.IDLE),
            ],
            total_agents=3,
            completed_count=1,
            running_count=1,
        )
        assert grid.total_agents == 3
        assert grid.completed_count == 1


class TestWorkflowMetrics:
    """Tests for WorkflowMetrics model."""

    def test_metrics_creation_defaults(self):
        """Test creating metrics with defaults."""
        metrics = WorkflowMetrics(workflow_id="wf-123")
        assert metrics.workflow_id == "wf-123"
        assert metrics.total_duration_ms == 0.0
        assert metrics.success_rate == 1.0

    def test_metrics_with_data(self):
        """Test metrics with agent data."""
        metrics = WorkflowMetrics(
            workflow_id="wf-123",
            total_duration_ms=5000.0,
            total_steps=3,
            agent_durations={"a1": 1000.0, "a2": 2000.0, "a3": 2000.0},
            agent_call_counts={"a1": 1, "a2": 1, "a3": 1},
        )
        assert metrics.total_steps == 3
        assert len(metrics.agent_durations) == 3


class TestMetricSeries:
    """Tests for MetricSeries model."""

    def test_series_creation(self):
        """Test creating metric series."""
        now = datetime.now(timezone.utc)
        series = MetricSeries(
            name="duration",
            unit="ms",
            points=[
                MetricPoint(timestamp=now, value=100.0),
                MetricPoint(timestamp=now, value=150.0),
            ],
        )
        assert series.name == "duration"
        assert len(series.points) == 2


class TestWorkflowDashboardData:
    """Tests for WorkflowDashboardData model."""

    def test_dashboard_creation_minimal(self):
        """Test creating dashboard with minimal data."""
        dashboard = WorkflowDashboardData(
            workflow_id="wf-123",
            status=AgentWorkflowStatus.RUNNING,
        )
        assert dashboard.workflow_id == "wf-123"
        assert dashboard.status == AgentWorkflowStatus.RUNNING

    def test_dashboard_with_all_components(self):
        """Test dashboard with all visualization components."""
        dashboard = WorkflowDashboardData(
            workflow_id="wf-123",
            status=AgentWorkflowStatus.COMPLETED,
            goal="Process documents",
            graph=ExecutionGraph(workflow_id="wf-123"),
            timeline=WorkflowTimeline(
                workflow_id="wf-123",
                start_time=datetime.now(timezone.utc),
                status=AgentWorkflowStatus.COMPLETED,
            ),
            metrics=WorkflowMetrics(workflow_id="wf-123"),
            total_agents=3,
            completed_agents=3,
        )
        assert dashboard.graph is not None
        assert dashboard.timeline is not None
        assert dashboard.metrics is not None


class TestBuildWorkflowDashboard:
    """Tests for build_workflow_dashboard function."""

    def test_build_dashboard_basic(self):
        """Test building dashboard from workflow state."""
        state = create_workflow_state("Test goal", "wf-test")

        dashboard = build_workflow_dashboard(state, ["agent1", "agent2"])

        assert dashboard.workflow_id == "wf-test"
        assert dashboard.goal == "Test goal"
        assert dashboard.total_agents == 2

    def test_build_dashboard_with_history(self):
        """Test building dashboard with step history."""
        state = create_workflow_state("Test", "wf-test")
        state.step_history = ["planner", "executor"]
        state.status = AgentWorkflowStatus.COMPLETED

        dashboard = build_workflow_dashboard(
            state, ["planner", "executor", "validator"]
        )

        assert dashboard.completed_agents == 2
        assert dashboard.status == AgentWorkflowStatus.COMPLETED

    def test_build_dashboard_includes_graph(self):
        """Test that dashboard includes execution graph."""
        state = create_workflow_state("Test", "wf-test")

        dashboard = build_workflow_dashboard(state, ["agent1"], include_graph=True)

        assert dashboard.graph is not None
        assert len(dashboard.graph.nodes) > 0

    def test_build_dashboard_excludes_components(self):
        """Test building dashboard without optional components."""
        state = create_workflow_state("Test", "wf-test")

        dashboard = build_workflow_dashboard(
            state,
            ["agent1"],
            include_graph=False,
            include_timeline=False,
            include_metrics=False,
        )

        assert dashboard.graph is None
        assert dashboard.timeline is None
        assert dashboard.metrics is None


class TestSystemDashboardData:
    """Tests for SystemDashboardData model."""

    def test_system_dashboard_creation(self):
        """Test creating system-wide dashboard."""
        dashboard = SystemDashboardData(
            active_workflows=5,
            completed_today=10,
            failed_today=1,
            total_agents_running=3,
            success_rate_24h=0.95,
        )
        assert dashboard.active_workflows == 5
        assert dashboard.success_rate_24h == 0.95

    def test_system_dashboard_with_workflows(self):
        """Test system dashboard with workflow list."""
        dashboard = SystemDashboardData(
            active_workflows=2,
            recent_workflows=[
                WorkflowDashboardData(
                    workflow_id="wf-1",
                    status=AgentWorkflowStatus.RUNNING,
                ),
                WorkflowDashboardData(
                    workflow_id="wf-2",
                    status=AgentWorkflowStatus.COMPLETED,
                ),
            ],
        )
        assert len(dashboard.recent_workflows) == 2
