"""Agent-specific query planners for Marie agent framework.

This module provides query planners that create DAGs for agent workflows,
including orchestration patterns like Qwen meta-planner with delegated
Haystack RAG and AutoGen teams.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.logging_core.predefined import default_logger as logger
from marie.query_planner.base import (
    Query,
    QueryDefinition,
    QueryPlanRegistry,
    QueryType,
    QueryTypeRegistry,
    register_query_plan,
)

# ============================================================================
# Agent Query Definitions
# ============================================================================


@QueryTypeRegistry.register("AGENT")
class AgentQueryDefinition(QueryDefinition):
    """Query definition for agent execution.

    Represents a task executed by an agent backend.
    """

    method: str = "AGENT"
    agent_backend: str = Field(
        default="qwen_agent",
        description="Agent backend type (qwen_agent, haystack, autogen)",
    )
    agent_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific configuration",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for continuity",
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum agent iterations",
    )
    tools: List[str] = Field(
        default_factory=list,
        description="Tools available to the agent",
    )
    params: dict = Field(
        default_factory=lambda: {
            "layout": None,
            "memory_type": "chat_buffer",
        }
    )

    def validate_params(self):
        """Validate agent query parameters."""
        if self.agent_backend not in ("qwen_agent", "haystack", "autogen"):
            logger.warning(
                f"Unknown agent backend: {self.agent_backend}. "
                "Ensure it's registered with AgentExecutor."
            )


@QueryTypeRegistry.register("AGENT_TOOL_CALL")
class AgentToolCallQueryDefinition(QueryDefinition):
    """Query definition for agent-to-agent tool calls.

    Used when an agent spawns a sub-task that should become a tracked
    DAG node rather than inline execution.
    """

    method: str = "AGENT_TOOL_CALL"
    caller_agent_id: str = Field(
        ...,
        description="ID of the agent making the call",
    )
    target_tool: str = Field(
        ...,
        description="Tool/agent being called",
    )
    tool_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the tool call",
    )
    params: dict = Field(default_factory=lambda: {"layout": None})

    def validate_params(self):
        """Validate tool call parameters."""
        if not self.target_tool:
            raise ValueError("target_tool is required for AGENT_TOOL_CALL")


@QueryTypeRegistry.register("RAG")
class RAGQueryDefinition(QueryDefinition):
    """Query definition for RAG (Retrieval-Augmented Generation).

    Specialized definition for Haystack RAG pipelines.
    """

    method: str = "RAG"
    endpoint: str = "agent://haystack/rag"
    retriever_type: str = Field(
        default="bm25",
        description="Type of retriever (bm25, embedding, hybrid)",
    )
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    generator_model: Optional[str] = Field(
        default=None,
        description="Model for answer generation",
    )
    params: dict = Field(
        default_factory=lambda: {
            "layout": None,
            "return_documents": True,
        }
    )

    def validate_params(self):
        """Validate RAG parameters."""
        pass


@QueryTypeRegistry.register("MULTI_AGENT")
class MultiAgentQueryDefinition(QueryDefinition):
    """Query definition for multi-agent team execution.

    Used for AutoGen teams or similar multi-agent workflows.
    """

    method: str = "MULTI_AGENT"
    endpoint: str = "agent://autogen/team"
    team_type: str = Field(
        default="research",
        description="Team type (research, coding, custom)",
    )
    agents: List[str] = Field(
        default_factory=list,
        description="Agent names in the team",
    )
    max_rounds: int = Field(
        default=10,
        description="Maximum conversation rounds",
    )
    speaker_selection: str = Field(
        default="auto",
        description="Speaker selection method",
    )
    params: dict = Field(default_factory=lambda: {"layout": None})

    def validate_params(self):
        """Validate multi-agent parameters."""
        if not self.agents:
            logger.warning(
                "No agents specified for MULTI_AGENT query. "
                "Using default team configuration."
            )


# ============================================================================
# Agent Query Planners
# ============================================================================


class PlannerInfo(BaseModel):
    """Information passed to query planners."""

    query_str: str = Field(..., description="User query string")
    ref_id: str = Field(..., description="Reference ID for the job")
    ref_type: str = Field(default="agent", description="Reference type")
    pages: List[int] = Field(default_factory=list, description="Page numbers")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class QueryPlan(BaseModel):
    """A plan consisting of Query nodes."""

    nodes: List[Query] = Field(
        default_factory=list, description="Query nodes in the plan"
    )

    def add_node(self, node: Query) -> None:
        """Add a node to the plan."""
        self.nodes.append(node)

    def get_node(self, task_id: str) -> Optional[Query]:
        """Get a node by task ID."""
        for node in self.nodes:
            if node.task_id == task_id:
                return node
        return None


@register_query_plan("agent_qwen_orchestrator")
def query_planner_qwen_orchestrator(
    planner_info: PlannerInfo,
    **kwargs: Any,
) -> QueryPlan:
    """DAG where Qwen-Agent is the top-level planner.

    Creates a simple orchestration DAG:
        START -> QWEN_PLAN -> END

    The agent can spawn additional nodes dynamically during execution
    via the DAGExpansionService.

    Args:
        planner_info: Planner information
        **kwargs: Additional arguments

    Returns:
        QueryPlan with orchestrator DAG
    """
    ref_id = planner_info.ref_id
    query_str = planner_info.query_str

    plan = QueryPlan()

    # START node
    start_node = Query(
        task_id=f"{ref_id}_START",
        query_str="Initialize agent workflow",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition={"method": "NOOP", "endpoint": "noop", "params": {"layout": None}},
    )
    plan.add_node(start_node)

    # QWEN_PLAN node - main agent execution
    qwen_node = Query(
        task_id=f"{ref_id}_QWEN_PLAN",
        query_str=query_str,
        dependencies=[f"{ref_id}_START"],
        node_type=QueryType.COMPUTE,
        definition={
            "method": "AGENT",
            "endpoint": "agent://qwen/plan",
            "agent_backend": kwargs.get("agent_backend", "qwen_agent"),
            "max_iterations": kwargs.get("max_iterations", 10),
            "tools": kwargs.get("tools", []),
            "params": {"layout": None},
        },
    )
    plan.add_node(qwen_node)

    # END node
    end_node = Query(
        task_id=f"{ref_id}_END",
        query_str="Finalize agent workflow",
        dependencies=[f"{ref_id}_QWEN_PLAN"],
        node_type=QueryType.MERGER,
        definition={"method": "NOOP", "endpoint": "noop", "params": {"layout": None}},
    )
    plan.add_node(end_node)

    logger.info(f"Created Qwen orchestrator plan with {len(plan.nodes)} nodes")
    return plan


@register_query_plan("agent_haystack_rag")
def query_planner_haystack_rag(
    planner_info: PlannerInfo,
    **kwargs: Any,
) -> QueryPlan:
    """Sub-plan for Haystack RAG execution.

    Creates a simple RAG DAG:
        ROOT -> RETRIEVAL -> READER -> END

    Args:
        planner_info: Planner information
        **kwargs: Additional arguments

    Returns:
        QueryPlan for RAG workflow
    """
    ref_id = planner_info.ref_id
    query_str = planner_info.query_str

    plan = QueryPlan()

    # ROOT node
    root_node = Query(
        task_id=f"{ref_id}_ROOT",
        query_str="Initialize RAG workflow",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition={"method": "NOOP", "endpoint": "noop", "params": {"layout": None}},
    )
    plan.add_node(root_node)

    # RETRIEVAL node
    retrieval_node = Query(
        task_id=f"{ref_id}_RETRIEVAL",
        query_str=query_str,
        dependencies=[f"{ref_id}_ROOT"],
        node_type=QueryType.COMPUTE,
        definition={
            "method": "RAG",
            "endpoint": "agent://haystack/retrieve",
            "retriever_type": kwargs.get("retriever_type", "bm25"),
            "top_k": kwargs.get("top_k", 5),
            "params": {"layout": None},
        },
    )
    plan.add_node(retrieval_node)

    # READER node
    reader_node = Query(
        task_id=f"{ref_id}_READER",
        query_str=query_str,
        dependencies=[f"{ref_id}_RETRIEVAL"],
        node_type=QueryType.COMPUTE,
        definition={
            "method": "RAG",
            "endpoint": "agent://haystack/read",
            "generator_model": kwargs.get("generator_model"),
            "params": {"layout": None},
        },
    )
    plan.add_node(reader_node)

    # END node
    end_node = Query(
        task_id=f"{ref_id}_END",
        query_str="Finalize RAG workflow",
        dependencies=[f"{ref_id}_READER"],
        node_type=QueryType.MERGER,
        definition={"method": "NOOP", "endpoint": "noop", "params": {"layout": None}},
    )
    plan.add_node(end_node)

    logger.info(f"Created Haystack RAG plan with {len(plan.nodes)} nodes")
    return plan


@register_query_plan("agent_autogen_team")
def query_planner_autogen_team(
    planner_info: PlannerInfo,
    **kwargs: Any,
) -> QueryPlan:
    """Sub-plan for AutoGen team execution.

    Creates a multi-agent team DAG:
        START -> [Agent1, Agent2, Agent3] (sequential or parallel) -> COORDINATOR -> END

    Args:
        planner_info: Planner information
        **kwargs: Additional arguments

    Returns:
        QueryPlan for team workflow
    """
    ref_id = planner_info.ref_id
    query_str = planner_info.query_str

    agents = kwargs.get("agents", ["researcher", "analyst", "writer"])
    parallel = kwargs.get("parallel", False)

    plan = QueryPlan()

    # START node
    start_node = Query(
        task_id=f"{ref_id}_START",
        query_str="Initialize team workflow",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition={"method": "NOOP", "endpoint": "noop", "params": {"layout": None}},
    )
    plan.add_node(start_node)

    # TEAM node - single node for team execution (AutoGen handles internal coordination)
    team_node = Query(
        task_id=f"{ref_id}_TEAM",
        query_str=query_str,
        dependencies=[f"{ref_id}_START"],
        node_type=QueryType.COMPUTE,
        definition={
            "method": "MULTI_AGENT",
            "endpoint": "agent://autogen/team",
            "team_type": kwargs.get("team_type", "research"),
            "agents": agents,
            "max_rounds": kwargs.get("max_rounds", 10),
            "speaker_selection": kwargs.get("speaker_selection", "round_robin"),
            "params": {"layout": None},
        },
    )
    plan.add_node(team_node)

    # END node
    end_node = Query(
        task_id=f"{ref_id}_END",
        query_str="Finalize team workflow",
        dependencies=[f"{ref_id}_TEAM"],
        node_type=QueryType.MERGER,
        definition={"method": "NOOP", "endpoint": "noop", "params": {"layout": None}},
    )
    plan.add_node(end_node)

    logger.info(f"Created AutoGen team plan with {len(plan.nodes)} nodes")
    return plan


@register_query_plan("agent_composite")
def query_planner_composite(
    planner_info: PlannerInfo,
    **kwargs: Any,
) -> QueryPlan:
    """Composite plan with Qwen orchestrator delegating to sub-workflows.

    Creates a DAG where Qwen can delegate to Haystack RAG or AutoGen teams:
        START -> QWEN_ORCHESTRATOR -> [RAG_BRANCH | TEAM_BRANCH] -> MERGE -> END

    Args:
        planner_info: Planner information
        **kwargs: Additional arguments

    Returns:
        QueryPlan for composite workflow
    """
    ref_id = planner_info.ref_id
    query_str = planner_info.query_str

    plan = QueryPlan()

    # START node
    start_node = Query(
        task_id=f"{ref_id}_START",
        query_str="Initialize composite workflow",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition={"method": "NOOP", "endpoint": "noop", "params": {"layout": None}},
    )
    plan.add_node(start_node)

    # ORCHESTRATOR node
    orchestrator_node = Query(
        task_id=f"{ref_id}_ORCHESTRATOR",
        query_str=query_str,
        dependencies=[f"{ref_id}_START"],
        node_type=QueryType.COMPUTE,
        definition={
            "method": "AGENT",
            "endpoint": "agent://qwen/orchestrate",
            "agent_backend": "qwen_agent",
            "max_iterations": kwargs.get("max_iterations", 15),
            "tools": ["run_haystack_rag", "run_autogen_team"],
            "params": {"layout": None},
        },
    )
    plan.add_node(orchestrator_node)

    # Note: Additional nodes can be dynamically added during execution
    # via DAGExpansionService when the orchestrator decides to delegate

    # END node
    end_node = Query(
        task_id=f"{ref_id}_END",
        query_str="Finalize composite workflow",
        dependencies=[f"{ref_id}_ORCHESTRATOR"],
        node_type=QueryType.MERGER,
        definition={"method": "NOOP", "endpoint": "noop", "params": {"layout": None}},
    )
    plan.add_node(end_node)

    logger.info(f"Created composite plan with {len(plan.nodes)} nodes")
    return plan
