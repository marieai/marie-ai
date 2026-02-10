"""
Query planner package for Marie-AI.
Provides tools for creating and managing query execution plans with branching support.
"""

from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    LlmQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    PlannerMetadata,
    PythonFunctionQueryDefinition,
    Query,
    QueryDefinition,
    QueryPlan,
    QueryPlanRegistry,
    QueryType,
    register_query_plan,
)
from marie.query_planner.branching import (
    BranchCondition,
    BranchConditionGroup,
    BranchEvaluationMode,
    BranchPath,
    BranchQueryDefinition,
    EnhancedMergerQueryDefinition,
    MergerStrategy,
    PythonBranchQueryDefinition,
    SkipReason,
    SwitchQueryDefinition,
)
from marie.query_planner.hitl import (
    HitlApprovalQueryDefinition,
    HitlCorrectionQueryDefinition,
    HitlRouterQueryDefinition,
)
from marie.query_planner.jsonpath_evaluator import (
    ComparisonOperator,
    JSONPathCondition,
    JSONPathConditionGroup,
    JSONPathEvaluator,
)
from marie.query_planner.planner import (
    ConditionalQueryPlanBuilder,
    compute_job_levels,
    plan_to_json,
    plan_to_yaml,
    query_planner,
    topological_sort,
)
from marie.query_planner.rag import (
    ContextCacheQueryDefinition,
    RAGDeleteQueryDefinition,
    RAGIngestQueryDefinition,
    RAGSearchQueryDefinition,
)

__all__ = [
    # Base classes
    "Query",
    "QueryPlan",
    "QueryType",
    "QueryDefinition",
    "PlannerInfo",
    "PlannerMetadata",
    "QueryPlanRegistry",
    "register_query_plan",
    # Query definitions
    "NoopQueryDefinition",
    "LlmQueryDefinition",
    "PythonFunctionQueryDefinition",
    "ExecutorEndpointQueryDefinition",
    # Planner functions
    "query_planner",
    "topological_sort",
    "compute_job_levels",
    "plan_to_yaml",
    "plan_to_json",
    # Builder
    "ConditionalQueryPlanBuilder",
    # Branching
    "BranchCondition",
    "BranchConditionGroup",
    "BranchPath",
    "BranchQueryDefinition",
    "PythonBranchQueryDefinition",
    "SwitchQueryDefinition",
    "EnhancedMergerQueryDefinition",
    "MergerStrategy",
    "BranchEvaluationMode",
    "SkipReason",
    # JSONPath
    "JSONPathEvaluator",
    "JSONPathCondition",
    "JSONPathConditionGroup",
    "ComparisonOperator",
    # HITL
    "HitlApprovalQueryDefinition",
    "HitlCorrectionQueryDefinition",
    "HitlRouterQueryDefinition",
    # RAG (Agentic Retrieval)
    "RAGSearchQueryDefinition",
    "RAGIngestQueryDefinition",
    "RAGDeleteQueryDefinition",
    "ContextCacheQueryDefinition",
]
