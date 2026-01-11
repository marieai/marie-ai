"""
Base module for mock query plans.

Provides common imports, helper functions, and type definitions used across
all mock query plan modules.
"""

from marie.job.job_manager import generate_job_id, increment_uuid7str
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    LlmQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
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
    SwitchQueryDefinition,
)

# Re-export everything for convenience
__all__ = [
    # Job utilities
    "generate_job_id",
    "increment_uuid7str",
    # Base query planner types
    "ExecutorEndpointQueryDefinition",
    "LlmQueryDefinition",
    "NoopQueryDefinition",
    "PlannerInfo",
    "Query",
    "QueryPlan",
    "QueryPlanRegistry",
    "QueryType",
    "register_query_plan",
    # Branching types
    "BranchCondition",
    "BranchConditionGroup",
    "BranchEvaluationMode",
    "BranchPath",
    "BranchQueryDefinition",
    "EnhancedMergerQueryDefinition",
    "MergerStrategy",
    "SwitchQueryDefinition",
]
