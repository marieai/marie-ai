"""
Mock Query Plans Package

This package provides modular mock query plans for testing the Marie-AI scheduler.
Plans are organized by node type category for easier maintenance.

Modules:
    - base: Common imports and utilities
    - traditional: Linear and parallel execution patterns
    - branching: BRANCH and SWITCH conditional routing
    - guardrail: GUARDRAIL node validation plans
    - hitl: Human-in-the-Loop workflow plans

Usage:
    from tests.integration.scheduler.mock_plans import (
        # Traditional plans
        query_planner_mock_simple,
        query_planner_mock_medium,
        query_planner_mock_complex,
        # Branching plans
        query_planner_mock_branch_simple,
        query_planner_mock_switch_complexity,
        # Guardrail plans
        query_planner_mock_guardrail_simple,
        # HITL plans
        query_planner_mock_hitl_approval,
    )

    # Or import by category
    from tests.integration.scheduler.mock_plans.traditional import *
    from tests.integration.scheduler.mock_plans.branching import *
"""

# Import base utilities
from .base import (
    ExecutorEndpointQueryDefinition,
    LlmQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryPlanRegistry,
    QueryType,
    generate_job_id,
    increment_uuid7str,
    register_query_plan,
)

# Import branching plans (this registers them via @register_query_plan decorator)
from .branching import (
    query_planner_mock_branch_all_match,
    query_planner_mock_branch_jsonpath_advanced,
    query_planner_mock_branch_multi_condition,
    query_planner_mock_branch_python_function,
    query_planner_mock_branch_regex_matching,
    query_planner_mock_branch_simple,
    query_planner_mock_nested_branches,
    query_planner_mock_switch_complexity,
)

# Import guardrail plans (this registers them via @register_query_plan decorator)
from .guardrail import (
    query_planner_mock_guardrail_executor_metric,
    query_planner_mock_guardrail_multi_metric,
    query_planner_mock_guardrail_retry_loop,
    query_planner_mock_guardrail_simple,
)

# Import HITL plans (this registers them via @register_query_plan decorator)
from .hitl import (
    query_planner_mock_hitl_approval,
    query_planner_mock_hitl_complete_workflow,
    query_planner_mock_hitl_correction,
    query_planner_mock_hitl_router,
)

# Import traditional plans (this registers them via @register_query_plan decorator)
from .traditional import (
    query_planner_mock_complex,
    query_planner_mock_medium,
    query_planner_mock_parallel_subgraphs,
    query_planner_mock_simple,
    query_planner_mock_with_subgraphs,
)

__all__ = [
    # Base utilities
    "ExecutorEndpointQueryDefinition",
    "LlmQueryDefinition",
    "NoopQueryDefinition",
    "PlannerInfo",
    "Query",
    "QueryPlan",
    "QueryPlanRegistry",
    "QueryType",
    "generate_job_id",
    "increment_uuid7str",
    "register_query_plan",
    # Traditional plans
    "query_planner_mock_simple",
    "query_planner_mock_medium",
    "query_planner_mock_complex",
    "query_planner_mock_with_subgraphs",
    "query_planner_mock_parallel_subgraphs",
    # Branching plans
    "query_planner_mock_branch_simple",
    "query_planner_mock_switch_complexity",
    "query_planner_mock_branch_multi_condition",
    "query_planner_mock_nested_branches",
    "query_planner_mock_branch_python_function",
    "query_planner_mock_branch_jsonpath_advanced",
    "query_planner_mock_branch_all_match",
    "query_planner_mock_branch_regex_matching",
    # Guardrail plans
    "query_planner_mock_guardrail_simple",
    "query_planner_mock_guardrail_retry_loop",
    "query_planner_mock_guardrail_executor_metric",
    "query_planner_mock_guardrail_multi_metric",
    # HITL plans
    "query_planner_mock_hitl_approval",
    "query_planner_mock_hitl_correction",
    "query_planner_mock_hitl_router",
    "query_planner_mock_hitl_complete_workflow",
]
