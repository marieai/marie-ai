"""
Mock Query Plans for Job Distribution Performance Testing

This module provides mock query plans that mimic the structure of production query plans
from testing job distribution performance, scheduling behavior, and
resource allocation in the Marie-AI scheduler.

Includes both traditional linear/parallel plans and advanced branching/conditional plans.

NOTE: This file is now a thin wrapper that imports from the modular mock_plans package.
The actual plan definitions are organized by category in mock_plans/:
    - mock_plans/traditional.py: Linear and parallel execution patterns
    - mock_plans/branching.py: BRANCH and SWITCH conditional routing
    - mock_plans/guardrail.py: GUARDRAIL node validation plans
    - mock_plans/hitl.py: Human-in-the-Loop workflow plans

Traditional Plans:
    - query_planner_mock_simple: Basic linear execution (3 nodes)
    - query_planner_mock_medium: Parallel execution with merge (7 nodes)
    - query_planner_mock_complex: Complex multi-stage pipeline (12 nodes)
    - query_planner_mock_with_subgraphs: Nested subgraph pattern
    - query_planner_mock_parallel_subgraphs: Parallel subgraphs (20+ nodes)

Branching/Conditional Plans:
    - query_planner_mock_branch_simple: Simple BRANCH with JSONPath conditions
    - query_planner_mock_switch_complexity: SWITCH-based value routing
    - query_planner_mock_branch_multi_condition: Complex AND/OR condition groups
    - query_planner_mock_nested_branches: Nested branching (branch within branch)
    - query_planner_mock_branch_python_function: Python function evaluation
    - query_planner_mock_branch_jsonpath_advanced: Advanced JSONPath (arrays, nested fields)
    - query_planner_mock_branch_all_match: ALL_MATCH evaluation mode (multiple paths)
    - query_planner_mock_branch_regex_matching: Regex pattern matching

Guardrail/Quality Control Plans:
    - query_planner_mock_guardrail_simple: Basic guardrail with length/regex metrics
    - query_planner_mock_guardrail_retry_loop: Guardrail with retry on failure
    - query_planner_mock_guardrail_executor_metric: Custom Python evaluation via executor
    - query_planner_mock_guardrail_multi_metric: Multiple metrics with weighted aggregation

HITL (Human-in-the-Loop) Plans:
    - query_planner_mock_hitl_approval: HITL approval workflow
    - query_planner_mock_hitl_correction: HITL correction workflow
    - query_planner_mock_hitl_router: HITL routing based on approval decision
    - query_planner_mock_hitl_complete_workflow: Complete HITL workflow with all patterns

Usage:
    from tests.integration.scheduler.mock_query_plans import (
        query_planner_mock_simple,
        query_planner_mock_medium,
        query_planner_mock_complex,
        query_planner_mock_with_subgraphs,
        query_planner_mock_parallel_subgraphs,
        query_planner_mock_branch_simple,
        query_planner_mock_switch_complexity,
        query_planner_mock_branch_multi_condition,
        query_planner_mock_nested_branches,
    )

    planner_info = PlannerInfo(name="mock_simple", base_id=generate_job_id())
    plan = query_planner_mock_simple(planner_info)
"""

# Re-export base utilities for backward compatibility
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

# Branching plans (BRANCH and SWITCH conditional routing)
from tests.integration.scheduler.mock_plans.branching import (
    query_planner_mock_branch_all_match,
    query_planner_mock_branch_jsonpath_advanced,
    query_planner_mock_branch_multi_condition,
    query_planner_mock_branch_python_function,
    query_planner_mock_branch_regex_matching,
    query_planner_mock_branch_simple,
    query_planner_mock_nested_branches,
    query_planner_mock_switch_complexity,
)

# Guardrail plans (quality validation nodes)
from tests.integration.scheduler.mock_plans.guardrail import (
    query_planner_mock_guardrail_executor_metric,
    query_planner_mock_guardrail_multi_metric,
    query_planner_mock_guardrail_retry_loop,
    query_planner_mock_guardrail_simple,
)

# HITL plans (Human-in-the-Loop workflows)
from tests.integration.scheduler.mock_plans.hitl import (
    query_planner_mock_hitl_approval,
    query_planner_mock_hitl_complete_workflow,
    query_planner_mock_hitl_correction,
    query_planner_mock_hitl_router,
)

# Traditional plans (linear and parallel execution patterns)
from tests.integration.scheduler.mock_plans.traditional import (
    query_planner_mock_complex,
    query_planner_mock_medium,
    query_planner_mock_parallel_subgraphs,
    query_planner_mock_simple,
    query_planner_mock_with_subgraphs,
)

# Import all plans from modular structure
# This registers them via @register_query_plan decorator and re-exports them





__all__ = [
    # Base utilities
    "generate_job_id",
    "increment_uuid7str",
    "ExecutorEndpointQueryDefinition",
    "LlmQueryDefinition",
    "NoopQueryDefinition",
    "PlannerInfo",
    "Query",
    "QueryPlan",
    "QueryPlanRegistry",
    "QueryType",
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


if __name__ == "__main__":
    """
    Persist each mock query plan as JSON and validate that it can be restored.

    Writes JSON files to `tmp/query_plans` and for each plan:
      - generates the plan
      - dumps it to disk using the plan's model dump
      - reloads the JSON and re-validates into a QueryPlan
      - asserts node count and task id set equality
    """
    import json
    import traceback
    from pathlib import Path

    output_dir = Path("tmp/query_plans")
    output_dir.mkdir(parents=True, exist_ok=True)

    plans_to_test = [
        ("mock_simple", "Simple Mock Plan"),
        ("mock_medium", "Medium Mock Plan"),
        ("mock_complex", "Complex Mock Plan"),
        ("mock_with_subgraphs", "Plan With Subgraphs"),
        ("mock_parallel_subgraphs", "Parallel Subgraphs Plan"),
        ("mock_branch_simple", "Branch Simple Plan"),
        ("mock_switch_complexity", "Switch Complexity Plan"),
        ("mock_branch_multi_condition", "Branch Multi Condition Plan"),
        ("mock_nested_branches", "Nested Branches Plan"),
        ("mock_branch_python_function", "Branch Python Function Plan"),
        ("mock_branch_jsonpath_advanced", "Branch JSONPath Advanced Plan"),
        ("mock_branch_all_match", "Branch All Match Plan"),
        ("mock_branch_regex_matching", "Branch Regex Matching Plan"),
        ("mock_hitl_approval", "HITL Approval Plan"),
        ("mock_hitl_correction", "HITL Correction Plan"),
        ("mock_hitl_router", "HITL Router Plan"),
        ("mock_hitl_complete_workflow", "HITL Complete Workflow Plan"),
        # Guardrail Plans
        ("mock_guardrail_simple", "Guardrail Simple Plan"),
        ("mock_guardrail_retry_loop", "Guardrail Retry Loop Plan"),
        ("mock_guardrail_executor_metric", "Guardrail Executor Metric Plan"),
        ("mock_guardrail_multi_metric", "Guardrail Multi-Metric Plan"),
    ]

    for plan_name, description in plans_to_test:
        print(f"\n{'=' * 80}")
        print(f"{description.upper()} - {plan_name}")
        print(f"{'=' * 80}\n")
        try:
            # Create planner info and generate plan
            planner_info = PlannerInfo(name=plan_name, base_id=generate_job_id())
            planner_func = QueryPlanRegistry.get(plan_name)
            if planner_func is None:
                print(f"Skipping {plan_name}: no registered planner found.")
                continue

            plan = planner_func(planner_info)

            # Dump to JSON
            plan_dict = plan.model_dump()  # safe dict representation
            out_path = output_dir / f"{plan_name}.json"
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(plan_dict, fh, ensure_ascii=False, indent=2)

            print(f"Persisted plan to: {out_path} (nodes: {len(plan.nodes)})")

            # Load back and validate
            with out_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)

            restored_plan = QueryPlan.model_validate(loaded)

            # Basic validations
            assert len(plan.nodes) == len(restored_plan.nodes), "Node count mismatch"
            original_ids = {n.task_id for n in plan.nodes}
            restored_ids = {n.task_id for n in restored_plan.nodes}
            assert original_ids == restored_ids, "Task ID sets differ after restore"

            print(f"Restore validation succeeded for {plan_name}: node_count={len(plan.nodes)}")
        except Exception:
            print(f"ERROR processing plan {plan_name}:")
            traceback.print_exc()
