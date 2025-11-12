"""
Branching and conditional execution models for query plans.
Supports JSONPath-based conditions and Python function evaluation.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from marie.query_planner.base import QueryDefinition, QueryTypeRegistry
from marie.query_planner.jsonpath_evaluator import ComparisonOperator


class BranchEvaluationMode(str, Enum):
    """How to evaluate branch conditions"""

    FIRST_MATCH = "first_match"  # Take first matching path, skip others
    ALL_MATCH = "all_match"  # Take all matching paths, skip non-matching
    PRIORITY_MATCH = "priority_match"  # Highest priority match, skip others


class BranchCondition(BaseModel):
    """
    Condition for branch path selection using JSONPath.

    Examples:
        # Simple equality
        BranchCondition(
            jsonpath="$.metadata.document_type",
            operator="==",
            value="invoice"
        )

        # Numeric comparison
        BranchCondition(
            jsonpath="$.results[0].confidence",
            operator=">=",
            value=0.85
        )

        # Array filter
        BranchCondition(
            jsonpath="$.errors[?(@.severity=='critical')]",
            operator="exists"
        )

        # Regex match
        BranchCondition(
            jsonpath="$.metadata.filename",
            operator="matches",
            value=r"^invoice_\d{4}\.pdf$"
        )
    """

    jsonpath: str = Field(
        ...,
        description="JSONPath expression to evaluate. Examples: '$.metadata.type', '$.data[*].score'",
    )

    operator: str = Field(
        default="==",
        description="Comparison operator: ==, !=, >, <, >=, <=, in, not_in, contains, matches, exists, is_empty",
    )

    value: Optional[Any] = Field(
        None,
        description="Value to compare against (not needed for exists/is_empty operators)",
    )

    description: Optional[str] = Field(
        None, description="Human-readable description of this condition"
    )

    @field_validator('jsonpath')
    @classmethod
    def validate_jsonpath(cls, v):
        """Validate JSONPath syntax at model creation"""
        if not v.strip():
            raise ValueError("JSONPath expression cannot be empty")

        # Basic validation - must start with $ or @
        if not (v.startswith('$') or v.startswith('@')):
            raise ValueError(
                f"JSONPath expression must start with $ (root) or @ (current): {v}"
            )

        return v

    @field_validator('operator')
    @classmethod
    def validate_operator(cls, v):
        """Validate operator is supported"""
        valid_operators = [op.value for op in ComparisonOperator]
        if v not in valid_operators:
            raise ValueError(
                f"Invalid operator '{v}'. Must be one of: {', '.join(valid_operators)}"
            )
        return v


class BranchConditionGroup(BaseModel):
    """
    Group of conditions with AND/OR logic.

    Example:
        BranchConditionGroup(
            conditions=[
                BranchCondition(jsonpath="$.type", operator="==", value="invoice"),
                BranchCondition(jsonpath="$.confidence", operator=">=", value=0.8)
            ],
            combinator="AND"
        )
    """

    conditions: List[Union[BranchCondition, 'BranchConditionGroup']] = Field(
        ..., min_length=1, description="List of conditions to evaluate"
    )

    combinator: str = Field(
        default="AND", pattern="^(AND|OR)$", description="Logical operator: AND or OR"
    )


class BranchPath(BaseModel):
    """
    Defines one possible execution path with JSONPath-based conditions.
    All paths are created in the QueryPlan at build time.
    At runtime, non-selected paths are marked SKIPPED.
    """

    path_id: str = Field(..., description="Unique identifier for this path")

    # JSONPath-based condition (RECOMMENDED)
    condition: Optional[Union[BranchCondition, BranchConditionGroup]] = Field(
        None, description="JSONPath condition or condition group"
    )

    # Python function fallback (for complex logic)
    condition_function: Optional[str] = Field(
        None,
        description="Python function path as fallback. Signature: fn(context: Dict) -> bool",
    )

    target_node_ids: List[str] = Field(
        ...,
        description="Node IDs to activate if condition matches (nodes must exist in QueryPlan)",
    )

    priority: int = Field(default=0, description="Higher priority evaluated first")

    description: Optional[str] = Field(None, description="Human-readable description")


@QueryTypeRegistry.register("BRANCH")
class BranchQueryDefinition(QueryDefinition):
    """
    BRANCH node - controls execution flow by marking paths as ACTIVE or SKIPPED.

    Execution flow:
    1. Query planner creates ALL paths and their target nodes in QueryPlan
    2. Target nodes have BRANCH node as dependency
    3. When BRANCH node executes:
       - Evaluate conditions for each path
       - Mark selected path(s) target nodes as READY
       - Mark non-selected path(s) target nodes as SKIPPED with reason
    4. Scheduler skips SKIPPED nodes (doesn't dispatch them)
    5. MERGER nodes see which branches were SKIPPED vs COMPLETED
    """

    method: str = "BRANCH"
    endpoint: str = "branch"  # Special endpoint handled by scheduler

    paths: List[BranchPath] = Field(
        ...,
        min_length=1,
        description="All possible execution paths. ALL are created in QueryPlan.",
    )

    default_path_id: Optional[str] = Field(
        None, description="Path to activate if no conditions match (fallback)"
    )

    evaluation_mode: BranchEvaluationMode = Field(
        default=BranchEvaluationMode.FIRST_MATCH,
        description="How to select among matching paths",
    )

    params: dict = Field(default_factory=dict)


@QueryTypeRegistry.register("PYTHON_BRANCH")
class PythonBranchQueryDefinition(BranchQueryDefinition):
    """
    Pure Python function-based branching.
    Function receives context and returns path_id(s) to activate.

    Example:
        def my_branch_function(context: dict) -> str:
            if context['metadata']['confidence'] > 0.9:
                return 'high_confidence_path'
            return 'low_confidence_path'
    """

    branch_function: str = Field(
        ...,
        description="Python function path. Signature: fn(context: Dict) -> Union[str, List[str]]",
    )

    function_timeout: int = Field(
        default=30, description="Max seconds for function execution"
    )


@QueryTypeRegistry.register("SWITCH")
class SwitchQueryDefinition(QueryDefinition):
    """
    Simple value-based routing - syntactic sugar for common pattern.
    All cases are created in QueryPlan, non-matching cases are SKIPPED.

    Example:
        SwitchQueryDefinition(
            switch_field="$.metadata.document_type",
            cases={
                "invoice": ["invoice_node_1", "invoice_node_2"],
                "contract": ["contract_node_1"],
                "receipt": ["receipt_node_1"]
            },
            default_case=["generic_node_1"]
        )
    """

    method: str = "SWITCH"
    endpoint: str = "switch"

    switch_field: str = Field(
        ..., description="JSONPath expression to field to evaluate"
    )

    cases: Dict[Any, List[str]] = Field(
        ...,
        description="Map of value -> target_node_ids (node IDs must exist in QueryPlan)",
    )

    default_case: Optional[List[str]] = Field(
        None, description="Default target_node_ids if no case matches"
    )

    params: dict = Field(default_factory=dict)


class MergerStrategy(str, Enum):
    """How MERGER combines results from branches"""

    WAIT_ALL_ACTIVE = "wait_all_active"  # Wait for all non-skipped branches (default)
    WAIT_ANY = "wait_any"  # Proceed when any branch completes
    WAIT_N = "wait_n"  # Wait for N branches
    WAIT_ALL_INCLUDING_SKIPPED = (
        "wait_all_including_skipped"  # Wait for all (even skipped)
    )


@QueryTypeRegistry.register("MERGER_ENHANCED")
class EnhancedMergerQueryDefinition(QueryDefinition):
    """
    Enhanced MERGER aware of branch skipping.
    By default, only waits for ACTIVE (non-skipped) branches.

    Example:
        EnhancedMergerQueryDefinition(
            method="NOOP",
            endpoint="noop",
            merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
            params={"layout": None}
        )
    """

    method: str = "NOOP"
    endpoint: str = "noop"

    merge_strategy: MergerStrategy = Field(
        default=MergerStrategy.WAIT_ALL_ACTIVE,
        description="Strategy for waiting on branch dependencies",
    )

    min_required: Optional[int] = Field(
        None, description="For WAIT_N strategy - minimum dependencies required"
    )

    timeout_seconds: Optional[int] = Field(
        None, description="Max wait time for dependencies"
    )

    # Optional merge function
    merge_function: Optional[str] = Field(
        None,
        description="Python function to merge results. Signature: fn(results: List[Dict]) -> Dict",
    )

    params: dict = Field(default_factory=lambda: {"layout": None})


class SkipReason(BaseModel):
    """
    Records why a node was skipped.
    Stored in WorkInfo.data['skip_metadata'] for debugging.
    """

    branch_node_id: str = Field(..., description="ID of BRANCH node that caused skip")

    reason: str = Field(..., description="Human-readable reason")

    evaluated_condition: Optional[Dict[str, Any]] = Field(
        None, description="The condition that was evaluated (for debugging)"
    )

    selected_paths: List[str] = Field(
        default_factory=list,
        description="Path IDs that were selected (this node's path was not selected)",
    )

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
