"""
Guardrail/Validation node models for query plans.
Supports inline quality evaluation during workflow execution with configurable metrics.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from marie.query_planner.base import QueryDefinition, QueryTypeRegistry


class GuardrailMetricType(str, Enum):
    """Types of guardrail metrics for quality evaluation"""

    FAITHFULNESS = "faithfulness"  # RAG faithfulness score
    RELEVANCE = "relevance"  # Query-response relevance
    JSON_SCHEMA = "json_schema"  # JSON schema validation
    REGEX_MATCH = "regex_match"  # Pattern matching
    LENGTH_CHECK = "length_check"  # Min/max length validation
    CONTAINS_KEYWORDS = "contains_keywords"  # Keyword presence check
    LLM_JUDGE = "llm_judge"  # LLM-as-judge evaluation
    EXECUTOR = "executor"  # Custom Python function via executor


class GuardrailMetric(BaseModel):
    """
    Single metric configuration for guardrail evaluation.

    Examples:
        # JSON Schema validation
        GuardrailMetric(
            type=GuardrailMetricType.JSON_SCHEMA,
            name="Output Schema",
            threshold=1.0,
            params={"schema": {"type": "object", "required": ["name", "value"]}}
        )

        # Regex pattern matching
        GuardrailMetric(
            type=GuardrailMetricType.REGEX_MATCH,
            name="No PII",
            threshold=1.0,
            params={"pattern": "^(?!.*\\d{3}-\\d{2}-\\d{4}).*$"}
        )

        # Length check
        GuardrailMetric(
            type=GuardrailMetricType.LENGTH_CHECK,
            name="Response Length",
            threshold=0.5,
            params={"min": 10, "max": 1000}
        )

        # Custom executor evaluation
        GuardrailMetric(
            type=GuardrailMetricType.EXECUTOR,
            name="PII Check",
            threshold=1.0,
            params={
                "endpoint": "guardrail_executor://evaluate",
                "function": "check_no_pii",
                "config": {}
            }
        )
    """

    type: GuardrailMetricType = Field(..., description="Type of metric evaluation")

    name: str = Field(..., description="Human-readable name for this metric")

    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight for aggregate score calculation",
    )

    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Pass threshold (0-1). Metric passes if score >= threshold",
    )

    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific parameters for the metric evaluation",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validate metric name is not empty"""
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        return v.strip()


class GuardrailResult(BaseModel):
    """Result from a single metric evaluation"""

    metric_name: str = Field(..., description="Name of the evaluated metric")

    passed: bool = Field(
        ..., description="Whether the metric passed (score >= threshold)"
    )

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Numeric score between 0.0 and 1.0"
    )

    feedback: Optional[str] = Field(
        None, description="Explanation or reasoning for the score"
    )

    execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Time taken to evaluate this metric in ms"
    )


class GuardrailEvaluationResult(BaseModel):
    """Aggregate result from all guardrail metrics"""

    overall_passed: bool = Field(
        ..., description="Whether the guardrail check passed overall"
    )

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate score based on aggregation mode",
    )

    individual_results: List[GuardrailResult] = Field(
        default_factory=list, description="Results from each individual metric"
    )

    selected_path_id: str = Field(..., description="Selected path ID: 'pass' or 'fail'")

    active_target_nodes: List[str] = Field(
        default_factory=list,
        description="Node IDs to activate (remain in CREATED state)",
    )

    skipped_target_nodes: List[str] = Field(
        default_factory=list, description="Node IDs to mark as SKIPPED"
    )

    total_execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total evaluation time in ms"
    )

    error: Optional[str] = Field(None, description="Error message if evaluation failed")


class GuardrailPath(BaseModel):
    """
    Defines pass or fail path with target nodes.
    Follows the same pattern as BranchPath for consistency.
    """

    path_id: str = Field(
        ...,
        pattern="^(pass|fail)$",
        description="Path identifier: 'pass' or 'fail'",
    )

    target_node_ids: List[str] = Field(
        default_factory=list,
        description="Node IDs to activate if this path is selected",
    )

    description: Optional[str] = Field(
        None, description="Human-readable description of this path"
    )


class GuardrailAggregationMode(str, Enum):
    """How to aggregate multiple metric results"""

    ALL = "all"  # Pass only if ALL metrics pass their thresholds
    ANY = "any"  # Pass if ANY metric passes its threshold
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted score vs pass_threshold


@QueryTypeRegistry.register("GUARDRAIL")
class GuardrailQueryDefinition(QueryDefinition):
    """
    GUARDRAIL node - inline quality validation during workflow execution.

    Execution flow:
    1. GUARDRAIL node executes (may gather data from upstream)
    2. After SUCCEEDED status, scheduler calls GuardrailEvaluator
    3. Evaluator runs each configured metric against input data
    4. Results are aggregated based on aggregation_mode
    5. Based on pass/fail:
       - Selected path target nodes remain CREATED (ready to execute)
       - Non-selected path target nodes marked SKIPPED
    6. Cascade skip to descendants of skipped nodes

    Example:
        GuardrailQueryDefinition(
            input_source="$.nodes.llm_node.output",
            metrics=[
                GuardrailMetric(
                    type=GuardrailMetricType.LENGTH_CHECK,
                    name="Response Length",
                    threshold=0.5,
                    params={"min": 10, "max": 1000}
                ),
                GuardrailMetric(
                    type=GuardrailMetricType.REGEX_MATCH,
                    name="No SSN",
                    threshold=1.0,
                    params={"pattern": "^(?!.*\\d{3}-\\d{2}-\\d{4}).*$"}
                )
            ],
            aggregation_mode=GuardrailAggregationMode.ALL,
            pass_threshold=0.8,
            paths=[
                GuardrailPath(
                    path_id="pass",
                    target_node_ids=["success_node"],
                    description="Validation passed"
                ),
                GuardrailPath(
                    path_id="fail",
                    target_node_ids=["error_node"],
                    description="Validation failed"
                )
            ]
        )
    """

    method: str = "GUARDRAIL"
    endpoint: str = "guardrail://control"

    # What to evaluate
    input_source: str = Field(
        default="",
        description="JSONPath expression to the data to evaluate (e.g., '$.nodes.llm_node.output')",
    )

    context_source: Optional[str] = Field(
        None,
        description="JSONPath expression to context data (for RAG metrics like faithfulness)",
    )

    query_source: Optional[str] = Field(
        None,
        description="JSONPath expression to original query (for relevance metrics)",
    )

    # Metrics to run
    metrics: List[GuardrailMetric] = Field(
        default_factory=list,
        description="List of metrics to evaluate",
    )

    # Aggregation
    aggregation_mode: GuardrailAggregationMode = Field(
        default=GuardrailAggregationMode.ALL,
        description="How to aggregate multiple metric results",
    )

    pass_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for weighted_average aggregation mode",
    )

    # Paths (matches BRANCH pattern)
    paths: List[GuardrailPath] = Field(
        default_factory=lambda: [
            GuardrailPath(
                path_id="pass", target_node_ids=[], description="Validation passed"
            ),
            GuardrailPath(
                path_id="fail", target_node_ids=[], description="Validation failed"
            ),
        ],
        description="Pass and fail paths with target nodes",
    )

    # Options
    fail_fast: bool = Field(
        default=False,
        description="Stop evaluation on first metric failure",
    )

    include_feedback: bool = Field(
        default=True,
        description="Include detailed feedback in results",
    )

    evaluation_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum seconds for evaluation (1-300)",
    )

    params: dict = Field(default_factory=dict)

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, v):
        """Validate that exactly pass and fail paths are defined"""
        if len(v) != 2:
            raise ValueError("GUARDRAIL must have exactly 2 paths (pass and fail)")

        path_ids = {p.path_id for p in v}
        if path_ids != {"pass", "fail"}:
            raise ValueError("GUARDRAIL paths must be 'pass' and 'fail'")

        return v

    def get_pass_path(self) -> Optional[GuardrailPath]:
        """Get the pass path configuration"""
        for path in self.paths:
            if path.path_id == "pass":
                return path
        return None

    def get_fail_path(self) -> Optional[GuardrailPath]:
        """Get the fail path configuration"""
        for path in self.paths:
            if path.path_id == "fail":
                return path
        return None
