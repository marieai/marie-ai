"""
Mock Query Plans for GUARDRAIL Node Testing

This module provides mock query plans for testing GUARDRAIL node functionality,
including quality validation, metric evaluation, and pass/fail path routing.

Plans:
    - mock_guardrail_simple: Basic guardrail with length and regex metrics
    - mock_guardrail_retry_loop: Guardrail with retry on failure
    - mock_guardrail_executor_metric: Custom Python evaluation via executor
"""

from marie.query_planner.guardrail import (
    GuardrailMetric,
    GuardrailMetricType,
    GuardrailPath,
    GuardrailQueryDefinition,
)

from .base import (
    ExecutorEndpointQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryType,
    increment_uuid7str,
    register_query_plan,
)


@register_query_plan("mock_guardrail_simple")
def query_planner_mock_guardrail_simple(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Simple GUARDRAIL test: validate output before returning.

    Structure:
        START -> PROCESS -> GUARDRAIL -> [pass: SUCCESS, fail: ERROR]

    Tests:
        - Basic guardrail evaluation with length and regex metrics
        - Pass/fail path routing
        - Skip cascade to non-selected path

    Args:
        planner_info: Contains configuration and state information for the plan
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A guardrail validation plan
    """
    base_id = planner_info.base_id
    nodes = []

    # START node
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # PROCESS node (simulates LLM or data processing)
    process = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Process data",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://document/process",
            params={"generate_output": True},
        ),
    )
    planner_info.current_id += 1
    nodes.append(process)

    # Generate IDs for downstream nodes (needed for guardrail path config)
    guardrail_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    success_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    error_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1

    # GUARDRAIL node with metrics
    guardrail = Query(
        task_id=guardrail_id,
        query_str=f"{planner_info.current_id - 2}: Validate response quality",
        dependencies=[process.task_id],
        node_type=QueryType.GUARDRAIL,
        definition=GuardrailQueryDefinition(
            input_source=f"$.nodes.{process.task_id}.output",
            metrics=[
                GuardrailMetric(
                    type=GuardrailMetricType.LENGTH_CHECK,
                    name="Response Length",
                    threshold=0.5,
                    weight=1.0,
                    params={"min": 10, "max": 1000},
                ),
                GuardrailMetric(
                    type=GuardrailMetricType.REGEX_MATCH,
                    name="No PII (SSN)",
                    threshold=1.0,
                    weight=1.0,
                    params={"pattern": r"^(?!.*\d{3}-\d{2}-\d{4}).*$"},
                ),
            ],
            aggregation_mode="all",
            pass_threshold=0.8,
            paths=[
                GuardrailPath(
                    path_id="pass",
                    target_node_ids=[success_id],
                    description="Response passed validation",
                ),
                GuardrailPath(
                    path_id="fail",
                    target_node_ids=[error_id],
                    description="Response failed validation",
                ),
            ],
            fail_fast=False,
            include_feedback=True,
            evaluation_timeout=30,
        ),
    )
    nodes.append(guardrail)

    # SUCCESS terminal node (pass path)
    success = Query(
        task_id=success_id,
        query_str=f"{planner_info.current_id - 1}: SUCCESS - Return response",
        dependencies=[guardrail_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(success)

    # ERROR terminal node (fail path)
    error = Query(
        task_id=error_id,
        query_str=f"{planner_info.current_id}: ERROR - Validation failed",
        dependencies=[guardrail_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(error)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_guardrail_retry_loop")
def query_planner_mock_guardrail_retry_loop(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    GUARDRAIL with retry: if validation fails, retry with different parameters.

    Structure:
        START -> LLM_1 -> GUARDRAIL_1 -> [pass: END, fail: LLM_2 -> GUARDRAIL_2 -> END]

    Tests:
        - Retry pattern on guardrail failure
        - Multiple guardrail nodes in sequence
        - Graceful degradation

    Args:
        planner_info: Contains configuration and state information for the plan
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A guardrail retry loop plan
    """
    base_id = planner_info.base_id
    nodes = []

    # START node
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # First LLM attempt
    llm_1 = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: LLM Generation (Attempt 1)",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_llm://generate",
            params={"temperature": 0.7, "attempt": 1},
        ),
    )
    planner_info.current_id += 1
    nodes.append(llm_1)

    # Pre-generate all IDs for path configuration
    guardrail_1_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    end_success_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    llm_2_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    guardrail_2_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    end_retry_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1

    # First GUARDRAIL
    guardrail_1 = Query(
        task_id=guardrail_1_id,
        query_str="Guardrail: Check quality (Attempt 1)",
        dependencies=[llm_1.task_id],
        node_type=QueryType.GUARDRAIL,
        definition=GuardrailQueryDefinition(
            input_source=f"$.nodes.{llm_1.task_id}.output",
            metrics=[
                GuardrailMetric(
                    type=GuardrailMetricType.LENGTH_CHECK,
                    name="Minimum Length",
                    threshold=1.0,
                    weight=1.0,
                    params={"min": 50, "max": 2000},
                ),
                GuardrailMetric(
                    type=GuardrailMetricType.CONTAINS_KEYWORDS,
                    name="Required Keywords",
                    threshold=0.5,
                    weight=1.0,
                    params={"keywords": ["result", "analysis", "conclusion"]},
                ),
            ],
            aggregation_mode="all",
            pass_threshold=0.8,
            paths=[
                GuardrailPath(
                    path_id="pass",
                    target_node_ids=[end_success_id],
                    description="Quality check passed",
                ),
                GuardrailPath(
                    path_id="fail",
                    target_node_ids=[llm_2_id],
                    description="Quality check failed - retry",
                ),
            ],
            evaluation_timeout=30,
        ),
    )
    nodes.append(guardrail_1)

    # END SUCCESS (first attempt passed)
    end_success = Query(
        task_id=end_success_id,
        query_str="END: Success (Attempt 1)",
        dependencies=[guardrail_1_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end_success)

    # Second LLM attempt (retry with different params)
    llm_2 = Query(
        task_id=llm_2_id,
        query_str="LLM Generation (Attempt 2 - Retry)",
        dependencies=[guardrail_1_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_llm://generate",
            params={"temperature": 0.3, "attempt": 2, "more_detailed": True},
        ),
    )
    nodes.append(llm_2)

    # Second GUARDRAIL (more lenient)
    guardrail_2 = Query(
        task_id=guardrail_2_id,
        query_str="Guardrail: Check quality (Attempt 2)",
        dependencies=[llm_2_id],
        node_type=QueryType.GUARDRAIL,
        definition=GuardrailQueryDefinition(
            input_source=f"$.nodes.{llm_2_id}.output",
            metrics=[
                GuardrailMetric(
                    type=GuardrailMetricType.LENGTH_CHECK,
                    name="Minimum Length",
                    threshold=1.0,
                    weight=1.0,
                    params={"min": 20, "max": 5000},  # More lenient
                ),
            ],
            aggregation_mode="all",
            pass_threshold=0.5,  # Lower threshold for retry
            paths=[
                GuardrailPath(
                    path_id="pass",
                    target_node_ids=[end_retry_id],
                    description="Retry succeeded",
                ),
                GuardrailPath(
                    path_id="fail",
                    target_node_ids=[end_retry_id],  # Same end node - accept best effort
                    description="Retry failed - return best effort",
                ),
            ],
            evaluation_timeout=30,
        ),
    )
    nodes.append(guardrail_2)

    # END RETRY (second attempt result)
    end_retry = Query(
        task_id=end_retry_id,
        query_str="END: Retry result",
        dependencies=[guardrail_2_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end_retry)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_guardrail_executor_metric")
def query_planner_mock_guardrail_executor_metric(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    GUARDRAIL using custom executor for Python evaluation.

    Structure:
        START -> EXTRACT -> GUARDRAIL (executor metric) -> [pass: STORE, fail: REVIEW]

    Tests:
        - Custom Python evaluation via executor
        - External validation service integration
        - Complex business logic validation

    Args:
        planner_info: Contains configuration and state information for the plan
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A guardrail plan with executor-based metrics
    """
    base_id = planner_info.base_id
    nodes = []

    # START node
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # EXTRACT node (simulates field extraction)
    extract = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Extract document fields",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_extract://document/extract",
            params={"fields": ["invoice_number", "total_amount", "date", "vendor"]},
        ),
    )
    planner_info.current_id += 1
    nodes.append(extract)

    # Pre-generate IDs
    guardrail_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    store_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    review_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1

    # GUARDRAIL with executor-based custom evaluation
    guardrail = Query(
        task_id=guardrail_id,
        query_str="Validate extracted fields",
        dependencies=[extract.task_id],
        node_type=QueryType.GUARDRAIL,
        definition=GuardrailQueryDefinition(
            input_source=f"$.nodes.{extract.task_id}.output.extracted_fields",
            metrics=[
                # Built-in JSON schema validation
                GuardrailMetric(
                    type=GuardrailMetricType.JSON_SCHEMA,
                    name="Schema Validation",
                    threshold=1.0,
                    weight=1.0,
                    params={
                        "schema": {
                            "type": "object",
                            "required": ["invoice_number", "total_amount"],
                            "properties": {
                                "invoice_number": {"type": "string"},
                                "total_amount": {"type": "number"},
                                "date": {"type": "string"},
                                "vendor": {"type": "string"},
                            },
                        }
                    },
                ),
                # Custom executor-based evaluation
                GuardrailMetric(
                    type=GuardrailMetricType.EXECUTOR,
                    name="Business Rules Validation",
                    threshold=0.8,
                    weight=2.0,  # Higher weight for business rules
                    params={
                        "endpoint": "guardrail_executor://evaluate",
                        "function": "check_json_structure",
                        "config": {
                            "required_fields": ["invoice_number", "total_amount"],
                            "amount_min": 0,
                            "amount_max": 1000000,
                        },
                    },
                ),
            ],
            aggregation_mode="weighted_average",
            pass_threshold=0.7,
            paths=[
                GuardrailPath(
                    path_id="pass",
                    target_node_ids=[store_id],
                    description="Validation passed - store results",
                ),
                GuardrailPath(
                    path_id="fail",
                    target_node_ids=[review_id],
                    description="Validation failed - manual review",
                ),
            ],
            fail_fast=False,
            include_feedback=True,
            evaluation_timeout=60,  # Longer timeout for executor call
        ),
    )
    nodes.append(guardrail)

    # STORE node (pass path)
    store = Query(
        task_id=store_id,
        query_str="Store validated results",
        dependencies=[guardrail_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_store://document/store",
            params={"validated": True},
        ),
    )
    nodes.append(store)

    # REVIEW node (fail path)
    review = Query(
        task_id=review_id,
        query_str="Queue for manual review",
        dependencies=[guardrail_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_review://document/queue",
            params={"reason": "validation_failed"},
        ),
    )
    nodes.append(review)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_guardrail_multi_metric")
def query_planner_mock_guardrail_multi_metric(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    GUARDRAIL with multiple metrics and aggregation modes.

    Structure:
        START -> PROCESS -> GUARDRAIL (5 metrics, weighted_average) -> [pass: END, fail: ERROR]

    Tests:
        - Multiple metric types in one guardrail
        - Weighted average aggregation
        - Fail-fast behavior

    Args:
        planner_info: Contains configuration and state information for the plan
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A guardrail plan with multiple metrics
    """
    base_id = planner_info.base_id
    nodes = []

    # START node
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # PROCESS node
    process = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Generate response",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_llm://generate",
            params={},
        ),
    )
    planner_info.current_id += 1
    nodes.append(process)

    # Pre-generate IDs
    guardrail_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    end_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1
    error_id = f"{increment_uuid7str(base_id, planner_info.current_id)}"
    planner_info.current_id += 1

    # GUARDRAIL with multiple metrics
    guardrail = Query(
        task_id=guardrail_id,
        query_str="Multi-metric quality check",
        dependencies=[process.task_id],
        node_type=QueryType.GUARDRAIL,
        definition=GuardrailQueryDefinition(
            input_source=f"$.nodes.{process.task_id}.output",
            metrics=[
                GuardrailMetric(
                    type=GuardrailMetricType.LENGTH_CHECK,
                    name="Length Check",
                    threshold=0.5,
                    weight=1.0,
                    params={"min": 100, "max": 5000},
                ),
                GuardrailMetric(
                    type=GuardrailMetricType.REGEX_MATCH,
                    name="No Profanity",
                    threshold=1.0,
                    weight=2.0,  # Higher weight - critical
                    params={"pattern": r"^(?!.*(badword1|badword2)).*$"},
                ),
                GuardrailMetric(
                    type=GuardrailMetricType.CONTAINS_KEYWORDS,
                    name="Contains Summary",
                    threshold=0.5,
                    weight=1.0,
                    params={"keywords": ["summary", "conclusion", "result"]},
                ),
                GuardrailMetric(
                    type=GuardrailMetricType.REGEX_MATCH,
                    name="Proper Formatting",
                    threshold=0.8,
                    weight=0.5,  # Lower weight - nice to have
                    params={"pattern": r"^[A-Z].*[.!?]$"},  # Starts with capital, ends with punctuation
                ),
            ],
            aggregation_mode="weighted_average",
            pass_threshold=0.75,
            paths=[
                GuardrailPath(
                    path_id="pass",
                    target_node_ids=[end_id],
                    description="All quality checks passed",
                ),
                GuardrailPath(
                    path_id="fail",
                    target_node_ids=[error_id],
                    description="Quality checks failed",
                ),
            ],
            fail_fast=True,  # Stop on first critical failure
            include_feedback=True,
            evaluation_timeout=30,
        ),
    )
    nodes.append(guardrail)

    # END node (pass)
    end = Query(
        task_id=end_id,
        query_str="END: Success",
        dependencies=[guardrail_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    # ERROR node (fail)
    error = Query(
        task_id=error_id,
        query_str="END: Quality check failed",
        dependencies=[guardrail_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(error)

    return QueryPlan(nodes=nodes)


# Export all plan functions
__all__ = [
    "query_planner_mock_guardrail_simple",
    "query_planner_mock_guardrail_retry_loop",
    "query_planner_mock_guardrail_executor_metric",
    "query_planner_mock_guardrail_multi_metric",
]
