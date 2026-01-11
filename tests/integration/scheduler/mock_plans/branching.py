"""
Branching Mock Query Plans

BRANCH and SWITCH node patterns for conditional routing.

Plans:
    - query_planner_mock_branch_simple: Simple BRANCH with JSONPath conditions
    - query_planner_mock_switch_complexity: SWITCH-based value routing
    - query_planner_mock_branch_multi_condition: Complex AND/OR condition groups
    - query_planner_mock_nested_branches: Nested branching (branch within branch)
    - query_planner_mock_branch_python_function: Python function evaluation
    - query_planner_mock_branch_jsonpath_advanced: Advanced JSONPath (arrays, nested fields)
    - query_planner_mock_branch_all_match: ALL_MATCH evaluation mode (multiple paths)
    - query_planner_mock_branch_regex_matching: Regex pattern matching
"""

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

from .base import (
    ExecutorEndpointQueryDefinition,
    LlmQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryType,
    increment_uuid7str,
    register_query_plan,
)


@register_query_plan("mock_branch_simple")
def query_planner_mock_branch_simple(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Simple branching mock query plan with JSONPath conditions.

    Structure:
        START -> CLASSIFY -> BRANCH (by document type) ->
            Path A: Invoice processing
            Path B: Contract processing
            Path C: Other processing
        -> MERGER -> END

    This demonstrates basic BRANCH functionality with JSONPath evaluation.
    """
    base_id = planner_info.base_id
    layout = planner_info.name
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

    # CLASSIFY - Determines document type
    classify = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Classify document type",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(classify)

    # Create target node IDs for each branch path
    invoice_task_id = increment_uuid7str(base_id, planner_info.current_id + 1)
    contract_task_id = increment_uuid7str(base_id, planner_info.current_id + 2)
    other_task_id = increment_uuid7str(base_id, planner_info.current_id + 3)

    # BRANCH node - Routes based on document type
    branch_paths = [
        BranchPath(
            path_id="invoice_path",
            condition=BranchCondition(
                jsonpath="$.metadata.document_type",
                operator="==",
                value="invoice",
                description="Route invoices to specialized processing",
            ),
            target_node_ids=[invoice_task_id],
            priority=1,
        ),
        BranchPath(
            path_id="contract_path",
            condition=BranchCondition(
                jsonpath="$.metadata.document_type",
                operator="==",
                value="contract",
                description="Route contracts to specialized processing",
            ),
            target_node_ids=[contract_task_id],
            priority=2,
        ),
        BranchPath(
            path_id="other_path",
            condition=None,  # Default path
            target_node_ids=[other_task_id],
            priority=3,
        ),
    ]

    branch = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: BRANCH by document type",
        dependencies=[classify.task_id],
        node_type=QueryType.BRANCH,
        definition=BranchQueryDefinition(
            method="BRANCH",
            endpoint="branch",
            paths=branch_paths,
            default_path_id="other_path",
            evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
        ),
    )
    planner_info.current_id += 1
    nodes.append(branch)

    # Invoice processing node
    invoice_process = Query(
        task_id=invoice_task_id,
        query_str=f"{planner_info.current_id}: Process invoice",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_b://document/process",
            params={"layout": layout, "doc_type": "invoice"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(invoice_process)

    # Contract processing node
    contract_process = Query(
        task_id=contract_task_id,
        query_str=f"{planner_info.current_id}: Process contract",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://document/process",
            params={"layout": layout, "doc_type": "contract"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(contract_process)

    # Other document processing node
    other_process = Query(
        task_id=other_task_id,
        query_str=f"{planner_info.current_id}: Process other document",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_d://document/process",
            params={"layout": layout, "doc_type": "other"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(other_process)

    # MERGER - Waits only for active (non-skipped) paths
    merger = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: MERGE results",
        dependencies=[invoice_process.task_id, contract_process.task_id, other_process.task_id],
        node_type=QueryType.MERGER,
        definition=EnhancedMergerQueryDefinition(
            method="MERGER_ENHANCED",
            endpoint="merger",
            merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
        ),
    )
    planner_info.current_id += 1
    nodes.append(merger)

    # END node
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[merger.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_switch_complexity")
def query_planner_mock_switch_complexity(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Mock query plan using SWITCH for value-based routing.

    Structure:
        START -> ANALYZE -> SWITCH (by complexity score) ->
            Simple: Fast processing
            Medium: Standard processing
            Complex: Advanced processing
        -> MERGER -> END

    This demonstrates SWITCH functionality for simple value-based routing.
    """
    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # ANALYZE - Calculates complexity score
    analyze = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Analyze complexity",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(analyze)

    # Create target node IDs
    fast_task_id = increment_uuid7str(base_id, planner_info.current_id + 1)
    standard_task_id = increment_uuid7str(base_id, planner_info.current_id + 2)
    advanced_task_id = increment_uuid7str(base_id, planner_info.current_id + 3)

    # SWITCH node - Routes based on complexity score
    switch = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: SWITCH by complexity",
        dependencies=[analyze.task_id],
        node_type=QueryType.SWITCH,
        definition=SwitchQueryDefinition(
            method="SWITCH",
            endpoint="switch",
            switch_field="$.metadata.complexity_score",
            cases={
                "simple": [fast_task_id],
                "medium": [standard_task_id],
                "complex": [advanced_task_id],
            },
            default_case=[standard_task_id],
        ),
    )
    planner_info.current_id += 1
    nodes.append(switch)

    # Fast processing for simple docs
    fast_process = Query(
        task_id=fast_task_id,
        query_str=f"{planner_info.current_id}: Fast processing",
        dependencies=[switch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_b://document/process",
            params={"layout": layout, "mode": "fast"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(fast_process)

    # Standard processing for medium docs
    standard_process = Query(
        task_id=standard_task_id,
        query_str=f"{planner_info.current_id}: Standard processing",
        dependencies=[switch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://document/process",
            params={"layout": layout, "mode": "standard"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(standard_process)

    # Advanced processing for complex docs
    advanced_process = Query(
        task_id=advanced_task_id,
        query_str=f"{planner_info.current_id}: Advanced processing",
        dependencies=[switch.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="deepseek_r1_32",
            endpoint="mock_executor_d://document/process",
            params={"layout": layout, "mode": "advanced"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(advanced_process)

    # MERGER
    merger = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: MERGE results",
        dependencies=[fast_process.task_id, standard_process.task_id, advanced_process.task_id],
        node_type=QueryType.MERGER,
        definition=EnhancedMergerQueryDefinition(
            method="MERGER_ENHANCED",
            endpoint="merger",
            merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
        ),
    )
    planner_info.current_id += 1
    nodes.append(merger)

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[merger.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_branch_multi_condition")
def query_planner_mock_branch_multi_condition(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Advanced branching with multi-condition groups (AND/OR logic).

    Structure:
        START -> CLASSIFY -> BRANCH (complex conditions) ->
            Path A: High priority AND (invoice OR contract)
            Path B: Large documents (page_count > 50)
            Path C: Standard processing (default)
        -> MERGER -> VALIDATE -> END

    This demonstrates complex condition groups with AND/OR combinators.
    """
    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # CLASSIFY
    classify = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Classify and analyze document",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(classify)

    # Create target node IDs
    expedited_task_id = increment_uuid7str(base_id, planner_info.current_id + 1)
    batch_task_id = increment_uuid7str(base_id, planner_info.current_id + 2)
    standard_task_id = increment_uuid7str(base_id, planner_info.current_id + 3)

    # BRANCH with complex conditions
    # Path 1: High priority AND (invoice OR contract)
    high_priority_condition = BranchConditionGroup(
        conditions=[
            BranchCondition(
                jsonpath="$.metadata.priority",
                operator="==",
                value="high",
                description="High priority documents",
            ),
            BranchConditionGroup(
                conditions=[
                    BranchCondition(
                        jsonpath="$.metadata.document_type",
                        operator="==",
                        value="invoice",
                    ),
                    BranchCondition(
                        jsonpath="$.metadata.document_type",
                        operator="==",
                        value="contract",
                    ),
                ],
                combinator="OR",
            ),
        ],
        combinator="AND",
    )

    branch_paths = [
        BranchPath(
            path_id="expedited",
            condition=high_priority_condition,
            target_node_ids=[expedited_task_id],
            priority=1,
        ),
        BranchPath(
            path_id="batch",
            condition=BranchCondition(
                jsonpath="$.metadata.page_count",
                operator=">",
                value=50,
                description="Large documents with more than 50 pages",
            ),
            target_node_ids=[batch_task_id],
            priority=2,
        ),
        BranchPath(
            path_id="standard",
            condition=None,  # Default path
            target_node_ids=[standard_task_id],
            priority=3,
        ),
    ]

    branch = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: BRANCH by priority and size",
        dependencies=[classify.task_id],
        node_type=QueryType.BRANCH,
        definition=BranchQueryDefinition(
            method="BRANCH",
            endpoint="branch",
            paths=branch_paths,
            default_path_id="standard",
            evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
        ),
    )
    planner_info.current_id += 1
    nodes.append(branch)

    # Expedited processing
    expedited = Query(
        task_id=expedited_task_id,
        query_str=f"{planner_info.current_id}: Expedited processing",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="deepseek_r1_32",
            endpoint="mock_executor_b://document/process",
            params={"layout": layout, "mode": "expedited"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(expedited)

    # Batch processing
    batch = Query(
        task_id=batch_task_id,
        query_str=f"{planner_info.current_id}: Batch processing",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://document/process",
            params={"layout": layout, "mode": "batch"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(batch)

    # Standard processing
    standard = Query(
        task_id=standard_task_id,
        query_str=f"{planner_info.current_id}: Standard processing",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_d://document/process",
            params={"layout": layout, "mode": "standard"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(standard)

    # MERGER
    merger = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: MERGE results",
        dependencies=[expedited.task_id, batch.task_id, standard.task_id],
        node_type=QueryType.MERGER,
        definition=EnhancedMergerQueryDefinition(
            method="MERGER_ENHANCED",
            endpoint="merger",
            merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
        ),
    )
    planner_info.current_id += 1
    nodes.append(merger)

    # VALIDATE
    validate = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Validate results",
        dependencies=[merger.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_e://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(validate)

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[validate.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_nested_branches")
def query_planner_mock_nested_branches(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Complex mock with nested branching - branch within a branch.

    Structure:
        START -> CLASSIFY -> BRANCH_1 (by type) ->
            Invoice path -> CHECK_AMOUNT -> BRANCH_2 (by amount) ->
                Small: Fast validation
                Large: Enhanced validation
            Contract path -> Standard validation
        -> MERGER -> END

    This demonstrates nested branching scenarios.
    """
    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # CLASSIFY
    classify = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Classify document",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(classify)

    # BRANCH_1 - By document type
    branch1_id = increment_uuid7str(base_id, planner_info.current_id)
    check_amount_id = increment_uuid7str(base_id, planner_info.current_id + 1)
    contract_validate_id = increment_uuid7str(base_id, planner_info.current_id + 2)

    branch1 = Query(
        task_id=branch1_id,
        query_str=f"{planner_info.current_id}: BRANCH_1 by type",
        dependencies=[classify.task_id],
        node_type=QueryType.BRANCH,
        definition=BranchQueryDefinition(
            method="BRANCH",
            endpoint="branch",
            paths=[
                BranchPath(
                    path_id="invoice",
                    condition=BranchCondition(
                        jsonpath="$.metadata.document_type",
                        operator="==",
                        value="invoice",
                    ),
                    target_node_ids=[check_amount_id],
                    priority=1,
                ),
                BranchPath(
                    path_id="contract",
                    condition=BranchCondition(
                        jsonpath="$.metadata.document_type",
                        operator="==",
                        value="contract",
                    ),
                    target_node_ids=[contract_validate_id],
                    priority=2,
                ),
            ],
            evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
        ),
    )
    planner_info.current_id += 1
    nodes.append(branch1)

    # CHECK_AMOUNT - For invoice path
    check_amount = Query(
        task_id=check_amount_id,
        query_str=f"{planner_info.current_id}: Check invoice amount",
        dependencies=[branch1.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_b://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(check_amount)

    # CONTRACT VALIDATION - For contract path (from first branch)
    contract_validate = Query(
        task_id=contract_validate_id,
        query_str=f"{planner_info.current_id}: Standard contract validation",
        dependencies=[branch1.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_e://document/process",
            params={"layout": layout, "doc_type": "contract"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(contract_validate)

    # BRANCH_2 - Nested branch by amount
    branch2_id = increment_uuid7str(base_id, planner_info.current_id)
    small_validate_id = increment_uuid7str(base_id, planner_info.current_id + 1)
    large_validate_id = increment_uuid7str(base_id, planner_info.current_id + 2)

    branch2 = Query(
        task_id=branch2_id,
        query_str=f"{planner_info.current_id}: BRANCH_2 by amount",
        dependencies=[check_amount.task_id],
        node_type=QueryType.BRANCH,
        definition=BranchQueryDefinition(
            method="BRANCH",
            endpoint="branch",
            paths=[
                BranchPath(
                    path_id="small_amount",
                    condition=BranchCondition(
                        jsonpath="$.metadata.amount",
                        operator="<=",
                        value=1000,
                    ),
                    target_node_ids=[small_validate_id],
                    priority=1,
                ),
                BranchPath(
                    path_id="large_amount",
                    condition=BranchCondition(
                        jsonpath="$.metadata.amount",
                        operator=">",
                        value=1000,
                    ),
                    target_node_ids=[large_validate_id],
                    priority=2,
                ),
            ],
            evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
        ),
    )
    planner_info.current_id += 1
    nodes.append(branch2)

    # Small amount validation
    small_validate = Query(
        task_id=small_validate_id,
        query_str=f"{planner_info.current_id}: Fast validation (small amount)",
        dependencies=[branch2.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://document/process",
            params={"layout": layout, "mode": "fast"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(small_validate)

    # Large amount validation
    large_validate = Query(
        task_id=large_validate_id,
        query_str=f"{planner_info.current_id}: Enhanced validation (large amount)",
        dependencies=[branch2.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="deepseek_r1_32",
            endpoint="mock_executor_d://document/process",
            params={"layout": layout, "mode": "enhanced"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(large_validate)

    # MERGER - Combines all validation paths
    merger = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: MERGE all validations",
        dependencies=[small_validate.task_id, large_validate.task_id, contract_validate.task_id],
        node_type=QueryType.MERGER,
        definition=EnhancedMergerQueryDefinition(
            method="MERGER_ENHANCED",
            endpoint="merger",
            merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
        ),
    )
    planner_info.current_id += 1
    nodes.append(merger)

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[merger.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_branch_python_function")
def query_planner_mock_branch_python_function(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Branching mock using Python function evaluation for complex business logic.

    Structure:
        START -> ANALYZE -> BRANCH (Python function) ->
            Path A: High-value documents (custom function)
            Path B: Standard documents (fallback)
        -> MERGER -> END

    This demonstrates using Python functions for complex conditions that
    can't be expressed in JSONPath.
    """
    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # ANALYZE - Enriches document with business metrics
    analyze = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Analyze document metrics",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://document/process",
            params={"layout": layout, "compute_risk_score": True},
        ),
    )
    planner_info.current_id += 1
    nodes.append(analyze)

    # Pre-allocate IDs
    premium_task_id = increment_uuid7str(base_id, planner_info.current_id + 1)
    standard_task_id = increment_uuid7str(base_id, planner_info.current_id + 2)

    # BRANCH using Python function
    branch_paths = [
        BranchPath(
            path_id="premium",
            condition_function="marie.scheduler.branch_conditions.is_high_value_document",
            target_node_ids=[premium_task_id],
            priority=1,
        ),
        BranchPath(
            path_id="standard",
            condition=None,  # Default fallback
            target_node_ids=[standard_task_id],
            priority=2,
        ),
    ]

    branch = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: BRANCH by business value (Python function)",
        dependencies=[analyze.task_id],
        node_type=QueryType.BRANCH,
        definition=BranchQueryDefinition(
            method="BRANCH",
            endpoint="branch",
            paths=branch_paths,
            default_path_id="standard",
            evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
        ),
    )
    planner_info.current_id += 1
    nodes.append(branch)

    # Premium processing
    premium_process = Query(
        task_id=premium_task_id,
        query_str=f"{planner_info.current_id}: Premium processing with manual review",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="deepseek_r1_32",
            endpoint="mock_executor_b://document/process",
            params={"layout": layout, "mode": "premium", "manual_review": True},
        ),
    )
    planner_info.current_id += 1
    nodes.append(premium_process)

    # Standard processing
    standard_process = Query(
        task_id=standard_task_id,
        query_str=f"{planner_info.current_id}: Standard automated processing",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://document/process",
            params={"layout": layout, "mode": "standard"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(standard_process)

    # MERGER
    merger = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: MERGE results",
        dependencies=[premium_process.task_id, standard_process.task_id],
        node_type=QueryType.MERGER,
        definition=EnhancedMergerQueryDefinition(
            method="MERGER_ENHANCED",
            endpoint="merger",
            merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
        ),
    )
    planner_info.current_id += 1
    nodes.append(merger)

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[merger.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_branch_jsonpath_advanced")
def query_planner_mock_branch_jsonpath_advanced(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Advanced JSONPath expressions with array filtering and complex queries.

    Structure:
        START -> EXTRACT -> BRANCH (advanced JSONPath) ->
            Path A: Documents with multiple recipients (array length check)
            Path B: Documents with high confidence scores (nested field)
            Path C: Documents with specific tags (array contains)
            Path D: Default
        -> MERGER -> END

    This demonstrates advanced JSONPath features.
    """
    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # EXTRACT - Extract metadata and structure
    extract = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Extract document metadata",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://document/process",
            params={"layout": layout, "extract_recipients": True, "extract_tags": True},
        ),
    )
    planner_info.current_id += 1
    nodes.append(extract)

    # Pre-allocate IDs
    multi_recipient_id = increment_uuid7str(base_id, planner_info.current_id + 1)
    high_confidence_id = increment_uuid7str(base_id, planner_info.current_id + 2)
    tagged_id = increment_uuid7str(base_id, planner_info.current_id + 3)
    default_id = increment_uuid7str(base_id, planner_info.current_id + 4)

    # BRANCH with advanced JSONPath expressions
    branch_paths = [
        BranchPath(
            path_id="multi_recipient",
            condition=BranchCondition(
                jsonpath="$.metadata.recipients",
                operator="is_empty",
                value=False,
                description="Documents with multiple recipients (non-empty array)",
            ),
            target_node_ids=[multi_recipient_id],
            priority=1,
        ),
        BranchPath(
            path_id="high_confidence",
            condition=BranchCondition(
                jsonpath="$.metadata.confidence_score",
                operator=">=",
                value=0.95,
                description="Documents with confidence >= 95%",
            ),
            target_node_ids=[high_confidence_id],
            priority=2,
        ),
        BranchPath(
            path_id="urgent_tagged",
            condition=BranchCondition(
                jsonpath="$.metadata.tags",
                operator="contains",
                value="urgent",
                description="Documents tagged as urgent",
            ),
            target_node_ids=[tagged_id],
            priority=3,
        ),
        BranchPath(
            path_id="default",
            condition=None,
            target_node_ids=[default_id],
            priority=4,
        ),
    ]

    branch = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: BRANCH by advanced metadata (JSONPath)",
        dependencies=[extract.task_id],
        node_type=QueryType.BRANCH,
        definition=BranchQueryDefinition(
            method="BRANCH",
            endpoint="branch",
            paths=branch_paths,
            default_path_id="default",
            evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
        ),
    )
    planner_info.current_id += 1
    nodes.append(branch)

    # Multi-recipient processing
    multi_recipient = Query(
        task_id=multi_recipient_id,
        query_str=f"{planner_info.current_id}: Multi-recipient distribution",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_b://document/distribute",
            params={"layout": layout, "mode": "multi_recipient"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(multi_recipient)

    # High confidence processing
    high_confidence = Query(
        task_id=high_confidence_id,
        query_str=f"{planner_info.current_id}: Auto-approve high confidence",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://document/approve",
            params={"layout": layout, "mode": "auto_approve"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(high_confidence)

    # Urgent tagged processing
    urgent_tagged = Query(
        task_id=tagged_id,
        query_str=f"{planner_info.current_id}: Expedite urgent documents",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="deepseek_r1_32",
            endpoint="mock_executor_d://document/process",
            params={"layout": layout, "mode": "urgent"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(urgent_tagged)

    # Default processing
    default_process = Query(
        task_id=default_id,
        query_str=f"{planner_info.current_id}: Standard document processing",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_e://document/process",
            params={"layout": layout, "mode": "standard"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(default_process)

    # MERGER
    merger = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: MERGE all results",
        dependencies=[
            multi_recipient.task_id,
            high_confidence.task_id,
            urgent_tagged.task_id,
            default_process.task_id,
        ],
        node_type=QueryType.MERGER,
        definition=EnhancedMergerQueryDefinition(
            method="MERGER_ENHANCED",
            endpoint="merger",
            merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
        ),
    )
    planner_info.current_id += 1
    nodes.append(merger)

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[merger.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_branch_all_match")
def query_planner_mock_branch_all_match(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Branching with ALL_MATCH evaluation mode (activates multiple paths).

    Structure:
        START -> CLASSIFY -> BRANCH (ALL_MATCH) ->
            Path A: Legal review (if contains legal terms)
            Path B: Financial review (if contains financial data)
            Path C: Compliance check (if regulated industry)
        -> MERGER (WAIT_ALL_ACTIVE) -> VALIDATE -> END

    This demonstrates ALL_MATCH mode where multiple paths can be activated
    simultaneously if multiple conditions are met.
    """
    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # CLASSIFY - Classify document content
    classify = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Classify document content",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="deepseek_r1_32",
            endpoint="mock_executor_a://document/process",
            params={"layout": layout, "detect_content_types": True},
        ),
    )
    planner_info.current_id += 1
    nodes.append(classify)

    # Pre-allocate IDs
    legal_review_id = increment_uuid7str(base_id, planner_info.current_id + 1)
    financial_review_id = increment_uuid7str(base_id, planner_info.current_id + 2)
    compliance_check_id = increment_uuid7str(base_id, planner_info.current_id + 3)

    # BRANCH with ALL_MATCHING mode
    branch_paths = [
        BranchPath(
            path_id="legal_review",
            condition=BranchCondition(
                jsonpath="$.metadata.contains_legal_terms",
                operator="==",
                value=True,
                description="Document contains legal terminology",
            ),
            target_node_ids=[legal_review_id],
            priority=1,
        ),
        BranchPath(
            path_id="financial_review",
            condition=BranchCondition(
                jsonpath="$.metadata.contains_financial_data",
                operator="==",
                value=True,
                description="Document contains financial information",
            ),
            target_node_ids=[financial_review_id],
            priority=2,
        ),
        BranchPath(
            path_id="compliance_check",
            condition=BranchCondition(
                jsonpath="$.metadata.regulated_industry",
                operator="==",
                value=True,
                description="Document from regulated industry",
            ),
            target_node_ids=[compliance_check_id],
            priority=3,
        ),
    ]

    branch = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: BRANCH for reviews (ALL_MATCHING)",
        dependencies=[classify.task_id],
        node_type=QueryType.BRANCH,
        definition=BranchQueryDefinition(
            method="BRANCH",
            endpoint="branch",
            paths=branch_paths,
            default_path_id=None,  # No default - only matching paths activate
            evaluation_mode=BranchEvaluationMode.ALL_MATCH,  # KEY: ALL_MATCH mode
        ),
    )
    planner_info.current_id += 1
    nodes.append(branch)

    # Legal review path
    legal_review = Query(
        task_id=legal_review_id,
        query_str=f"{planner_info.current_id}: Legal review",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_b://document/process",
            params={"layout": layout, "review_type": "legal"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(legal_review)

    # Financial review path
    financial_review = Query(
        task_id=financial_review_id,
        query_str=f"{planner_info.current_id}: Financial review",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://document/process",
            params={"layout": layout, "review_type": "financial"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(financial_review)

    # Compliance check path
    compliance_check = Query(
        task_id=compliance_check_id,
        query_str=f"{planner_info.current_id}: Compliance check",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_d://document/process",
            params={"layout": layout, "review_type": "compliance"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(compliance_check)

    # MERGER - Waits for all active review paths
    merger = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: MERGE review results",
        dependencies=[
            legal_review.task_id,
            financial_review.task_id,
            compliance_check.task_id,
        ],
        node_type=QueryType.MERGER,
        definition=EnhancedMergerQueryDefinition(
            method="MERGER_ENHANCED",
            endpoint="merger",
            merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
        ),
    )
    planner_info.current_id += 1
    nodes.append(merger)

    # VALIDATE - Final validation
    validate = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Validate all reviews",
        dependencies=[merger.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_e://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1
    nodes.append(validate)

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[validate.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_branch_regex_matching")
def query_planner_mock_branch_regex_matching(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Branching using regex pattern matching in JSONPath conditions.

    Structure:
        START -> EXTRACT -> BRANCH (regex patterns) ->
            Path A: Email addresses (matches email pattern)
            Path B: Phone numbers (matches phone pattern)
            Path C: URLs (matches URL pattern)
            Path D: Plain text (default)
        -> MERGER -> END

    This demonstrates regex matching with the 'matches' operator.
    """
    base_id = planner_info.base_id
    layout = planner_info.name
    nodes = []

    # START
    start = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start)

    # EXTRACT - Extract primary content identifier
    extract = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Extract content identifier",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://document/process",
            params={"layout": layout, "extract_primary_field": True},
        ),
    )
    planner_info.current_id += 1
    nodes.append(extract)

    # Pre-allocate IDs
    email_id = increment_uuid7str(base_id, planner_info.current_id + 1)
    phone_id = increment_uuid7str(base_id, planner_info.current_id + 2)
    url_id = increment_uuid7str(base_id, planner_info.current_id + 3)
    text_id = increment_uuid7str(base_id, planner_info.current_id + 4)

    # BRANCH with regex matching
    branch_paths = [
        BranchPath(
            path_id="email",
            condition=BranchCondition(
                jsonpath="$.metadata.primary_field",
                operator="matches",
                value=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                description="Email address pattern",
            ),
            target_node_ids=[email_id],
            priority=1,
        ),
        BranchPath(
            path_id="phone",
            condition=BranchCondition(
                jsonpath="$.metadata.primary_field",
                operator="matches",
                value=r"^\+?1?\d{9,15}$",
                description="Phone number pattern",
            ),
            target_node_ids=[phone_id],
            priority=2,
        ),
        BranchPath(
            path_id="url",
            condition=BranchCondition(
                jsonpath="$.metadata.primary_field",
                operator="matches",
                value=r"^https?://[^\s]+$",
                description="URL pattern",
            ),
            target_node_ids=[url_id],
            priority=3,
        ),
        BranchPath(
            path_id="text",
            condition=None,
            target_node_ids=[text_id],
            priority=4,
        ),
    ]

    branch = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: BRANCH by content pattern (regex)",
        dependencies=[extract.task_id],
        node_type=QueryType.BRANCH,
        definition=BranchQueryDefinition(
            method="BRANCH",
            endpoint="branch",
            paths=branch_paths,
            default_path_id="text",
            evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
        ),
    )
    planner_info.current_id += 1
    nodes.append(branch)

    # Email processing
    email_process = Query(
        task_id=email_id,
        query_str=f"{planner_info.current_id}: Process email address",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_b://document/process",
            params={"layout": layout, "content_type": "email"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(email_process)

    # Phone processing
    phone_process = Query(
        task_id=phone_id,
        query_str=f"{planner_info.current_id}: Process phone number",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_c://document/process",
            params={"layout": layout, "content_type": "phone"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(phone_process)

    # URL processing
    url_process = Query(
        task_id=url_id,
        query_str=f"{planner_info.current_id}: Process URL",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_d://document/process",
            params={"layout": layout, "content_type": "url"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(url_process)

    # Text processing
    text_process = Query(
        task_id=text_id,
        query_str=f"{planner_info.current_id}: Process plain text",
        dependencies=[branch.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_e://document/process",
            params={"layout": layout, "content_type": "text"},
        ),
    )
    planner_info.current_id += 1
    nodes.append(text_process)

    # MERGER
    merger = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: MERGE results",
        dependencies=[
            email_process.task_id,
            phone_process.task_id,
            url_process.task_id,
            text_process.task_id,
        ],
        node_type=QueryType.MERGER,
        definition=EnhancedMergerQueryDefinition(
            method="MERGER_ENHANCED",
            endpoint="merger",
            merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
        ),
    )
    planner_info.current_id += 1
    nodes.append(merger)

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[merger.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)
