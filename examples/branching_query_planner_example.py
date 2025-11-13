"""
Example demonstrating branching and conditional logic in Marie-AI Query Planners.

This example shows how to create query plans with:
1. BRANCH nodes with JSONPath conditions
2. SWITCH nodes for value-based routing
3. Enhanced MERGER nodes aware of branch skipping
4. Multiple execution paths with skip logic

The example implements a document processing workflow that routes documents
based on their type and complexity.
"""

from datetime import datetime, timezone

from marie.query_planner import (
    BranchCondition,
    BranchConditionGroup,
    BranchEvaluationMode,
    BranchPath,
    ConditionalQueryPlanBuilder,
    MergerStrategy,
    NoopQueryDefinition,
    PlannerInfo,
)


def create_document_routing_plan():
    """
    Create a query plan that routes documents based on type and complexity.

    Workflow:
    1. START (noop) - Entry point
    2. CLASSIFY - Classify document type
    3. BRANCH_BY_TYPE - Route based on document type (JSONPath)
        - Path A: Invoice processing
        - Path B: Contract processing
        - Path C: Generic processing
    4. COMPLEXITY_CHECK - Check document complexity
    5. SWITCH_BY_COMPLEXITY - Route based on complexity score
        - Simple: Fast processing
        - Complex: Advanced processing
    6. MERGER - Combine results from all branches
    7. FINALIZE - Final processing step
    """

    planner_info = PlannerInfo(
        name="document_routing_workflow",
        version="1.0",
        description="Route documents based on type and complexity",
    )

    builder = ConditionalQueryPlanBuilder(planner_info)

    # 1. Start node (entry point)
    start_id = builder.add_node(
        query_str="START",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(
            name="start",
            description="Workflow entry point",
        ),
        dependencies=None,
    )

    # 2. Classify document
    classify_id = builder.add_node(
        query_str="CLASSIFY_DOCUMENT",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(
            name="classify",
            description="Classify document type using ML model",
        ),
        dependencies=[start_id],
    )

    # 3. Branch based on document type (JSONPath conditions)
    # Define branching paths
    invoice_path = BranchPath(
        path_id="invoice_path",
        condition=BranchCondition(
            jsonpath="$.metadata.document_type",
            operator="==",
            value="invoice",
            description="Route to invoice processing",
        ),
        target_node_ids=["EXTRACT_INVOICE_DATA"],
        priority=1,
    )

    contract_path = BranchPath(
        path_id="contract_path",
        condition=BranchCondition(
            jsonpath="$.metadata.document_type",
            operator="==",
            value="contract",
            description="Route to contract processing",
        ),
        target_node_ids=["EXTRACT_CONTRACT_DATA"],
        priority=2,
    )

    generic_path = BranchPath(
        path_id="generic_path",
        condition=BranchCondition(
            jsonpath="$.metadata.document_type",
            operator="in",
            value=["other", "unknown"],
            description="Route to generic processing",
        ),
        target_node_ids=["EXTRACT_GENERIC_DATA"],
        priority=3,
    )

    branch_type_id = builder.add_branch(
        query_str="BRANCH_BY_DOCUMENT_TYPE",
        paths=[invoice_path, contract_path, generic_path],
        default_path_id="generic_path",  # Fallback if no condition matches
        evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
        dependencies=[classify_id],
    )

    # 4. Create processing nodes for each path
    extract_invoice_id = builder.add_node(
        query_str="EXTRACT_INVOICE_DATA",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(
            name="extract_invoice",
            description="Extract invoice-specific fields",
        ),
        dependencies=[branch_type_id],
    )
    builder.mark_as_conditional(extract_invoice_id, branch_type_id, "invoice_path")

    extract_contract_id = builder.add_node(
        query_str="EXTRACT_CONTRACT_DATA",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(
            name="extract_contract",
            description="Extract contract-specific fields",
        ),
        dependencies=[branch_type_id],
    )
    builder.mark_as_conditional(extract_contract_id, branch_type_id, "contract_path")

    extract_generic_id = builder.add_node(
        query_str="EXTRACT_GENERIC_DATA",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(
            name="extract_generic",
            description="Extract generic document fields",
        ),
        dependencies=[branch_type_id],
    )
    builder.mark_as_conditional(extract_generic_id, branch_type_id, "generic_path")

    # 5. Complexity check (runs after any extraction path)
    complexity_check_id = builder.add_node(
        query_str="COMPLEXITY_CHECK",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(
            name="complexity_check",
            description="Calculate document complexity score",
        ),
        dependencies=[extract_invoice_id, extract_contract_id, extract_generic_id],
    )

    # 6. Switch based on complexity score
    switch_complexity_id = builder.add_switch(
        query_str="SWITCH_BY_COMPLEXITY",
        switch_field="$.metadata.complexity_score",
        cases={
            "simple": ["FAST_PROCESSING"],
            "medium": ["STANDARD_PROCESSING"],
            "complex": ["ADVANCED_PROCESSING"],
        },
        default_case=["STANDARD_PROCESSING"],
        dependencies=[complexity_check_id],
    )

    # 7. Create processing nodes for each complexity level
    fast_processing_id = builder.add_node(
        query_str="FAST_PROCESSING",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(
            name="fast_processing",
            description="Fast processing for simple documents",
        ),
        dependencies=[switch_complexity_id],
    )
    builder.mark_as_conditional(fast_processing_id, switch_complexity_id, "simple")

    standard_processing_id = builder.add_node(
        query_str="STANDARD_PROCESSING",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(
            name="standard_processing",
            description="Standard processing for medium documents",
        ),
        dependencies=[switch_complexity_id],
    )
    builder.mark_as_conditional(standard_processing_id, switch_complexity_id, "medium")

    advanced_processing_id = builder.add_node(
        query_str="ADVANCED_PROCESSING",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(
            name="advanced_processing",
            description="Advanced processing for complex documents",
        ),
        dependencies=[switch_complexity_id],
    )
    builder.mark_as_conditional(advanced_processing_id, switch_complexity_id, "complex")

    # 8. Enhanced merger that waits only for active (non-skipped) paths
    merger_id = builder.add_merger(
        query_str="MERGE_RESULTS",
        merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
        dependencies=[
            fast_processing_id,
            standard_processing_id,
            advanced_processing_id,
        ],
    )

    # 9. Final processing
    finalize_id = builder.add_node(
        query_str="FINALIZE",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(
            name="finalize",
            description="Finalize and output results",
        ),
        dependencies=[merger_id],
    )

    # Build and return the plan
    plan = builder.build()
    return plan


def create_multi_condition_branch_plan():
    """
    Example showing complex multi-condition branching with AND/OR logic.
    """

    planner_info = PlannerInfo(
        name="multi_condition_workflow",
        version="1.0",
        description="Demonstrate complex condition groups",
    )

    builder = ConditionalQueryPlanBuilder(planner_info)

    start_id = builder.add_node(
        query_str="START",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(name="start"),
    )

    # Branch with complex condition groups (AND/OR logic)
    # Path 1: High priority AND (invoice OR contract)
    condition_group_1 = BranchConditionGroup(
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

    high_priority_path = BranchPath(
        path_id="high_priority",
        condition=condition_group_1,
        target_node_ids=["EXPEDITED_PROCESSING"],
        priority=1,
    )

    # Path 2: Large documents (page_count > 50)
    large_doc_path = BranchPath(
        path_id="large_documents",
        condition=BranchCondition(
            jsonpath="$.metadata.page_count",
            operator=">",
            value=50,
            description="Documents with more than 50 pages",
        ),
        target_node_ids=["BATCH_PROCESSING"],
        priority=2,
    )

    # Path 3: Everything else
    standard_path = BranchPath(
        path_id="standard",
        condition=None,  # No condition = default path
        target_node_ids=["STANDARD_PROCESSING"],
        priority=3,
    )

    branch_id = builder.add_branch(
        query_str="BRANCH_BY_PRIORITY_AND_SIZE",
        paths=[high_priority_path, large_doc_path, standard_path],
        evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
        dependencies=[start_id],
    )

    # Add processing nodes
    expedited_id = builder.add_node(
        query_str="EXPEDITED_PROCESSING",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(name="expedited"),
        dependencies=[branch_id],
    )

    batch_id = builder.add_node(
        query_str="BATCH_PROCESSING",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(name="batch"),
        dependencies=[branch_id],
    )

    standard_id = builder.add_node(
        query_str="STANDARD_PROCESSING",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(name="standard"),
        dependencies=[branch_id],
    )

    # Merger
    merger_id = builder.add_merger(
        query_str="MERGE_RESULTS",
        merge_strategy=MergerStrategy.WAIT_ALL_ACTIVE,
        dependencies=[expedited_id, batch_id, standard_id],
    )

    plan = builder.build()
    return plan


def create_python_function_branch_plan():
    """
    Example showing branching using Python function evaluation instead of JSONPath.
    Useful for complex business logic that can't be expressed in JSONPath.
    """

    planner_info = PlannerInfo(
        name="python_function_workflow",
        version="1.0",
        description="Use Python functions for branch conditions",
    )

    builder = ConditionalQueryPlanBuilder(planner_info)

    start_id = builder.add_node(
        query_str="START",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(name="start"),
    )

    # Path using Python function for complex logic
    # The function path should be a module path to a callable
    # Example: "my_module.conditions.is_high_value_document"
    high_value_path = BranchPath(
        path_id="high_value",
        condition_function="marie.examples.branch_conditions.is_high_value_document",
        target_node_ids=["PREMIUM_PROCESSING"],
        priority=1,
    )

    standard_path = BranchPath(
        path_id="standard",
        condition=None,
        target_node_ids=["STANDARD_PROCESSING"],
        priority=2,
    )

    branch_id = builder.add_branch(
        query_str="BRANCH_BY_VALUE",
        paths=[high_value_path, standard_path],
        evaluation_mode=BranchEvaluationMode.FIRST_MATCH,
        dependencies=[start_id],
    )

    premium_id = builder.add_node(
        query_str="PREMIUM_PROCESSING",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(name="premium"),
        dependencies=[branch_id],
    )

    standard_id = builder.add_node(
        query_str="STANDARD_PROCESSING",
        node_type="COMPUTE",
        definition=NoopQueryDefinition(name="standard"),
        dependencies=[branch_id],
    )

    plan = builder.build()
    return plan


if __name__ == "__main__":
    # Example 1: Document routing with type and complexity
    print("=" * 80)
    print("Example 1: Document Routing Workflow")
    print("=" * 80)
    plan1 = create_document_routing_plan()
    print(f"Plan Name: {plan1.planner_info.name}")
    print(f"Total Nodes: {len(plan1.queries)}")
    print(f"Description: {plan1.planner_info.description}")
    print("\nQuery Nodes:")
    for query in plan1.queries:
        print(
            f"  - {query.query} (type: {query.definition.method if hasattr(query.definition, 'method') else 'N/A'})"
        )

    # Example 2: Multi-condition branching
    print("\n" + "=" * 80)
    print("Example 2: Multi-Condition Branching")
    print("=" * 80)
    plan2 = create_multi_condition_branch_plan()
    print(f"Plan Name: {plan2.planner_info.name}")
    print(f"Total Nodes: {len(plan2.queries)}")

    # Example 3: Python function branching
    print("\n" + "=" * 80)
    print("Example 3: Python Function Branching")
    print("=" * 80)
    plan3 = create_python_function_branch_plan()
    print(f"Plan Name: {plan3.planner_info.name}")
    print(f"Total Nodes: {len(plan3.queries)}")

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)

    # You can also export to YAML/JSON for inspection
    from marie.query_planner import plan_to_json, plan_to_yaml

    print("\n" + "=" * 80)
    print("Example 1 as YAML:")
    print("=" * 80)
    yaml_output = plan_to_yaml(plan1)
    print(yaml_output)
