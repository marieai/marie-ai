"""
Mock Query Plans for Job Distribution Performance Testing

This module provides mock query plans that mimic the structure of production query plans
from grapnel-g5 for testing job distribution performance, scheduling behavior, and
resource allocation in the Marie-AI scheduler.

Usage:
    from tests.integration.scheduler.mock_query_plans import (
        query_planner_mock_simple,
        query_planner_mock_medium,
        query_planner_mock_complex,
    )

    planner_info = PlannerInfo(name="mock_simple", base_id=generate_job_id())
    plan = query_planner_mock_simple(planner_info)
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


@register_query_plan("mock_simple")
def query_planner_mock_simple(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Simple mock query plan for basic performance testing.

    Structure: START -> PROCESS -> END (3 nodes, linear)

    This plan is useful for testing basic job scheduling without complex dependencies.

    Args:
        planner_info: Contains configuration and state information for the plan
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A simple linear execution plan
    """
    base_id = planner_info.base_id
    layout = planner_info.name

    # START node
    root = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1

    # PROCESS node
    process = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Process document",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor://process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1

    # END node
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[process.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    return QueryPlan(nodes=[root, process, end])


@register_query_plan("mock_medium")
def query_planner_mock_medium(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Medium complexity mock query plan for performance testing.

    Structure:
        START -> PREPARE -> [WORKER_1, WORKER_2, WORKER_3] -> MERGE -> FINALIZE -> END
        (7 nodes, with parallel execution branch)

    This plan tests parallel job execution and merge operations.

    Args:
        planner_info: Contains configuration and state information for the plan
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A medium complexity execution plan with parallel branches
    """
    base_id = planner_info.base_id
    layout = planner_info.name

    # START node
    root = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1

    # PREPARE node
    prepare = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Prepare data",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://prepare",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1

    # PARALLEL WORKERS
    workers = []
    worker_configs = [
        ("Worker 1: Extract text", "mock_executor_b://extract"),
        ("Worker 2: Classify", "mock_executor_b://classify"),
        ("Worker 3: Analyze", "mock_executor_c://analyze"),
    ]

    for name, endpoint in worker_configs:
        worker = Query(
            task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
            query_str=f"{planner_info.current_id}: {name}",
            dependencies=[prepare.task_id],
            node_type=QueryType.COMPUTE,
            definition=LlmQueryDefinition(
                model_name="mock_model",
                endpoint=endpoint,
                params={"layout": layout, "key": name.lower()},
            ),
        )
        planner_info.current_id += 1
        workers.append(worker)

    # MERGE node
    merge = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: MERGE results",
        dependencies=[w.task_id for w in workers],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout}),
    )
    planner_info.current_id += 1

    # FINALIZE node
    finalize = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Finalize results",
        dependencies=[merge.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://finalize",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1

    # END node
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[finalize.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    return QueryPlan(nodes=[root, prepare] + workers + [merge, finalize, end])


@register_query_plan("mock_complex")
def query_planner_mock_complex(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Complex mock query plan mimicking production grapnel-g5 structure.

    Structure:
        START -> INIT -> [5 parallel annotators] -> MERGE ->
        [Sequential post-processing: Parser, Extractor, Validator] -> END
        (12 nodes, with parallel and sequential stages)

    This plan closely mimics the structure found in grapnel-g5 query plans
    and is suitable for comprehensive performance testing.

    Args:
        planner_info: Contains configuration and state information for the plan
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A complex execution plan with multiple stages
    """
    base_id = planner_info.base_id
    layout = planner_info.name

    # START node
    root = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1

    # INIT node - similar to Document Annotator
    init_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Initialize processing",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1

    # PARALLEL ANNOTATORS - mimicking KV, Tables, Code, Name, Embeddings
    annotators = []
    annotator_configs = [
        ("KV Annotator", "mock_executor_llm://annotator/kv", "LLM"),
        ("Table Annotator", "mock_executor_llm://annotator/table", "LLM"),
        ("Code Annotator", "mock_executor_llm://annotator/code", "LLM"),
        ("Name Annotator", "mock_executor_llm://annotator/name", "LLM"),
        ("Embedding Generator", "mock_executor_embed://embeddings", "EXECUTOR"),
    ]

    for name, endpoint, def_type in annotator_configs:
        if def_type == "LLM":
            definition = LlmQueryDefinition(
                model_name="deepseek_r1_32",
                endpoint=endpoint,
                params={"layout": layout, "key": name.lower().replace(" ", "_")},
            )
        else:
            definition = ExecutorEndpointQueryDefinition(
                endpoint=endpoint,
                params={"layout": layout, "key": name.lower().replace(" ", "_")},
            )

        annotator = Query(
            task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
            query_str=f"{planner_info.current_id}: {name}",
            dependencies=[init_node.task_id],
            node_type=QueryType.COMPUTE,
            definition=definition,
        )
        planner_info.current_id += 1
        annotators.append(annotator)

    # MERGE node - combines all annotator results
    merge = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: MERGE annotators",
        dependencies=[a.task_id for a in annotators],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout}),
    )
    planner_info.current_id += 1

    # POST-PROCESSING SEQUENTIAL CHAIN
    # Table Parser
    table_parser = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Table parser",
        dependencies=[merge.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_parser://parser/table",
            params={"layout": layout, "key": "table-extract"},
        ),
    )
    planner_info.current_id += 1

    # Table Extractor
    table_extractor = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Table extractor",
        dependencies=[table_parser.task_id],
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name="qwen_v2_5_vl",
            endpoint="mock_executor_table://extractor/table",
            params={"layout": layout, "key": "table-extract"},
        ),
    )
    planner_info.current_id += 1

    # Result Validator
    validator = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Validate results",
        dependencies=[table_extractor.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_validator://validator",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1

    # END node
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[validator.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(params={"layout": layout}),
    )

    return QueryPlan(
        nodes=[root, init_node]
        + annotators
        + [merge, table_parser, table_extractor, validator, end]
    )


def _build_subgraph_mock(
    planner_info: PlannerInfo,
    layout: str,
    parent_task_id: str,
    num_parallel_tasks: int = 3,
) -> dict:
    """
    Helper function to build a mock subgraph with parallel execution.

    This demonstrates the subgraph pattern used in grapnel-g5 for complex workflows.

    Args:
        planner_info: Planner configuration
        layout: Layout identifier
        parent_task_id: ID of the parent node this subgraph depends on
        num_parallel_tasks: Number of parallel tasks to create

    Returns:
        dict: Dictionary with 'root', 'end', and 'nodes' keys
    """
    nodes = []

    # Subgraph ROOT
    root = Query(
        task_id=f"{increment_uuid7str(planner_info.base_id, planner_info.current_id)}",
        query_str="SUBGRAPH: Start parallel processing",
        dependencies=[parent_task_id],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout}),
    )
    planner_info.current_id += 1
    nodes.append(root)

    # Parallel tasks
    parallel_tasks = []
    for i in range(num_parallel_tasks):
        task = Query(
            task_id=f"{increment_uuid7str(planner_info.base_id, planner_info.current_id)}",
            query_str=f"Parallel task {i+1}",
            dependencies=[root.task_id],
            node_type=QueryType.COMPUTE,
            definition=ExecutorEndpointQueryDefinition(
                endpoint=f"mock_executor_{chr(97+i)}://parallel",
                params={"layout": layout, "task_num": i},
            ),
        )
        planner_info.current_id += 1
        nodes.append(task)
        parallel_tasks.append(task)

    # Subgraph END
    end = Query(
        task_id=f"{increment_uuid7str(planner_info.base_id, planner_info.current_id)}",
        query_str="SUBGRAPH: End parallel processing",
        dependencies=[t.task_id for t in parallel_tasks],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout}),
    )
    planner_info.current_id += 1
    nodes.append(end)

    return {"root": root, "end": end, "nodes": nodes}


@register_query_plan("mock_with_subgraphs")
def query_planner_mock_with_subgraphs(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Advanced mock query plan with nested subgraphs.

    This demonstrates the subgraph pattern used in grapnel-g5 for organizing
    complex workflows with clear boundaries.

    Structure:
        START -> [Subgraph 1: 3 parallel] -> Process ->
        [Subgraph 2: 4 parallel] -> END

    Args:
        planner_info: Contains configuration and state information
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: An advanced plan with nested subgraphs
    """
    base_id = planner_info.base_id
    layout = planner_info.name

    # START
    root = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1

    # First subgraph
    subgraph1 = _build_subgraph_mock(
        planner_info=planner_info,
        layout=layout,
        parent_task_id=root.task_id,
        num_parallel_tasks=3,
    )

    # Middle processing node
    process = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Process intermediate results",
        dependencies=[subgraph1["end"].task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_main://process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1

    # Second subgraph
    subgraph2 = _build_subgraph_mock(
        planner_info=planner_info,
        layout=layout,
        parent_task_id=process.task_id,
        num_parallel_tasks=4,
    )

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[subgraph2["end"].task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    return QueryPlan(
        nodes=[root] + subgraph1["nodes"] + [process] + subgraph2["nodes"] + [end]
    )


if __name__ == "__main__":
    """
    Example usage and validation of mock query plans.
    """
    from pprint import pprint

    from marie.query_planner.planner import (
        print_query_plan,
        print_sorted_nodes,
        topological_sort,
        visualize_query_plan_graph,
    )

    # Test each plan
    plans_to_test = [
        ("mock_simple", "Simple Mock Plan"),
        ("mock_medium", "Medium Mock Plan"),
        ("mock_complex", "Complex Mock Plan"),
        ("mock_with_subgraphs", "Mock Plan with Subgraphs"),
    ]

    for plan_name, description in plans_to_test:
        print(f"\n{'='*80}")
        print(f"{description.upper()}")
        print(f"{'='*80}\n")

        # Create planner info
        planner_info = PlannerInfo(name=plan_name, base_id=generate_job_id())

        # Get the registered planner function
        planner_func = QueryPlanRegistry.get(plan_name)

        # Generate the plan
        plan = planner_func(planner_info)

        # Print plan details
        print(f"Plan: {plan_name}")
        print(f"Number of nodes: {len(plan.nodes)}")
        print(f"\nNode details:")
        pprint(plan.model_dump(), width=120)

        # Topological sort
        sorted_nodes = topological_sort(plan)
        print(f"\nTopological order:")
        print_sorted_nodes(sorted_nodes, plan)

        # Print formatted plan
        print_query_plan(plan, plan_name)

        # Visualize (if graphviz is available)
        try:
            visualize_query_plan_graph(plan)
            print(f"\nGraph visualization saved for {plan_name}")
        except Exception as e:
            print(f"\nSkipping visualization (graphviz not available): {e}")
