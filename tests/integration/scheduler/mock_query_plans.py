"""
Mock Query Plans for Job Distribution Performance Testing

This module provides mock query plans that mimic the structure of production query plans
from grapnel-g5 for testing job distribution performance, scheduling behavior, and
resource allocation in the Marie-AI scheduler.

Includes both traditional linear/parallel plans and advanced branching/conditional plans.

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
            endpoint="mock_executor_a://document/process",
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
    """T
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
            endpoint="mock_executor_a://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1

    # PARALLEL WORKERS
    workers = []
    worker_configs = [
        ("Worker 1: Extract text", "mock_executor_b://document/process", "LLM"),
        ("Worker 2: Classify", "mock_executor_c://document/process", "LLM"),
        ("Worker 3: Analyze", "mock_executor_d://document/process", "EXECUTOR"),
    ]

    for name, endpoint, def_type in worker_configs:
        if def_type == "LLM":
            definition = LlmQueryDefinition(
                model_name="mock_model",
                endpoint=endpoint,
                params={"layout": layout, "key": name.lower()},
            )
        else:
            definition = ExecutorEndpointQueryDefinition(
                endpoint=endpoint,
                params={"layout": layout, "key": name.lower()},
            )

        worker = Query(
            task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
            query_str=f"{planner_info.current_id}: {name}",
            dependencies=[prepare.task_id],
            node_type=QueryType.COMPUTE,
            definition=definition,
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
            endpoint="mock_executor_e://document/process",
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
        ("KV Annotator", "mock_executor_a://document/process", "LLM"),
        ("Table Annotator", "mock_executor_b://document/process", "LLM"),
        ("Code Annotator", "mock_executor_c://document/process", "LLM"),
        ("Name Annotator", "mock_executor_d://document/process", "LLM"),
        ("Embedding Generator", "mock_executor_e://document/process", "EXECUTOR"),
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
            endpoint="mock_executor_f://document/process",
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
            endpoint="mock_executor_g://document/process",
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
            endpoint="mock_executor_h://document/process",
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
        # Limit to 8 executors (a-h), wrap around if needed
        executor_letter = chr(97 + (i % 8))
        task = Query(
            task_id=f"{increment_uuid7str(planner_info.base_id, planner_info.current_id)}",
            query_str=f"Parallel task {i+1}",
            dependencies=[root.task_id],
            node_type=QueryType.COMPUTE,
            definition=ExecutorEndpointQueryDefinition(
                endpoint=f"mock_executor_{executor_letter}://document/process",
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
            endpoint="mock_executor_d://document/process",
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


@register_query_plan("mock_parallel_subgraphs")
def query_planner_mock_parallel_subgraphs(
    planner_info: PlannerInfo, **kwargs
) -> QueryPlan:
    """
    Highly complex mock query plan with multiple parallel subgraphs.

    This demonstrates advanced DAG execution where multiple independent subgraphs
    run concurrently, each containing their own internal parallel tasks. This
    pattern is useful for testing complex scheduling scenarios with nested
    parallelism and resource contention.

    Structure:
        START -> INIT ->
        [
            Subgraph 1 (Text Processing): 4 parallel tasks,
            Subgraph 2 (Image Processing): 5 parallel tasks,
            Subgraph 3 (Data Analysis): 3 parallel tasks
        ] (all running in parallel) ->
        MERGE ALL -> POST_PROCESS -> VALIDATE -> END

    Total: ~20+ nodes with deep parallelism

    Args:
        planner_info: Contains configuration and state information
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A highly complex plan with parallel subgraphs
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

    # INIT - Prepare data for parallel subgraphs
    init_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Initialize parallel subgraphs",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_a://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1

    # SUBGRAPH 1: Text Processing Pipeline
    # This subgraph handles text extraction, NER, sentiment, and summarization
    text_subgraph_root = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: TEXT_SUBGRAPH: Start text processing",
        dependencies=[init_node.task_id],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout, "subgraph": "text"}),
    )
    planner_info.current_id += 1

    text_tasks = []
    text_task_configs = [
        ("Extract text", "mock_executor_a://document/process", "EXECUTOR"),
        ("NER extraction", "mock_executor_b://document/process", "LLM"),
        ("Sentiment analysis", "mock_executor_c://document/process", "LLM"),
        ("Text summarization", "mock_executor_d://document/process", "LLM"),
    ]

    for name, endpoint, def_type in text_task_configs:
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

        task = Query(
            task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
            query_str=f"{planner_info.current_id}: TEXT: {name}",
            dependencies=[text_subgraph_root.task_id],
            node_type=QueryType.COMPUTE,
            definition=definition,
        )
        planner_info.current_id += 1
        text_tasks.append(task)

    text_subgraph_end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: TEXT_SUBGRAPH: Merge text results",
        dependencies=[t.task_id for t in text_tasks],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout, "subgraph": "text"}),
    )
    planner_info.current_id += 1

    # SUBGRAPH 2: Image Processing Pipeline
    # This subgraph handles OCR, object detection, image classification, etc.
    image_subgraph_root = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: IMAGE_SUBGRAPH: Start image processing",
        dependencies=[init_node.task_id],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout, "subgraph": "image"}),
    )
    planner_info.current_id += 1

    image_tasks = []
    image_task_configs = [
        ("OCR processing", "mock_executor_b://document/process", "EXECUTOR"),
        ("Object detection", "mock_executor_c://document/process", "EXECUTOR"),
        ("Image classification", "mock_executor_d://document/process", "LLM"),
        ("Visual QA", "mock_executor_e://document/process", "LLM"),
        ("Image captioning", "mock_executor_f://document/process", "LLM"),
    ]

    for name, endpoint, def_type in image_task_configs:
        if def_type == "LLM":
            definition = LlmQueryDefinition(
                model_name="qwen_v2_5_vl",
                endpoint=endpoint,
                params={"layout": layout, "key": name.lower().replace(" ", "_")},
            )
        else:
            definition = ExecutorEndpointQueryDefinition(
                endpoint=endpoint,
                params={"layout": layout, "key": name.lower().replace(" ", "_")},
            )

        task = Query(
            task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
            query_str=f"{planner_info.current_id}: IMAGE: {name}",
            dependencies=[image_subgraph_root.task_id],
            node_type=QueryType.COMPUTE,
            definition=definition,
        )
        planner_info.current_id += 1
        image_tasks.append(task)

    image_subgraph_end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: IMAGE_SUBGRAPH: Merge image results",
        dependencies=[t.task_id for t in image_tasks],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout, "subgraph": "image"}),
    )
    planner_info.current_id += 1

    # SUBGRAPH 3: Data Analysis Pipeline
    # This subgraph handles structured data analysis, statistics, and insights
    data_subgraph_root = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: DATA_SUBGRAPH: Start data analysis",
        dependencies=[init_node.task_id],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout, "subgraph": "data"}),
    )
    planner_info.current_id += 1

    data_tasks = []
    data_task_configs = [
        ("Statistical analysis", "mock_executor_e://document/process", "EXECUTOR"),
        ("Pattern recognition", "mock_executor_f://document/process", "LLM"),
        ("Anomaly detection", "mock_executor_g://document/process", "EXECUTOR"),
    ]

    for name, endpoint, def_type in data_task_configs:
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

        task = Query(
            task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
            query_str=f"{planner_info.current_id}: DATA: {name}",
            dependencies=[data_subgraph_root.task_id],
            node_type=QueryType.COMPUTE,
            definition=definition,
        )
        planner_info.current_id += 1
        data_tasks.append(task)

    data_subgraph_end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: DATA_SUBGRAPH: Merge data results",
        dependencies=[t.task_id for t in data_tasks],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout, "subgraph": "data"}),
    )
    planner_info.current_id += 1

    # MERGE ALL SUBGRAPHS - Combines results from all three parallel subgraphs
    global_merge = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: GLOBAL_MERGE: Merge all subgraph results",
        dependencies=[
            text_subgraph_end.task_id,
            image_subgraph_end.task_id,
            data_subgraph_end.task_id,
        ],
        node_type=QueryType.MERGER,
        definition=NoopQueryDefinition(params={"layout": layout, "merge_type": "global"}),
    )
    planner_info.current_id += 1

    # POST-PROCESSING - Final processing of combined results
    post_process = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Post-process combined results",
        dependencies=[global_merge.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_g://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1

    # VALIDATION - Validate final output quality
    validate = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Validate final output",
        dependencies=[post_process.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="mock_executor_h://document/process",
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[validate.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    # Build complete node list
    all_nodes = (
        [root, init_node]
        + [text_subgraph_root]
        + text_tasks
        + [text_subgraph_end]
        + [image_subgraph_root]
        + image_tasks
        + [image_subgraph_end]
        + [data_subgraph_root]
        + data_tasks
        + [data_subgraph_end]
        + [global_merge, post_process, validate, end]
    )

    return QueryPlan(nodes=all_nodes)


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

    Args:
        planner_info: Contains configuration and state information
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A plan with simple branching based on document type
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

    Args:
        planner_info: Contains configuration and state information
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A plan with SWITCH-based routing by complexity
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

    Args:
        planner_info: Contains configuration and state information
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A plan with complex multi-condition branching
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

    Args:
        planner_info: Contains configuration and state information
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A plan with nested branching logic
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
    # We need to create this first to get the ID, then define paths
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
    # This needs to be created BEFORE branch2 because it was pre-allocated at current_id + 2
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

    Args:
        planner_info: Contains configuration and state information
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A plan with Python function-based branching
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
    # The function path should be a module path to a callable
    # In real usage: "my_app.business_rules.is_premium_document"
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

    This demonstrates advanced JSONPath features:
    - Array length checks
    - Nested field access
    - Array membership testing
    - Complex field extraction

    Args:
        planner_info: Contains configuration and state information
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A plan with advanced JSONPath expressions
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
    # Note: These use jsonpath-ng-ext extended syntax
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

    Args:
        planner_info: Contains configuration and state information
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A plan with ALL_MATCHING evaluation mode
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
    # Multiple paths can be activated if their conditions match
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

    Args:
        planner_info: Contains configuration and state information
        **kwargs: Additional keyword arguments

    Returns:
        QueryPlan: A plan with regex-based branching
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
        ("mock_parallel_subgraphs", "Mock Plan with Parallel Subgraphs"),
        ("mock_branch_simple", "Simple Branching Mock Plan (JSONPath)"),
        ("mock_switch_complexity", "SWITCH-based Complexity Routing"),
        ("mock_branch_multi_condition", "Multi-Condition Branching (AND/OR)"),
        ("mock_nested_branches", "Nested Branching (Branch within Branch)"),
        ("mock_branch_python_function", "Python Function Branching"),
        ("mock_branch_jsonpath_advanced", "Advanced JSONPath Expressions"),
        ("mock_branch_all_match", "ALL_MATCH Evaluation Mode"),
        ("mock_branch_regex_matching", "Regex Pattern Matching"),
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
