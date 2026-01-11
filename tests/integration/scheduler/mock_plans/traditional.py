"""
Traditional Mock Query Plans

Linear and parallel execution patterns for basic performance testing.

Plans:
    - query_planner_mock_simple: Basic linear execution (3 nodes)
    - query_planner_mock_medium: Parallel execution with merge (7 nodes)
    - query_planner_mock_complex: Complex multi-stage pipeline (12 nodes)
    - query_planner_mock_with_subgraphs: Nested subgraph pattern
    - query_planner_mock_parallel_subgraphs: Parallel subgraphs (20+ nodes)
"""

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
            query_str=f"Parallel task {i + 1}",
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
