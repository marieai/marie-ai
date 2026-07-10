"""
KB Indexing Query Planner.

Creates DAGs for indexing documents into a knowledge base's vector store.
Uses TextExtractionExecutor (the deployed OCR/extraction executor, same one
ocr_planner.py targets) for extraction and VectorStoreExecutor for embedding
and storage.

Runtime parameters (provided at job submission, NOT at plan creation):
    uri: S3 or file URI of the document
    ref_id: Document reference ID
    ref_type: Document type classification
    source_id: KB source filter (e.g., "submission:s1")
    index_name: Vector store index name
    node_type: Node classification (document, image, text)
    workflow_id: Workflow record ID for status updates
"""

from marie.job.job_manager import generate_job_id, increment_uuid7str
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryType,
)

PLAN_ID = "kb_indexing"


# Note: Registration is done in builtin.py via QueryPlanRegistry.register()
def query_planner_kb_indexing(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Plan a query execution graph for KB document indexing.

    Uses TextExtractionExecutor (extract_executor://document/extract) for
    OCR-based extraction - the same executor ocr_planner.py targets.

    Pipeline: START -> EXTRACT -> EMBED_AND_STORE -> END

    Note: Runtime parameters (uri, ref_id, source_id, etc.) are provided
    at job submission time, not during plan creation. The planner only
    defines the DAG structure and static configuration.
    """
    layout = planner_info.name
    base_id = planner_info.base_id

    metadata = planner_info.metadata or {}
    run_params = metadata.get("run_params", {})

    nodes = []

    # START node
    start_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1
    nodes.append(start_node)

    # EXTRACT node - uses TextExtractionExecutor, the deployed extraction
    # executor (same node ocr_planner.py targets). document_backend_executor
    # was never part of a real deployment and is not used here.
    #
    # Param vocabulary of TextExtractionExecutor.extract (/document/extract):
    #   - "layout": consumed today, but only via op_params for the tracing
    #     span's LAYOUT_ID attribute - not read by the extraction logic itself.
    #   - real business params it reads from `payload` (none populated by this
    #     planner yet): "regions" (bounding boxes), "format" (coordinate
    #     format), "mode" (PSMode), "return_ocr", "features" (pipeline config).
    #   - "parse_mode", "layout_options", "cache_options": ride along inert -
    #     TextExtractionExecutor does not read any of them today. Kept for
    #     forward-compat / observability only.
    # Runtime params (uri, ref_id, ref_type) provided at job submission
    extract_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: EXTRACT text (format-aware)",
        dependencies=[start_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="extract_executor://document/extract",
            params={
                "layout": layout,
                **(
                    {"parse_mode": run_params["parse_mode"]}
                    if "parse_mode" in run_params
                    else {}
                ),
                **(
                    {"layout_options": run_params["layout_options"]}
                    if "layout_options" in run_params
                    else {}
                ),
                **(
                    {"cache_options": run_params["cache_options"]}
                    if "cache_options" in run_params
                    else {}
                ),
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(extract_node)

    # EMBED_AND_STORE node - uses VectorStoreExecutor
    # Runtime params (source_id, index_name, ref_id, etc.) provided at job submission
    embed_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: EMBED and STORE",
        dependencies=[extract_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="vector_store_executor://embed_and_store",
            params={
                "layout": layout,
                **(
                    {"source_id": metadata["source_id"]}
                    if "source_id" in metadata
                    else {}
                ),
                **(
                    {"index_name": metadata["index_name"]}
                    if "index_name" in metadata
                    else {}
                ),
                **(
                    {"multimodal": run_params["multimodal"]}
                    if "multimodal" in run_params
                    else {}
                ),
                "node_type": "document",
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(embed_node)

    # END node
    end_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[embed_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end_node)

    return QueryPlan(nodes=nodes)


if __name__ == "__main__":
    from pprint import pprint

    from marie.query_planner.base import QueryPlanRegistry
    from marie.query_planner.planner import (
        print_query_plan,
        query_planner,
        visualize_query_plan_graph,
    )

    # Register the planner for testing (normally done via builtin.py)
    QueryPlanRegistry.register(PLAN_ID, query_planner_kb_indexing)

    # Test the planner - runtime params (uri, ref_id, etc.) are NOT passed here
    # They are provided at job submission time via the job scheduler
    planner_info = PlannerInfo(name=PLAN_ID, base_id=generate_job_id())

    plan = query_planner(planner_info)
    pprint(plan.model_dump())
    visualize_query_plan_graph(plan)
    print_query_plan(plan, PLAN_ID)
