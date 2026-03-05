"""
RAG Indexing Query Planner.

Creates DAGs for indexing documents into vector stores.
Uses document backend for format-aware extraction and
VectorStoreExecutor for embedding and storage.
"""

from marie.job.job_manager import generate_job_id, increment_uuid7str
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryPlanRegistry,
    QueryType,
    register_query_plan,
)

PLAN_ID = "rag_indexing"


@register_query_plan(PLAN_ID)
def query_planner_rag_indexing(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Plan a query execution graph for RAG document indexing.

    Uses document backend for format-aware extraction:
    - Parsed mode (DOCX, XLSX, etc.): Direct text extraction (fast)
    - Frames mode (PDF, images): OCR pipeline (accurate)

    Pipeline: START -> EXTRACT -> EMBED_AND_STORE -> END

    Config parameters (from run_config):
        uri: S3 or file URI of the document
        ref_id: Document reference ID
        ref_type: Document type classification
        source_id: RAG source filter (e.g., "submission:s1")
        index_name: Vector store index name
        node_type: Node classification (document, image, text)
        workflow_id: Workflow record ID for status updates
    """
    base_id = planner_info.base_id
    config = kwargs.get("config", {})

    # Validate required params
    required = ["uri", "ref_id", "source_id", "index_name"]
    missing = [p for p in required if not config.get(p)]
    if missing:
        raise ValueError(f"RAG indexing planner missing required params: {missing}")

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

    # EXTRACT node - uses DocumentBackendExecutor
    # Automatically routes to:
    # - Parsed mode: DOCX, XLSX, PPTX, HTML, Markdown, CSV, Email, EPUB
    # - Frames mode + OCR: PDF, images, legacy Office, LaTeX, RST, DjVu
    extract_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: EXTRACT text (format-aware)",
        dependencies=[start_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="document_backend_executor://extract",
            params={
                "uri": config.get("uri"),
                "ref_id": config.get("ref_id"),
                "ref_type": config.get("ref_type", "document"),
                "ocr_fallback": True,
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(extract_node)

    # EMBED_AND_STORE node - uses VectorStoreExecutor
    embed_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: EMBED and STORE",
        dependencies=[extract_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="vector_store_executor://embed_and_store",
            params={
                "source_id": config.get("source_id"),
                "index_name": config.get("index_name", "default"),
                "node_type": config.get("node_type", "document"),
                "ref_doc_id": config.get("ref_id"),
                # Workflow tracking for status updates
                "workflow_id": config.get("workflow_id"),
                "ref_id": config.get("ref_id"),
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

    from marie.query_planner.planner import (
        print_query_plan,
        query_planner,
        visualize_query_plan_graph,
    )

    # Test the planner
    planner_info = PlannerInfo(name=PLAN_ID, base_id=generate_job_id())
    config = {
        "uri": "s3://test-bucket/tenants/t1/submissions/s1/doc.pdf",
        "ref_id": "doc_001",
        "ref_type": "submission_document",
        "source_id": "submission:s1",
        "index_name": "test_index",
        "node_type": "document",
        "workflow_id": "wf_001",
    }

    plan = query_planner(planner_info, config=config)
    pprint(plan.model_dump())
    visualize_query_plan_graph(plan)
    print_query_plan(plan, PLAN_ID)
