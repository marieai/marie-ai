"""
Mock plans for KB document indexing workflows.

Plans for testing the KB indexing pipeline with different document types
and processing paths (parsed vs frames/OCR).
"""

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


@register_query_plan("mock_kb_indexing_simple")
def query_planner_mock_kb_indexing_simple(
    planner_info: PlannerInfo, **kwargs
) -> QueryPlan:
    """
    Basic KB indexing pipeline (4 nodes).

    Structure:
        START -> EXTRACT -> EMBED_AND_STORE -> END

    Tests: Linear execution, format-aware extraction, vector storage
    """
    base_id = planner_info.base_id
    config = kwargs.get("config", {})
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

    # EXTRACT - format-aware document extraction
    extract = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: EXTRACT text (format-aware)",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="extract_executor://document/extract",
            params={
                "uri": config.get("uri"),
                "ref_id": config.get("ref_id"),
                "ref_type": config.get("ref_type", "document"),
                "ocr_fallback": True,
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(extract)

    # EMBED_AND_STORE
    embed = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: EMBED and STORE",
        dependencies=[extract.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="vector_store_executor://embed_and_store",
            params={
                "source_id": config.get("source_id"),
                "index_name": config.get("index_name", "default"),
                "node_type": config.get("node_type", "document"),
                "ref_doc_id": config.get("ref_id"),
                "workflow_id": config.get("workflow_id"),
                "ref_id": config.get("ref_id"),
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(embed)

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[embed.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_kb_indexing_with_chunking")
def query_planner_mock_kb_indexing_with_chunking(
    planner_info: PlannerInfo, **kwargs
) -> QueryPlan:
    """
    KB indexing with explicit chunking stage (5 nodes).

    Structure:
        START -> EXTRACT -> CHUNK -> EMBED_AND_STORE -> END

    Tests: Separate chunking step before embedding
    """
    base_id = planner_info.base_id
    config = kwargs.get("config", {})
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

    # EXTRACT
    extract = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: EXTRACT text",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="extract_executor://document/extract",
            params={
                "uri": config.get("uri"),
                "ref_id": config.get("ref_id"),
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(extract)

    # CHUNK - explicit chunking stage
    chunk = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: CHUNK text",
        dependencies=[extract.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="chunking_executor://chunk",
            params={
                "chunk_size": config.get("chunk_size", 512),
                "chunk_overlap": config.get("chunk_overlap", 50),
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(chunk)

    # EMBED_AND_STORE
    embed = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: EMBED and STORE",
        dependencies=[chunk.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="vector_store_executor://embed_and_store",
            params={
                "source_id": config.get("source_id"),
                "index_name": config.get("index_name", "default"),
                "workflow_id": config.get("workflow_id"),
                "ref_id": config.get("ref_id"),
            },
        ),
    )
    planner_info.current_id += 1
    nodes.append(embed)

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[embed.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)


@register_query_plan("mock_kb_indexing_noop")
def query_planner_mock_kb_indexing_noop(
    planner_info: PlannerInfo, **kwargs
) -> QueryPlan:
    """
    Minimal KB indexing pipeline (2 nodes) - for testing only.

    Structure:
        START -> END

    Tests: Pipeline setup without actual processing
    """
    base_id = planner_info.base_id
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

    # END
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    nodes.append(end)

    return QueryPlan(nodes=nodes)
