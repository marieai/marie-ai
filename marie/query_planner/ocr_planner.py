from marie.job.job_manager import increment_uuid7str
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryType,
    register_query_plan,
)


@register_query_plan("extract")
def query_planner_extract(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Plan a structured query execution graph for document OCR extraction.
    :param planner_info: A transfer object that holds relevant information for constructing a structured query execution graph for document OCR extraction.
    :param kwargs: Additional keyword arguments for future use.
    :return: The QueryPlan for the structured query execution graph.

    Example:
    START -> ROOT (Annotator) -> Frame1 -> Frame2 -> MERGE -> COLLATOR -> END

    """

    layout = planner_info.name
    base_id = planner_info.base_id
    # Root node for the entire process
    root = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    planner_info.current_id += 1

    segment_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: EXTRACT data",
        dependencies=[root.task_id],  # Depends on global merged results
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="extract_executor://document/extract",
            # this is the endpoint for the extraction service, we will need to make this configurable in the future
            params={
                'layout': layout,
            },
        ),
    )
    planner_info.current_id += 1

    end_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[segment_node.task_id],  # Depends on extraction step
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(params={'layout': layout}),
    )

    return QueryPlan(nodes=[root] + [segment_node] + [end_node])
