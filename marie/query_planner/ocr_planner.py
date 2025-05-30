from pprint import pprint

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
from marie.query_planner.planner import (
    print_query_plan,
    print_sorted_nodes,
    query_planner,
    topological_sort,
    visualize_query_plan_graph,
)

PLAN_ID = "extract"


# @register_query_plan(PLAN_ID)
# QueryPlanRegistry.register(PLAN_ID, query_planner_extract)
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


if __name__ == "__main__":
    question = "Annotate documents with named entities."
    # this can alsow be registered via a decorator
    QueryPlanRegistry.register(PLAN_ID, query_planner_extract)
    print(QueryPlanRegistry.list_planners())

    planner_info = PlannerInfo(name=PLAN_ID, base_id=generate_job_id())
    plan = query_planner(planner_info)
    pprint(plan.model_dump())
    visualize_query_plan_graph(plan)

    sorted_nodes = topological_sort(plan)
    print_sorted_nodes(sorted_nodes, plan)
    print_query_plan(plan, PLAN_ID)
