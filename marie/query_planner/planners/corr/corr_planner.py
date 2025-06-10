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

PLAN_ID = "corr"


### All Nodes
@register_query_plan(PLAN_ID)
# QueryPlanRegistry.register(PLAN_ID, query_planner_corr)
def query_planner_corr(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Plan a structured query execution graph for Corr classification and indexing.
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

    med_classify_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: CLASSIFY medical pages",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="med_page_classify_executor://document/classify",
            params={
                'layout': layout,
            },
        ),
    )
    planner_info.current_id += 1

    corr_classify_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: CLASSIFY corr routing by page",
        dependencies=[med_classify_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="corr_routing_executor://document/classify",
            params={
                'layout': layout,
            },
        ),
    )
    planner_info.current_id += 1

    llm_index_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: INDEX corr document data",
        dependencies=[corr_classify_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="corr_indexing_executor://document/index",
            params={
                'layout': layout,
            },
        ),
    )
    planner_info.current_id += 1

    end_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[llm_index_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(params={'layout': layout}),
    )

    return QueryPlan(
        nodes=[root]
        + [med_classify_node, corr_classify_node, llm_index_node]
        + [end_node]
    )


### Classify Only
@register_query_plan(f"{PLAN_ID}-classify")
def query_planner_corr_classify(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """
    Plan a structured query execution graph for Corr classification and indexing.
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

    med_classify_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: CLASSIFY medical pages",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="med_page_classify_executor://document/classify",
            params={
                'layout': layout,
            },
        ),
    )
    planner_info.current_id += 1

    corr_classify_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: CLASSIFY corr routing by page",
        dependencies=[med_classify_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="corr_routing_executor://document/classify",
            params={
                'layout': layout,
            },
        ),
    )
    planner_info.current_id += 1

    end_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[corr_classify_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(params={'layout': layout}),
    )

    return QueryPlan(
        nodes=[root] + [med_classify_node, corr_classify_node] + [end_node]
    )


### LLM index graph
@register_query_plan(f"{PLAN_ID}-index")
def query_planner_corr_index(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """LLM ONLY"""

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

    llm_index_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: INDEX corr document data",
        # dependencies=[corr_classify_node.task_id],
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="corr_indexing_executor://document/index",
            params={
                'layout': layout,
            },
        ),
    )
    planner_info.current_id += 1

    end_node = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[llm_index_node.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(params={'layout': layout}),
    )

    return QueryPlan(nodes=[root] + [llm_index_node] + [end_node])


if __name__ == "__main__":
    question = "Annotate documents with named entities."
    # this can be registered via a decorator
    QueryPlanRegistry.register(PLAN_ID, query_planner_corr)
    print(QueryPlanRegistry.list_planners())

    planner_info = PlannerInfo(name=PLAN_ID, base_id=generate_job_id())
    plan = query_planner(planner_info)
    pprint(plan.model_dump())
    visualize_query_plan_graph(plan)

    sorted_nodes = topological_sort(plan)
    print_sorted_nodes(sorted_nodes, plan)
    print_query_plan(plan, PLAN_ID)
