from marie.job.job_manager import increment_uuid7str
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryType,
)


def query_planner_kb_repository(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """Refresh a saved native repository source on the KB executor."""
    nodes = []
    for label, definition in (
        ('START', NoopQueryDefinition()),
        (
            'REFRESH repository',
            ExecutorEndpointQueryDefinition(
                endpoint='repository_executor://refresh_repository'
            ),
        ),
        ('END', NoopQueryDefinition()),
    ):
        nodes.append(
            Query(
                task_id=increment_uuid7str(
                    planner_info.base_id, planner_info.current_id
                ),
                query_str=label,
                dependencies=[nodes[-1].task_id] if nodes else [],
                node_type=QueryType.COMPUTE,
                definition=definition,
            )
        )
        planner_info.current_id += 1
    return QueryPlan(nodes=nodes)
