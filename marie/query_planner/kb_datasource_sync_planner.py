from marie.job.job_manager import increment_uuid7str
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryType,
)


def query_planner_kb_datasource_sync(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """Run a saved datasource binding on the Plugin Executor."""
    nodes = []
    for label, definition in (
        ('START', NoopQueryDefinition()),
        (
            'SYNC datasource',
            ExecutorEndpointQueryDefinition(
                endpoint='plugin_daemon_executor://execute',
                params={'operation': 'kb_datasource_sync'},
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
