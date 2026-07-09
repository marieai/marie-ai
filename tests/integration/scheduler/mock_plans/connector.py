"""
Connector Mock Query Plans (W2)

A CONNECTOR node serialized per the W2 wire contract: ``method ==
"EXECUTOR_ENDPOINT"`` with the FIXED endpoint ``plugin_daemon_executor://execute``
and ALL plugin identity + credential requirements inside ``params`` (``params.plugin``).
The scheduler routes the ``plugin_daemon_executor`` scheme to
``MariePluginDaemonExecutor``'s ``/execute`` handler, which reads ``params.plugin``,
resolves credentials, and dispatches to the marie-plugin-daemon.

This is the EXECUTOR_ENDPOINT form Studio actually emits — distinct from
``mock_plans/plugin.py`` (the typed ``method == "PLUGIN"`` route, the rejected
wire alternative kept only for reference).

Plans:
    - query_planner_mock_connector_tool: START -> CONNECTOR(EXECUTOR_ENDPOINT) -> END
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

# Fixed, routing-only endpoint (identity rides in params.plugin, not the path).
CONNECTOR_ENDPOINT = "plugin_daemon_executor://execute"


def _connector_plan(
    planner_info: PlannerInfo,
    *,
    plugin_ref: str,
    tool_ref: str,
    resource: str,
    operation: str,
) -> QueryPlan:
    """Build a minimal START -> CONNECTOR -> END plan.

    The CONNECTOR node uses ``ExecutorEndpointQueryDefinition`` with the fixed
    ``plugin_daemon_executor://execute`` endpoint; the daemon identity lives in
    ``params.plugin`` (``tool_ref`` + ``package_ref`` + ``package_digest`` are
    mandatory at dispatch).
    """
    base_id = planner_info.base_id
    layout = planner_info.name

    root = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: START",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )
    planner_info.current_id += 1

    connector = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: CONNECTOR {plugin_ref}/{operation}",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint=CONNECTOR_ENDPOINT,
            params={
                "layout": layout,
                "resource": resource,
                "operation": operation,
                "plugin": {
                    "tool_ref": tool_ref,
                    "package_ref": plugin_ref,
                    "package_digest": "sha256:mockdigest",
                    "package_trust_level": "community",
                    "install_id": "mock-install",
                    "provider_id": "mock-provider",
                    "package_id": "mock-package",
                    "action_type": "tool",
                },
                "credential_requirements": [],
            },
        ),
    )
    planner_info.current_id += 1

    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[connector.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    return QueryPlan(nodes=[root, connector, end])


@register_query_plan("mock_connector_tool")
def query_planner_mock_connector_tool(
    planner_info: PlannerInfo, **kwargs
) -> QueryPlan:
    """CONNECTOR route: START -> CONNECTOR(EXECUTOR_ENDPOINT, plugin daemon) -> END."""
    return _connector_plan(
        planner_info,
        plugin_ref="ext.m3forge.reader",
        tool_ref="web_reader",
        resource="pages",
        operation="web_reader",
    )
