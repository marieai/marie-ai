"""
Plugin mock query plans.

Minimal single-node plugin routes, one per plugin family. Each plan exercises a
PLUGIN node that the scheduler dispatches to the plugin daemon executor
(``MariePluginDaemonExecutor``), which runs the installed plugin through the
marie-plugin-daemon.

Plans:
    - query_planner_mock_plugin_tool: START -> PLUGIN(tool) -> END
    - query_planner_mock_plugin_model: START -> PLUGIN(model) -> END
    - query_planner_mock_plugin_datasource: START -> PLUGIN(datasource) -> END
    - query_planner_mock_plugin_trigger: START -> PLUGIN(trigger) -> END
"""

from .base import (
    NoopQueryDefinition,
    PlannerInfo,
    PluginQueryDefinition,
    Query,
    QueryPlan,
    QueryType,
    increment_uuid7str,
    register_query_plan,
)


def _plugin_plan(
    planner_info: PlannerInfo,
    plugin_type: str,
    plugin_ref: str,
    action: str,
) -> QueryPlan:
    """
    Build a minimal START -> PLUGIN -> END plan for a single plugin family.

    The PLUGIN node carries a PluginQueryDefinition whose ``method == "PLUGIN"``
    is the routing signal for the plugin daemon executor; the endpoint encodes
    the executor scheme so the DAG is self-describing.
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

    # PLUGIN node
    invoke = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: Invoke {plugin_type} plugin {plugin_ref}/{action}",
        dependencies=[root.task_id],
        node_type=QueryType.COMPUTE,
        definition=PluginQueryDefinition(
            endpoint=f"plugin_daemon_executor://{plugin_ref}/{action}",
            plugin_type=plugin_type,
            plugin_ref=plugin_ref,
            action=action,
            params={"layout": layout},
        ),
    )
    planner_info.current_id += 1

    # END node
    end = Query(
        task_id=f"{increment_uuid7str(base_id, planner_info.current_id)}",
        query_str=f"{planner_info.current_id}: END",
        dependencies=[invoke.task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(),
    )

    return QueryPlan(nodes=[root, invoke, end])


@register_query_plan("mock_plugin_tool")
def query_planner_mock_plugin_tool(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """Tool-plugin route: START -> PLUGIN(tool) -> END."""
    return _plugin_plan(
        planner_info,
        plugin_type="tool",
        plugin_ref="ext.m3forge.reader",
        action="web_reader",
    )


@register_query_plan("mock_plugin_model")
def query_planner_mock_plugin_model(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """Model-plugin route: START -> PLUGIN(model) -> END."""
    return _plugin_plan(
        planner_info,
        plugin_type="model",
        plugin_ref="ext.m3forge.openai",
        action="text_embedding",
    )


@register_query_plan("mock_plugin_datasource")
def query_planner_mock_plugin_datasource(
    planner_info: PlannerInfo, **kwargs
) -> QueryPlan:
    """Datasource-plugin route: START -> PLUGIN(datasource) -> END."""
    return _plugin_plan(
        planner_info,
        plugin_type="datasource",
        plugin_ref="ext.m3forge.notion",
        action="fetch_pages",
    )


@register_query_plan("mock_plugin_trigger")
def query_planner_mock_plugin_trigger(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """Trigger-plugin route: START -> PLUGIN(trigger) -> END."""
    return _plugin_plan(
        planner_info,
        plugin_type="trigger",
        plugin_ref="ext.m3forge.schedule",
        action="cron_tick",
    )
