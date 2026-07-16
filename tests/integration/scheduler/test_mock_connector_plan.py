"""W2 acceptance (plan shape): the CONNECTOR mock plan serializes to the locked
wire contract — method EXECUTOR_ENDPOINT, the fixed endpoint
``plugin_daemon_executor://execute``, and all plugin identity + credential
requirements inside ``params`` (``params.plugin``).

This is the plan→dispatch contract the scheduler routes to
``MariePluginDaemonExecutor.connector_invoke`` (see
tests/unit/executor/extensions/test_plugin_daemon_executor.py for the executor
side and tests/unit/job/test_gateway_job_distributor.py for the routing).
"""

from marie.query_planner import mock_query_plans
from marie.query_planner.base import ExecutorEndpointQueryDefinition
from marie.query_planner.mock_query_plans import (
    PlannerInfo,
    QueryPlanRegistry,
    generate_job_id,
)


def test_mock_connector_planner_is_exported() -> None:
    assert hasattr(mock_query_plans, "query_planner_mock_connector_tool")
    assert QueryPlanRegistry.get("mock_connector_tool") is not None


def test_mock_connector_serializes_to_plugin_daemon_execute() -> None:
    planner = QueryPlanRegistry.get("mock_connector_tool")
    assert planner is not None

    plan = planner(PlannerInfo(name="mock_connector_tool", base_id=generate_job_id()))

    connectors = [
        node
        for node in plan.nodes
        if getattr(node.definition, "endpoint", None)
        == "plugin_daemon_executor://execute"
    ]
    assert len(connectors) == 1

    definition = connectors[0].definition
    assert isinstance(definition, ExecutorEndpointQueryDefinition)
    assert definition.method == "EXECUTOR_ENDPOINT"

    # Identity rides in params (mapper.from_task forwards only params); the
    # endpoint is routing-only and carries none.
    plugin = definition.params["plugin"]
    assert plugin["package_ref"] == "ext.m3forge.reader"
    assert plugin["tool_ref"] == "web_reader"
    assert plugin["package_digest"]  # mandatory at dispatch (envelope raises without it)
    assert plugin["action_type"] == "tool"
    assert definition.params["operation"] == "web_reader"
    assert "credential_requirements" in definition.params
