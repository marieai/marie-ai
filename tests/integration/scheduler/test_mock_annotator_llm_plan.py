from tests.integration.scheduler import mock_query_plans
from tests.integration.scheduler.mock_query_plans import (
    LlmQueryDefinition,
    PlannerInfo,
    QueryPlanRegistry,
    generate_job_id,
)


def test_mock_annotator_llm_planner_is_exported() -> None:
    assert hasattr(mock_query_plans, "query_planner_mock_annotator_llm")
    assert QueryPlanRegistry.get("mock_annotator_llm") is not None


def test_mock_annotator_llm_targets_production_shaped_endpoint() -> None:
    planner = QueryPlanRegistry.get("mock_annotator_llm")
    assert planner is not None

    plan = planner(PlannerInfo(name="mock_annotator_llm", base_id=generate_job_id()))
    executor_nodes = [
        node
        for node in plan.nodes
        if getattr(node.definition, "endpoint", None) == "annotator_llm://annotator/llm"
    ]

    assert len(executor_nodes) == 1
    assert isinstance(executor_nodes[0].definition, LlmQueryDefinition)
    assert executor_nodes[0].definition.model_name == "gpt-5.2-mock"
    assert executor_nodes[0].definition.params["layout"] == "mock-llm"
    assert executor_nodes[0].definition.params["key"] == "mock-llm"
