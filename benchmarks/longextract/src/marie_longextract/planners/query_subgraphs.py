from __future__ import annotations

from collections.abc import Callable
from typing import Any

from marie_longextract.agents.unit_extract import build_unit_task_contract

from marie.query_planner import (
    ExecutorEndpointQueryDefinition,
    LlmQueryDefinition,
    PlannerInfo,
    Query,
    QueryType,
)

MODEL_NAME = "qwen_3_instruct"


def build_table_context_node(
    planner_info: PlannerInfo,
    node_id_factory: Callable[[PlannerInfo], str],
    layout: str,
    dependencies: list[str],
) -> Query:
    return Query(
        task_id=node_id_factory(planner_info),
        query_str="Detect document tables and continuations",
        dependencies=dependencies,
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name=MODEL_NAME,
            endpoint="/annotator/llm",
            params={"layout": layout, "key": "tables"},
        ),
    )


def build_aggregation_policy_node(
    planner_info: PlannerInfo,
    node_id_factory: Callable[[PlannerInfo], str],
    layout: str,
    dependencies: list[str],
    units: list[dict[str, Any]],
) -> Query:
    contracts = [build_unit_task_contract(unit) for unit in units]
    return Query(
        task_id=node_id_factory(planner_info),
        query_str="Compile LongExtract aggregation policy",
        dependencies=dependencies,
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name=MODEL_NAME,
            endpoint="/annotator/llm",
            params={
                "layout": layout,
                "key": "longextract-aggregation-policy",
                "extraction_units": contracts,
            },
        ),
    )


def build_extraction_subgraph(
    planner_info: PlannerInfo,
    node_id_factory: Callable[[PlannerInfo], str],
    layout: str,
    dependencies: list[str],
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    contracts = [build_unit_task_contract(unit) for unit in units]
    annotator = Query(
        task_id=node_id_factory(planner_info),
        query_str="Extract LongExtract schema units",
        dependencies=dependencies,
        node_type=QueryType.COMPUTE,
        definition=LlmQueryDefinition(
            model_name=MODEL_NAME,
            endpoint="/annotator/llm",
            params={
                "layout": layout,
                "key": "longextract-unit-extract",
                "extraction_units": contracts,
            },
        ),
    )
    parser = Query(
        task_id=node_id_factory(planner_info),
        query_str="Aggregate LongExtract continuation records",
        dependencies=[annotator.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="annotator_parser://annotator/result-parser",
            params={"layout": layout, "function": "longextract-aggregated"},
        ),
    )
    return {"nodes": [annotator, parser], "end": parser}
