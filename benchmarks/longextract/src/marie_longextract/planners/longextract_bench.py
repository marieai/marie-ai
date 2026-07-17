from __future__ import annotations

from typing import Any

from marie_longextract.ops.schema import build_extraction_units
from marie_longextract.planners.query_subgraphs import (
    build_aggregation_policy_node,
    build_extraction_subgraph,
    build_table_context_node,
)
from marie_longextract.tools.artifacts import read_json, require_benchmark_metadata

from marie.job.job_manager import increment_uuid7str
from marie.query_planner import (
    ExecutorEndpointQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryType,
    register_query_plan,
)

PLAN_ID = "longextract_bench"
LAYOUT_ID = "longextract-bench"


def _node_id(planner_info: PlannerInfo) -> str:
    value = increment_uuid7str(planner_info.base_id, planner_info.current_id)
    planner_info.current_id += 1
    return value


@register_query_plan(PLAN_ID)
def query_planner_longextract_bench(
    planner_info: PlannerInfo,
    **_kwargs: Any,
) -> QueryPlan:
    metadata = planner_info.metadata
    if not isinstance(metadata, dict):
        raise ValueError("LongExtractBench planner metadata is required")
    schema_uri, _output_uri, _work_uri = require_benchmark_metadata(metadata)
    schema = read_json(schema_uri)
    units = build_extraction_units(schema)
    if not units:
        raise ValueError("LongExtractBench schema produced no extraction units")

    start = Query(
        task_id=_node_id(planner_info),
        query_str="START LongExtractBench",
        dependencies=[],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(params={"layout": LAYOUT_ID}),
    )
    prepare = Query(
        task_id=_node_id(planner_info),
        query_str="Extract and prepare document context",
        dependencies=[start.task_id],
        node_type=QueryType.COMPUTE,
        definition=ExecutorEndpointQueryDefinition(
            endpoint="extract_executor://document/extract",
            params={"layout": LAYOUT_ID},
        ),
    )
    tables = build_table_context_node(
        planner_info=planner_info,
        node_id_factory=_node_id,
        layout=LAYOUT_ID,
        dependencies=[prepare.task_id],
    )
    aggregation_policy = build_aggregation_policy_node(
        planner_info=planner_info,
        node_id_factory=_node_id,
        layout=LAYOUT_ID,
        dependencies=[prepare.task_id],
        units=units,
    )
    extraction = build_extraction_subgraph(
        planner_info=planner_info,
        node_id_factory=_node_id,
        layout=LAYOUT_ID,
        dependencies=[tables.task_id, aggregation_policy.task_id],
        units=units,
    )
    end = Query(
        task_id=_node_id(planner_info),
        query_str="END LongExtractBench",
        dependencies=[extraction["end"].task_id],
        node_type=QueryType.COMPUTE,
        definition=NoopQueryDefinition(params={"layout": LAYOUT_ID}),
    )
    return QueryPlan(
        nodes=[
            start,
            prepare,
            tables,
            aggregation_policy,
            *extraction["nodes"],
            end,
        ]
    )
