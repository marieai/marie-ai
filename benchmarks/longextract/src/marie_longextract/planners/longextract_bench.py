from __future__ import annotations

from typing import Any

from marie_longextract.ops.schema import build_extraction_units
from marie_longextract.planners import PLAN_ID
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


def _main() -> None:
    import argparse
    from pathlib import Path
    from pprint import pprint

    from marie.job.job_manager import generate_job_id
    from marie.query_planner import QueryPlanRegistry
    from marie.query_planner.planner import (
        print_query_plan,
        print_sorted_nodes,
        query_planner,
        topological_sort,
        visualize_query_plan_graph,
    )

    parser = argparse.ArgumentParser(description="Visualize the LongExtract query plan")
    parser.add_argument(
        "--schema",
        required=True,
        type=Path,
        help="Path to a LongExtract JSON schema",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("query_plan_graph.png"),
        help="PNG path for the rendered query plan graph",
    )
    args = parser.parse_args()

    schema_path = args.schema.expanduser().resolve()
    if not schema_path.is_file():
        parser.error(f"schema does not exist: {schema_path}")

    output_path = args.output.expanduser().resolve()
    artifact_root = output_path.parent.resolve()
    planner_info = PlannerInfo(
        name=PLAN_ID,
        base_id=generate_job_id(),
        metadata={
            "content_type": "application/pdf",
            "benchmark": {
                "schema_uri": str(schema_path),
                "output_uri": (artifact_root / "result.json").as_uri(),
                "work_uri": f"{(artifact_root / 'work').as_uri()}/",
            },
        },
    )

    print(QueryPlanRegistry.list_planners())
    plan = query_planner(planner_info)
    pprint(plan.model_dump())
    visualize_query_plan_graph(plan, output_path=str(output_path))

    sorted_nodes = topological_sort(plan)
    print_sorted_nodes(sorted_nodes, plan)
    print_query_plan(plan, PLAN_ID)


if __name__ == "__main__":
    _main()
