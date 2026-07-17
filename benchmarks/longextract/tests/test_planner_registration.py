from __future__ import annotations

import marie_longextract.planners.longextract_bench  # noqa: F401

from marie.job.job_manager import generate_job_id
from marie.query_planner import PlannerInfo, QueryPlanRegistry
from marie.query_planner.mapper import JobMetadata


def _schema() -> dict:
    return {
        'type': 'object',
        'properties': {
            'claim_number': {'type': 'string'},
            'service_lines': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {'code': {'type': 'string'}},
                },
            },
        },
    }


def test_longextract_planner_registers() -> None:
    assert 'longextract_bench' in QueryPlanRegistry.list_planners()


def test_longextract_planner_builds_existing_extraction_route(monkeypatch) -> None:
    monkeypatch.setattr(
        'marie_longextract.planners.longextract_bench.read_json', lambda _uri: _schema()
    )
    planner = QueryPlanRegistry.get('longextract_bench')
    plan = planner(
        PlannerInfo(
            name='longextract_bench',
            base_id=generate_job_id(),
            metadata={
                'uri': 's3://bucket/doc.pdf',
                'content_type': 'application/pdf',
                'benchmark': {
                    'schema_uri': 's3://bucket/schema.json',
                    'work_uri': 's3://bucket/work/',
                    'output_uri': 's3://bucket/result.json',
                },
            },
        )
    )

    executor_endpoints = [
        node.definition.endpoint
        for node in plan.nodes
        if node.definition.method == 'EXECUTOR_ENDPOINT'
    ]
    assert executor_endpoints == [
        'extract_executor://document/extract',
        'annotator_parser://annotator/result-parser',
    ]
    prepare = next(
        node
        for node in plan.nodes
        if getattr(node.definition, 'endpoint', None)
        == 'extract_executor://document/extract'
    )
    assert 'parse_mode' not in prepare.definition.params

    unit_nodes = [
        node
        for node in plan.nodes
        if getattr(node.definition, 'endpoint', None) == '/annotator/llm'
        and node.definition.params['key'] == 'longextract-unit-extract'
    ]
    assert len(unit_nodes) == 1
    table_node = next(
        node
        for node in plan.nodes
        if getattr(node.definition, 'endpoint', None) == '/annotator/llm'
        and node.definition.params['key'] == 'tables'
    )
    policy_node = next(
        node
        for node in plan.nodes
        if getattr(node.definition, 'endpoint', None) == '/annotator/llm'
        and node.definition.params['key'] == 'longextract-aggregation-policy'
    )
    assert table_node.dependencies == [prepare.task_id]
    assert policy_node.dependencies == [prepare.task_id]
    assert unit_nodes[0].dependencies == [table_node.task_id, policy_node.task_id]
    assert unit_nodes[0].definition.model_name == 'qwen_3_instruct'
    extraction_units = unit_nodes[0].definition.params['extraction_units']
    assert [unit['unit_name'] for unit in extraction_units] == [
        'document_fields',
        'service_lines',
    ]
    assert all(unit['prompt_variables'] for unit in extraction_units)
    assert policy_node.definition.params['extraction_units'] == extraction_units
    assert 'output_name' not in unit_nodes[0].definition.params
    assert 'output_uri' not in unit_nodes[0].definition.params
    unit_metadata = JobMetadata.from_task(unit_nodes[0], 'longextract_bench').metadata
    assert unit_metadata.op_params['extraction_units'] == extraction_units

    parser_node = next(
        node
        for node in plan.nodes
        if getattr(node.definition, 'endpoint', None)
        == 'annotator_parser://annotator/result-parser'
    )
    assert parser_node.dependencies == [unit_nodes[0].task_id]
    assert parser_node.definition.params == {
        'layout': 'longextract-bench',
        'function': 'longextract-aggregated',
    }

    assert not any(node.definition.method == 'PYTHON_FUNCTION' for node in plan.nodes)
    assert not any(
        str(getattr(node.definition, 'endpoint', '')).startswith('longextract_bench://')
        for node in plan.nodes
    )

    routes = [
        JobMetadata.from_task(node, 'longextract_bench').metadata.on
        for node in plan.nodes
    ]
    assert 'extract_executor://document/extract' in routes
    assert 'annotator_llm://annotator/llm' in routes
    assert 'annotator_parser://annotator/result-parser' in routes
    assert not any('serverless' in route for route in routes)
