from __future__ import annotations

from marie.job.job_manager import increment_uuid7str
from marie.query_planner.base import (
    ExecutorEndpointQueryDefinition,
    NoopQueryDefinition,
    PlannerInfo,
    Query,
    QueryPlan,
    QueryType,
)

PLAN_ID = 'schema_extract'


def query_planner_schema_extract(planner_info: PlannerInfo, **kwargs) -> QueryPlan:
    """Run OCR followed by extraction from the current saved layout/key definition."""
    run_params = (planner_info.metadata or {}).get('run_params') or {}
    layout, key = run_params.get('layout'), run_params.get('key')
    if not isinstance(layout, str) or not layout or not isinstance(key, str) or not key:
        raise ValueError('schema_extract requires saved layout and key run parameters')
    if set(run_params) - {'layout', 'key', 'parse_mode', 'processor_id', 'document_id'}:
        raise ValueError(
            'schema_extract accepts a saved profile and document identity, without inline definitions'
        )
    if run_params.get('parse_mode', 'ocr') != 'ocr':
        raise ValueError('schema_extract requires OCR parsing')

    nodes = []
    definitions = [
        ('START', NoopQueryDefinition()),
        (
            'OCR',
            ExecutorEndpointQueryDefinition(
                endpoint='extract_executor://document/extract',
                params={'layout': layout, 'parse_mode': 'ocr', 'strict_result': True},
            ),
        ),
        (
            'SCHEMA EXTRACT',
            ExecutorEndpointQueryDefinition(
                endpoint='annotator_llm_executor://annotator/schema',
                params={'layout': layout, 'key': key},
            ),
        ),
        ('END', NoopQueryDefinition()),
    ]
    for label, definition in definitions:
        nodes.append(
            Query(
                task_id=increment_uuid7str(
                    planner_info.base_id, planner_info.current_id
                ),
                query_str=f'{planner_info.current_id}: {label}',
                dependencies=[nodes[-1].task_id] if nodes else [],
                node_type=QueryType.COMPUTE,
                definition=definition,
            )
        )
        planner_info.current_id += 1
    return QueryPlan(nodes=nodes)
