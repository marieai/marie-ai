import pytest

from marie.query_planner.base import PlannerInfo


def test_schema_extract_planner_routes_ocr_then_configured_annotation() -> None:
    from marie.query_planner.schema_extract_planner import query_planner_schema_extract

    plan = query_planner_schema_extract(
        PlannerInfo(
            name='schema_extract',
            base_id='0' * 32,
            metadata={
                'run_params': {
                    'layout': 'claims',
                    'key': 'extract',
                    'parse_mode': 'ocr',
                }
            },
        )
    )
    _, ocr, extract, end = plan.nodes
    assert ocr.definition.endpoint == 'extract_executor://document/extract'
    assert ocr.definition.params == {
        'layout': 'claims',
        'parse_mode': 'ocr',
        'strict_result': True,
    }
    assert extract.definition.endpoint == 'annotator_llm_executor://annotator/schema'
    assert extract.definition.params == {'layout': 'claims', 'key': 'extract'}
    assert extract.dependencies == [ocr.task_id]
    assert end.dependencies == [extract.task_id]


@pytest.mark.parametrize(
    'run_params',
    [
        {},
        {'layout': 'claims'},
        {'key': 'extract'},
        {'layout': 'claims', 'key': 'extract', 'extractor_schema': {}},
        {'layout': 'claims', 'key': 'extract', 'prompt': 'inline'},
    ],
)
def test_schema_extract_requires_saved_profile_and_rejects_inline_definitions(
    run_params,
) -> None:
    from marie.query_planner.schema_extract_planner import query_planner_schema_extract

    with pytest.raises(ValueError):
        query_planner_schema_extract(
            PlannerInfo(
                name='schema_extract',
                base_id='0' * 32,
                metadata={'run_params': run_params},
            )
        )


def test_builtin_registration_makes_schema_extract_available(monkeypatch) -> None:
    from marie.query_planner.base import QueryPlanRegistry
    from marie.query_planner.builtin import register_all_known_planners
    from marie.query_planner.model import QueryPlannersConf

    monkeypatch.setattr(QueryPlanRegistry, '_plans', {})
    monkeypatch.setattr(QueryPlanRegistry, '_metadata', {})
    monkeypatch.setattr(QueryPlanRegistry, '_id_to_name', {})
    register_all_known_planners(QueryPlannersConf(planners=[]))
    assert 'schema_extract' in QueryPlanRegistry.list_planners()
