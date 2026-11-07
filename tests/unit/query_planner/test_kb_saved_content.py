import pytest

from marie.query_planner import kb_indexing_planner
from marie.query_planner.base import ExecutorEndpointQueryDefinition, PlannerInfo


def plan(metadata):
    factory = getattr(kb_indexing_planner, 'query_planner_kb_index_content', None)
    assert callable(factory), 'Register an explicit saved-content workflow'
    return factory(PlannerInfo(name='kb_index_content', base_id='0' * 32, metadata=metadata))


def test_saved_content_workflow_has_no_extraction_node():
    result = plan({'input_ref': {'kind': 'extraction', 'ref_id': 'output', 'ref_type': 'document'}})
    endpoints = [node.definition.endpoint for node in result.nodes if isinstance(node.definition, ExecutorEndpointQueryDefinition)]
    assert endpoints == ['vector_store_executor://embed_and_store']
    assert len(result.nodes) == 3
    assert result.nodes[1].definition.params['input_mode'] == 'saved_content'


def test_catalog_can_preview_the_actual_saved_content_graph():
    assert len(plan(None).nodes) == 3


@pytest.mark.parametrize('reference', [None, {}, {'kind': 'chess', 'game': {'moves': ['e4']}}])
def test_document_workflow_rejects_missing_or_unsupported_content(reference):
    with pytest.raises(ValueError, match='content|workflow'):
        plan({'input_ref': reference})


def test_waiting_upstream_input_is_not_treated_as_ready_content():
    with pytest.raises(ValueError, match='ready|content'):
        plan({'input_ref': {'kind': 'upstream', 'planner': 'extract'}})
