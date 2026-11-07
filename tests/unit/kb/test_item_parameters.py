from uuid import uuid4

import pytest

from marie.kb.item_state import item_indexing_run, item_run_parameters


def registered(**changes):
    return {
        'tenant_id': str(uuid4()),
        'index_id': str(uuid4()),
        'source_id': str(uuid4()),
        'item_id': 'custom:game',
        'indexing_dag_id': str(uuid4()),
        'owner_type': 'game',
        'owner_id': 'one',
        'uri': 's3://fixture/one.json',
        'run_params': {'parse_mode': 'ocr'},
        **changes,
    }


def test_authoritative_item_parameters_preserve_opaque_identity():
    item = registered()
    params = item_run_parameters(item)
    scope, attempt = item_indexing_run(params)
    assert scope.item_id == 'custom:game'
    assert attempt == item['indexing_dag_id']
    assert params['run_params'] == item['run_params']
    assert params['uri'] == item['uri']
    assert params['index_id'] == params['index_name']


def test_artifacts_are_isolated_by_item_scope_and_attempt():
    first = registered()
    retry = {**first, 'indexing_dag_id': str(uuid4())}
    other = {**first, 'index_id': str(uuid4())}
    refs = {item_run_parameters(item)['ref_id'] for item in (first, retry, other)}
    assert len(refs) == 3
    assert all('/' not in ref and ':' not in ref for ref in refs)


@pytest.mark.parametrize(
    'field', ['tenant_id', 'index_id', 'source_id', 'item_id', 'kb_indexing_dag_id']
)
def test_item_indexing_run_requires_full_registered_scope(field):
    params = item_run_parameters(registered())
    del params[field]
    with pytest.raises(ValueError, match='KB item'):
        item_indexing_run(params)


def test_conflicting_index_names_do_not_override_registered_scope():
    params = item_run_parameters(registered())
    params['index_name'] = str(uuid4())
    with pytest.raises(ValueError, match='KB item'):
        item_indexing_run(params)


def test_persisted_legacy_job_envelope_resolves_the_same_indexing_dag():
    item = registered()
    parameters = item_run_parameters(item)
    parameters['kb_attempt_id'] = parameters.pop('kb_indexing_dag_id')
    _, indexing_dag_id = item_indexing_run(parameters)
    assert indexing_dag_id == item['indexing_dag_id']
