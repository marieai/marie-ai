import pytest
from marie.extension.datasource import (
    DatasourceDefinition,
    DatasourceInputError,
    DatasourceOutputError,
    DatasourcePage,
    invoke_datasource,
)


def definition():
    return DatasourceDefinition.model_validate(
        {
            'ref': 'datasources/sample/wiki',
            'name': 'Wiki',
            'providerRef': 'providers/wiki',
            'configurationSchema': {
                'parameters': [
                    {'key': 'space', 'type': 'string', 'required': True},
                    {'key': 'archived', 'type': 'boolean', 'default': False},
                ]
            },
            'outputSchema': {
                'type': 'object',
                'required': ['id', 'score'],
                'properties': {
                    'id': {'type': 'string'},
                    'score': {'type': 'number'},
                },
            },
        }
    )


def payload(**updates):
    return {
        'type': 'datasource',
        'action': 'fetch_page',
        'datasource': 'datasources/sample/wiki',
        'source_id': 'source-1',
        'index_id': 'index-1',
        'tenant_id': 'tenant-1',
        'configuration': {'space': 'Engineering'},
        'limit': 2,
        **updates,
    }


def test_page_preserves_typed_records_and_source_configuration():
    seen = []

    def fetch(request):
        seen.append(request)
        return DatasourcePage(
            records=[{'id': 'row-1', 'score': 0}], next_cursor='page-2'
        )

    result = invoke_datasource(definition(), payload(), fetch)
    assert result.records == [{'id': 'row-1', 'score': 0}]
    assert result.next_cursor == 'page-2'
    assert seen[0].configuration == {'space': 'Engineering', 'archived': False}
    assert seen[0].source_id == 'source-1'


@pytest.mark.parametrize(
    'updates',
    [
        {'datasource': 'datasources/other/wiki'},
        {'type': 'tool'},
        {'limit': 101},
        {'source_id': ''},
        {'configuration': {'space': 42}},
        {'configuration': {'space': 'x', 'api_key': 'forged'}},
        {'implementation': 'evil.py'},
    ],
)
def test_invalid_requests_never_call_the_handler(updates):
    called = []
    with pytest.raises(DatasourceInputError):
        invoke_datasource(
            definition(), payload(**updates), lambda request: called.append(request)
        )
    assert called == []


@pytest.mark.parametrize(
    'page',
    [
        {'records': [{'id': 'x', 'score': 'wrong'}]},
        {'records': [{'id': str(i), 'score': i} for i in range(3)]},
        {'records': [{'id': 'x' * (1024 * 1024), 'score': 0}]},
    ],
)
def test_invalid_or_unbounded_output_is_rejected(page):
    with pytest.raises(DatasourceOutputError):
        invoke_datasource(definition(), payload(), lambda request: page)


def test_runtime_uses_the_existing_session_frames():
    from marie_plugins.runtime.datasource import dispatch_datasource

    frames = list(
        dispatch_datasource(
            {'session_id': 's1', 'event': 'request', 'data': payload()},
            definition(),
            lambda request: DatasourcePage(records=[{'id': 'x', 'score': 0}]),
        )
    )
    assert [frame['data']['type'] for frame in frames] == ['stream', 'end']
    assert frames[0]['data']['data']['records'] == [{'id': 'x', 'score': 0}]
    assert frames[0]['session_id'] == 's1'


def test_runtime_errors_do_not_echo_private_input_or_output():
    from marie_plugins.runtime.datasource import dispatch_datasource

    frames = list(
        dispatch_datasource(
            {
                'session_id': 's1',
                'event': 'request',
                'data': payload(configuration={'space': 42}),
            },
            definition(),
            lambda request: None,
        )
    )
    assert frames[0]['data']['type'] == 'error'
    assert frames[0]['data']['data']['code'] == 'invalid_datasource_request'
    assert '42' not in str(frames)


def test_output_schema_never_fetches_external_references(monkeypatch):
    import urllib.request

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('External schema request')

    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    datasource = definition()
    datasource.output_schema = {'$ref': 'https://example.invalid/private-schema'}
    with pytest.raises(DatasourceOutputError):
        invoke_datasource(
            datasource, payload(), lambda request: {'records': [{'id': 'x'}]}
        )
    assert calls == []


def test_missing_output_declaration_does_not_call_the_handler():
    datasource = definition()
    datasource.output_schema = None
    calls = []
    with pytest.raises(DatasourceOutputError):
        invoke_datasource(datasource, payload(), lambda request: calls.append(request))
    assert calls == []
