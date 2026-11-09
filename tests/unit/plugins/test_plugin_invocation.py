from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError

import pytest
from pydantic import ValidationError

from marie.plugin_daemon import PluginInvocationSpec, build_invocation_envelope
from marie.plugins import daemon_client
from marie.plugins.daemon_client import PluginDaemonClient


def datasource_spec(**overrides: object) -> PluginInvocationSpec:
    return PluginInvocationSpec(
        **{
            'action_type': 'datasource',
            'action_id': 'datasources/confluence',
            'install_id': 'install-1',
            'package_id': 'package-1',
            'package_ref': 'ext.test.sources',
            'package_digest': 'sha256:abc',
            **overrides,
        }
    )


def test_datasource_envelope_needs_no_tool_or_provider() -> None:
    spec = datasource_spec(credential_binding_ids=['binding-1'])
    envelope = build_invocation_envelope(
        spec,
        payload={'source_id': 'source-1', 'configuration': {'space': 'Engineering'}},
        organization_id='org-1',
        workspace_id='workspace-1',
    )
    assert envelope['actionType'] == 'datasource'
    assert envelope['actionId'] == 'datasources/confluence'
    assert envelope['providerId'] == ''
    assert envelope['credentialBindingIds'] == ['binding-1']
    assert 'tool' not in envelope['payload']


@pytest.mark.parametrize('action_id', ['', '   '])
def test_invocation_requires_explicit_action_identity(action_id: str) -> None:
    with pytest.raises(ValidationError, match='action_id'):
        datasource_spec(action_id=action_id)


def test_invocation_rejects_unknown_action_type() -> None:
    with pytest.raises(ValidationError, match='action_type'):
        datasource_spec(action_type='unknown')


def test_client_returns_structured_frames_and_resolves_credentials(monkeypatch) -> None:
    monkeypatch.setenv('PLUGIN_TEST_TOKEN', 'test-token')
    captured = {}
    frames = [
        {'type': 'stream', 'data': {'records': [{'id': 'record-1', 'count': 0}]}},
        {
            'type': 'artifact',
            'artifactId': 'artifact-1',
            'payload': {'uri': 's3://kb/data.json'},
        },
        {'type': 'end', 'final': True},
    ]

    def respond(request, timeout):
        captured['envelope'] = json.loads(request.data)
        captured['timeout'] = timeout
        return io.BytesIO(json.dumps(frames).encode())

    monkeypatch.setattr(daemon_client, 'urlopen', respond)
    client = PluginDaemonClient(
        base_url='http://daemon',
        organization_id='org-1',
        workspace_id='workspace-1',
        user_id='user-1',
        signing_key_id='test-key',
        signing_secret='test-secret',
    )
    payload = {'source_id': 'source-1', 'credentials': {'forged': 'value'}}
    result = client.invoke(
        datasource_spec(
            credential_requirements=[
                {
                    'name': 'token',
                    'secret_ref': 'env:PLUGIN_TEST_TOKEN',
                    'required': True,
                },
            ],
            credential_binding_ids=['binding-1'],
        ),
        payload,
        request_id='request-1',
        runtime_policy={'timeoutMs': 12000},
    )
    assert [frame['type'] for frame in result] == ['stream', 'artifact', 'end']
    assert result[0]['data'] == frames[0]['data']
    assert result[1]['artifactId'] == 'artifact-1'
    assert result[1]['payload'] == frames[1]['payload']
    assert all(frame['requestId'] == 'request-1' for frame in result)
    envelope = captured['envelope']
    assert envelope['actionId'] == 'datasources/confluence'
    assert envelope['credentialBindingIds'] == ['binding-1']
    assert envelope['payload']['credentials'] == {'token': 'test-token'}
    assert envelope['payload']['user_id'] == 'user-1'
    assert envelope['signature']['keyId'] == 'test-key'
    assert captured['timeout'] == 12
    assert payload['credentials'] == {'forged': 'value'}


def test_client_bounds_daemon_response(monkeypatch) -> None:
    response = io.BytesIO(b'x' * 1024)
    monkeypatch.setattr(daemon_client, 'urlopen', lambda *args, **kwargs: response)
    with PluginDaemonClient(
        base_url='http://daemon',
        organization_id='org-1',
        workspace_id='workspace-1',
        max_response_bytes=512,
    ) as client:
        with pytest.raises(RuntimeError, match='response exceeds'):
            client.invoke(datasource_spec(), {})
    assert response.closed


def test_client_cancellation_does_not_return_a_partial_success(monkeypatch) -> None:
    started = threading.Event()
    released = threading.Event()

    class Response:
        def read(self, size):
            started.set()
            assert released.wait(2)
            return b'[]'

        def close(self):
            released.set()

    monkeypatch.setattr(daemon_client, 'urlopen', lambda *args, **kwargs: Response())
    with (
        PluginDaemonClient(
            base_url='http://daemon',
            organization_id='org-1',
            workspace_id='workspace-1',
        ) as client,
        ThreadPoolExecutor() as pool,
    ):
        result = pool.submit(
            client.invoke, datasource_spec(), {}, request_id='request-1'
        )
        try:
            assert started.wait(2)
            client.cancel('request-1')
            with pytest.raises(RuntimeError, match='cancelled'):
                result.result(timeout=2)
        finally:
            released.set()


def test_client_preserves_timeout_failure(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise TimeoutError('test timeout')

    monkeypatch.setattr(daemon_client, 'urlopen', timeout)
    with PluginDaemonClient(
        base_url='http://daemon',
        organization_id='org-1',
        workspace_id='workspace-1',
    ) as client:
        with pytest.raises(RuntimeError, match='test timeout'):
            client.invoke(datasource_spec(), {})


def test_datasource_http_errors_do_not_expose_resolved_credentials(monkeypatch, caplog):
    monkeypatch.setenv('PLUGIN_TEST_TOKEN', 'private-datasource-token')

    def denied(*args, **kwargs):
        raise HTTPError(
            'http://daemon', 502, 'failure', {}, io.BytesIO(b'private-datasource-token')
        )

    monkeypatch.setattr(daemon_client, 'urlopen', denied)
    with PluginDaemonClient(
        base_url='http://daemon', organization_id='org-1', workspace_id='ws-1'
    ) as client:
        with pytest.raises(RuntimeError) as error:
            client.invoke(
                datasource_spec(
                    credential_requirements=[
                        {
                            'name': 'token',
                            'secret_ref': 'env:PLUGIN_TEST_TOKEN',
                            'required': True,
                        }
                    ]
                ),
                {},
                redact_output=True,
            )
    assert 'private-datasource-token' not in str(error.value)
    assert 'private-datasource-token' not in caplog.text
