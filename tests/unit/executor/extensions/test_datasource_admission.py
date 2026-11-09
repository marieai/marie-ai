import asyncio
import copy
import json
from datetime import datetime, timedelta, timezone

import pytest
from docarray import DocList
from docarray.documents import TextDoc

from marie.executor.extensions.plugin_daemon_executor import MariePluginDaemonExecutor
from marie.plugin_daemon import (
    PluginInvocationSpec,
    build_invocation_envelope,
    sign_envelope,
)

KEY_ID = 'test-admission'
SECRET = 'test-admission-only-secret'


def admission():
    spec = PluginInvocationSpec(
        action_type='datasource',
        action_id='datasources/team/wiki',
        install_id='install-1',
        package_id='package-1',
        package_ref='ext.test.wiki',
        package_digest='sha256:test',
    )
    envelope = build_invocation_envelope(
        spec,
        organization_id='org-1',
        workspace_id='ws-1',
        user_id='user-1',
        request_id='request-1',
        payload={
            'type': 'datasource',
            'action': 'fetch_page',
            'datasource': spec.action_id,
            'source_id': 'source-1',
            'index_id': 'index-1',
            'tenant_id': 'tenant-1',
            'configuration': {'space': 'Engineering'},
            'cursor': None,
            'limit': 2,
        },
    )
    envelope['input'] = {
        'credential_requirements': [],
        'output_schema': {
            'type': 'object',
            'required': ['id'],
            'properties': {'id': {'type': 'string'}},
        },
    }
    return sign_envelope(envelope, key_id=KEY_ID, secret=SECRET)


def executor(monkeypatch, output=None):
    worker = MariePluginDaemonExecutor(
        daemon_url='http://unused',
        start=False,
        env={
            'MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID': KEY_ID,
            'MARIE_PLUGIN_DAEMON_SIGNING_SECRET': SECRET,
        },
    )
    calls = []

    def invoke(spec, payload, request_id, **context):
        calls.append((spec, payload, context))
        return (
            output
            if output is not None
            else [
                {
                    'type': 'stream',
                    'data': {'records': [{'id': 'row-1'}], 'next_cursor': None},
                },
                {'type': 'end'},
            ]
        )

    monkeypatch.setattr(worker, '_invoke_daemon', invoke)
    return worker, calls


def run(worker, parameters):
    return [
        json.loads(doc.text)
        for doc in asyncio.run(
            worker.invoke_plugin(DocList[TextDoc](), parameters=parameters)
        )
    ]


def test_signed_binding_executes_only_its_admitted_configuration(monkeypatch):
    worker, calls = executor(monkeypatch)
    frames = run(
        worker,
        {
            'invocation_envelope': admission(),
            'source_id': 'forged',
            'configuration': {'space': 'forged'},
        },
    )
    assert [frame['type'] for frame in frames] == ['stream', 'end']
    assert calls[0][0].action_type == 'datasource'
    assert calls[0][1]['source_id'] == 'source-1'
    assert calls[0][1]['configuration'] == {'space': 'Engineering'}
    assert calls[0][2]['organization_id'] == 'org-1'


@pytest.mark.parametrize(
    'change',
    [
        'tamper',
        'expire',
        'key',
        'missing-signature',
        'missing-source',
        'action-mismatch',
        'missing-policy',
        'missing-schema',
        'missing-actor',
    ],
)
def test_invalid_admission_never_dispatches(monkeypatch, change):
    worker, calls = executor(monkeypatch)
    envelope = admission()
    if change == 'tamper':
        envelope['payload']['configuration']['space'] = 'forged'
    if change == 'expire':
        envelope['expiresAt'] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        envelope = sign_envelope(envelope, key_id=KEY_ID, secret=SECRET)
    if change == 'key':
        envelope['signature']['keyId'] = 'other-key'
    if change == 'missing-signature':
        del envelope['signature']
    if change == 'missing-source':
        envelope['payload']['source_id'] = ''
        envelope = sign_envelope(envelope, key_id=KEY_ID, secret=SECRET)
    if change == 'action-mismatch':
        envelope['payload']['datasource'] = 'datasources/another/wiki'
        envelope = sign_envelope(envelope, key_id=KEY_ID, secret=SECRET)
    if change == 'missing-policy':
        del envelope['runtimePolicy']
        envelope = sign_envelope(envelope, key_id=KEY_ID, secret=SECRET)
    if change == 'missing-schema':
        del envelope['input']['output_schema']
        envelope = sign_envelope(envelope, key_id=KEY_ID, secret=SECRET)
    if change == 'missing-actor':
        del envelope['userId']
        envelope = sign_envelope(envelope, key_id=KEY_ID, secret=SECRET)
    frames = run(worker, {'invocation_envelope': envelope})
    assert frames[0]['type'] == 'error'
    assert calls == []


def test_unsigned_datasource_cannot_use_the_tool_compatibility_path(monkeypatch):
    worker, calls = executor(monkeypatch)
    original = admission()
    frames = run(
        worker,
        {
            'plugin': {
                'action_type': 'datasource',
                'action_id': original['actionId'],
                'package_ref': original['packageRef'],
                'package_digest': original['packageDigest'],
                'install_id': 'install-1',
            },
            'source_id': 'source-1',
            'organization_id': 'org-1',
            'workspace_id': 'ws-1',
        },
    )
    assert frames[0]['type'] == 'error'
    assert calls == []


@pytest.mark.parametrize('action_type', ['tool', 'stub'])
def test_datasource_payload_cannot_hide_behind_a_legacy_action(
    monkeypatch, action_type
):
    worker, calls = executor(monkeypatch)
    payload = admission()['payload']
    frames = run(
        worker,
        {
            **payload,
            'plugin': {
                'action_type': action_type,
                'action_id': 'tools/echo',
                'tool_ref': 'echo',
                'package_ref': 'ext.test.datasource',
                'package_digest': 'sha256:test',
            },
        },
    )
    assert frames[0]['type'] == 'error'
    assert calls == []


@pytest.mark.parametrize(
    'output',
    [
        [
            {'type': 'stream', 'data': {'records': [{'wrong': 'record'}]}},
            {'type': 'end'},
        ],
        [{'type': 'stream', 'data': {'records': []}}],
        [{'type': 'end'}],
        [
            {'type': 'stream', 'data': {'records': [{'id': str(i)} for i in range(3)]}},
            {'type': 'end'},
        ],
    ],
)
def test_worker_validates_output_before_returning_it_to_studio(monkeypatch, output):
    worker, calls = executor(monkeypatch, output)
    frames = run(worker, {'invocation_envelope': admission()})
    assert frames[0]['error']['code'] == 'invalid_datasource_output'
    assert len(calls) == 1


def test_worker_does_not_fetch_external_schema_references(monkeypatch):
    import urllib.request

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('External schema request')

    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    worker, _ = executor(monkeypatch)
    envelope = admission()
    envelope['input']['output_schema'] = {
        '$ref': 'https://example.invalid/private-schema'
    }
    envelope = sign_envelope(envelope, key_id=KEY_ID, secret=SECRET)
    frames = run(worker, {'invocation_envelope': envelope})
    assert frames[0]['error']['code'] == 'invalid_datasource_output'
    assert calls == []


def test_unresolvable_credential_returns_safe_structured_error(monkeypatch):
    worker, _ = executor(monkeypatch)

    def unavailable(*args, **kwargs):
        raise ValueError('env:PRIVATE_BINDING_REFERENCE could not be resolved')

    monkeypatch.setattr(worker, '_invoke_daemon', unavailable)
    frames = run(worker, {'invocation_envelope': admission()})
    assert frames[0]['error']['code'] == 'credential_unavailable'
    assert 'PRIVATE_BINDING_REFERENCE' not in str(frames)
