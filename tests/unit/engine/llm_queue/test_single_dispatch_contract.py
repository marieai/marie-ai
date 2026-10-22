from __future__ import annotations

import inspect
import json
from dataclasses import asdict, fields

import pytest
from marie.engine import completion_contract
from marie.engine.completion_contract import CompletionCallParams
from marie.engine.llm_queue import producer
from marie.engine.llm_queue.config import LlmQueueRuntimeConfig
from marie.engine.llm_queue.queue_keys import QueueKeys
from marie.engine.llm_queue.store import RequestStore, StoreLimits


def test_runtime_configuration_has_one_queue_selector(monkeypatch) -> None:
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'true')
    monkeypatch.setenv('LLM_QUEUE_CONTRACT_VERSION', 'unsupported-old-setting')
    config = LlmQueueRuntimeConfig.from_env()
    assert config.enabled
    assert 'queue_contract_version' not in {item.name for item in fields(config)}


def test_producer_is_named_for_its_responsibility() -> None:
    assert hasattr(producer, 'QueuedDispatchProducer')
    assert not hasattr(producer, 'V3Producer')


def test_store_does_not_select_an_execution_version() -> None:
    assert 'version' not in inspect.signature(RequestStore).parameters


def test_engine_rejects_retired_list_client_override(monkeypatch) -> None:
    from marie.engine.openai_engine import OpenAIEngine

    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    with pytest.raises(ValueError, match='caller_pool_forbidden'):
        OpenAIEngine(model_name='model', is_multimodal=False, queue_client=object())


def test_store_namespace_preserves_fabric_isolation_without_protocol_label() -> None:
    assert QueueKeys('Fabric-A').prefix == 'llm:dispatch:{fabric:fabric-a}:'
    assert QueueKeys('fabric-b').prefix != QueueKeys('fabric-a').prefix


def durable_request():
    envelope = completion_contract.QueuedCompletionEnvelope
    assert 'schema_version' in {item.name for item in fields(envelope)}
    return envelope(
        fabric_group_id='fabric',
        producer_id='producer',
        attempt_id='attempt',
        pool_id='pool',
        endpoint_id='endpoint',
        config_revision='r1',
        logical_batch_id='batch',
        logical_task_id='task',
        item_index=0,
        expires_at_ms=1000,
        call=CompletionCallParams(model='model', messages=[]),
    )


def test_durable_envelope_uses_numeric_schema_and_retains_trace() -> None:
    from dataclasses import replace

    envelope = replace(
        durable_request(), traceparent='trace', tracestate='vendor=value'
    )
    data = json.loads(envelope.to_json())
    assert data['schema_version'] == 1
    assert type(data['schema_version']) is int
    assert 'contract_version' not in data
    restored = type(envelope).from_json(envelope.to_json())
    assert restored == envelope


@pytest.mark.parametrize('schema', [None, True, '1', 1.0, 2])
def test_durable_envelope_rejects_missing_or_unsupported_schema(schema) -> None:
    request = durable_request()
    data = json.loads(request.to_json())
    if schema is None:
        data.pop('schema_version')
    else:
        data['schema_version'] = schema
    with pytest.raises(ValueError, match='schema'):
        type(request).from_json(json.dumps(data))


@pytest.mark.parametrize('attach_producer', [False, True])
def test_store_startup_uses_only_its_fabric_policy(
    monkeypatch, attach_producer
) -> None:
    from marie.engine.llm_queue import store as store_module

    class Client:
        __module__ = 'valkey.client'

        def __init__(self):
            self.reads = []
            self.operations = []

        def register_script(self, source):
            def execute(*, keys, args):
                self.operations.append(json.loads(args[0])['op'])
                return json.dumps({'disposition': 'initialized'})

            return execute

        def get(self, key):
            self.reads.append(key)
            if attach_producer:
                return json.dumps(
                    asdict(StoreLimits()), sort_keys=True, separators=(',', ':')
                )
            return None

        def scan(self, *args, **kwargs):
            raise AssertionError('Startup must not scan unrelated store namespaces')

        def close(self):
            pass

    client = Client()
    monkeypatch.setattr(store_module, '_build_sync_client', lambda *a, **kw: client)
    store = (
        RequestStore.for_producer('redis://unused', fabric_id='fabric')
        if attach_producer
        else RequestStore('redis://unused', fabric_id='fabric')
    )
    try:
        assert client.reads == ['llm:dispatch:{fabric:fabric}:limits']
        assert client.operations == ([] if attach_producer else ['initialize'])
        assert store.limits == StoreLimits()
    finally:
        store.close()
