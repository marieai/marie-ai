import asyncio
import json
import threading
import time
from copy import deepcopy
from dataclasses import asdict, replace
from unittest import mock

import pytest
from marie.engine.batch_processor import BatchProcessor
from marie.engine.completion_contract import (
    CompletionCallParams,
    QueuedCompletionEnvelope,
    RequestContext,
    build_completion_call,
)
from marie.engine.llm_queue.adapters.openai_compatible import (
    OpenAICompatibleExecutionAdapter,
)
from marie.engine.llm_queue.config import LlmQueueProducerConfig
from marie.engine.llm_queue.endpoint import RegisteredEndpointGroup, RegisteredReplica
from marie.engine.llm_queue.producer import QueuedDispatchProducer
from marie.engine.llm_queue.queue_keys import QueueKeys, validate_identifier
from marie.engine.llm_queue.result_types import BatchResult
from marie.engine.llm_queue.store import (
    AdmissionConflict,
    ClaimRecord,
    OwnerToken,
    PreparedAdmission,
    RequestStore,
    RoutingManifestProjection,
    StoreLimits,
    StoreReply,
)
from marie.engine.openai_engine import OpenAIEngine


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def test_endpoint_status_defaults_uninitialized_circuit_to_closed() -> None:
    store = object.__new__(RequestStore)
    store.keys = QueueKeys("default")
    store._read = mock.Mock(
        return_value=["open", None, "0", None, None, "0", None, "0", "0"]
    )

    status = store.endpoint_status("endpoint")

    assert status["circuit"] == "closed"
    assert status["waiting_reason"] is None


def test_execution_history_marks_semantic_failure_as_error(monkeypatch) -> None:
    from marie.engine.llm_queue import request_dispatcher
    from marie.engine.llm_queue.endpoint import ExecutionOutcome

    class Span:
        def set_status(self, status):
            self.status = status

        def end(self):
            pass

    span = Span()
    monkeypatch.setattr(
        request_dispatcher._tracer,
        "start_span",
        lambda *_args, **_kwargs: span,
    )
    attributes = {"marie.llm_dispatch.queue_wait_ms": 5}

    request_dispatcher._emit_execution_history(
        attributes,
        ExecutionOutcome(
            response={"usage": {"total_tokens": 192}},
            category="repetition",
            remote_settled=True,
            availability_success=True,
        ),
        time.time_ns(),
        time.monotonic(),
    )

    assert attributes["marie.llm_dispatch.status"] == "error"
    assert attributes["marie.llm_dispatch.error_type"] == "repetition"
    assert attributes["llm.token_count.total"] == 192


def test_endpoint_group_requires_compatible_unique_replicas() -> None:
    first = RegisteredReplica(
        replica_id="replica-a",
        base_url="https://a.example/v1",
        credential_env="REPLICA_A_TOKEN",
        capability_digest="a" * 64,
    )
    second = RegisteredReplica(
        replica_id="replica-b",
        base_url="https://b.example/v1",
        capability_digest="a" * 64,
    )

    group = RegisteredEndpointGroup(
        group_id="document-llm", revision="r1", replicas=(first, second)
    )

    assert tuple(replica.replica_id for replica in group.replicas) == (
        "replica-a",
        "replica-b",
    )
    assert "REPLICA_A_TOKEN" not in repr(group)
    with pytest.raises(ValueError, match="capability"):
        RegisteredEndpointGroup(
            group_id="document-llm",
            revision="r1",
            replicas=(
                first,
                replace(second, capability_digest="b" * 64),
            ),
        )


@pytest.mark.asyncio
async def test_replica_selection_skips_unavailable_and_saturated_targets() -> None:
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )

    replicas = (
        RegisteredReplica(
            replica_id="replica-a",
            base_url="https://a.example/v1",
            capability_digest="a" * 64,
            execution_limit=2,
        ),
        RegisteredReplica(
            replica_id="replica-b",
            base_url="https://b.example/v1",
            capability_digest="a" * 64,
            execution_limit=2,
        ),
    )
    group = RegisteredEndpointGroup("document-llm", "r1", replicas)
    statuses = {
        "replica-a": {
            "gate": "open",
            "circuit": "open",
            "next_probe": 0,
            "probe_claim": None,
            "reserved_items": 0,
            "reserved_bytes": 0,
        },
        "replica-b": {
            "gate": "open",
            "circuit": "closed",
            "next_probe": 0,
            "probe_claim": None,
            "reserved_items": 1,
            "reserved_bytes": 10,
        },
    }
    store = mock.Mock()
    store.limits = StoreLimits()
    store.metadata.return_value = mock.Mock(payload_bytes=1)
    store.server_time_ms.return_value = 1
    store.endpoint_status.side_effect = statuses.__getitem__
    dispatcher = RequestDispatcher(
        store=store,
        endpoint_groups=[group],
        lanes=[DispatchLane("document-small", endpoint_group_id="document-llm")],
    )
    claim = ClaimRecord(
        attempt_id="attempt-1",
        claim_id="claim-1",
        execution_sequence=0,
        owner_generation=1,
        pool_id="document-small",
        endpoint_group_id="document-llm",
        charged_cost=1,
        charge_sequence=1,
        refund_state="not_refunded",
    )
    try:
        selected = await dispatcher.select_replica(claim)
        assert selected.replica_id == "replica-b"
        assert claim.endpoint_group_id == "document-llm"
        statuses["replica-a"]["circuit"] = "closed"
        selected = await dispatcher.select_replica(claim)
        assert selected.replica_id == "replica-a"
    finally:
        dispatcher._store_workers.shutdown()
        dispatcher._lease_worker.shutdown()
        dispatcher._refresh_worker.shutdown()


def test_routing_manifest_projection_is_idempotent_and_conflict_safe() -> None:
    records = {}
    store = object.__new__(RequestStore)
    store.keys = QueueKeys('default')
    store._client_error = RuntimeError

    def script(*, keys, args):
        import json

        payload = json.loads(args[0])
        prior = records.get(keys[0])
        if prior is not None:
            disposition = (
                'existing'
                if prior['route_digest'] == payload['route_digest']
                else 'conflict'
            )
            return json.dumps({'disposition': disposition})
        records[keys[0]] = payload
        return json.dumps({'disposition': 'projected'})

    store._script = script
    projection = RoutingManifestProjection(
        job_id='job-1',
        work_unit_id='node-1',
        fabric_group_id='default',
        policy_generation=1,
        route_digest='a' * 64,
        pool_id='document-small',
        endpoint_group_id='primary',
        endpoint_revision='r1',
    )

    assert store.project_routing_manifest(projection).disposition == 'projected'
    assert store.project_routing_manifest(projection).disposition == 'existing'
    with pytest.raises(AdmissionConflict, match='routing_binding_conflict'):
        store.project_routing_manifest(replace(projection, route_digest='b' * 64))


def test_routing_manifest_keys_share_the_fabric_slot() -> None:
    keys = QueueKeys('Default')

    assert keys.routing_manifest('job-1', 'node-1').startswith(keys.prefix)
    assert keys.routing_manifests.startswith(keys.prefix)


def test_routing_manifest_rejects_unsafe_policy_generation() -> None:
    store = object.__new__(RequestStore)
    store.keys = QueueKeys('default')
    projection = RoutingManifestProjection(
        job_id='job-1',
        work_unit_id='node-1',
        fabric_group_id='default',
        policy_generation=2**53,
        route_digest='a' * 64,
        pool_id='document-small',
        endpoint_group_id='primary',
        endpoint_revision='r1',
    )

    with pytest.raises(ValueError, match='policy generation'):
        store.project_routing_manifest(projection)


def test_manifest_admission_uses_projected_pool_without_pool_metadata() -> None:
    projection = RoutingManifestProjection(
        job_id="job-1",
        work_unit_id="node-1",
        fabric_group_id="default",
        policy_generation=1,
        route_digest="a" * 64,
        pool_id="document-small",
        endpoint_group_id="primary",
        endpoint_revision="r1",
    )
    store = object.__new__(RequestStore)
    store.keys = QueueKeys("default")
    store.limits = StoreLimits()
    store._limits_json = json.dumps(
        asdict(store.limits),
        sort_keys=True,
        separators=(",", ":"),
    )
    store._client_error = RuntimeError
    store.resolve_routing_manifest = lambda *_args: projection
    captured = {}

    def script(*, keys, args):
        import json

        captured["keys"] = keys
        captured["payload"] = json.loads(args[0])
        return json.dumps({"disposition": "admitted"})

    store._script = script
    call = _call(
        [{"role": "user", "content": "route me"}],
        context=RequestContext(job_id="job-1", work_unit_id="node-1"),
    )
    request = PreparedAdmission.create(
        fabric_group_id="default",
        producer_id="producer-1",
        attempt_id="attempt-1",
        job_id="job-1",
        work_unit_id="node-1",
        logical_batch_id="batch-1",
        logical_task_id="task-1",
        item_index=0,
        expires_at_ms=1000,
        call=call,
        estimated_cost_units=1,
    )

    assert store.admit_from_manifest(request).disposition == "admitted"
    assert captured["keys"][3] == store.keys.ready("document-small")
    assert captured["payload"]["pool_id"] == "document-small"
    assert captured["payload"]["job_id"] == "job-1"
    assert captured["payload"]["work_unit_id"] == "node-1"


def test_manifest_admission_rejects_tampered_content_digest() -> None:
    call = _call(
        [{"role": "user", "content": "route me"}],
        context=RequestContext(job_id="job-1", work_unit_id="node-1"),
    )
    request = PreparedAdmission.create(
        fabric_group_id="default",
        producer_id="producer-1",
        attempt_id="attempt-1",
        job_id="job-1",
        work_unit_id="node-1",
        logical_batch_id="batch-1",
        logical_task_id="task-1",
        item_index=0,
        expires_at_ms=1000,
        call=call,
        estimated_cost_units=1,
    )
    store = object.__new__(RequestStore)
    store.keys = QueueKeys("default")

    with pytest.raises(ValueError, match="content digest"):
        store.admit_from_manifest(replace(request, content_digest="b" * 64))


def test_manifest_admission_rejects_missing_route() -> None:
    call = _call(
        [{"role": "user", "content": "route me"}],
        context=RequestContext(job_id="job-1", work_unit_id="node-1"),
    )
    request = PreparedAdmission.create(
        fabric_group_id="default",
        producer_id="producer-1",
        attempt_id="attempt-1",
        job_id="job-1",
        work_unit_id="node-1",
        logical_batch_id="batch-1",
        logical_task_id="task-1",
        item_index=0,
        expires_at_ms=1000,
        call=call,
        estimated_cost_units=1,
    )
    store = object.__new__(RequestStore)
    store.keys = QueueKeys("default")
    store.resolve_routing_manifest = lambda *_args: None

    assert store.admit_from_manifest(request).disposition == "routing_manifest_missing"


@pytest.mark.parametrize('metadata', [{}, {'pool_id': 'caller-chosen'}])
def test_queued_producer_submits_durable_provenance_without_pool_metadata(
    metadata: dict,
) -> None:
    producer = QueuedDispatchProducer(
        config=_queue_config(
            queue_url="redis://unused:6379/0",
            fabric_group_id="default",
        )
    )
    producer.producer_id = "producer-1"
    captured = []

    class Store:
        @staticmethod
        def server_time_ms():
            return 1000

        @staticmethod
        def admit_from_manifest(request):
            captured.append(request)
            producer._pending[request.attempt_id].result = BatchResult(
                request.logical_task_id, "done", None
            )
            return StoreReply(disposition="admitted")

        @staticmethod
        def ack_result(*_args):
            return StoreReply(disposition="acked")

    producer._store = Store()
    producer._start = lambda *_args: producer._store
    producer._check = lambda: None
    call = _call(
        [{"role": "user", "content": "route me"}],
        context=RequestContext(job_id="job-1", work_unit_id="node-1"),
    )

    results = producer.execute(
        calls=[call],
        batch_request_id="batch-1",
        batch_timeout=1,
        metadata=metadata,
    )

    assert [result.response for result in results] == ["done"]
    assert captured[0].job_id == "job-1"
    assert captured[0].work_unit_id == "node-1"
    assert not hasattr(captured[0], "pool_id")
    assert captured[0].call.context == call.context


def test_request_context_round_trip_preserves_durable_provenance() -> None:
    context = RequestContext(
        job_id="job-1",
        work_unit_id="node-1",
        ref_id="doc-1",
        ref_type="stress",
    )

    assert RequestContext.from_dict(context.to_dict()) == context


def test_claim_and_charge_returns_durable_accounting_record() -> None:
    store = object.__new__(RequestStore)
    store.metadata = lambda _attempt: None
    captured = {}

    def invoke(op, **kwargs):
        captured.update(op=op, **kwargs)
        return {
            "disposition": "claimed",
            "attempt_id": "attempt-1",
            "claim_id": "claim-1",
            "execution_seq": 0,
            "owner_generation": 7,
            "pool_id": "document-small",
            "endpoint_group_id": "primary",
            "charged_cost": 4,
            "charge_sequence": 11,
            "refund_state": "not_refunded",
        }

    store._invoke = invoke
    owner = OwnerToken("gateway", 7)

    claim = store.claim_and_charge(
        owner,
        "document-small",
        expected_attempt="attempt-1",
        expected_cost=4,
        claim_id="claim-1",
    )

    assert claim == ClaimRecord(
        attempt_id="attempt-1",
        claim_id="claim-1",
        execution_sequence=0,
        owner_generation=7,
        pool_id="document-small",
        endpoint_group_id="primary",
        charged_cost=4,
        charge_sequence=11,
        refund_state="not_refunded",
    )
    assert captured["op"] == "claim"
    assert captured["expected_cost"] == 4


def test_return_untransmitted_and_refund_is_bound_to_claim_record() -> None:
    store = object.__new__(RequestStore)
    captured = {}
    store._change = lambda op, **kwargs: (
        captured.update(op=op, **kwargs)
        or StoreReply(
            disposition="returned",
            charged_cost=4,
            refunded_cost=4,
            charge_sequence=11,
            refund_state="refunded",
        )
    )
    claim = ClaimRecord(
        attempt_id="attempt-1",
        claim_id="claim-1",
        execution_sequence=0,
        owner_generation=7,
        pool_id="document-small",
        endpoint_group_id="primary",
        charged_cost=4,
        charge_sequence=11,
        refund_state="not_refunded",
    )

    reply = store.return_untransmitted_and_refund(OwnerToken("gateway", 8), claim)

    assert reply.disposition == "returned"
    assert reply.refund_state == "refunded"
    assert captured == {
        "op": "return_untransmitted",
        "owner": OwnerToken("gateway", 8),
        "attempt_id": "attempt-1",
        "claim_id": "claim-1",
        "charge_sequence": 11,
        "execution_sequence": 0,
    }


def _call(
    messages,
    *,
    context: RequestContext | None = None,
    **overrides,
) -> CompletionCallParams:
    return build_completion_call(
        model="test-model",
        messages=messages,
        completion_params=overrides or None,
        stream=False,
        context=context,
    )


def _completion_payload(text: str, *, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "message": {"content": text},
                "finish_reason": finish_reason,
            }
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class _CircuitBreaker:
    def is_available(self, address):
        return True

    def get_state(self, address):
        return "closed"

    def record_success(self, address):
        pass

    def record_failure(self, address):
        pass


def _build_processor(*, client, queue_enabled: bool) -> BatchProcessor:
    processor = object.__new__(BatchProcessor)
    processor.client = client
    processor.model_string = "test-model"
    processor.logger = _Logger()
    processor.max_concurrency = 4
    processor.batch_timeout = 2.0
    processor.backend_address = "http://test-backend:8000"
    processor.default_completion_params = {}
    processor._shared_request_semaphore = None
    processor._shared_request_semaphore_loop = None
    processor._direct_runner = None
    processor._direct_runner_lock = threading.Lock()
    processor._circuit_breaker = _CircuitBreaker()
    processor._gate_lock = None
    processor._gate_lock_loop = None
    processor._queued_executor = None
    processor._queue_config = _queue_config(enabled=queue_enabled)
    processor._queue_mode_logged = False
    return processor


def test_producer_config_ignores_removed_pool_and_version_environment(monkeypatch):
    monkeypatch.setenv("LLM_QUEUE_POOL_ID", "document-small")
    monkeypatch.setenv("LLM_QUEUE_CONTRACT_VERSION", "v2")

    config = LlmQueueProducerConfig.from_env(enabled=True)

    assert not hasattr(config, "pool_id")
    assert not hasattr(config, "queue_contract_version")


@pytest.mark.parametrize(
    "selector",
    [
        "pool_id",
        "queue_pool_id",
        "llm_queue_pool_id",
        "queue_contract_version",
        "llm_queue_contract_version",
        "queue_producer_id",
    ],
)
def test_openai_engine_rejects_legacy_caller_queue_selectors(selector):
    with pytest.raises(ValueError, match="caller_pool_forbidden"):
        OpenAIEngine(model_name="mock", **{selector: "caller-value"})


def test_queue_keys_use_dispatch_namespace_and_validate_generic_identifiers():
    keys = QueueKeys('Fabric-A')
    assert keys.fabric_id == 'fabric-a'
    assert keys.prefix == 'llm:dispatch:{fabric:fabric-a}:'
    assert keys.owner == 'llm:dispatch:{fabric:fabric-a}:owner'
    assert (
        keys.request('attempt-1') == 'llm:dispatch:{fabric:fabric-a}:request:attempt-1'
    )
    assert keys.ready('pool-1') == 'llm:dispatch:{fabric:fabric-a}:ready:pool-1'
    assert (
        keys.alive('producer-1')
        == 'llm:dispatch:{fabric:fabric-a}:producer:producer-1:alive'
    )
    assert (
        keys.members('producer-1')
        == 'llm:dispatch:{fabric:fabric-a}:producer:producer-1:requests'
    )
    assert keys.route('pool-1') == 'llm:dispatch:{fabric:fabric-a}:route:pool-1'
    assert (
        keys.endpoint('endpoint-1')
        == 'llm:dispatch:{fabric:fabric-a}:endpoint:endpoint-1'
    )
    assert validate_identifier('pool_1') == 'pool_1'
    with pytest.raises(ValueError, match='Identifiers must contain'):
        validate_identifier('bad:{identifier}')


def test_batch_processor_uses_queued_executor_when_enabled(monkeypatch):
    processor = object.__new__(BatchProcessor)
    processor.client = None
    processor.model_string = "test-model"
    processor.logger = _Logger()
    processor.max_concurrency = 20
    processor.batch_timeout = 5.0
    processor.backend_address = "http://test-backend:8000"
    processor.default_completion_params = {}
    processor._shared_request_semaphore = None
    processor._shared_request_semaphore_loop = None
    processor._circuit_breaker = None
    processor._gate_lock = None
    processor._gate_lock_loop = None
    processor._queued_executor = None
    processor._queue_config = _queue_config()
    processor._queue_mode_logged = False

    class _QueuedExecutor:
        def execute(self, **kwargs):
            return [
                BatchResult("req_task_0", "resp-0", None),
                BatchResult("req_task_1", "resp-1", None),
            ]

    monkeypatch.setattr(processor, "_get_queued_executor", lambda: _QueuedExecutor())

    responses = processor.batch_generate(
        messages_list=[
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
        guided_json=None,
    )

    assert responses == ["resp-0", "resp-1"]


def test_openai_compatible_execution_adapter_enforces_timeout():
    class _Completions:
        async def create(self, **kwargs):
            await asyncio.sleep(0.2)

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    adapter = OpenAICompatibleExecutionAdapter(
        client=_Client(),
        logger=_Logger(),
        default_timeout=600.0,
    )
    with pytest.raises(TimeoutError):
        asyncio.run(
            adapter.execute(
                _call([{"role": "user", "content": "slow"}]),
                timeout_seconds=0.05,
            )
        )


def test_openai_compatible_execution_adapter_adds_endpoint_context():
    class _Completions:
        async def create(self, **kwargs):
            raise ConnectionError("connection refused")

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    adapter = OpenAICompatibleExecutionAdapter(
        client=_Client(),
        logger=_Logger(),
        default_timeout=600.0,
        backend_address="http://llm-backend:8000/v1",
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            adapter.execute(
                _call([{"role": "user", "content": "hello"}]),
                timeout_seconds=0.05,
            )
        )

    assert "http://llm-backend:8000/v1" in str(exc_info.value)
    assert "ConnectionError: connection refused" in str(exc_info.value)


def test_openai_compatible_execution_adapter_factory_closes_owned_client():
    payload = _completion_payload("done")
    clients = []

    class _Completions:
        async def create(self, **kwargs):
            return deepcopy(payload)

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    def client_factory():
        client = _Client()
        clients.append(client)
        return client

    adapter = OpenAICompatibleExecutionAdapter(
        client_factory=client_factory,
        logger=_Logger(),
        default_timeout=600.0,
        backend_address="http://llm-backend:8000/v1",
    )

    completion = asyncio.run(
        adapter.execute(
            _call([{"role": "user", "content": "hello"}]),
            timeout_seconds=0.05,
        )
    )

    assert completion == payload
    assert len(clients) == 1
    assert clients[0].closed is True


def test_batch_processor_forwards_guided_json_to_queued_executor(monkeypatch):
    processor = object.__new__(BatchProcessor)
    processor.client = None
    processor.model_string = "test-model"
    processor.logger = _Logger()
    processor.max_concurrency = 20
    processor.batch_timeout = 5.0
    processor.backend_address = "http://test-backend:8000"
    processor.default_completion_params = {}
    processor._shared_request_semaphore = None
    processor._shared_request_semaphore_loop = None
    processor._circuit_breaker = None
    processor._gate_lock = None
    processor._gate_lock_loop = None
    processor._queued_executor = None
    processor._queue_config = _queue_config()
    processor._queue_mode_logged = False

    captured = {}

    class _QueuedExecutor:
        def execute(self, **kwargs):
            captured.update(kwargs)
            return [BatchResult("req_task_0", "resp-0", None)]

    monkeypatch.setattr(processor, "_get_queued_executor", lambda: _QueuedExecutor())

    processor.batch_generate(
        messages_list=[[{"role": "user", "content": "a"}]],
        guided_json={"type": "object"},
        metadata={"job_id": "job-1"},
    )

    assert captured["calls"][0].extra_body["guided_json"] == {"type": "object"}
    assert "request_contexts" not in captured


def _queue_config(**overrides) -> LlmQueueProducerConfig:
    values = dict(
        enabled=True,
        queue_url='redis://unused:6379/0',
        fabric_group_id='default',
        producer_ttl_seconds=5,
        producer_refresh_interval_seconds=0.05,
        reply_queue_ttl_seconds=60,
        max_buffered_requests=8,
    )
    values.update(overrides)
    return LlmQueueProducerConfig(**values)


@pytest.mark.parametrize(
    'environment_enabled,caller_enabled,expected',
    [
        ('true', False, False),
        ('false', True, True),
        ('true', None, True),
    ],
)
def test_batch_processor_caller_enablement_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment_enabled: str,
    caller_enabled: bool | None,
    expected: bool,
) -> None:
    from openai import AsyncOpenAI

    monkeypatch.setenv('LLM_QUEUE_ENABLED', environment_enabled)
    monkeypatch.delenv('LLM_QUEUE_URL', raising=False)
    monkeypatch.delenv('LLM_QUEUE_VALKEY_URL', raising=False)
    client = AsyncOpenAI(api_key='fixture-key', base_url='http://unused.invalid')
    processor = BatchProcessor(
        client=client,
        model_string='model',
        logger=_Logger(),
        queue_enabled=caller_enabled,
        queue_url='redis://unused:6379/0' if expected else None,
        queue_fabric_group_id='fabric' if expected else None,
    )
    try:
        assert processor.uses_queued_dispatch is expected
        if expected:
            assert isinstance(processor._get_queued_executor(), QueuedDispatchProducer)
        else:
            assert processor._queue_config.queue_url is None
            assert processor._queued_executor is None
    finally:
        processor.close()
        asyncio.run(client.close())


def test_enabled_batch_processor_constructs_canonical_producer_without_fallback() -> (
    None
):
    import os

    processor = _build_processor(client=None, queue_enabled=True)
    processor._queued_executor_lock = threading.Lock()
    processor._queue_pid = os.getpid()
    executor = processor._get_queued_executor()
    try:
        assert isinstance(executor, QueuedDispatchProducer)
        assert processor._get_queued_executor() is executor
    finally:
        executor.close()


def test_multimodal_durable_envelope_preserves_content_and_context() -> None:
    context = RequestContext(
        ref_id='doc', requested_pages=(0,), job_id='job', work_unit_id='node'
    )
    call = _call(
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'describe'},
                    {
                        'type': 'image_url',
                        'image_url': {'url': 'data:image/png;base64,ZmFrZQ=='},
                        'min_pixels': 401408,
                    },
                ],
            }
        ],
        context=context,
    )
    envelope = QueuedCompletionEnvelope(
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
        call=call,
    )
    restored = QueuedCompletionEnvelope.from_json(envelope.to_json())
    assert restored.call.messages == call.messages
    assert restored.call.context == context
    assert restored.schema_version == 1


@pytest.mark.parametrize('schema', [None, True, '1', 0, 2])
def test_durable_envelope_rejects_missing_or_unsupported_integer_schema(
    schema: object,
) -> None:
    envelope = QueuedCompletionEnvelope(
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
        call=_call([]),
    )
    payload = json.loads(envelope.to_json())
    if schema is None:
        payload.pop('schema_version')
    else:
        payload['schema_version'] = schema
    with pytest.raises(ValueError, match='Unsupported queued completion schema'):
        QueuedCompletionEnvelope.from_json(json.dumps(payload))
