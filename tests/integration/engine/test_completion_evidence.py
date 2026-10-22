import asyncio
import json
import subprocess
import time
from dataclasses import replace
from uuid import uuid4

import pytest
from marie.engine.completion_contract import (
    CompletionCallParams,
    QueuedCompletionEnvelope,
    RequestContext,
)
from marie.engine.completion_observer import CompletionAttempt
from marie.engine.llm_queue.config import LlmQueueProducerConfig
from marie.engine.llm_queue.endpoint import ExecutionOutcome, RegisteredEndpoint
from marie.engine.llm_queue.producer import QueuedDispatchProducer
from marie.engine.llm_queue.request_dispatcher import DispatchLane, RequestDispatcher
from marie.engine.llm_queue.store import (
    OwnerToken,
    RequestStore,
    RoutingManifestProjection,
    StaleOwner,
    StoreLimits,
)


@pytest.fixture(scope="module")
def queue_url():
    container = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "-p",
            "127.0.0.1::6379",
            "valkey/valkey:9-alpine",
            "valkey-server",
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        text=True,
    ).strip()
    try:
        address = subprocess.check_output(
            ["docker", "port", container, "6379/tcp"], text=True
        ).strip()
        yield "redis://" + address + "/0"
    finally:
        subprocess.run(["docker", "stop", container], check=True, capture_output=True)


def executing_store(url, *, allowance=32768):
    store = RequestStore(
        url, fabric_id=uuid4().hex, limits=StoreLimits(result_allowance=allowance)
    )
    owner = store.acquire_owner("dispatcher", lease_ms=60000)
    store.configure_route(
        owner,
        "pool",
        "endpoint",
        revision="r1",
        execution_limit=2,
        execution_bytes=1024**2,
    )
    producer = store.create_producer(lease_ms=60000)
    request = QueuedCompletionEnvelope(
        fabric_group_id=store.keys.fabric_id,
        producer_id=producer,
        attempt_id="attempt",
        pool_id="pool",
        endpoint_id="endpoint",
        config_revision="r1",
        logical_batch_id="batch",
        logical_task_id="batch_task_0",
        item_index=0,
        expires_at_ms=store.server_time_ms() + 30000,
        call=CompletionCallParams(model="model", messages=[]),
        capture_attempts=True,
    )
    assert store.admit(request).disposition == "admitted"
    assert (
        store.claim(owner, "attempt", pool_id="pool", claim_id="claim").disposition
        == "claimed"
    )
    assert (
        store.authorize_start(owner, "attempt", claim_id="claim").disposition
        == "started"
    )
    return store, owner, producer


def test_durable_attempts_are_idempotent_fenced_and_retained_until_ack(queue_url):
    store, owner, producer = executing_store(queue_url)
    try:
        attempt = CompletionAttempt(
            "batch_task_0", 1, "  raw response  ", "length", "model", 2, 3, 4, None
        )
        arguments = dict(
            claim_id="claim", execution_seq=1, physical_attempt=1, attempt=attempt
        )
        assert (
            store.record_completion_attempt(owner, "attempt", **arguments).disposition
            == "recorded"
        )
        assert (
            store.record_completion_attempt(owner, "attempt", **arguments).disposition
            == "existing"
        )
        with pytest.raises(StaleOwner):
            store.record_completion_attempt(
                OwnerToken("other", owner.generation), "attempt", **arguments
            )
        assert (
            store.record_completion_attempt(
                owner, "attempt", **dict(arguments, claim_id="other")
            ).disposition
            == "stale_claim"
        )
        assert (
            store.finish(
                owner,
                "attempt",
                claim_id="claim",
                execution_seq=1,
                result={"choices": []},
            ).disposition
            == "finished"
        )
        result = store.read_result(producer, "attempt")
        assert result["attempts"][0]["raw_response"] == "  raw response  "
        assert len(result["attempts"]) == 1
        assert store.usage()["reserved_items"] == 0
        assert store.usage()["storage_bytes"] == store.limits.metadata_bytes + len(
            json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode()
        )
        store.ack_result(producer, "attempt")
        assert store.usage()["records"] == 0
    finally:
        store.close()


def test_evidence_overflow_fails_result_and_releases_capacity(queue_url):
    store, owner, producer = executing_store(queue_url, allowance=512)
    try:
        attempt = CompletionAttempt(
            "batch_task_0", 1, "x" * 2048, "stop", "model", 2, 3, 4, None
        )
        reply = store.record_completion_attempt(
            owner,
            "attempt",
            claim_id="claim",
            execution_seq=1,
            physical_attempt=1,
            attempt=attempt,
        )
        assert reply.disposition == "evidence_too_large"
        store.finish(
            owner, "attempt", claim_id="claim", execution_seq=1, result={"choices": []}
        )
        assert store.metadata("attempt").state == "failed"
        assert (
            store.read_result(producer, "attempt")["evidence_error"]
            == "evidence_too_large"
        )
        assert store.usage()["reserved_items"] == 0
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [None, "observer", "store", "validation", "cancel", "interrupt"]
)
@pytest.mark.parametrize("prewrite_failure", [False, True])
async def test_dispatcher_delivers_raw_attempts_before_callback_and_settles(
    queue_url, failure, prewrite_failure, monkeypatch
):
    store = RequestStore(queue_url, fabric_id=uuid4().hex)
    calls = []
    provider_waiting = asyncio.Event()
    provider_release = asyncio.Event()
    if failure == "store":

        def fail_write(*args, **kwargs):
            raise OSError("evidence store write failed")

        monkeypatch.setattr(store, "record_completion_attempt", fail_write)

    class Client:
        def __init__(self, endpoint):
            pass

        async def execute(self, call, **kwargs):
            calls.append(call)
            if len(calls) == 1:
                if prewrite_failure:
                    return ExecutionOutcome(
                        category="connection_refused",
                        retryable=True,
                        remote_settled=True,
                        request_started=False,
                        availability_failure=True,
                    )
                return ExecutionOutcome(
                    category="rate_limited", retryable=True, remote_settled=True
                )
            if failure == "interrupt":
                raise asyncio.CancelledError("provider interrupted")
            if failure == "cancel":
                provider_waiting.set()
                await provider_release.wait()
            return ExecutionOutcome(
                response={
                    "model": "model",
                    "choices": [
                        {
                            "message": {"content": ' {"ok":true} '},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                },
                remote_settled=True,
                availability_success=True,
            )

        async def close(self):
            pass

    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint("endpoint", "http://127.0.0.1:9/v1", allow_loopback=True)
        ],
        lanes=[DispatchLane("pool", "endpoint")],
        client_factory=Client,
        poll_seconds=0.01,
        retry_min_ms=1,
        retry_max_ms=2,
    )
    producer = QueuedDispatchProducer(
        config=LlmQueueProducerConfig.from_env(
            enabled=True, queue_url=queue_url, fabric_group_id=store.keys.fabric_id
        )
    )
    attempts, results = [], []

    def observe(attempt):
        attempts.append(attempt)
        if failure == "observer":
            raise OSError("evidence disk full")

    def result(task_id, response):
        assert len(attempts) == 2
        if failure == "validation":
            raise ValueError("candidate failed validation")
        results.append(response)

    try:
        await runtime.start()
        deadline = time.monotonic() + 5
        while runtime.owner is None or store.resolve_route("pool") is None:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        store.project_routing_manifest(
            RoutingManifestProjection(
                "job",
                "work",
                store.keys.fabric_id,
                1,
                "a" * 64,
                "pool",
                "endpoint",
                "r1",
            )
        )
        execution = asyncio.create_task(
            asyncio.to_thread(
                producer.execute,
                calls=[
                    CompletionCallParams(
                        model="model",
                        messages=[],
                        response_format={"type": "json_object"},
                        context=RequestContext(job_id="job", work_unit_id="work"),
                    )
                ],
                batch_request_id="batch",
                batch_timeout=1 if failure == "interrupt" else 5,
                attempt_observer=observe,
                on_result=result,
            )
        )
        if failure == "cancel":
            await asyncio.wait_for(provider_waiting.wait(), 5)
            attempt_id = next(
                iter(store.client.smembers(store.keys.members(producer.producer_id)))
            )
            store.cancel_or_expire(producer.producer_id, attempt_id)
            provider_release.set()
        batch = await asyncio.wait_for(execution, timeout=10)
        if failure == "interrupt":
            attempt_id = next(
                iter(store.client.smembers(store.keys.members(producer.producer_id)))
            )
            for _ in range(100):
                saved = store.read_result(producer.producer_id, attempt_id)
                if saved is not None:
                    break
                await asyncio.sleep(0.01)
            assert len(saved["attempts"]) == 2
            assert saved["attempts"][-1]["error"] == "execution_interrupted"
            assert batch[0].error is not None
            assert store.usage()["reserved_items"] == 1
            return
        assert len(calls) == (1 if failure == "store" else 2)
        if attempts:
            assert attempts[0].error == (
                "connection_refused" if prewrite_failure else "rate_limited"
            )
        if failure:
            assert batch[0].error is not None
            assert not results
        else:
            assert [a.attempt_number for a in attempts] == [1, 2]
            if prewrite_failure:
                assert [a.execution_sequence for a in attempts] == [1, 1]
                assert attempts[0].claim_id != attempts[1].claim_id
            assert attempts[1].raw_response == ' {"ok":true} '
            assert attempts[1].finish_reason == "length"
            assert attempts[1].input_tokens == 2
            assert results == ['{"ok":true}']
        assert store.usage()["reserved_items"] == 0
        if failure == "observer":
            assert store.usage()["records"] == 1
        elif failure in {None, "validation"}:
            assert store.usage()["records"] == 0
    finally:
        await asyncio.to_thread(producer.close)
        await runtime.stop()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"choices": {"bad": "shape"}},
        {"choices": "invalid"},
        {"choices": 1},
        {"choices": []},
        {},
        {"choices": [None]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": 1}}]},
        {"choices": [{"message": {"content": [{"type": "text", "text": 1}]}}]},
    ],
    ids=[
        "mapping-choices",
        "string-choices",
        "scalar-choices",
        "empty-choices",
        "missing-choices",
        "missing-choice",
        "missing-message",
        "scalar-content",
        "scalar-text-part",
    ],
)
async def test_malformed_completion_preserves_attempt_and_settles_remote(
    queue_url, response
):
    store = RequestStore(queue_url, fabric_id=uuid4().hex)
    calls = []
    attempts = []

    class Client:
        def __init__(self, endpoint):
            pass

        async def execute(self, call, **kwargs):
            calls.append(call)
            return ExecutionOutcome(response=response, remote_settled=True)

        async def close(self):
            pass

    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint("endpoint", "http://127.0.0.1:9/v1", allow_loopback=True)
        ],
        lanes=[DispatchLane("pool", "endpoint")],
        client_factory=Client,
        poll_seconds=0.01,
    )
    producer = QueuedDispatchProducer(
        config=LlmQueueProducerConfig.from_env(
            enabled=True, queue_url=queue_url, fabric_group_id=store.keys.fabric_id
        )
    )
    try:
        await runtime.start()
        for _ in range(500):
            if runtime.owner is not None and store.resolve_route("pool"):
                break
            await asyncio.sleep(0.01)
        store.project_routing_manifest(
            RoutingManifestProjection(
                "job",
                "work",
                store.keys.fabric_id,
                1,
                "a" * 64,
                "pool",
                "endpoint",
                "r1",
            )
        )
        results = await asyncio.wait_for(
            asyncio.to_thread(
                producer.execute,
                calls=[
                    CompletionCallParams(
                        model="model",
                        messages=[],
                        context=RequestContext(job_id="job", work_unit_id="work"),
                    )
                ],
                batch_request_id="batch",
                batch_timeout=1,
                attempt_observer=attempts.append,
            ),
            5,
        )
        assert len(calls) == 1
        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].error.category == "invalid_completion"
        assert results[0].response is None
        assert len(attempts) == 1
        assert attempts[0].raw_response is None
        assert attempts[0].error == "invalid_response"
        assert store.usage()["reserved_items"] == 0
        assert store.usage()["reserved_bytes"] == 0
    finally:
        await asyncio.to_thread(producer.close)
        await runtime.stop()
        store.close()


def test_cancelled_delivery_retains_evidence_without_releasing_unsettled_remote(
    queue_url,
):
    store, owner, producer = executing_store(queue_url)
    try:
        evidence = CompletionAttempt(
            "batch_task_0", 1, None, None, None, None, None, 4, "connection_lost"
        )
        store.record_completion_attempt(
            owner,
            "attempt",
            claim_id="claim",
            execution_seq=1,
            physical_attempt=1,
            attempt=evidence,
        )
        store.cancel_or_expire(producer, "attempt")
        assert (
            store.read_result(producer, "attempt")["attempts"][0]["error"]
            == "connection_lost"
        )
        assert store.usage()["reserved_items"] == 1
        store.ack_result(producer, "attempt")
        assert store.usage()["reserved_items"] == 1
        store.settle_remote(
            owner,
            "attempt",
            claim_id="claim",
            execution_seq=1,
            evidence="remote_cancelled",
        )
        assert store.usage()["reserved_items"] == 0
    finally:
        store.close()


def test_provider_can_record_completion_after_delivery_cancellation(queue_url):
    store, owner, producer = executing_store(queue_url)
    try:
        store.cancel_or_expire(producer, "attempt")
        assert (
            store.usage()["storage_bytes"]
            == store.limits.metadata_bytes + store.limits.result_allowance
        )
        evidence = CompletionAttempt(
            "batch_task_0", 1, "late raw response", "stop", "model", 2, 3, 4, None
        )
        assert (
            store.record_completion_attempt(
                owner,
                "attempt",
                claim_id="claim",
                execution_seq=1,
                physical_attempt=1,
                attempt=evidence,
            ).disposition
            == "recorded"
        )
        assert (
            store.read_result(producer, "attempt")["attempts"][0]["raw_response"]
            == "late raw response"
        )
        assert store.usage()["reserved_items"] == 1
        store.settle_remote(
            owner,
            "attempt",
            claim_id="claim",
            execution_seq=1,
            evidence="remote_completed",
        )
        result = store.read_result(producer, "attempt")
        assert store.usage()["reserved_items"] == 0
        assert store.usage()["storage_bytes"] == store.limits.metadata_bytes + len(
            json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode()
        )
    finally:
        store.close()


def test_late_evidence_overflow_keeps_cancelled_storage_bounded(queue_url):
    store, owner, producer = executing_store(queue_url, allowance=512)
    try:
        store.cancel_or_expire(producer, "attempt")
        evidence = CompletionAttempt(
            "batch_task_0", 1, "x" * 2048, "stop", "model", 2, 3, 4, None
        )
        assert (
            store.record_completion_attempt(
                owner,
                "attempt",
                claim_id="claim",
                execution_seq=1,
                physical_attempt=1,
                attempt=evidence,
            ).disposition
            == "evidence_too_large"
        )
        assert store.usage()["storage_bytes"] == store.limits.metadata_bytes + 512
        result = store.read_result(producer, "attempt")
        assert result["evidence_error"] == "evidence_too_large"
        assert result["attempts"] == []
        store.settle_remote(
            owner,
            "attempt",
            claim_id="claim",
            execution_seq=1,
            evidence="remote_completed",
        )
        assert store.usage()["reserved_items"] == 0
        assert store.usage()["storage_bytes"] == store.limits.metadata_bytes + len(
            json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode()
        )
    finally:
        store.close()
