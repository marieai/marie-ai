import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from marie.engine.batch_processor import (
    BatchProcessor,
    BatchResult,
    _create_retry_decorator,
)
from marie.engine.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from marie.engine.completion_contract import (
    CompletionCallParams,
    RequestContext,
    apply_repetition_recovery,
    build_completion_call,
    has_terminal_repetition,
    recover_complete_json,
)
from marie.engine.exceptions import MaxTokensExceededError, RepetitionError
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.producer import _batch_result

from marie.excepts import BatchExecutionError, CircuitOpenError


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _call(content: str, **completion_params) -> CompletionCallParams:
    return build_completion_call(
        model="test-model",
        messages=[{"role": "user", "content": content}],
        completion_params=completion_params or None,
        stream=False,
    )


def _build_processor(
    max_concurrency: int = 20,
    batch_timeout: float = 600.0,
    backend_address: str = "http://test-backend:8000",
) -> BatchProcessor:
    processor = object.__new__(BatchProcessor)
    processor.client = None
    processor.model_string = "test-model"
    processor.logger = _Logger()
    processor.max_concurrency = max_concurrency
    processor.batch_timeout = batch_timeout
    processor.backend_address = backend_address
    processor.default_completion_params = {}
    processor._shared_request_semaphore = None
    processor._shared_request_semaphore_loop = None
    processor._direct_runner = None
    processor._direct_runner_lock = threading.Lock()
    processor._circuit_breaker = CircuitBreaker(
        config=CircuitBreakerConfig(),
        logger=_Logger(),
    )
    processor._gate_lock = None
    processor._gate_lock_loop = None
    processor._queue_client = None
    processor._queued_executor = None
    processor._queue_config = LlmQueueConfig.from_env(enabled=False)
    processor._queue_mode_logged = False
    return processor


# ---------------------------------------------------------------------------
# Existing test: shared concurrency limit across overlapping calls
# ---------------------------------------------------------------------------
def test_load_batched_completion_calls_shares_concurrency_limit_across_overlapping_calls():
    processor = _build_processor(max_concurrency=2)
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def fake_completion(**kwargs):
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            task_id = kwargs["task_id"]
            return task_id, f"response:{task_id}"
        finally:
            async with lock:
                active -= 1

    processor.acompletion_call_with_retry = fake_completion

    async def run():
        results = await asyncio.gather(
            processor.load_batched_completion_calls(
                calls=[_call("a"), _call("b"), _call("c")],
                request_id="req-1",
            ),
            processor.load_batched_completion_calls(
                calls=[_call("d"), _call("e"), _call("f")],
                request_id="req-2",
            ),
        )

        assert len(results) == 2
        assert peak == 2

    asyncio.run(run())


def test_batch_generate_calls_shares_concurrency_limit_across_caller_threads():
    processor = _build_processor(max_concurrency=2)
    active = 0
    peak = 0
    loop_ids = set()
    lock = threading.Lock()

    async def fake_completion(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            loop_ids.add(id(asyncio.get_running_loop()))
        try:
            await asyncio.sleep(0.02)
            task_id = kwargs["task_id"]
            return task_id, f"response:{task_id}"
        finally:
            with lock:
                active -= 1

    processor.acompletion_call_with_retry = fake_completion

    def run_batch(request_id: str):
        return processor.batch_generate_calls(
            calls=[_call("a"), _call("b"), _call("c")],
            request_id=request_id,
        )

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(run_batch, ["req-1", "req-2", "req-3"]))
    finally:
        processor.close()

    assert results == [
        ["response:req-1_task_0", "response:req-1_task_1", "response:req-1_task_2"],
        ["response:req-2_task_0", "response:req-2_task_1", "response:req-2_task_2"],
        ["response:req-3_task_0", "response:req-3_task_1", "response:req-3_task_2"],
    ]
    assert peak == 2
    assert len(loop_ids) == 1


# ---------------------------------------------------------------------------
# Test: SDK retries disabled (max_retries=0)
# ---------------------------------------------------------------------------
def test_openai_sdk_retries_disabled():
    """Verify that OpenAIEngine builds its client through the shared helper."""
    import os
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with (
            mock.patch(
                "marie.engine.openai_engine.build_async_openai_client"
            ) as mock_builder,
            mock.patch(
                "marie.engine.batch_processor.AsyncOpenAI",
                new=object,
            ),
        ):
            mock_client = mock.MagicMock()
            mock_client.models.list.return_value = []
            mock_builder.return_value = mock_client

            from marie.engine.openai_engine import OpenAIEngine

            OpenAIEngine(
                model_name="test-model",
                base_url="http://localhost:8000",
                is_multimodal=False,
            )

            mock_builder.assert_called_once_with(
                api_key="test-key",
                base_url="http://localhost:8000",
            )


def test_openai_engine_normalizes_calls_before_strategy_split():
    import os
    import unittest.mock as mock

    captured = {}

    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with (
            mock.patch(
                "marie.engine.openai_engine.build_async_openai_client"
            ) as mock_builder,
            mock.patch(
                "marie.engine.batch_processor.AsyncOpenAI",
                new=object,
            ),
        ):
            mock_client = mock.MagicMock()
            mock_client.models.list.return_value = []
            mock_builder.return_value = mock_client

            from marie.engine.openai_engine import OpenAIEngine

            engine = OpenAIEngine(
                model_name="test-model",
                base_url="http://localhost:8000",
                is_multimodal=False,
            )

            def fake_batch_generate_calls(
                *, calls, on_result=None, metadata=None, **kwargs
            ):
                captured["calls"] = calls
                captured["metadata"] = metadata
                captured["extra_kwargs"] = kwargs
                return ["ok"]

            engine.batch_processor.batch_generate_calls = fake_batch_generate_calls

            context = RequestContext(
                ref_id="PID_1",
                ref_type="stress",
                page_number=1,
            )

            responses = engine.batch_generate(
                ["hello"],
                system_prompt="Return only the requested data.",
                guided_json={"type": "object"},
                metadata={"source": "unit-test"},
                request_contexts=[context],
                completion_params={"temperature": 0.25},
            )

    assert responses == ["ok"]
    assert len(captured["calls"]) == 1
    call = captured["calls"][0]
    assert isinstance(call, CompletionCallParams)
    assert call.messages[0]["role"] == "system"
    assert call.messages[0]["content"] == "Return only the requested data."
    assert call.messages[1]["role"] == "user"
    assert call.messages[1]["content"] == "hello"
    assert call.temperature == 0.25
    assert call.extra_body["guided_json"] == {"type": "object"}
    assert call.context == context
    assert captured["metadata"] == {"source": "unit-test"}
    assert "request_contexts" not in captured["extra_kwargs"]


def test_completion_call_context_is_not_provider_payload():
    context = RequestContext(
        ref_id="PID_1.tif",
        ref_type="stress",
        page_number=1,
    )
    call = build_completion_call(
        model="test-model",
        messages=[{"role": "user", "content": "a"}],
        completion_params={"temperature": 0.25, "context": {"ref_id": "bad"}},
        context=context,
    )

    create_kwargs = call.to_create_kwargs()

    assert call.context == context
    assert create_kwargs["temperature"] == 0.25
    assert "context" not in create_kwargs


@pytest.mark.parametrize(
    ("recovery", "unsupported"),
    [
        ({1: 0.7}, "1"),
        ({1: 0.7, "unexpected": 0.8}, "1, unexpected"),
        ({None: 0.7, 1: 0.8}, "1, None"),
    ],
)
def test_repetition_recovery_rejects_non_string_keys(
    recovery: dict[object, object], unsupported: str
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _call("hello", repetition_recovery=recovery)

    assert str(exc_info.value) == (
        "Unsupported repetition recovery parameters: " + unsupported
    )


def test_repetition_recovery_round_trips_without_entering_provider_payload():
    call = build_completion_call(
        model="qwen",
        messages=[{"role": "user", "content": "extract"}],
        completion_params={
            "temperature": 0.0,
            "repetition_recovery": {
                "temperature": 0.7,
                "top_p": 0.8,
                "presence_penalty": 1.5,
                "extra_body": {"top_k": 20},
            },
        },
    )

    restored = CompletionCallParams.from_dict(call.to_dict())
    recovered = apply_repetition_recovery(restored)

    assert "repetition_recovery" not in call.to_create_kwargs()
    assert recovered is not None
    assert recovered.temperature == 0.7
    assert recovered.top_p == 0.8
    assert recovered.presence_penalty == 1.5
    assert recovered.extra_body == {"top_k": 20}
    assert recovered.messages == call.messages
    assert recovered.repetition_recovery is None


@pytest.mark.parametrize(
    "text",
    [
        "value value value value value value value value value value value value",
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
        "abc def abc def abc def abc def abc def",
    ],
)
def test_terminal_repetition_detects_runaway_suffixes(text: str):
    assert has_terminal_repetition(text) is True


@pytest.mark.parametrize(
    "text",
    [
        '{"values":[1,1,1,1]}',
        "a short phrase a short phrase",
        "ordinary model output",
    ],
)
def test_terminal_repetition_ignores_short_valid_content(text: str):
    assert has_terminal_repetition(text) is False


def test_complete_json_is_recoverable_when_length_stops_before_eos():
    text = '  {"document_type":"invoice","fields":[]}  '

    assert recover_complete_json(text) == '{"document_type":"invoice","fields":[]}'


def test_complete_json_prefix_is_recoverable_before_repetitive_tail():
    text = '{"ok":true}\n' + "loop " * 20

    assert recover_complete_json(text) == '{"ok":true}'


def test_incomplete_json_is_not_recovered():
    assert recover_complete_json('{"document_type":"invoice","fields":') is None


def test_queued_repetition_result_preserves_terminal_error_type():
    result = _batch_result(
        "task-1",
        "failed",
        {"error": "repetition", "category": "repetition"},
    )

    assert isinstance(result.error, RepetitionError)


def test_openai_engine_request_context_length_mismatch_warns_and_falls_back():
    class _RecordingLogger(_Logger):
        def __init__(self):
            self.warnings = []

        def warning(self, *args, **kwargs):
            self.warnings.append(args)

    from marie.engine.openai_engine import OpenAIEngine

    engine = object.__new__(OpenAIEngine)
    engine.logger = _RecordingLogger()
    engine.model_string = "test-model"
    engine.system_prompt = None
    engine.is_multimodal = False
    engine.batch_processor = type(
        "BatchProcessor",
        (),
        {"default_completion_params": {}},
    )()

    calls = engine._build_completion_calls(
        batch_content=["a", "b"],
        request_contexts=[RequestContext(page_number=1)],
    )

    assert [call.context for call in calls] == [None, None]
    assert engine.logger.warnings
    assert engine.logger.warnings[0][0].startswith(
        "Ignoring request_contexts length mismatch"
    )


# ---------------------------------------------------------------------------
# Test: Circuit breaker fast-fail when circuit is open
# ---------------------------------------------------------------------------
def test_circuit_breaker_fast_fail_when_open():
    """Tasks should fail fast with CircuitOpenError when the circuit is open."""
    processor = _build_processor(max_concurrency=10)
    cb = processor._circuit_breaker
    address = processor.backend_address

    # Force circuit open by recording enough failures
    for _ in range(5):
        cb.record_failure(address)
    assert cb.get_state(address) == CircuitState.OPEN

    async def fake_completion(**kwargs):
        # Should never be called
        raise AssertionError("Should not reach completion when circuit is open")

    processor.completion_non_streaming_call = fake_completion

    async def run():
        results = await processor.load_batched_completion_calls(
            calls=[_call("a"), _call("b"), _call("c")],
            request_id="req-cb-open",
        )

        # All tasks should have CircuitOpenError
        for br in results:
            assert br is not None
            assert isinstance(br.error, CircuitOpenError)
            assert br.response is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test: HALF_OPEN probe slot reservation and release
# ---------------------------------------------------------------------------
def test_half_open_probe_slot_reservation():
    """In HALF_OPEN state, only half_open_max_calls tasks should proceed."""
    processor = _build_processor(max_concurrency=10)
    cb = processor._circuit_breaker
    address = processor.backend_address

    # Force circuit open then let recovery timeout pass
    for _ in range(5):
        cb.record_failure(address)
    assert cb.get_state(address) == CircuitState.OPEN

    # Manually transition to HALF_OPEN
    import time as _time

    stats = cb._stats[address]
    stats.open_time = _time.monotonic() - 60  # 60s ago, recovery_timeout=30s

    completion_count = 0

    async def fake_completion(**kwargs):
        nonlocal completion_count
        completion_count += 1
        await asyncio.sleep(0.01)
        task_id = kwargs["task_id"]
        return task_id, f"response:{task_id}"

    processor.completion_non_streaming_call = fake_completion

    async def run():
        results = await processor.load_batched_completion_calls(
            calls=[_call("a"), _call("b"), _call("c")],
            request_id="req-half-open",
        )

        # With half_open_max_calls=1, only 1 task should proceed to completion
        # The rest should get CircuitOpenError
        succeeded = [br for br in results if br and br.error is None]
        circuit_rejected = [
            br for br in results if br and isinstance(br.error, CircuitOpenError)
        ]

        assert len(succeeded) == 1
        assert len(circuit_rejected) == 2

        # The half_open_calls counter should be released back to 0
        final_stats = cb.get_stats(address)
        assert final_stats.half_open_calls == 0

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test: BatchExecutionError raised instead of generic RuntimeError
# ---------------------------------------------------------------------------
def test_batch_generate_calls_raises_batch_execution_error_on_failure():
    """batch_generate_calls should raise BatchExecutionError when tasks fail."""
    processor = _build_processor(max_concurrency=10)

    async def failing_completion(**kwargs):
        task_id = kwargs["task_id"]
        raise RuntimeError(f"Simulated failure for {task_id}")

    processor.acompletion_call_with_retry = failing_completion

    try:
        with pytest.raises(BatchExecutionError) as exc_info:
            processor.batch_generate_calls(
                calls=[_call("msg1"), _call("msg2")],
            )
    finally:
        processor.close()

    err = exc_info.value
    assert err.request_id  # Should have a request_id
    assert err.total == 2
    assert len(err.failed_results) == 2
    assert isinstance(err.primary_error, RuntimeError)
    assert err.primary_task_id is not None
    assert "Simulated failure" in str(err)
    # Each failed result should have the exception
    for fr in err.failed_results:
        assert fr.error is not None


# ---------------------------------------------------------------------------
# Test: unexpected completion exceptions propagate as task failures
# ---------------------------------------------------------------------------
def test_completion_non_streaming_call_reraises_unexpected_exception():
    """Unexpected task errors must propagate so the batch marks failure."""
    import os
    import unittest.mock as mock
    from types import SimpleNamespace

    processor = object.__new__(BatchProcessor)
    processor.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=mock.AsyncMock(side_effect=ValueError("unexpected failure"))
            )
        )
    )
    processor.model_string = "test-model"
    processor.logger = _Logger()
    processor.backend_address = "http://test-backend:8000"
    processor.default_completion_params = {}
    processor._circuit_breaker = CircuitBreaker(
        config=CircuitBreakerConfig(),
        logger=_Logger(),
    )

    async def run():
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="unexpected failure"):
                await processor.completion_non_streaming_call(
                    call=_call("hello"),
                    task_id="task-1",
                    request_id="req-1",
                )

    asyncio.run(run())


def test_max_tokens_exceeded_is_not_retried() -> None:
    attempts = 0

    @_create_retry_decorator(3)
    async def truncated_completion() -> None:
        nonlocal attempts
        attempts += 1
        raise MaxTokensExceededError()

    with pytest.raises(MaxTokensExceededError):
        asyncio.run(truncated_completion())

    assert attempts == 1


@pytest.mark.parametrize(
    ("content", "complete"),
    [
        ('{"document_type":"invoice"}', True),
        ('["invoice"]', True),
        ('{"document_type":', False),
        ('["invoice",', False),
    ],
)
@pytest.mark.parametrize(
    ("completion_params", "expect_recovery"),
    [
        ({}, False),
        ({"response_format": {"type": "text"}}, False),
        ({"extra_body": {"guided_json": None}}, False),
        ({"extra_body": {"top_k": 20}}, False),
        ({"response_format": {"type": "json_object"}}, True),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "test", "schema": {}},
                }
            },
            True,
        ),
        ({"extra_body": {"guided_json": {}}}, True),
    ],
)
def test_length_response_recovers_json_only_with_json_contract(
    content: str, complete: bool, completion_params: dict, expect_recovery: bool
) -> None:
    import unittest.mock as mock

    import marie.engine.batch_processor as batch_processor_module

    processor = _build_processor(max_concurrency=1)
    calls = []

    async def complete_json(client, call):
        calls.append(call)
        return {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "length",
                }
            ]
        }

    async def run():
        with (
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            mock.patch.object(
                batch_processor_module,
                "execute_completion_call",
                new=complete_json,
            ),
            mock.patch.object(processor, "save_debug_msg", new=mock.AsyncMock()),
        ):
            return await processor.acompletion_call_with_retry(
                max_retries=3,
                call=_call("hello", **completion_params),
                task_id="task-1",
                request_id="request-1",
            )

    if expect_recovery and complete:
        assert asyncio.run(run()) == ("task-1", content)
    else:
        with pytest.raises(MaxTokensExceededError):
            asyncio.run(run())
    assert len(calls) == 1


def test_repetitive_length_response_retries_once_with_recovery_sampling() -> None:
    import unittest.mock as mock

    import marie.engine.batch_processor as batch_processor_module

    processor = _build_processor(max_concurrency=1)
    calls = []
    responses = [
        {
            "choices": [
                {
                    "message": {"content": '{"value":"' + "loop " * 20},
                    "finish_reason": "length",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {"content": '{"value":"recovered"}'},
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    async def respond(client, call):
        calls.append(call)
        return responses.pop(0)

    async def run():
        with (
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            mock.patch.object(
                batch_processor_module,
                "execute_completion_call",
                new=respond,
            ),
            mock.patch.object(
                batch_processor_module,
                "wait_exponential",
                return_value=lambda retry_state: 0,
            ),
        ):
            return await processor.acompletion_call_with_retry(
                max_retries=3,
                call=_call(
                    "hello",
                    temperature=0.0,
                    repetition_recovery={"temperature": 0.7, "top_p": 0.8},
                ),
                task_id="task-1",
                request_id="request-1",
            )

    assert asyncio.run(run()) == ("task-1", '{"value":"recovered"}')
    assert [call.temperature for call in calls] == [0.0, 0.7]
    assert [call.top_p for call in calls] == [1.0, 0.8]


def test_persistent_repetition_stops_after_one_recovery_call() -> None:
    import unittest.mock as mock

    import marie.engine.batch_processor as batch_processor_module

    processor = _build_processor(max_concurrency=1)
    calls = []

    async def repeat(client, call):
        calls.append(call)
        return {
            "choices": [
                {
                    "message": {"content": '{"value":"' + "loop " * 20},
                    "finish_reason": "length",
                }
            ]
        }

    async def run():
        with (
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            mock.patch.object(
                batch_processor_module,
                "execute_completion_call",
                new=repeat,
            ),
            mock.patch.object(
                batch_processor_module,
                "wait_exponential",
                return_value=lambda retry_state: 0,
            ),
        ):
            with pytest.raises(RepetitionError):
                await processor.acompletion_call_with_retry(
                    max_retries=3,
                    call=_call(
                        "hello",
                        repetition_recovery={"temperature": 0.7, "top_p": 0.8},
                    ),
                    task_id="task-1",
                    request_id="request-1",
                )

    asyncio.run(run())
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Test: Batch timeout cancels and drains pending tasks
# ---------------------------------------------------------------------------
def test_batch_timeout_cancels_tasks():
    """Batch should raise asyncio.TimeoutError and cancel tasks on timeout."""
    processor = _build_processor(max_concurrency=10, batch_timeout=0.1)

    async def slow_completion(**kwargs):
        await asyncio.sleep(10)  # Way longer than timeout
        task_id = kwargs["task_id"]
        return task_id, "done"

    processor.acompletion_call_with_retry = slow_completion

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await processor.load_batched_completion_calls(
                calls=[_call("a"), _call("b")],
                request_id="req-timeout",
            )

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test: callback failure cancels unfinished inference
# ---------------------------------------------------------------------------
def test_callback_failure_cancels_pending_tasks():
    processor = _build_processor(max_concurrency=2)
    second_started = asyncio.Event()
    second_cancelled = asyncio.Event()

    async def fake_completion(**kwargs):
        task_id = kwargs["task_id"]
        if task_id.endswith("_0"):
            await second_started.wait()
            return task_id, "invalid response"

        second_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            second_cancelled.set()
            raise

    def failing_callback(task_id, response):
        raise ValueError("invalid structured response")

    processor.acompletion_call_with_retry = fake_completion

    async def run():
        with pytest.raises(ValueError, match="invalid structured response"):
            await processor.load_batched_completion_calls(
                calls=[_call("a"), _call("b")],
                request_id="req-callback",
                on_result=failing_callback,
            )

        assert second_cancelled.is_set()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test: failed inference does not invoke result callback
# ---------------------------------------------------------------------------
def test_failed_inference_does_not_invoke_result_callback():
    processor = _build_processor(max_concurrency=1)
    callbacks = []

    async def fake_completion(**kwargs):
        raise RuntimeError("simulated inference failure")

    processor.acompletion_call_with_retry = fake_completion

    async def run():
        results = await processor.load_batched_completion_calls(
            calls=[_call("a")],
            request_id="req-failed",
            on_result=lambda task_id, response: callbacks.append((task_id, response)),
        )
        assert isinstance(results[0].error, RuntimeError)

    asyncio.run(run())

    assert callbacks == []


# ---------------------------------------------------------------------------
# Test: retries re-check the circuit breaker
# ---------------------------------------------------------------------------
def test_retry_stops_when_circuit_opens():
    """A retry must not call a backend after its first failure opens the circuit."""
    import os
    import unittest.mock as mock

    import httpx
    import marie.engine.batch_processor as batch_processor_module
    from openai import APIConnectionError

    processor = _build_processor(max_concurrency=1)
    processor._circuit_breaker = CircuitBreaker(
        config=CircuitBreakerConfig(failure_threshold=1),
        logger=_Logger(),
    )
    attempts = 0
    wait_calls = 0

    def wait_strategy(retry_state):
        nonlocal wait_calls
        wait_calls += 1
        return 0

    async def failing_call(client, call):
        nonlocal attempts
        attempts += 1
        raise APIConnectionError(request=httpx.Request("POST", "http://test"))

    async def run():
        with (
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            mock.patch.object(
                batch_processor_module,
                "execute_completion_call",
                new=failing_call,
            ),
            mock.patch.object(
                batch_processor_module,
                "wait_exponential",
                return_value=wait_strategy,
            ),
        ):
            with pytest.raises(CircuitOpenError):
                await processor.acompletion_call_with_retry(
                    max_retries=3,
                    call=_call("hello"),
                    task_id="task-1",
                    request_id="request-1",
                )

    asyncio.run(run())

    assert attempts == 1
    assert wait_calls == 0


# ---------------------------------------------------------------------------
# Test: timed-out inference is not replayed
# ---------------------------------------------------------------------------
def test_api_timeout_is_not_retried():
    import unittest.mock as mock

    import httpx
    import marie.engine.batch_processor as batch_processor_module
    from openai import APITimeoutError

    processor = _build_processor(max_concurrency=1)
    attempts = 0
    wait_calls = 0

    def wait_strategy(retry_state):
        nonlocal wait_calls
        wait_calls += 1
        return 0

    async def timing_out_call(client, call):
        nonlocal attempts
        attempts += 1
        raise APITimeoutError(request=httpx.Request("POST", "http://test"))

    async def run():
        with (
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            mock.patch.object(
                batch_processor_module,
                "execute_completion_call",
                new=timing_out_call,
            ),
            mock.patch.object(
                batch_processor_module,
                "wait_exponential",
                return_value=wait_strategy,
            ),
        ):
            with pytest.raises(APITimeoutError):
                await processor.acompletion_call_with_retry(
                    max_retries=3,
                    call=_call("hello"),
                    task_id="task-1",
                    request_id="request-1",
                )

    asyncio.run(run())

    assert attempts == 1
    assert wait_calls == 0


# ---------------------------------------------------------------------------
# Test: process_batch does not retry BatchExecutionError
# ---------------------------------------------------------------------------
def test_process_batch_does_not_retry_batch_execution_error():
    """process_batch should propagate BatchExecutionError without retrying."""
    import unittest.mock as mock

    call_count = 0

    async def fake_acall(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise BatchExecutionError(
            request_id="test-req",
            failed_results=[BatchResult("t0", None, RuntimeError("fail"))],
            total=1,
        )

    from marie.extract.annotators.util import process_batch

    mock_engine = mock.MagicMock()
    batch = [(mock.MagicMock(), "prompt", "img.png")]

    async def run():
        nonlocal call_count
        # We patch the llm_call.acall to raise BatchExecutionError
        with mock.patch("marie.engine.llm_ops.LLMCall.acall", side_effect=fake_acall):
            with pytest.raises(BatchExecutionError):
                await process_batch(
                    batch=batch,
                    engine=mock_engine,
                    output_path="/tmp/test",
                    is_multimodal=False,
                    expect_output="json",
                )

        # Should have been called exactly once (no retries)
        assert call_count == 1

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test: process_batch does not replay a timed-out batch
# ---------------------------------------------------------------------------
def test_process_batch_does_not_retry_batch_timeout():
    """A batch timeout should propagate without replaying completed requests."""
    import unittest.mock as mock

    from marie.extract.annotators.util import process_batch

    call_count = 0

    async def fake_acall(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise asyncio.TimeoutError("simulated timeout")

    mock_engine = mock.MagicMock()
    batch = [(mock.MagicMock(), "prompt", "img.png")]

    async def run():
        with mock.patch("marie.engine.llm_ops.LLMCall.acall", side_effect=fake_acall):
            with pytest.raises(asyncio.TimeoutError, match="simulated timeout"):
                await process_batch(
                    batch=batch,
                    engine=mock_engine,
                    output_path="/tmp/test",
                    is_multimodal=False,
                    expect_output="json",
                )

    asyncio.run(run())

    assert call_count == 1


# ---------------------------------------------------------------------------
# Test: process_batch forwards its system prompt
# ---------------------------------------------------------------------------
def test_process_batch_forwards_system_prompt(tmp_path):
    import unittest.mock as mock

    from marie.extract.annotators import util

    captured = {}

    class FakeLLMCall:
        def __init__(self, engine, system_prompt=None):
            captured["system_prompt"] = system_prompt

        async def acall(self, prompts, **kwargs):
            kwargs["on_result"]("request_task_0", '{"ok": true}')
            return ['{"ok": true}']

    async def run():
        with mock.patch.object(util, "LLMCall", FakeLLMCall):
            await util.process_batch(
                batch=[(mock.MagicMock(), "prompt", "00001.png")],
                engine=mock.MagicMock(),
                output_path=str(tmp_path),
                is_multimodal=False,
                expect_output="json",
                system_prompt="Return valid JSON only.",
            )

    asyncio.run(run())

    assert captured["system_prompt"] == "Return valid JSON only."
    assert (tmp_path / "00001.json").is_file()


# ---------------------------------------------------------------------------
# Test: one malformed result does not discard completed siblings
# ---------------------------------------------------------------------------
def test_process_batch_finishes_siblings_before_raising_parse_error(tmp_path):
    import unittest.mock as mock

    from marie.engine.output_parser import JSONOutputParserError

    from marie.extract.annotators import util

    class FakeLLMCall:
        def __init__(self, engine, system_prompt=None):
            pass

        async def acall(self, prompts, **kwargs):
            kwargs["on_result"]("request_task_0", "invalid JSON")
            kwargs["on_result"]("request_task_1", '{"ok": true}')
            return ["invalid JSON", '{"ok": true}']

    async def run():
        with mock.patch.object(util, "LLMCall", FakeLLMCall):
            with pytest.raises(JSONOutputParserError):
                await util.process_batch(
                    batch=[
                        (mock.MagicMock(), "prompt", "00001.png"),
                        (mock.MagicMock(), "prompt", "00002.png"),
                    ],
                    engine=mock.MagicMock(),
                    output_path=str(tmp_path),
                    is_multimodal=False,
                    expect_output="json",
                )

    asyncio.run(run())

    assert not (tmp_path / "00001.json").exists()
    assert (tmp_path / "00002.json").is_file()


# ---------------------------------------------------------------------------
# Test: invalid JSON never creates a result file
# ---------------------------------------------------------------------------
def test_invalid_json_does_not_create_result_file(tmp_path):
    import unittest.mock as mock

    from marie.engine.output_parser import JSONOutputParserError

    from marie.extract.annotators.util import _write_single_result

    with pytest.raises(JSONOutputParserError):
        _write_single_result(
            b_image=mock.MagicMock(),
            b_prompt="prompt",
            b_image_path="00001.png",
            task_id="request_task_0",
            response="This response contains no JSON.",
            output_path=str(tmp_path),
            expect_output="json",
        )

    assert not (tmp_path / "00001.json").exists()


# ---------------------------------------------------------------------------
# Test: failed mini-batch lets active siblings finish
# ---------------------------------------------------------------------------
def test_scan_stops_new_work_and_finishes_active_batches(tmp_path):
    import unittest.mock as mock
    from types import SimpleNamespace

    from marie.extract.annotators import util

    (tmp_path / "00001.png").touch()
    (tmp_path / "00002.png").touch()
    (tmp_path / "00003.png").touch()
    second_started = asyncio.Event()
    second_finished = asyncio.Event()
    processed = []

    def fake_prepare_batch(*, file_batch, **kwargs):
        yield file_batch

    async def fake_process_batch(batch, *args, **kwargs):
        processed.extend(batch)
        if batch == ["00001.png"]:
            await second_started.wait()
            raise RuntimeError("first mini-batch failed")
        if batch == ["00002.png"]:
            second_started.set()
            await asyncio.sleep(0.01)
            second_finished.set()
            return
        raise AssertionError("new work started after a batch failure")

    async def run():
        with (
            mock.patch.object(
                util,
                "frames_from_file",
                side_effect=AssertionError("full-document frames must not be loaded"),
                create=True,
            ) as eager_frame_loader,
            mock.patch.object(
                util,
                "prepare_batch_with_meta_units",
                side_effect=fake_prepare_batch,
            ),
            mock.patch.object(
                util,
                "process_batch",
                side_effect=fake_process_batch,
            ),
        ):
            with pytest.raises(RuntimeError, match="first mini-batch failed"):
                await util.ascan_and_process_images(
                    source_dir=str(tmp_path),
                    output_dir=str(tmp_path),
                    prompt=mock.MagicMock(),
                    document=mock.MagicMock(),
                    engine=SimpleNamespace(
                        batch_processor=SimpleNamespace(max_concurrency=2)
                    ),
                    expect_output="json",
                    mini_batch_size=1,
                )

        assert second_finished.is_set()
        assert processed == ["00001.png", "00002.png"]
        eager_frame_loader.assert_not_called()

    asyncio.run(run())


def test_v3_scan_failure_cancels_active_batches(tmp_path):
    import unittest.mock as mock
    from types import SimpleNamespace

    from PIL import Image

    from marie.extract.annotators import util

    for name in ("00001.png", "00002.png", "00003.png"):
        with Image.new("RGB", (8, 8), "white") as image:
            image.save(tmp_path / name)
    second_started = asyncio.Event()
    second_cancelled = asyncio.Event()
    cancellations = []

    async def fake_process_batch(batch, *args, cancellation=None, **kwargs):
        cancellations.append(cancellation)
        if batch[0][2].endswith("00001.png"):
            await second_started.wait()
            raise RuntimeError("first mini-batch failed")
        if batch[0][2].endswith("00002.png"):
            second_started.set()
            while not cancellation.is_set():
                await asyncio.sleep(0.001)
            second_cancelled.set()
            return
        raise AssertionError("new work started after a batch failure")

    async def run():
        with mock.patch.object(
            util,
            "process_batch",
            side_effect=fake_process_batch,
        ):
            with pytest.raises(RuntimeError, match="first mini-batch failed"):
                await util.ascan_and_process_images(
                    source_dir=str(tmp_path),
                    output_dir=str(tmp_path),
                    prompt=mock.MagicMock(),
                    document=mock.MagicMock(),
                    engine=SimpleNamespace(
                        batch_processor=SimpleNamespace(
                            max_concurrency=2,
                            uses_v3_queue=True,
                        )
                    ),
                    expect_output="json",
                    mini_batch_size=1,
                )

        assert second_cancelled.is_set()
        assert len(cancellations) == 2
        assert cancellations[0] is cancellations[1]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test: reruns process only missing or invalid outputs
# ---------------------------------------------------------------------------
def test_scan_resumes_missing_and_invalid_outputs(tmp_path):
    import unittest.mock as mock
    from types import SimpleNamespace

    from marie.extract.annotators import util

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    for name in ("00001.png", "00002.png", "00003.png"):
        (source_dir / name).touch()
    (output_dir / "00001.json").write_text('{"complete": true}')
    (output_dir / "00002.json").write_text("incomplete")

    processed = []

    def fake_prepare_batch(*, file_batch, **kwargs):
        yield file_batch

    async def fake_process_batch(batch, *args, **kwargs):
        processed.extend(batch)

    async def run():
        with (
            mock.patch.object(
                util,
                "frames_from_file",
                side_effect=AssertionError("full-document frames must not be loaded"),
                create=True,
            ) as eager_frame_loader,
            mock.patch.object(
                util,
                "prepare_batch_with_meta_units",
                side_effect=fake_prepare_batch,
            ),
            mock.patch.object(
                util,
                "process_batch",
                side_effect=fake_process_batch,
            ),
        ):
            await util.ascan_and_process_images(
                source_dir=str(source_dir),
                output_dir=str(output_dir),
                prompt=mock.MagicMock(),
                document=mock.MagicMock(),
                engine=SimpleNamespace(
                    batch_processor=SimpleNamespace(max_concurrency=1)
                ),
                expect_output="json",
                mini_batch_size=1,
            )

        eager_frame_loader.assert_not_called()

    asyncio.run(run())

    assert processed == ["00002.png", "00003.png"]


# ---------------------------------------------------------------------------
# Test: load_batched_completion_calls returns BatchResult objects
# ---------------------------------------------------------------------------
def test_load_batched_completion_calls_returns_batch_results():
    """load_batched_completion_calls should return List[BatchResult] with full metadata."""
    processor = _build_processor(max_concurrency=10)

    async def fake_completion(**kwargs):
        task_id = kwargs["task_id"]
        return task_id, f"response:{task_id}"

    processor.acompletion_call_with_retry = fake_completion

    async def run():
        results = await processor.load_batched_completion_calls(
            calls=[_call("a"), _call("b")],
            request_id="req-br",
        )

        assert len(results) == 2
        for br in results:
            assert isinstance(br, BatchResult)
            assert br.error is None
            assert br.response is not None
            assert br.task_id.startswith("req-br_task_")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test: CircuitOpenError not counted as backend failure
# ---------------------------------------------------------------------------
def test_circuit_open_error_not_counted_as_failure():
    """CircuitOpenError should not increment failure count on the breaker."""
    processor = _build_processor(max_concurrency=10)
    cb = processor._circuit_breaker
    address = processor.backend_address

    # Force circuit open
    for _ in range(5):
        cb.record_failure(address)
    initial_failures = cb.get_stats(address).total_failures

    async def fake_completion(**kwargs):
        raise AssertionError("Should not be called")

    processor.acompletion_call_with_retry = fake_completion

    async def run():
        await processor.load_batched_completion_calls(
            calls=[_call("a")],
            request_id="req-no-count",
        )

    asyncio.run(run())

    # Total failures should not have increased
    assert cb.get_stats(address).total_failures == initial_failures
