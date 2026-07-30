import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from marie.engine.batch_processor import BatchProcessor, BatchResult
from marie.engine.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from marie.engine.completion_contract import (
    CompletionCallParams,
    RequestContext,
    build_completion_call,
)
from marie.engine.llm_queue.config import LlmQueueConfig

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

    processor.acompletion_call_with_retry = fake_completion

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

    processor.acompletion_call_with_retry = fake_completion

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
# Test: process_batch uses await asyncio.sleep instead of time.sleep
# ---------------------------------------------------------------------------
def test_process_batch_uses_async_sleep_on_retry():
    """Verify process_batch does not block the event loop on retry backoff."""
    import inspect
    import textwrap

    from marie.extract.annotators import util

    source = inspect.getsource(util.process_batch)

    # Should NOT contain time.sleep
    assert "time.sleep" not in source, (
        "process_batch still uses blocking time.sleep for retry backoff"
    )
    # Should contain asyncio.sleep
    assert "asyncio.sleep" in source, (
        "process_batch should use asyncio.sleep for retry backoff"
    )


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
