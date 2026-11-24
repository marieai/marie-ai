import asyncio
import logging
import threading
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from uuid import UUID

import pytest
from marie.engine.batch_processor import BatchProcessor
from marie.engine.completion_contract import (
    CompletionCallParams,
    PreparedCalls,
    RequestContext,
)
from marie.engine.openai_engine import OpenAIEngine
from openai import AsyncOpenAI
from PIL import Image

from marie.excepts import BatchExecutionError


def _engine(processor: BatchProcessor | SimpleNamespace) -> OpenAIEngine:
    engine = object.__new__(OpenAIEngine)
    engine.batch_processor = processor
    engine.logger = logging.getLogger(__name__)
    engine.model_string = 'test-model'
    engine.system_prompt = 'system'
    engine.is_multimodal = False
    return engine


def _call(content: str = 'hello') -> CompletionCallParams:
    return CompletionCallParams(
        model='test-model',
        messages=[{'role': 'user', 'content': content}],
        context=RequestContext(job_id='job', work_unit_id='unit'),
    )


@pytest.fixture
def processor() -> Iterator[BatchProcessor]:
    processor = BatchProcessor(
        AsyncOpenAI(api_key='test', base_url='http://127.0.0.1:1/v1'),
        'test-model',
        logging.getLogger(__name__),
        max_concurrency=3,
        batch_timeout=2,
        queue_enabled=False,
    )
    yield processor
    processor.close()


@pytest.mark.parametrize('queued', [False, True])
def test_complete_many_forwards_one_batch_and_controls_by_identity(
    queued: bool,
) -> None:
    outputs = ['first', None]
    processor = SimpleNamespace(
        uses_queued_dispatch=queued,
        batch_generate_calls=Mock(return_value=outputs),
        close=Mock(),
    )
    engine = _engine(processor)
    calls = [_call('first'), _call('second')]
    context = calls[0].context
    cancellation = threading.Event()
    on_result = Mock()
    observer = Mock()
    metadata = {'source': 'contract-test'}

    assert (
        engine.complete_many(
            calls,
            request_id='caller-request',
            on_result=on_result,
            metadata=metadata,
            queue_deadline=123.5,
            cancellation=cancellation,
            attempt_observer=observer,
        )
        is outputs
    )

    processor.batch_generate_calls.assert_called_once()
    forwarded = processor.batch_generate_calls.call_args.kwargs
    assert forwarded['calls'] is calls
    assert forwarded['calls'][0].context is context
    assert forwarded['cancellation'] is cancellation
    assert forwarded['on_result'] is on_result
    assert forwarded['attempt_observer'] is observer
    assert forwarded['metadata'] is metadata
    assert forwarded['request_id'] == 'caller-request'
    assert forwarded['queue_deadline'] == 123.5
    on_result.assert_not_called()
    observer.assert_not_called()


def test_complete_many_preserves_prepared_calls_identity_and_laziness() -> None:
    prepare = Mock(return_value=_call())
    validation = _call()
    calls = PreparedCalls(2, prepare, validation)
    processor = SimpleNamespace(
        uses_queued_dispatch=True,
        batch_generate_calls=Mock(return_value=['one', 'two']),
        close=Mock(),
    )

    assert _engine(processor).complete_many(calls) == ['one', 'two']
    assert processor.batch_generate_calls.call_args.kwargs['calls'] is calls
    assert len(calls) == 2
    assert calls.validation_call is validation
    prepare.assert_not_called()
    with pytest.raises(FrozenInstanceError):
        calls.count = 3


@pytest.mark.parametrize('count', [0, 2])
def test_complete_many_rejects_lazy_direct_calls_before_preparation(count: int) -> None:
    prepare = Mock(side_effect=AssertionError('must stay lazy'))
    processor = SimpleNamespace(
        uses_queued_dispatch=False,
        batch_generate_calls=Mock(),
        _get_direct_runner=Mock(),
        close=Mock(),
    )

    with pytest.raises(
        ValueError, match='^Lazy image preparation requires queued dispatch$'
    ):
        _engine(processor).complete_many(PreparedCalls(count, prepare, _call()))

    prepare.assert_not_called()
    processor.batch_generate_calls.assert_not_called()
    processor._get_direct_runner.assert_not_called()


@pytest.mark.parametrize('public', [False, True], ids=['processor', 'public'])
@pytest.mark.parametrize(
    'request_id', ['ordered', None], ids=['caller-id', 'generated-id']
)
def test_complete_many_matches_processor_ordered_text_and_nullable_slots(
    processor: BatchProcessor, public: bool, request_id: str | None
) -> None:
    first_released = asyncio.Event()

    async def complete(*, task_id: str, **kwargs: Any) -> str | tuple[str, str] | None:
        index = int(task_id.rsplit('_', 1)[1])
        if index == 0:
            await first_released.wait()
            return task_id, 'first'
        if index == 1:
            first_released.set()
            return 'second'
        return None

    processor.acompletion_call_with_retry = complete
    submit = (
        _engine(processor).complete_many if public else processor.batch_generate_calls
    )
    callbacks = []
    options = {} if request_id is None else {'request_id': request_id}
    outputs = submit(
        calls=[_call('first'), _call('second'), _call('third')],
        on_result=lambda task_id, response: callbacks.append((task_id, response)),
        **options,
    )

    assert outputs == ['first', 'second', None]
    if request_id is None:
        request_id = callbacks[0][0].rsplit('_task_', 1)[0]
        assert str(UUID(request_id)) == request_id
    assert sorted(callbacks) == [
        (f'{request_id}_task_0', 'first'),
        (f'{request_id}_task_1', 'second'),
        (f'{request_id}_task_2', None),
    ]
    assert callbacks.index((f'{request_id}_task_1', 'second')) < callbacks.index(
        (f'{request_id}_task_0', 'first')
    )


@pytest.mark.parametrize('public', [False, True], ids=['processor', 'public'])
def test_complete_many_matches_processor_batch_error_partitions(
    processor: BatchProcessor, public: bool
) -> None:
    original_error = ValueError('invalid response')

    async def complete(*, task_id: str, **kwargs: Any) -> tuple[str, str]:
        if task_id.endswith('_1'):
            raise original_error
        return task_id, f'text:{task_id}'

    processor.acompletion_call_with_retry = complete
    submit = (
        _engine(processor).complete_many if public else processor.batch_generate_calls
    )
    callbacks = []
    with pytest.raises(BatchExecutionError) as raised:
        submit(
            calls=[_call('first'), _call('second'), _call('third')],
            request_id='partial',
            on_result=lambda task_id, response: callbacks.append((task_id, response)),
        )

    error = raised.value
    assert error.request_id == 'partial'
    assert error.total == 3
    assert len(error.failed_results) == 1
    assert error.failed_results[0].task_id == 'partial_task_1'
    assert error.failed_results[0].error is original_error
    assert [result.task_id for result in error.successful_results] == [
        'partial_task_0',
        'partial_task_2',
    ]
    assert sorted(callbacks) == [
        ('partial_task_0', 'text:partial_task_0'),
        ('partial_task_2', 'text:partial_task_2'),
    ]


def test_complete_many_empty_queued_batch_has_no_producer(
    processor: BatchProcessor, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    processor._queue_config = replace(processor._queue_config, enabled=True)
    producer = Mock(side_effect=AssertionError('empty batch must not start a producer'))
    monkeypatch.setattr(processor, '_get_queued_executor', producer)

    assert _engine(processor).complete_many([]) == []
    producer.assert_not_called()


@pytest.mark.parametrize('queued', [False, True])
def test_prompt_call_delegates_once_without_replaying_callbacks(
    queued: bool,
) -> None:
    processor = SimpleNamespace(
        uses_queued_dispatch=queued,
        batch_timeout=20,
        default_completion_params={},
        batch_generate_calls=Mock(return_value=['answer']),
        close=Mock(),
    )
    engine = _engine(processor)
    engine.complete_many = Mock(wraps=engine.complete_many)
    calls = [_call()]
    engine.build_completion_calls = Mock(return_value=calls)
    on_result = Mock()
    observer = Mock()
    cancellation = threading.Event()
    metadata = {'source': 'prompt'}

    assert engine(
        ['hello'],
        request_contexts=[calls[0].context],
        metadata=metadata,
        queue_deadline=456.0,
        cancellation=cancellation,
        on_result=on_result,
        attempt_observer=observer,
    ) == ['answer']

    engine.complete_many.assert_called_once()
    processor.batch_generate_calls.assert_called_once()
    forwarded = processor.batch_generate_calls.call_args.kwargs
    assert forwarded['on_result'] is on_result
    assert forwarded['attempt_observer'] is observer
    assert forwarded['metadata'] is metadata
    if queued:
        assert isinstance(forwarded['calls'], PreparedCalls)
        engine.build_completion_calls.assert_not_called()
        assert forwarded['queue_deadline'] == 456.0
        assert forwarded['cancellation'] is cancellation
    else:
        assert forwarded['calls'] is calls
    on_result.assert_not_called()
    observer.assert_not_called()


@pytest.mark.parametrize('entry', ['call', 'generate'])
@pytest.mark.parametrize('shape', ['text', 'texts', 'empty', 'image', 'images'])
def test_prompt_entry_preserves_input_and_output_shapes(entry: str, shape: str) -> None:
    image = Image.new('RGB', (28, 28))
    inputs = {
        'text': 'hello',
        'texts': ['first', 'second'],
        'empty': [],
        'image': [image, 'hello'],
        'images': [[image, 'first'], [image, 'second']],
    }
    captured = []

    def execute(*, calls: list[CompletionCallParams], **kwargs: Any) -> list[str]:
        captured.extend(calls)
        return [f'answer-{i}' for i in range(len(calls))]

    processor = SimpleNamespace(
        uses_queued_dispatch=False,
        default_completion_params={},
        batch_generate_calls=execute,
        close=Mock(),
    )
    engine = _engine(processor)
    engine.is_multimodal = shape in ('image', 'images')
    invoke = engine if entry == 'call' else engine.generate
    try:
        result = invoke(
            inputs[shape],
            system_prompt='custom system',
            completion_params={'temperature': 0.25},
            mm_processor_kwargs={'min_pixels': 784, 'max_pixels': 1568},
        )
        expected = {
            'text': 'answer-0',
            'texts': ['answer-0', 'answer-1'],
            'empty': [],
            'image': 'answer-0',
            'images': ['answer-0', 'answer-1'],
        }
        assert result == expected[shape]
        assert all(call.messages[0]['content'] == 'custom system' for call in captured)
        assert all(call.temperature == 0.25 for call in captured)
        if shape in ('image', 'images'):
            assert all(
                call.messages[1]['content'][0]['type'] == 'image_url'
                for call in captured
            )
    finally:
        image.close()
