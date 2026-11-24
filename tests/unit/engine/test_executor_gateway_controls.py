import threading
import time
from pathlib import Path
from typing import Any

import pytest
from marie.engine.completion_contract import (
    CompletionCallParams,
    PreparedCalls,
    RequestContext,
)
from marie.engine.llm_ops import LLMCall
from marie.engine.multimodal_ops import MultimodalLLMCall
from marie.engine.openai_engine import OpenAIEngine
from PIL import Image

from marie.extract.annotators.util import process_batch


class _PublicEngine:
    uses_queued_dispatch = True
    batch_timeout = 5
    max_concurrency = 2
    system_prompt = 'system'
    model_string = 'test-model'

    def __init__(self) -> None:
        self.controls: dict[str, Any] = {}
        self.prepared: list[CompletionCallParams] = []
        self.submissions = 0

    @property
    def batch_processor(self) -> None:
        raise AssertionError('caller must use the public engine interface')

    def build_completion_calls(self, **kwargs: Any) -> list[CompletionCallParams]:
        contexts = kwargs.get('request_contexts')
        return [
            CompletionCallParams(
                model=self.model_string,
                messages=[],
                context=contexts[0] if contexts else None,
            )
        ]

    def complete_many(self, calls: PreparedCalls, **kwargs: Any) -> list[str]:
        self.submissions += 1
        self.controls = kwargs
        responses = [f'{{"index":{i}}}' for i in range(len(calls))]
        for i in reversed(range(len(calls))):
            self.prepared.append(calls[i])
            kwargs['on_result'](f'public_task_{i}', responses[i])
        return responses

    def __call__(self, content: Any, **kwargs: Any) -> list[str]:
        self.controls = kwargs
        return ['answer']


@pytest.mark.asyncio
async def test_process_batch_uses_public_completion_boundary(tmp_path: Path) -> None:
    engine = _PublicEngine()
    cancellation = threading.Event()
    metadata = {'source': 'public-boundary'}
    images = []
    items = []

    def prepare(index: int) -> list[Any]:
        image = Image.new('RGB', (2, 2))
        images.append(image)
        item = [
            image,
            f'prompt-{index}',
            str(tmp_path / f'frame_{index + 1:04d}.png'),
            '',
        ]
        items.append(item)
        return item

    started = time.monotonic()
    results = await process_batch(
        [(None, '', str(tmp_path / f'frame_{i + 1:04d}.png'), '') for i in range(2)],
        engine,
        str(tmp_path),
        is_multimodal=True,
        expect_output='json',
        request_context=RequestContext(job_id='job', work_unit_id='unit'),
        metadata=metadata,
        prepare_item=prepare,
        preparation_lock=threading.Lock(),
        cancellation=cancellation,
    )

    assert results == [{'index': 0}, {'index': 1}]
    assert engine.submissions == 1
    assert engine.controls['cancellation'] is cancellation
    assert engine.controls['metadata'] is metadata
    assert started + engine.batch_timeout <= engine.controls['queue_deadline']
    assert engine.controls['queue_deadline'] <= time.monotonic() + engine.batch_timeout
    assert [call.context.page_number for call in engine.prepared] == [2, 1]
    assert all(call.context.job_id == 'job' for call in engine.prepared)
    assert items == [[], []]
    for image in images:
        with pytest.raises(ValueError, match='closed image'):
            image.getpixel((0, 0))


@pytest.mark.asyncio
@pytest.mark.parametrize('multimodal', [False, True])
async def test_async_wrappers_use_public_execution_controls(multimodal: bool) -> None:
    engine = _PublicEngine()
    wrapper = MultimodalLLMCall(engine) if multimodal else LLMCall(engine)
    image = Image.new('RGB', (2, 2))
    try:
        inputs = [[image, 'prompt']] if multimodal else ['prompt']
        assert await wrapper.aforward(inputs) == ['answer']
        assert isinstance(engine.controls['cancellation'], threading.Event)
        assert engine.controls['queue_deadline'] > time.monotonic()
    finally:
        image.close()


@pytest.mark.parametrize(
    'name', ['batch_generate', 'openai_generate', '_build_completion_calls']
)
def test_obsolete_engine_entry_points_are_removed(name: str) -> None:
    assert not hasattr(OpenAIEngine, name)


def test_prepared_calls_has_one_public_import_path() -> None:
    from marie.engine.llm_queue import producer

    assert not hasattr(producer, 'PreparedCalls')
