from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

pytest.importorskip(
    'marie.engine', reason='Requires the existing Marie host environment'
)

import marie_parsebench.context_providers  # noqa: F401
from marie.engine.completion_observer import CompletionAttempt
from marie_parsebench.models import SourceTransform
from marie_parsebench.page_parsing import load_page_annotator_config

from marie.extract.annotators.llm_annotator import LLMAnnotator

CONFIG = (
    Path(__file__).parents[2] / 'config/extract/TID-parsebench/annotator/config.yml'
)


class _FakeEngine:
    model_string = 'qwen_v3_30b_instruct'
    system_prompt = None

    def __init__(self, response: str) -> None:
        self.response = response
        self.batch_processor = SimpleNamespace(
            max_concurrency=1, uses_queued_dispatch=False
        )

    def __call__(self, content, **kwargs):
        task_id = 'request_task_0'
        kwargs['attempt_observer'](
            CompletionAttempt(
                task_id=task_id,
                attempt_number=1,
                raw_response=self.response,
                finish_reason='stop',
                response_model='openai/qwen3-vl-30b-a3b-instruct',
                input_tokens=25,
                output_tokens=10,
                elapsed_ms=15.0,
                error=None,
            )
        )
        kwargs['on_result'](task_id, self.response)
        return [self.response]


def _runtime(tmp_path: Path, response: str):
    loaded = load_page_annotator_config(CONFIG)
    width, height = 64, 48
    transform = SourceTransform(
        source_width=float(width),
        source_height=float(height),
        source_unit='pixels',
        crop_xyxy=(0.0, 0.0, float(width), float(height)),
        rotation_degrees=0,
        render_width_px=width,
        render_height_px=height,
    )
    parsebench = {
        'identity': {
            'run_id': 'run-1',
            'example_id': 'example-1',
            'pipeline_name': 'parsebench-page',
            'source_sha256': 'a' * 64,
            'config_sha256': loaded.config_sha256,
            'ref_id': 'source-1',
            'ref_type': 'parsebench',
            'dag_id': 'dag-1',
        },
        'producer': {'name': 'marie-ai', 'version': 'test', 'job_id': 'job-1'},
        'pages': [
            {
                'local_page_number': 1,
                'page_index': 5,
                'original_source_page_index': None,
                'width_px': width,
                'height_px': height,
                'source_transform': transform.model_dump(mode='json'),
            }
        ],
    }
    run_context = SimpleNamespace(
        parameters={
            'parsebench': parsebench,
            'reference_answer': 'FORBIDDEN-REFERENCE',
        }
    )
    document = SimpleNamespace(
        page_count=1,
        source_metadata={
            'pages': 1,
            'ocr': [
                {
                    'meta': {
                        'page': 0,
                        'imageSize': {'width': width, 'height': height},
                        'format': 'xywh',
                    },
                    'lines': [
                        {
                            'line': 1,
                            'text': 'Visible title',
                            'bbox': [4, 5, 20, 8],
                            'wordids': [1],
                        }
                    ],
                    'words': [
                        {
                            'id': 1,
                            'text': 'FORBIDDEN-WORD',
                            'box': [4, 5, 20, 8],
                            'line': 1,
                            'word_index': 0,
                        }
                    ],
                }
            ],
            'expected_markdown': 'FORBIDDEN-EXPECTED',
        },
    )
    work = tmp_path / 'work'
    frames = work / 'frames'
    frames.mkdir(parents=True)
    image = Image.new('RGB', (width, height), 'white')
    image.save(frames / 'frame_0001.png')
    with patch(
        'marie.extract.annotators.llm_annotator.route_llm_engine',
        return_value=MagicMock(),
    ):
        annotator = LLMAnnotator(
            working_dir=str(work),
            annotator_conf=loaded.annotator_conf,
            layout_conf={'layout_id': 'parsebench'},
            run_context=run_context,
            prompt_dir=str(CONFIG.parent),
        )
    annotator.engine = _FakeEngine(response)
    return annotator, document, image, work


@pytest.mark.asyncio
async def test_actual_annotator_rejects_non_ocr_before_engine_invocation(
    tmp_path: Path,
) -> None:
    annotator, document, image, work = _runtime(
        tmp_path, '{"status":"blank","elements":[]}'
    )
    document.source_metadata['extraction'] = {
        'result_kind': 'semantic_document',
        'route': 'plugin',
        'ocr_invoked': False,
    }
    document.source_metadata['ocr'][0]['lines'] = [
        {'line': 0, 'text': 'source', 'bbox': [8, 8, 48, 20]}
    ]

    with patch.object(
        _FakeEngine,
        '__call__',
        side_effect=AssertionError('engine must not run for non-OCR extraction'),
    ) as engine_call:
        with pytest.raises(ValueError, match='non-OCR extraction'):
            await annotator.aannotate(document, [image])
        engine_call.assert_not_called()

    output = work / 'agent-output/parsebench-page'
    assert not (output / 'frame_0001.json').exists()
    assert not (output / '_SUCCESS.yaml').exists()


@pytest.mark.asyncio
async def test_actual_annotator_lifecycle_persists_host_bound_page(
    tmp_path: Path,
) -> None:
    response = json.dumps(
        {
            'status': 'complete',
            'elements': [
                {
                    'kind': 'heading',
                    'content': '**Visible title**',
                    'content_format': 'markdown',
                    'bbox_xyxy': [4, 5, 24, 13],
                    'attributes': {'level': '1'},
                }
            ],
            'warnings': [],
        }
    )
    annotator, document, image, work = _runtime(tmp_path, response)

    await annotator.aannotate(document, [image])

    output = work / 'agent-output/parsebench-page'
    artifact = json.loads((output / 'frame_0001.json').read_text())
    assert artifact['status'] == 'complete'
    assert artifact['page_index'] == 5
    assert artifact['identity']['source_sha256'] == 'a' * 64
    assert artifact['producer']['name'] == 'marie-ai'
    assert artifact['elements'][0]['geometry_source'] == 'vlm_localization'
    assert artifact['usage']['input_tokens'] == 25
    request = json.loads(next((output / 'attempts').glob('*/request.json')).read_text())
    assert 'Visible title' in request['prompt']
    assert 'FORBIDDEN' not in request['prompt']
    assert (output / '_SUCCESS.yaml').is_file()


@pytest.mark.parametrize(
    'kind,content',
    [
        ('heading', '**Visible title**'),
        ('code', '```python\nprint("source text")\n```'),
    ],
)
@pytest.mark.asyncio
async def test_actual_annotator_lifecycle_accepts_fence_and_preserves_raw_attempt(
    tmp_path: Path,
    kind: str,
    content: str,
) -> None:
    payload = json.dumps(
        {
            'status': 'complete',
            'elements': [
                {
                    'kind': kind,
                    'content': content,
                    'content_format': 'markdown',
                }
            ],
            'warnings': [],
        }
    )
    response = f'```json\n{payload}\n```'
    annotator, document, image, work = _runtime(tmp_path, response)

    await annotator.aannotate(document, [image])

    output = work / 'agent-output/parsebench-page'
    artifact = json.loads((output / 'frame_0001.json').read_text())
    attempt = json.loads(
        next((output / 'attempts').glob('*/attempt-0001.json')).read_text()
    )
    assert artifact['status'] == 'complete'
    assert artifact['elements'][0]['content'] == content
    assert attempt['raw_response'] == response
    assert (output / '_SUCCESS.yaml').is_file()
    assert not (output / 'failed/frame_0001.json').exists()


@pytest.mark.parametrize(
    'response',
    [
        '{"status":"complete","elements":[]}',
        '```json\n{"status":"complete","elements":[]}\n```',
    ],
)
@pytest.mark.asyncio
async def test_actual_annotator_lifecycle_keeps_invalid_candidate_failed(
    tmp_path: Path, response: str
) -> None:
    annotator, document, image, work = _runtime(tmp_path, response)

    with pytest.raises(Exception, match='complete page candidate'):
        await annotator.aannotate(document, [image])

    output = work / 'agent-output/parsebench-page'
    assert (
        json.loads((output / 'failed/frame_0001.json').read_text())['status']
        == 'failed'
    )
    assert not (output / 'frame_0001.json').exists()
    assert not (output / '_SUCCESS.yaml').exists()
    attempt = json.loads(
        next((output / 'attempts').glob('*/attempt-0001.json')).read_text()
    )
    assert attempt['raw_response'] == response
