from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import marie_parsebench.page_parsing as page_parsing
import pytest
from marie_parsebench.models import SourceTransform
from marie_parsebench.page_parsing import (
    CandidateElement,
    PageBinding,
    PageCandidate,
    ParseBenchParameters,
    config_sha256,
    handle_page_response,
    load_page_annotator_config,
)
from PIL import Image
from pydantic import ValidationError

CONFIG = (
    Path(__file__).parents[1] / 'config/extract/TID-parsebench/annotator/config.yml'
)


def _identity(config_hash: str = 'b' * 64) -> dict[str, object]:
    return {
        'run_id': 'run-1',
        'example_id': 'example-1',
        'pipeline_name': 'parsebench-page',
        'source_sha256': 'a' * 64,
        'config_sha256': config_hash,
        'ref_id': 'source-1',
        'ref_type': 'parsebench',
        'dag_id': 'dag-1',
    }


def _transform(width: int = 100, height: int = 200) -> dict[str, object]:
    return SourceTransform(
        source_width=100.0,
        source_height=200.0,
        source_unit='pixels',
        crop_xyxy=(0.0, 0.0, 100.0, 200.0),
        rotation_degrees=0,
        render_width_px=width,
        render_height_px=height,
    ).model_dump(mode='json')


def _parameters(config_hash: str = 'b' * 64) -> dict[str, object]:
    return {
        'identity': _identity(config_hash),
        'producer': {'name': 'marie-ai', 'version': 'test', 'job_id': 'job-1'},
        'pages': [
            {
                'local_page_number': 1,
                'page_index': 2,
                'original_source_page_index': 7,
                'width_px': 100,
                'height_px': 200,
                'source_transform': _transform(),
            }
        ],
    }


@dataclass(frozen=True)
class _Attempt:
    task_id: str
    attempt_number: int
    raw_response: str | None
    finish_reason: str | None
    response_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_ms: float
    error: str | None
    cost_usd: float | None = None


def test_nested_parameters_preserve_sparse_source_order() -> None:
    payload = _parameters()
    first = payload['pages'][0]
    assert isinstance(first, dict)
    payload['pages'] = [
        {**first, 'local_page_number': 1, 'page_index': 2},
        {**first, 'local_page_number': 2, 'page_index': 0},
        {**first, 'local_page_number': 3, 'page_index': 3},
    ]

    parameters = ParseBenchParameters.model_validate(payload)

    assert [page.page_index for page in parameters.pages] == [2, 0, 3]


@pytest.mark.parametrize(
    'mutation',
    [
        lambda payload: payload.update({'reference_answer': 'forbidden'}),
        lambda payload: payload['pages'][0].update({'group': 'tables'}),
        lambda payload: payload['identity'].update({'dataset_row': 'forbidden'}),
    ],
)
def test_nested_parameters_reject_unknown_fields(mutation) -> None:
    payload = _parameters()
    mutation(payload)

    with pytest.raises(ValidationError, match='extra_forbidden'):
        ParseBenchParameters.model_validate(payload)


def test_page_bindings_require_complete_local_coverage_and_unique_sources() -> None:
    payload = _parameters()
    first = payload['pages'][0]
    assert isinstance(first, dict)
    payload['pages'] = [
        {**first, 'local_page_number': 1, 'page_index': 2},
        {**first, 'local_page_number': 3, 'page_index': 2},
    ]

    with pytest.raises(ValidationError, match='local page numbers'):
        ParseBenchParameters.model_validate(payload)


def test_page_binding_dimensions_must_match_render_transform() -> None:
    payload = _parameters()
    payload['pages'][0]['width_px'] = 99

    with pytest.raises(ValidationError, match='render dimensions'):
        ParseBenchParameters.model_validate(payload)


@pytest.mark.parametrize(
    'field,value',
    [
        ('identity', {'spoofed': True}),
        ('geometry_source', 'ocr'),
        ('usage', {'input_tokens': 1}),
    ],
)
def test_candidate_rejects_direct_host_owned_fields(field: str, value: object) -> None:
    payload = {
        'status': 'complete',
        'elements': [
            {
                'kind': 'text',
                'content': 'source text',
                'content_format': 'markdown',
                field: value,
            }
        ],
    }

    with pytest.raises(ValidationError, match='extra_forbidden'):
        PageCandidate.model_validate(payload)


@pytest.mark.parametrize(
    'reserved',
    [
        'marie.element_id',
        'marie.page_id',
        'marie.geometry_source',
        'marie.original_source_page_index',
    ],
)
def test_candidate_rejects_reserved_marie_attributes(reserved: str) -> None:
    with pytest.raises(ValidationError, match='host-owned'):
        CandidateElement.model_validate(
            {
                'kind': 'text',
                'content': 'source text',
                'content_format': 'markdown',
                'attributes': {reserved: 'spoofed'},
            }
        )


def test_candidate_enforces_table_and_chart_semantics() -> None:
    with pytest.raises(ValidationError, match='table content must use HTML'):
        CandidateElement.model_validate(
            {'kind': 'table', 'content': '| A |', 'content_format': 'markdown'}
        )
    with pytest.raises(ValidationError, match='labeled HTML table'):
        CandidateElement.model_validate(
            {'kind': 'chart', 'content': '<p>42</p>', 'content_format': 'html'}
        )


@pytest.mark.parametrize(
    'kind,content',
    [
        ('table', '<tableau>not a table</table>'),
        ('chart', '<tableau><thread>42</thread></table>'),
    ],
)
def test_candidate_rejects_pseudo_table_tags(kind: str, content: str) -> None:
    with pytest.raises(ValidationError, match='HTML table'):
        CandidateElement.model_validate(
            {'kind': kind, 'content': content, 'content_format': 'html'}
        )


@pytest.mark.parametrize(
    'kind,content',
    [
        ('table', '<table><tr><td>42</table>'),
        ('chart', '<table><tr><th>Value</th><td>42</table>'),
    ],
)
def test_candidate_rejects_unbalanced_table_markup(kind: str, content: str) -> None:
    with pytest.raises(ValidationError, match='HTML table'):
        CandidateElement.model_validate(
            {'kind': kind, 'content': content, 'content_format': 'html'}
        )


@pytest.mark.parametrize(
    'kind,content',
    [
        (
            'table',
            '<table><tr><th rowspan="2">Name</th><td>A</td></tr>'
            '<tr><td>B</td></tr></table>',
        ),
        (
            'chart',
            '<table><thead><tr><th>Series</th><th>Value</th></tr></thead>'
            '<tbody><tr><td>A</td><td>42</td></tr></tbody></table>',
        ),
    ],
)
def test_candidate_preserves_valid_table_markup(kind: str, content: str) -> None:
    candidate = CandidateElement.model_validate(
        {'kind': kind, 'content': content, 'content_format': 'html'}
    )

    assert candidate.content == content


def test_bare_and_single_fenced_responses_validate_as_the_same_candidate() -> None:
    bare = '{"status":"blank","elements":[],"warnings":[]}'
    responses = (bare, f'```json\n{bare}\n```', f'```\n{bare}\n```')

    candidates = [
        PageCandidate.model_validate_json(
            page_parsing._response_candidate_json(response)
        )
        for response in responses
    ]

    assert candidates == [candidates[0], candidates[0], candidates[0]]


def test_bare_and_single_fenced_responses_preserve_markdown_code_fences() -> None:
    content = '```python\nprint("source text")\n```'
    bare = json.dumps(
        {
            'status': 'complete',
            'elements': [
                {
                    'kind': 'code',
                    'content': content,
                    'content_format': 'markdown',
                }
            ],
        }
    )
    responses = (bare, f'```json\n{bare}\n```', f'```\n{bare}\n```')

    candidates = [
        PageCandidate.model_validate_json(
            page_parsing._response_candidate_json(response)
        )
        for response in responses
    ]

    assert candidates == [candidates[0], candidates[0], candidates[0]]
    assert candidates[0].elements[0].content == content


@pytest.mark.parametrize(
    'response',
    [
        'before\n```json\n{"status":"blank","elements":[],"warnings":[]}\n```',
        '```json\n{"status":"blank","elements":[],"warnings":[]}\n```\nafter',
        '```json\n{"status":"blank","elements":[],"warnings":[]}\n```\n'
        '```\n{"status":"blank","elements":[],"warnings":[]}\n```',
        '```json\n{"status":"blank","elements":[],"warnings":[]}',
        '```python\n{"status":"blank","elements":[],"warnings":[]}\n```',
    ],
)
def test_response_rejects_prose_multiple_incomplete_or_labeled_fences(
    response: str,
) -> None:
    with pytest.raises(ValueError):
        PageCandidate.model_validate_json(
            page_parsing._response_candidate_json(response)
        )


@pytest.mark.parametrize(
    'response',
    [
        '```json\n{"status":\n```',
        '```json\n{"status":"complete","elements":[]}\n```',
        '```\n{"status":"blank","elements":[],"identity":{"run_id":"spoofed"}}\n```',
    ],
)
def test_fenced_response_still_rejects_invalid_candidate_payload(response: str) -> None:
    with pytest.raises(ValidationError):
        PageCandidate.model_validate_json(
            page_parsing._response_candidate_json(response)
        )


def test_load_config_returns_runtime_annotator_and_stable_hash() -> None:
    loaded = load_page_annotator_config(CONFIG)

    assert loaded.annotator_conf['name'] == 'parsebench-page'
    assert loaded.annotator_conf['model_config']['model_name'] == 'qwen_v3_30b_instruct'
    assert loaded.annotator_conf['model_config']['min_pixels'] == 401408
    assert loaded.annotator_conf['model_config']['max_pixels'] == 2007040
    assert loaded.config_sha256 == config_sha256(
        loaded.annotator_conf, prompt_template=loaded.prompt_template
    )

    moved = json.loads(json.dumps(loaded.annotator_conf))
    moved['model_config']['prompt_path'] = './copied/location/page.j2'
    assert (
        config_sha256(moved, prompt_template=loaded.prompt_template)
        == loaded.config_sha256
    )


def test_config_hash_changes_with_template_or_model_visible_schema() -> None:
    loaded = load_page_annotator_config(CONFIG)

    assert config_sha256(
        loaded.annotator_conf,
        prompt_template=loaded.prompt_template + '\nchanged',
    ) != config_sha256(
        loaded.annotator_conf,
        prompt_template=loaded.prompt_template,
    )
    assert config_sha256(
        loaded.annotator_conf,
        prompt_template=loaded.prompt_template,
        candidate_schema={'type': 'object', 'properties': {}},
    ) != config_sha256(
        loaded.annotator_conf,
        prompt_template=loaded.prompt_template,
    )


def test_config_hash_includes_handler_protocol_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_page_annotator_config(CONFIG)

    assert page_parsing.HANDLER_PROTOCOL_VERSION == 'marie.parsebench.page-handler.v5'
    v5_hash = config_sha256(
        loaded.annotator_conf,
        prompt_template=loaded.prompt_template,
    )
    monkeypatch.setattr(
        page_parsing,
        'HANDLER_PROTOCOL_VERSION',
        'marie.parsebench.page-handler.v4',
    )

    assert (
        config_sha256(
            loaded.annotator_conf,
            prompt_template=loaded.prompt_template,
        )
        != v5_hash
    )


def test_config_rejects_pixel_limits_that_are_not_forwarded() -> None:
    loaded = load_page_annotator_config(CONFIG)
    changed = json.loads(json.dumps(loaded.annotator_conf))
    changed['model_config']['max_pixels'] = 999999

    with pytest.raises(ValueError, match='max_pixels'):
        config_sha256(changed, prompt_template=loaded.prompt_template)


def test_config_rejects_integer_that_masquerades_as_multimodal_boolean() -> None:
    loaded = load_page_annotator_config(CONFIG)
    changed = json.loads(json.dumps(loaded.annotator_conf))
    changed['model_config']['multimodal'] = 1

    with pytest.raises(ValueError, match='multimodal'):
        config_sha256(changed, prompt_template=loaded.prompt_template)


def test_handler_binds_identity_geometry_usage_and_writes_sidecar(
    tmp_path: Path,
) -> None:
    loaded = load_page_annotator_config(CONFIG)
    binding = PageBinding.model_validate(_parameters(loaded.config_sha256)['pages'][0])
    run_context = SimpleNamespace(
        parameters={'parsebench': _parameters(loaded.config_sha256)}
    )
    document = SimpleNamespace(
        page_count=1,
        source_metadata={
            'pages': 1,
            'ocr': [
                {
                    'meta': {
                        'page': 0,
                        'imageSize': {'width': 100, 'height': 200},
                        'format': 'xywh',
                    },
                    'lines': [
                        {
                            'line': 1,
                            'text': 'Title',
                            'bbox': [10, 20, 30, 10],
                            'confidence': 0.99,
                            'wordids': [1],
                        }
                    ],
                }
            ],
        },
    )
    from marie_parsebench.page_parsing import render_page_prompt

    prompt = render_page_prompt(loaded.prompt_template, document, binding)
    response = json.dumps(
        {
            'status': 'complete',
            'elements': [
                {
                    'kind': 'heading',
                    'content': '**Title**',
                    'content_format': 'markdown',
                    'bbox_xyxy': [10.0, 20.0, 40.0, 30.0],
                    'attributes': {'level': '1'},
                }
            ],
            'warnings': [],
        }
    )
    attempts = (
        _Attempt('task-0', 1, None, None, 'model-a', 2, 1, 5.0, 'timeout', 0.1),
        _Attempt('task-0', 2, response, 'stop', 'model-b', 7, 3, 12.5, None, 0.2),
    )
    image = Image.new('RGB', (100, 200))

    artifact = handle_page_response(
        response=response,
        prompt=prompt,
        prompt_template=loaded.prompt_template,
        image_path=str(tmp_path / 'frame_0001.png'),
        image=image,
        unit=SimpleNamespace(page_number=1, data=binding.model_dump(mode='json')),
        document=document,
        run_context=run_context,
        annotator_conf=loaded.annotator_conf,
        attempts=attempts,
        error=None,
        output_path=str(tmp_path),
    )

    assert artifact['identity'] == _identity(loaded.config_sha256)
    assert artifact['page_index'] == 2
    assert artifact['original_source_page_index'] == 7
    assert artifact['page_id'] == 'example-1:page:2'
    assert artifact['elements'][0]['element_id'] == 'example-1:page:2:element:0001'
    assert artifact['elements'][0]['bbox_xyxy_normalized'] == [0.1, 0.1, 0.4, 0.15]
    assert artifact['elements'][0]['geometry_source'] == 'vlm_localization'
    assert artifact['usage'] == {
        'input_tokens': 9,
        'output_tokens': 4,
        'cost_usd': 0.30000000000000004,
        'latency_ms': 17.5,
    }
    evidence = json.loads((tmp_path / 'evidence/frame_0001.json').read_text())
    assert evidence['config_sha256'] == loaded.config_sha256
    assert evidence['attempts'][0]['response_model'] == 'model-a'
    assert 'raw_response' not in evidence['attempts'][0]


def test_unknown_attempt_usage_remains_null(tmp_path: Path) -> None:
    loaded = load_page_annotator_config(CONFIG)
    parameters = _parameters(loaded.config_sha256)
    binding = PageBinding.model_validate(parameters['pages'][0])
    document = SimpleNamespace(page_count=1, source_metadata={'pages': 1})
    from marie_parsebench.page_parsing import render_page_prompt

    prompt = render_page_prompt(loaded.prompt_template, document, binding)
    response = '{"status":"blank","elements":[],"warnings":[]}'
    attempt = _Attempt('task-0', 1, response, 'stop', None, None, None, 3.0, None)

    artifact = handle_page_response(
        response=response,
        prompt=prompt,
        prompt_template=loaded.prompt_template,
        image_path=str(tmp_path / 'frame_0001.png'),
        image=Image.new('RGB', (100, 200)),
        unit=SimpleNamespace(page_number=1, data=binding.model_dump(mode='json')),
        document=document,
        run_context=SimpleNamespace(parameters={'parsebench': parameters}),
        annotator_conf=loaded.annotator_conf,
        attempts=(attempt,),
        error=None,
        output_path=str(tmp_path),
    )

    assert artifact['usage']['input_tokens'] is None
    assert artifact['usage']['output_tokens'] is None
    assert artifact['usage']['cost_usd'] is None
    assert artifact['usage']['latency_ms'] == 3.0


def test_render_page_prompt_preserves_jinja_like_source_text() -> None:
    loaded = load_page_annotator_config(CONFIG)
    binding = PageBinding.model_validate(_parameters(loaded.config_sha256)['pages'][0])
    source_text = 'Printed {{ customer_name }} and {% raw %} are document text'
    document = SimpleNamespace(
        page_count=1,
        source_metadata={
            'pages': 1,
            'ocr': [
                {
                    'meta': {
                        'page': 0,
                        'imageSize': {'width': 100, 'height': 200},
                        'format': 'xywh',
                    },
                    'lines': [{'line': 1, 'text': source_text, 'bbox': [1, 2, 90, 10]}],
                }
            ],
        },
    )
    from marie_parsebench.page_parsing import render_page_prompt

    rendered = render_page_prompt(loaded.prompt_template, document, binding)

    assert source_text in rendered


def test_invalid_response_persists_failed_artifact_once_and_raises(
    tmp_path: Path,
) -> None:
    loaded = load_page_annotator_config(CONFIG)
    parameters = _parameters(loaded.config_sha256)
    binding = PageBinding.model_validate(parameters['pages'][0])
    document = SimpleNamespace(page_count=1, source_metadata={'pages': 1})
    from marie_parsebench.page_parsing import render_page_prompt

    kwargs = {
        'response': '{"status":"complete","elements":[]}',
        'prompt': render_page_prompt(loaded.prompt_template, document, binding),
        'prompt_template': loaded.prompt_template,
        'image_path': str(tmp_path / 'frame_0001.png'),
        'image': Image.new('RGB', (100, 200)),
        'unit': SimpleNamespace(page_number=1, data=binding.model_dump(mode='json')),
        'document': document,
        'run_context': SimpleNamespace(parameters={'parsebench': parameters}),
        'annotator_conf': loaded.annotator_conf,
        'attempts': (),
        'error': None,
        'output_path': str(tmp_path),
    }

    with pytest.raises(ValidationError, match='complete page candidate'):
        handle_page_response(**kwargs)

    failed_path = tmp_path / 'failed/frame_0001.json'
    failed = json.loads(failed_path.read_text())
    assert failed['status'] == 'failed'
    assert failed['identity'] == _identity(loaded.config_sha256)
    original = failed_path.read_bytes()
    with pytest.raises(FileExistsError):
        handle_page_response(**kwargs)
    assert failed_path.read_bytes() == original
