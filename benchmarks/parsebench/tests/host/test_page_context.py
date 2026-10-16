from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip(
    'marie.engine', reason='Requires the existing Marie host environment'
)

from marie_parsebench.context_providers import ParseBenchPageContextProvider
from marie_parsebench.models import SourceTransform
from marie_parsebench.page_parsing import load_page_annotator_config

from marie.prompt import PromptTemplate

CONFIG = (
    Path(__file__).parents[2] / 'config/extract/TID-parsebench/annotator/config.yml'
)


def _binding(local: int, source: int, width: int = 80, height: int = 60) -> dict:
    return {
        'local_page_number': local,
        'page_index': source,
        'original_source_page_index': None,
        'width_px': width,
        'height_px': height,
        'source_transform': SourceTransform(
            source_width=float(width),
            source_height=float(height),
            source_unit='pixels',
            crop_xyxy=(0.0, 0.0, float(width), float(height)),
            rotation_degrees=0,
            render_width_px=width,
            render_height_px=height,
        ).model_dump(mode='json'),
    }


def _parameters(pages: list[dict]) -> dict:
    loaded = load_page_annotator_config(CONFIG)
    return {
        'identity': {
            'run_id': 'run-1',
            'example_id': 'example-1',
            'pipeline_name': 'parsebench-page',
            'source_sha256': 'a' * 64,
            'config_sha256': loaded.config_sha256,
            'ref_id': 'ref-1',
            'ref_type': 'parsebench',
            'dag_id': 'dag-1',
        },
        'producer': {'name': 'marie-ai', 'version': 'test'},
        'pages': pages,
    }


def _ocr(page: int, width: int = 80, height: int = 60) -> dict:
    return {
        'meta': {
            'page': page,
            'imageSize': {'width': width, 'height': height},
            'format': 'xywh',
        },
        'lines': [
            {
                'line': 1,
                'text': f'source page {page}',
                'bbox': [1, 2, 30, 10],
                'confidence': 0.5,
                'wordids': [1],
                'forbidden_line_field': 'FORBIDDEN-LINE',
            }
        ],
        'words': [{'text': 'FORBIDDEN-WORD'}],
    }


@pytest.mark.parametrize(
    'extraction',
    [
        None,
        {'route': 'ocr', 'ocr_invoked': True},
        {'route': 'plugin', 'ocr_invoked': True},
        {'route': 'plugin'},
    ],
)
def test_provider_preserves_sparse_source_order_and_allowlists_ocr(
    extraction: dict | None,
) -> None:
    pages = [_binding(1, 2), _binding(2, 0), _binding(3, 3)]
    provider = ParseBenchPageContextProvider(
        SimpleNamespace(parameters={'parsebench': _parameters(pages)}),
        'parsebench-page',
    )
    document = SimpleNamespace(
        page_count=3,
        source_metadata={
            'pages': 3,
            'ocr': [_ocr(0), _ocr(1), _ocr(2)],
            'reference_answer': 'FORBIDDEN-METADATA',
        },
    )
    if extraction is not None:
        document.source_metadata['extraction'] = extraction

    units = provider.get_processing_units(document)
    contexts = [
        json.loads(
            provider.get_variables(document, unit.page_number, unit)[
                'SOURCE_CONTEXT_JSON'
            ]
        )
        for unit in units
    ]

    assert [unit.page_number for unit in units] == [1, 2, 3]
    assert [unit.data['page_index'] for unit in units] == [2, 0, 3]
    assert [context['source_page_index'] for context in contexts] == [2, 0, 3]
    assert contexts[0]['ocr_lines'] == [
        {
            'line_number': 1,
            'text': 'source page 0',
            'bbox_xywh': [1.0, 2.0, 30.0, 10.0],
            'provenance': 'source_ocr',
        }
    ]
    assert 'FORBIDDEN' not in json.dumps(contexts)


def test_rendered_request_excludes_other_runtime_and_metadata_keys() -> None:
    loaded = load_page_annotator_config(CONFIG)
    page = _binding(1, 4)
    run_context = SimpleNamespace(
        parameters={
            'parsebench': _parameters([page]),
            'prompt_variables': {'SOURCE_CONTEXT_JSON': 'FORBIDDEN-INVOCATION'},
            'reference_answer': 'FORBIDDEN-REFERENCE',
            'group': 'FORBIDDEN-GROUP',
        }
    )
    provider = ParseBenchPageContextProvider(run_context, 'parsebench-page')
    document = SimpleNamespace(
        page_count=1,
        source_metadata={
            'pages': 1,
            'ocr': [_ocr(0)],
            'reference': 'FORBIDDEN-DOCUMENT',
            'url': 'FORBIDDEN-URL',
        },
    )
    unit = provider.get_processing_units(document)[0]

    rendered = PromptTemplate.from_str(loaded.prompt_template).render(
        provider.get_variables(document, 1, unit)
    )

    assert 'source page 0' in rendered
    assert 'FORBIDDEN' not in rendered
    assert set(provider.get_variables(document, 1, unit)) == {
        'OUTPUT_CANDIDATE_SCHEMA_JSON',
        'SOURCE_CONTEXT_JSON',
    }


def test_provider_allows_legitimately_missing_ocr_as_empty_context() -> None:
    page = _binding(1, 0)
    provider = ParseBenchPageContextProvider(
        SimpleNamespace(parameters={'parsebench': _parameters([page])}),
        'parsebench-page',
    )
    document = SimpleNamespace(page_count=1, source_metadata={'pages': 1})
    unit = provider.get_processing_units(document)[0]

    context = json.loads(
        provider.get_variables(document, 1, unit)['SOURCE_CONTEXT_JSON']
    )

    assert context['ocr_lines'] == []


@pytest.mark.parametrize('method', ['get_processing_units', 'get_eligible_pages'])
def test_provider_rejects_explicit_non_ocr_compatibility_boxes(method: str) -> None:
    page = _binding(1, 0)
    provider = ParseBenchPageContextProvider(
        SimpleNamespace(parameters={'parsebench': _parameters([page])}),
        'parsebench-page',
    )
    ocr = _ocr(0)
    ocr['lines'] = [{'line': 0, 'text': 'source', 'bbox': [8, 8, 48, 20]}]
    document = SimpleNamespace(
        page_count=1,
        source_metadata={
            'pages': 1,
            'ocr': [ocr],
            'extraction': {
                'result_kind': 'semantic_document',
                'route': 'plugin',
                'ocr_invoked': False,
            },
        },
    )

    with pytest.raises(ValueError, match='non-OCR extraction'):
        getattr(provider, method)(document)


@pytest.mark.parametrize(
    'mutation,error',
    [
        (lambda ocr: ocr['meta'].update({'page': 4}), 'local page'),
        (
            lambda ocr: ocr['meta']['imageSize'].update({'width': 79}),
            'dimensions',
        ),
        (lambda ocr: ocr['lines'][0].update({'bbox': [70, 2, 30, 10]}), 'bbox'),
        (lambda ocr: ocr['meta'].update({'format': 'xyxy'}), 'xywh'),
    ],
)
def test_provider_rejects_inconsistent_source_ocr(mutation, error: str) -> None:
    page = _binding(1, 0)
    provider = ParseBenchPageContextProvider(
        SimpleNamespace(parameters={'parsebench': _parameters([page])}),
        'parsebench-page',
    )
    ocr = _ocr(0)
    mutation(ocr)
    document = SimpleNamespace(page_count=1, source_metadata={'pages': 1, 'ocr': [ocr]})

    with pytest.raises(ValueError, match=error):
        provider.get_processing_units(document)


def test_provider_rejects_nested_extras() -> None:
    parameters = _parameters([_binding(1, 0)])
    parameters['group'] = 'tables'

    with pytest.raises(ValueError, match='Extra inputs'):
        ParseBenchPageContextProvider(
            SimpleNamespace(parameters={'parsebench': parameters}),
            'parsebench-page',
        )
