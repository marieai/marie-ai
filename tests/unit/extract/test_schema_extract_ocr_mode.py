from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from marie.executor.text import text_extraction_executor as module
from marie.executor.text.text_extraction_executor import TextExtractionExecutor


def test_planner_operator_parse_mode_routes_ocr() -> None:
    assert (
        TextExtractionExecutor._parse_mode(
            {
                'payload': {'op_params': {'layout': 'workbench', 'parse_mode': 'ocr'}},
            }
        )
        == 'ocr'
    )


def test_explicit_run_params_and_top_level_modes_keep_precedence() -> None:
    parameters = {
        'run_params': {'parse_mode': 'fast'},
        'parse_mode': 'agent',
        'payload': {'op_params': {'parse_mode': 'ocr'}},
    }
    assert TextExtractionExecutor._parse_mode(parameters) == 'fast'
    del parameters['run_params']
    assert TextExtractionExecutor._parse_mode(parameters) == 'agent'


@pytest.mark.parametrize('strict_result', [True, False])
@pytest.mark.parametrize('failure', ['exception', 'empty', 'unsupported'])
def test_schema_planner_ocr_failure_raises_while_legacy_behavior_is_preserved(
    monkeypatch, strict_result: bool, failure: str
) -> None:
    executor = object.__new__(TextExtractionExecutor)
    executor.logger = MagicMock()
    executor.runtime_info = {'worker': 'test'}
    executor.show_error = True
    executor.persist = MagicMock()
    executor._route = lambda *_args: (
        'unsupported' if failure == 'unsupported' else 'ocr'
    )

    def execute(**_kwargs):
        if failure == 'exception':
            raise RuntimeError('OCR provider failed')
        return None

    executor.pipeline = SimpleNamespace(pipeline_name='default', execute=execute)
    monkeypatch.setattr(
        module, 'fetch_asset_to_temp', lambda _key: ('/doc.tif', 'tiff')
    )
    monkeypatch.setattr(module, 'get_frames_from_file', lambda *_args: ['frame'])
    monkeypatch.setattr(module, 'torch_gc', lambda: None)
    parameters = {
        'job_id': 'node-ocr',
        'ref_id': 'Doc.tif',
        'ref_type': 'extractor_document',
        'payload': {
            'op_params': {
                'parse_mode': 'ocr',
                **({'strict_result': True} if strict_result else {}),
            }
        },
    }
    docs = [SimpleNamespace(asset_key='s3://bucket/doc.tif', pages=None)]
    if strict_result:
        with pytest.raises((RuntimeError, ValueError), match='OCR|supports'):
            executor.extract(docs, parameters)
    else:
        result = executor.extract(docs, parameters)
        assert result['status'] == ('failed' if failure == 'empty' else 'error')
    assert not executor.persist.called
