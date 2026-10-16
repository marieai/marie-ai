from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from PIL import Image

from marie.executor.extract import document_annotator_executor as annotator_module
from marie.extract.annotators.context_provider import ProcessingUnit
from tests.unit.extract.annotators.test_response_handler import (
    _ContextManager,
    _document,
    _FakeEngine,
    _make_annotator,
)


async def test_process_annotation_uses_configured_layout_prompt_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured_root = tmp_path / 'alternate-extract-config'
    captured: dict = {}

    class RecordingAnnotator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def aannotate(self, _document: object, _frames: list) -> None:
            return None

    executor = object.__new__(annotator_module.DocumentAnnotatorExecutor)
    executor.logger = MagicMock()
    executor.root_config_dir = str(configured_root)
    executor.runtime_info = {'worker': 'annotator-test'}
    executor.show_error = True
    executor.asset_tracking_enabled = False
    executor._setup_request = lambda *_args, **_kwargs: None

    source_doc = SimpleNamespace(asset_key='s3://input/document.pdf', pages=None)
    source = tmp_path / 'document.pdf'
    source.write_bytes(b'document')
    metadata_file = tmp_path / 'metadata.json'
    metadata_file.write_text('{"ocr": []}', encoding='utf-8')
    config = OmegaConf.create(
        {'annotators': {'page-parser': {'enabled': True, 'model_config': {}}}}
    )
    monkeypatch.setattr(annotator_module, 'layout_config', lambda *_args: config)
    monkeypatch.setattr(
        annotator_module,
        'docs_from_asset',
        lambda *_args, **_kwargs: ([source_doc], str(source)),
    )
    monkeypatch.setattr(annotator_module, 'frames_from_docs', lambda _docs: [])
    monkeypatch.setattr(
        annotator_module,
        'prepare_asset_directory',
        lambda **_kwargs: (str(tmp_path), str(tmp_path / 'frames'), str(metadata_file)),
    )
    monkeypatch.setattr(
        annotator_module.MetaReader,
        'from_data',
        lambda **_kwargs: SimpleNamespace(page_count=1),
    )
    monkeypatch.setattr(
        annotator_module, 'get_payload_features', lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(annotator_module, 'MARIE_KERNEL_AVAILABLE', False)
    monkeypatch.setattr(
        annotator_module, 'store_assets', lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(annotator_module, 'torch_gc', lambda: None)

    result = await executor._process_annotation_request(
        [source_doc],
        {
            'job_id': 'annotation-job',
            'ref_id': 'document.pdf',
            'ref_type': 'parse',
            'payload': {'op_params': {'key': 'page-parser', 'layout': 'custom-layout'}},
        },
        RecordingAnnotator,
    )

    assert result['status'] == 'success'
    assert captured['prompt_dir'] == str(
        configured_root / 'TID-custom-layout' / 'annotator'
    )


async def test_response_handler_executes_after_executor_releases_runtime_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    annotator = _make_annotator(tmp_path)
    annotator.context_manager = _ContextManager([ProcessingUnit(page_number=1)])
    annotator.engine = _FakeEngine()
    annotate = AsyncMock(wraps=annotator.aannotate)
    monkeypatch.setattr(annotator, 'aannotate', annotate)
    image = Image.new('RGB', (32, 32))
    image.save(tmp_path / 'work' / 'frames' / 'frame_0001.png')
    document = _document()

    executor = object.__new__(annotator_module.DocumentAnnotatorExecutor)
    executor.logger = MagicMock()
    executor.root_config_dir = str(tmp_path / 'config')
    executor.runtime_info = {'worker': 'annotator-test'}
    executor.show_error = True
    executor.asset_tracking_enabled = False
    executor._setup_request = lambda *_args, **_kwargs: None

    source_doc = SimpleNamespace(asset_key='s3://input/document.pdf', pages=None)
    source = tmp_path / 'document.pdf'
    source.write_bytes(b'document')
    metadata_file = tmp_path / 'metadata.json'
    metadata_file.write_text('{"ocr": []}', encoding='utf-8')
    config = OmegaConf.create({'annotators': {'test-ann': annotator.annotator_conf}})
    monkeypatch.setattr(annotator_module, 'layout_config', lambda *_args: config)
    monkeypatch.setattr(
        annotator_module,
        'docs_from_asset',
        lambda *_args, **_kwargs: ([source_doc], str(source)),
    )
    monkeypatch.setattr(annotator_module, 'frames_from_docs', lambda _docs: [image])
    monkeypatch.setattr(
        annotator_module,
        'prepare_asset_directory',
        lambda **_kwargs: (
            str(tmp_path / 'work'),
            annotator.frames_dir,
            str(metadata_file),
        ),
    )
    monkeypatch.setattr(
        annotator_module.MetaReader, 'from_data', lambda **_kwargs: document
    )
    monkeypatch.setattr(
        annotator_module, 'get_payload_features', lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(annotator_module, 'MARIE_KERNEL_AVAILABLE', False)
    monkeypatch.setattr(
        annotator_module, 'store_assets', lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(annotator_module, 'torch_gc', lambda: None)

    result = await executor._process_annotation_request(
        [source_doc],
        {
            'job_id': 'annotation-job',
            'ref_id': 'document.pdf',
            'ref_type': 'parse',
            'payload': {'op_params': {'key': 'test-ann', 'layout': 'custom-layout'}},
        },
        lambda **_kwargs: annotator,
    )

    annotate.assert_awaited_once_with(document, [])
    assert result['status'] == 'success', result
    output_dir = tmp_path / 'work' / 'agent-output' / 'test-ann'
    assert (output_dir / 'frame_0001.json').exists()
    assert (output_dir / '_SUCCESS.yaml').exists()
