import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from marie.executor.text import text_extraction_executor as executor_module
from marie.executor.text.text_extraction_executor import TextExtractionExecutor
from marie.extraction import result_adapter
from marie.pipe import extract_pipeline as pipeline_module
from marie.pipe.extract_pipeline import ExtractPipeline
from tests.unit.executor.text.test_format_router import (
    _bare_executor,
    _docs,
    _params,
    _StubPlugins,
)


def _enable_tracking(executor: TextExtractionExecutor) -> None:
    executor.pipeline = None
    executor.storage_enabled = False
    executor.asset_tracking_enabled = True
    executor.asset_tracker = MagicMock()
    executor._get_upstream_versions = lambda *_args: []
    executor._get_upstream_asset_tuples = lambda *_args: []


def test_tracks_each_returned_uri_once_without_rebuilding_file_metadata() -> None:
    executor = object.__new__(TextExtractionExecutor)
    executor.logger = logging.getLogger(__name__)
    _enable_tracking(executor)
    metadata_uri = 's3://results-bucket/actual-upload/metadata-result.json'
    pdf_uri = 's3://results-bucket/actual-upload/pdf/results.pdf'
    results = {
        'ocr': [{'lines': [{'text': 'Hello'}]}],
        '_stored_asset_uris': [metadata_uri, pdf_uri, metadata_uri],
    }

    executor.persist(
        'document.pdf',
        'extract',
        results,
        job_id='job-1',
        dag_id='dag-1',
        node_task_id='extract-1',
    )

    recorded = executor.asset_tracker.record_materializations.call_args.kwargs
    assets = recorded['assets']
    assert len(assets) == 2
    assert [asset['uri'] for asset in assets] == [metadata_uri, pdf_uri]
    assert [asset['asset_key'] for asset in assets] == [metadata_uri, pdf_uri]
    for asset in assets:
        assert asset.get('size_bytes') is None
        assert asset.get('checksum') is None
        assert asset['version'] == ''
    assert recorded['job_id'] == 'job-1'
    assert recorded['dag_id'] == 'dag-1'
    assert recorded['node_task_id'] == 'extract-1'
    assert '_stored_asset_uris' not in results


@pytest.mark.parametrize('uploaded', [None, []])
def test_does_not_track_files_without_confirmed_uploads(
    tmp_path: Path, uploaded: list[str] | None
) -> None:
    executor = object.__new__(TextExtractionExecutor)
    executor.logger = logging.getLogger(__name__)
    _enable_tracking(executor)
    (tmp_path / 'document.meta.json').write_text('{}')

    executor.persist(
        'document',
        'extract',
        {
            'work_dir': str(tmp_path),
            'assets': uploaded,
            'ocr': [{'lines': [{'text': 'Hello'}]}],
            '_stored_asset_uris': uploaded,
        },
        job_id='job-1',
    )

    executor.asset_tracker.record_materializations.assert_not_called()


@pytest.mark.parametrize('after_upload', ['replace', 'delete'])
def test_semantic_extraction_uses_upload_results_after_local_files_change(
    monkeypatch: pytest.MonkeyPatch,
    after_upload: str,
) -> None:
    monkeypatch.setenv('MARIE_S3_BUCKET', 'results-bucket')
    executor = _bare_executor(plugins=_StubPlugins(formats=('docx',)))
    _enable_tracking(executor)
    executor.persist = TextExtractionExecutor.persist.__get__(executor)
    monkeypatch.setattr(
        executor_module,
        'fetch_asset_to_temp',
        lambda _key: ('/tmp/source.docx', 'docx'),
    )
    uploaded = {}
    local_roots = []

    def upload(ref_id: str, ref_type: str, root: str) -> list[str]:
        local_roots.append(Path(root))
        for path in Path(root).rglob('*'):
            if path.is_file():
                uri = 's3://results-bucket/actual-upload/metadata-result.json'
                uploaded[uri] = path.read_bytes()
                if after_upload == 'replace':
                    path.write_bytes(b'replaced after upload')
                else:
                    path.unlink()
        return list(uploaded)

    monkeypatch.setattr(result_adapter, 'store_assets', upload)

    response = executor.extract(_docs('s3://inputs/source.docx'), _params())

    assert response['status'] == 'succeeded'
    recorded = executor.asset_tracker.record_materializations.call_args.kwargs
    assert len(recorded['assets']) == 1
    asset = recorded['assets'][0]
    assert asset['uri'] == 's3://results-bucket/actual-upload/metadata-result.json'
    content = uploaded[asset['uri']]
    assert json.loads(content)['ocr'][0]['lines'][0]['text'] == '# Title'
    assert '_stored_asset_uris' not in json.loads(content)
    assert '_stored_asset_uris' not in response['metadata']
    assert local_roots and all(not root.exists() for root in local_roots)


@pytest.mark.parametrize('uploaded', [None, []])
def test_semantic_metadata_returns_the_upload_result(
    monkeypatch: pytest.MonkeyPatch, uploaded: list[str] | None
) -> None:
    monkeypatch.setattr(result_adapter, 'store_assets', lambda *_args: uploaded)

    assert (
        result_adapter.write_extraction_metadata('document', 'extract', {}) == uploaded
    )


@pytest.mark.parametrize(
    'uploaded', [None, ['s3://results/extract/document/document.meta.json']]
)
@pytest.mark.parametrize('track_assets', [False, True])
def test_ocr_upload_metadata_is_consumed_before_persistence_and_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uploaded: list[str] | None,
    track_assets: bool,
) -> None:
    monkeypatch.setenv('MARIE_S3_BUCKET', 'results')
    pipeline = object.__new__(ExtractPipeline)
    pipeline.logger = logging.getLogger(__name__)
    pipeline.default_pipeline_config = {}
    pipeline.classifier_groups = {}
    pipeline.indexer_groups = {}
    pipeline.pipeline_name = 'default'
    pipeline.ocr_engines = {}
    pipeline.engine_name = 'test'
    pipeline.boundary = lambda _ref_id, frames, *_args, **_kwargs: (frames, {})
    pipeline.segment = lambda _ref_id, frames, *_args, **_kwargs: frames
    pipeline.template_matching = lambda *_args, **_kwargs: {}
    pipeline.execute_classifier_and_indexer_pipeline = lambda *_args: None
    pipeline.render_pdf = lambda *_args: None
    pipeline.render_blobs = lambda *_args: None
    pipeline.render_adlib = lambda *_args: None

    def pack(_ref_id, _ref_type, _root, metadata):
        metadata['assets'] = ['s3://results/extract/document/planned-only.pdf']

    pipeline.pack_assets = pack
    monkeypatch.setattr(
        pipeline_module, 'restore_assets', lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(pipeline_module, 'burst_frames', lambda *_args: None)
    monkeypatch.setattr(pipeline_module, 'ocr_frames', lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pipeline_module, 'store_assets', lambda *_args: uploaded)
    monkeypatch.setattr(
        pipeline_module, 'create_working_dir', lambda *_args, **_kwargs: str(tmp_path)
    )
    pipeline_results = []
    execute_frames = pipeline.execute_frames_pipeline

    def execute_frames_pipeline(*args, **kwargs):
        result = execute_frames(*args, **kwargs)
        pipeline_results.append(dict(result))
        return result

    pipeline.execute_frames_pipeline = execute_frames_pipeline
    executor = _bare_executor(plugins=_StubPlugins())
    _enable_tracking(executor)
    executor.pipeline = pipeline
    executor.asset_tracking_enabled = track_assets
    executor.persist = TextExtractionExecutor.persist.__get__(executor)
    executor.storage_enabled = True
    stored_results = []
    executor.store = lambda **kwargs: stored_results.append(
        dict(kwargs['docs'][0].content)
    )
    monkeypatch.setattr(
        executor_module,
        'fetch_asset_to_temp',
        lambda _key: ('/tmp/source.png', 'png'),
    )
    monkeypatch.setattr(
        executor_module,
        'get_frames_from_file',
        lambda *_args: [np.zeros((2, 2, 3), dtype=np.uint8)],
    )

    response = executor.extract(
        _docs('s3://inputs/source.png'), _params(ref_id='document', parse_mode='ocr')
    )

    assert response['status'] == 'succeeded'
    metadata = response['metadata']

    expected_assets = ['s3://results/extract/document/planned-only.pdf']
    assert pipeline_results[0]['_stored_asset_uris'] == uploaded
    assert metadata['assets'] == expected_assets
    assert '_stored_asset_uris' not in metadata
    persisted = json.loads((tmp_path / 'document.meta.json').read_bytes())
    assert persisted['assets'] == expected_assets
    assert '_stored_asset_uris' not in persisted
    assert len(stored_results) == 1
    assert '_stored_asset_uris' not in stored_results[0]
    if track_assets and uploaded:
        recorded = executor.asset_tracker.record_materializations.call_args.kwargs
        assert [asset['uri'] for asset in recorded['assets']] == uploaded
    else:
        executor.asset_tracker.record_materializations.assert_not_called()
