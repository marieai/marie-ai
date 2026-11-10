import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from marie.executor.extract import document_annotator_executor as annotator_module


@pytest.fixture
def executor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('MARIE_S3_BUCKET', 'results-bucket')
    instance = object.__new__(annotator_module.DocumentAnnotatorExecutor)
    instance.logger = MagicMock()
    instance.asset_tracking_enabled = True
    instance.asset_tracker = MagicMock()
    instance._get_upstream_versions = lambda *_args: []
    instance._get_upstream_asset_tuples = lambda *_args: []
    return instance


def record_outputs(executor, root: Path, stored_uris: list[str] | None) -> None:
    executor._record_annotation_assets(
        job_id='annotation-job',
        ref_id='Invoice.PDF',
        ref_type='LBX/Document',
        op_key='claims',
        op_layout='custom-layout',
        page_count=2,
        root_asset_dir=str(root),
        stored_uris=stored_uris,
        dag_id='annotation-dag',
        node_task_id='claims-task',
        partition_key='partition-1',
    )


def test_records_saved_file_uri_actual_size_and_checksum(
    executor, tmp_path: Path
) -> None:
    output = tmp_path / 'agent-output' / 'claims'
    output.mkdir(parents=True)
    content = b'{"claims": [1, 2]}'
    (output / 'frame_0001.json').write_bytes(content)
    (output / '_SUCCESS.yaml').write_text('files: [frame_0001.json]')
    uri = 's3://results-bucket/lbx_document/invoice/agent-output/claims/frame_0001.json'

    record_outputs(executor, tmp_path, [uri])

    args = executor.asset_tracker.record_materializations.call_args.kwargs
    asset = args['assets'][0]
    assert asset['uri'] == uri
    assert asset['size_bytes'] == len(content)
    assert asset['checksum'] == hashlib.sha256(content).hexdigest()
    assert asset['metadata']['checksum_scope'] == 'file'
    assert asset['metadata']['file_count'] == 1
    assert args['dag_id'] == 'annotation-dag'
    assert args['node_task_id'] == 'claims-task'
    assert args['partition_key'] == 'partition-1'

    (output / 'frame_0001.json').write_bytes(b'{"claims": [3, 4]}')
    record_outputs(executor, tmp_path, [uri])
    updated = executor.asset_tracker.record_materializations.call_args.kwargs['assets'][
        0
    ]
    assert updated['version'] != asset['version']


def test_multiple_outputs_link_to_folder_with_a_content_manifest(
    executor, tmp_path: Path
) -> None:
    output = tmp_path / 'agent-output' / 'claims'
    (output / 'nested').mkdir(parents=True)
    (output / 'nested' / 'result.json').write_bytes(b'{}')
    (output / 'page 1.json').write_bytes(b'[]')
    # An upstream annotator's files must not contribute to this materialization.
    (output.parent / 'upstream.json').write_bytes(b'upstream')
    prefix = 's3://results-bucket/lbx_document/invoice/agent-output/claims/'

    record_outputs(
        executor, tmp_path, [prefix + 'nested/result.json', prefix + 'page 1.json']
    )

    asset = executor.asset_tracker.record_materializations.call_args.kwargs['assets'][0]
    assert asset['uri'] == prefix
    assert asset['size_bytes'] == 4
    assert asset['metadata']['checksum_scope'] == 'manifest'
    assert asset['metadata']['file_count'] == 2
    assert [entry['path'] for entry in asset['metadata']['files']] == [
        'nested/result.json',
        'page 1.json',
    ]
    assert (
        asset['metadata']['files'][0]['checksum'] == hashlib.sha256(b'{}').hexdigest()
    )

    record_outputs(
        executor, tmp_path, [prefix + 'page 1.json', prefix + 'nested/result.json']
    )
    repeated = executor.asset_tracker.record_materializations.call_args.kwargs[
        'assets'
    ][0]
    assert repeated['version'] == asset['version']
    assert repeated['checksum'] == asset['checksum']


@pytest.mark.parametrize(
    'stored_uris',
    [
        None,
        [],
        ['s3://results-bucket/lbx_document/invoice'],
        [
            's3://results-bucket/lbx_document/invoice/agent-output/claims-other/result.json'
        ],
        ['s3://other-bucket/lbx_document/invoice/agent-output/claims/result.json'],
    ],
)
def test_missing_or_unconfirmed_uploads_are_not_materialized(
    executor, tmp_path: Path, stored_uris
) -> None:
    output = tmp_path / 'agent-output' / 'claims'
    output.mkdir(parents=True)
    (output / 'result.json').write_bytes(b'{}')

    record_outputs(executor, tmp_path, stored_uris)

    executor.asset_tracker.record_materializations.assert_not_called()


def test_partial_upload_is_not_materialized(executor, tmp_path: Path) -> None:
    output = tmp_path / 'agent-output' / 'claims'
    output.mkdir(parents=True)
    (output / 'one.json').write_bytes(b'{}')
    (output / 'two.json').write_bytes(b'[]')

    record_outputs(
        executor,
        tmp_path,
        ['s3://results-bucket/lbx_document/invoice/agent-output/claims/one.json'],
    )

    executor.asset_tracker.record_materializations.assert_not_called()


def test_success_marker_alone_is_not_an_output(executor, tmp_path: Path) -> None:
    output = tmp_path / 'agent-output' / 'claims'
    output.mkdir(parents=True)
    (output / '_SUCCESS.yaml').write_text('files: []')

    record_outputs(
        executor,
        tmp_path,
        ['s3://results-bucket/lbx_document/invoice/agent-output/claims/_SUCCESS.yaml'],
    )

    executor.asset_tracker.record_materializations.assert_not_called()


@pytest.mark.parametrize('upload_succeeds', [True, False])
async def test_annotation_records_assets_after_upload(
    executor, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, upload_succeeds: bool
) -> None:
    executor.root_config_dir = str(tmp_path / 'config')
    executor.runtime_info = {'worker': 'annotation-test'}
    executor.show_error = True
    executor._setup_request = lambda *_args, **_kwargs: None
    source_doc = SimpleNamespace(asset_key='s3://input/document.pdf', pages=None)
    metadata_file = tmp_path / 'metadata.json'
    metadata_file.write_text('{"ocr": []}')
    config = OmegaConf.create({'annotators': {'claims': {'enabled': True}}})
    monkeypatch.setattr(annotator_module, 'layout_config', lambda *_args: config)
    monkeypatch.setattr(
        annotator_module,
        'docs_from_asset',
        lambda *_args, **_kwargs: ([source_doc], str(tmp_path / 'source.pdf')),
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
        lambda **_kwargs: SimpleNamespace(page_count=2),
    )
    monkeypatch.setattr(
        annotator_module, 'get_payload_features', lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(annotator_module, 'MARIE_KERNEL_AVAILABLE', False)
    monkeypatch.setattr(annotator_module, 'torch_gc', lambda: None)

    class Annotator:
        def __init__(self, **_kwargs):
            pass

        async def aannotate(self, _document, _frames):
            output = tmp_path / 'agent-output' / 'claims'
            output.mkdir(parents=True)
            (output / 'result.json').write_bytes(b'{"saved": true}')

    uri = 's3://results-bucket/lbx_document/invoice/agent-output/claims/result.json'

    def upload(*_args, **_kwargs):
        executor.asset_tracker.record_materializations.assert_not_called()
        return [uri] if upload_succeeds else None

    monkeypatch.setattr(annotator_module, 'store_assets', upload)
    result = await executor._process_annotation_request(
        [source_doc],
        {
            'job_id': 'annotation-job',
            'ref_id': 'Invoice.PDF',
            'ref_type': 'LBX/Document',
            'payload': {'op_params': {'key': 'claims', 'layout': 'custom-layout'}},
        },
        Annotator,
    )

    assert result['status'] == 'success', result
    if upload_succeeds:
        assert (
            executor.asset_tracker.record_materializations.call_args.kwargs['assets'][
                0
            ]['uri']
            == uri
        )
    else:
        executor.asset_tracker.record_materializations.assert_not_called()
